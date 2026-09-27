import os
import sys
import json
import argparse
import torch
import torch.distributed as dist
from easydict import EasyDict as edict
from tqdm import tqdm
from torch.nn import functional as F

from utils.args import get_args, logging_cfgs
from utils.initialize import initialize
from utils.build_model import build_model
from utils.build_dataloader import create_val_dataloaders
from utils.distributed import all_gather_list, ddp_allgather


def is_dist():
    return dist.is_available() and dist.is_initialized()


def get_rank():
    if is_dist():
        return dist.get_rank()
    return 0


def safe_all_gather_list(x):
    if is_dist():
        gathered = all_gather_list(x)
        return [item for sublist in gathered for item in sublist]
    return x


def safe_ddp_allgather(x):
    if is_dist():
        return ddp_allgather(x)
    return x


def get_field(batch, names, default=None):
    for name in names:
        if isinstance(batch, dict) and name in batch:
            return batch[name]
        if hasattr(batch, name):
            return getattr(batch, name)
    return default


def flatten_one_level(x):
    if x is None:
        return None

    if isinstance(x, tuple):
        x = list(x)

    if not isinstance(x, list):
        return [x]

    if len(x) > 0 and isinstance(x[0], list):
        return [item for sublist in x for item in sublist]

    return x


def build_text_meta(batch):
    ids = get_field(batch, ["ids"], [])
    ids = flatten_one_level(ids)

    raw_captions = get_field(
        batch,
        ["raw_captions", "captions", "caption", "texts", "text"],
        None,
    )

    ids_txt = get_field(batch, ["ids_txt"], None)

    raw_captions_original = raw_captions
    raw_captions = flatten_one_level(raw_captions)

    if raw_captions is None:
        raw_captions = [""] * len(ids)

    if ids_txt is not None:
        text_video_ids = flatten_one_level(ids_txt)
    else:
        if (
            isinstance(raw_captions_original, list)
            and len(raw_captions_original) > 0
            and isinstance(raw_captions_original[0], list)
        ):
            text_video_ids = []
            for vid, caps in zip(ids, raw_captions_original):
                text_video_ids.extend([vid] * len(caps))
        else:
            text_video_ids = ids

    if len(text_video_ids) != len(raw_captions):
        if len(ids) == len(raw_captions):
            text_video_ids = ids
        else:
            text_video_ids = text_video_ids[: len(raw_captions)] + [""] * max(
                0, len(raw_captions) - len(text_video_ids)
            )

    text_meta = []

    for i, (vid, cap) in enumerate(zip(text_video_ids, raw_captions)):
        text_meta.append(
            {
                "text_index": i,
                "video_id": str(vid),
                "caption": "" if cap is None else str(cap),
            }
        )

    return text_meta


def build_video_meta(batch):
    ids = get_field(batch, ["ids"], [])
    ids = flatten_one_level(ids)

    subtitles = get_field(
        batch,
        [
            "raw_subtitles",
            "subtitles",
            "subtitle",
            "subtitle_texts",
            "asr",
            "raw_asr",
        ],
        None,
    )
    subtitles = flatten_one_level(subtitles)

    video_paths = get_field(
        batch,
        [
            "video_paths",
            "video_path",
            "vision_paths",
            "vision_path",
            "video_fnames",
            "video_fname",
        ],
        None,
    )
    video_paths = flatten_one_level(video_paths)

    if subtitles is None:
        subtitles = [""] * len(ids)

    if video_paths is None:
        video_paths = [""] * len(ids)

    if len(subtitles) != len(ids):
        subtitles = subtitles[: len(ids)] + [""] * max(0, len(ids) - len(subtitles))

    if len(video_paths) != len(ids):
        video_paths = video_paths[: len(ids)] + [""] * max(0, len(ids) - len(video_paths))

    video_meta = []

    for i, vid in enumerate(ids):
        video_meta.append(
            {
                "video_index": i,
                "video_id": str(vid),
                "subtitle": "" if subtitles[i] is None else str(subtitles[i]),
                "video_path": "" if video_paths[i] is None else str(video_paths[i]),
            }
        )

    return video_meta


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


@torch.no_grad()
def build_index(model, val_loader, task, output_dir):
    model.eval()

    text_feats = []
    video_feats = []
    audio_feats = []
    subtitle_feats = []

    condition_feats_store = {}

    text_meta_all = []
    video_meta_all = []

    for batch in tqdm(val_loader, desc="Building demo index"):
        batch = edict(batch)

        cur_text_meta = build_text_meta(batch)
        cur_video_meta = build_video_meta(batch)

        output = model(batch, task, compute_loss=False)

        if "feat_b" not in output:
            raise RuntimeError("model output does not contain feat_b.")

        if "feat_v" not in output:
            raise RuntimeError("model output does not contain feat_v.")

        text_feats.append(output["feat_b"].detach())
        video_feats.append(output["feat_v"].detach())

        if "feat_a" in output:
            audio_feats.append(output["feat_a"].detach())

        if "feat_s" in output:
            subtitle_feats.append(output["feat_s"].detach())

        for key, value in output.items():
            if key.startswith("condition_feats_"):
                if key not in condition_feats_store:
                    condition_feats_store[key] = []
                condition_feats_store[key].append(value.detach())

        text_meta_all.extend(cur_text_meta)
        video_meta_all.extend(cur_video_meta)

    text_feats = torch.cat(text_feats, dim=0)
    video_feats = torch.cat(video_feats, dim=0)

    text_feats = safe_ddp_allgather(text_feats)
    video_feats = safe_ddp_allgather(video_feats)

    if len(audio_feats) > 0:
        audio_feats = torch.cat(audio_feats, dim=0)
        audio_feats = safe_ddp_allgather(audio_feats)
    else:
        audio_feats = None

    if len(subtitle_feats) > 0:
        subtitle_feats = torch.cat(subtitle_feats, dim=0)
        subtitle_feats = safe_ddp_allgather(subtitle_feats)
    else:
        subtitle_feats = None

    for key in list(condition_feats_store.keys()):
        condition_feats_store[key] = torch.cat(condition_feats_store[key], dim=0)
        condition_feats_store[key] = safe_ddp_allgather(condition_feats_store[key])

    text_meta_all = safe_all_gather_list(text_meta_all)
    video_meta_all = safe_all_gather_list(video_meta_all)

    if get_rank() == 0:
        os.makedirs(output_dir, exist_ok=True)

        text_feats = F.normalize(text_feats.cpu(), dim=-1)
        video_feats = F.normalize(video_feats.cpu(), dim=-1)

        torch.save(text_feats, os.path.join(output_dir, "text_feats.pt"))
        torch.save(video_feats, os.path.join(output_dir, "video_feats.pt"))

        if audio_feats is not None:
            audio_feats = F.normalize(audio_feats.cpu(), dim=-1)
            torch.save(audio_feats, os.path.join(output_dir, "audio_feats.pt"))

        if subtitle_feats is not None:
            subtitle_feats = F.normalize(subtitle_feats.cpu(), dim=-1)
            torch.save(subtitle_feats, os.path.join(output_dir, "subtitle_feats.pt"))

        for key, value in condition_feats_store.items():
            value = value.cpu()
            torch.save(value, os.path.join(output_dir, f"{key}.pt"))

        save_json(text_meta_all, os.path.join(output_dir, "text_meta.json"))
        save_json(video_meta_all, os.path.join(output_dir, "video_meta.json"))

        print("")
        print("=" * 100)
        print(f"Saved demo index to: {output_dir}")
        print(f"text_feats      : {tuple(text_feats.shape)}")
        print(f"video_feats     : {tuple(video_feats.shape)}")
        print(f"audio_feats     : {None if audio_feats is None else tuple(audio_feats.shape)}")
        print(f"subtitle_feats  : {None if subtitle_feats is None else tuple(subtitle_feats.shape)}")

        for key, value in condition_feats_store.items():
            print(f"{key:<22}: {tuple(value.shape)}")

        print(f"text_meta       : {len(text_meta_all)}")
        print(f"video_meta      : {len(video_meta_all)}")
        print("=" * 100)


def parse_demo_args():
    parser = argparse.ArgumentParser(add_help=False)

    parser.add_argument("--demo_output_dir", type=str, default="./demo_index")
    parser.add_argument("--demo_loader_key", type=str, default="")
    parser.add_argument("--demo_task", type=str, default="ret%tv")

    demo_args, remaining = parser.parse_known_args()

    sys.argv = [sys.argv[0]] + remaining

    return demo_args


def main():
    demo_args = parse_demo_args()

    args = get_args()
    args.run_cfg.mode = "testing"

    initialize(args)
    logging_cfgs(args)

    model, _, _ = build_model(args)

    val_loaders = create_val_dataloaders(args)

    if demo_args.demo_loader_key:
        if demo_args.demo_loader_key not in val_loaders:
            raise KeyError(
                f"Cannot find loader key: {demo_args.demo_loader_key}. "
                f"Available keys: {list(val_loaders.keys())}"
            )
        loader_key = demo_args.demo_loader_key
    else:
        loader_key = list(val_loaders.keys())[0]

    if get_rank() == 0:
        print(f"Using val loader: {loader_key}")
        print(f"Using demo task : {demo_args.demo_task}")
        print(f"Saving index to : {demo_args.demo_output_dir}")

    build_index(
        model=model,
        val_loader=val_loaders[loader_key],
        task=demo_args.demo_task,
        output_dir=demo_args.demo_output_dir,
    )


if __name__ == "__main__":
    main()