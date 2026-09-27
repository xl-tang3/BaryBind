import sys
import argparse

import torch
import torch.distributed as dist
from easydict import EasyDict as edict
from torch.nn import functional as F

from utils.args import get_args, logging_cfgs
from utils.initialize import initialize
from utils.build_model import build_model

from demo_search_core import (
    load_demo_index,
    load_condition_feats,
    search_text_to_video,
    search_text_to_audio,
    search_text_to_subtitle,
    search_text_to_video_subtitle,
    search_text_to_video_audio_subtitle,
    search_video_to_text,
    search_video_to_audio,
    search_video_to_subtitle,
    search_video_to_text_subtitle,
    search_video_to_text_audio_subtitle,
    print_results,
)


def is_dist():
    return dist.is_available() and dist.is_initialized()


def get_rank():
    if is_dist():
        return dist.get_rank()
    return 0


def parse_demo_args():
    parser = argparse.ArgumentParser(add_help=False)

    parser.add_argument(
        "--demo_index_dir",
        type=str,
        default="./demo_index",
    )

    parser.add_argument(
        "--demo_mode",
        type=str,
        default="t2v",
        choices=[
            "t2v",
            "t2a",
            "t2s",
            "t2vs",
            "t2vas",
            "v2t",
            "v2a",
            "v2s",
            "v2ts",
            "v2tas",
        ],
    )

    parser.add_argument(
        "--query",
        type=str,
        default="",
    )

    parser.add_argument(
        "--query_input",
        type=str,
        default="",
    )

    parser.add_argument(
        "--video_id",
        type=str,
        default="",
    )

    parser.add_argument(
        "--topk",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--wv",
        type=float,
        default=0.7,
    )

    parser.add_argument(
        "--ws",
        type=float,
        default=0.3,
    )

    parser.add_argument(
        "--rerank_topk",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--small_batch",
        type=int,
        default=32,
    )

    demo_args, remaining = parser.parse_known_args()

    sys.argv = [sys.argv[0]] + remaining

    return demo_args


@torch.no_grad()
def encode_query_text(
    model,
    query,
    return_tokens=False,
):
    if not query:
        raise ValueError("--query is empty.")

    batch = edict(
        {
            "raw_captions": [query],
        }
    )

    feat_b = model.batch_get(
        batch,
        "feat_b",
    )

    feat_b = F.normalize(
        feat_b,
        dim=-1,
    )

    if return_tokens:
        caption_tokens = model.batch_get(
            batch,
            "caption_tokens",
        )

        return (
            feat_b,
            caption_tokens.input_ids,
            caption_tokens.attention_mask,
        )

    return feat_b


def load_model():
    args = get_args()

    args.run_cfg.mode = "testing"

    initialize(args)
    logging_cfgs(args)

    model, _, _ = build_model(args)

    model.eval()

    return model


def check_video_id(video_id, demo_mode):
    if not video_id:
        raise ValueError(
            f"--video_id is required for --demo_mode {demo_mode}."
        )


def main():
    demo_args = parse_demo_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    index = load_demo_index(
        index_dir=demo_args.demo_index_dir,
        device=device,
    )

    need_model = demo_args.demo_mode in [
        "t2v",
        "t2a",
        "t2s",
        "t2vs",
        "t2vas",
    ]

    model = None

    if need_model:
        model = load_model()

    # =====================================================
    # Text -> Video
    # =====================================================
    if demo_args.demo_mode == "t2v":
        query_feat = encode_query_text(
            model=model,
            query=demo_args.query,
            return_tokens=False,
        )

        results = search_text_to_video(
            query_feat=query_feat,
            index=index,
            topk=demo_args.topk,
        )

        if get_rank() == 0:
            print_results(
                f"Text-to-Bary-to-Video | query: {demo_args.query_input}",
                results,
            )

    # =====================================================
    # Text -> Audio
    # =====================================================
    elif demo_args.demo_mode == "t2a":
        query_feat = encode_query_text(
            model=model,
            query=demo_args.query,
            return_tokens=False,
        )

        results = search_text_to_audio(
            query_feat=query_feat,
            index=index,
            topk=demo_args.topk,
        )

        if get_rank() == 0:
            print_results(
                f"Text-to-Bary-to-Audio | query: {demo_args.query_input}",
                results,
            )

    # =====================================================
    # Text -> Subtitle
    # =====================================================
    elif demo_args.demo_mode == "t2s":
        query_feat = encode_query_text(
            model=model,
            query=demo_args.query,
            return_tokens=False,
        )

        results = search_text_to_subtitle(
            query_feat=query_feat,
            index=index,
            topk=demo_args.topk,
        )

        if get_rank() == 0:
            print_results(
                f"Text-to-Bary-to-Subtitle | query: {demo_args.query_input}",
                results,
            )

    # =====================================================
    # Text -> Video + Subtitle
    # =====================================================
    elif demo_args.demo_mode == "t2vs":
        query_feat = encode_query_text(
            model=model,
            query=demo_args.query,
            return_tokens=False,
        )

        results = search_text_to_video_subtitle(
            query_feat=query_feat,
            index=index,
            topk=demo_args.topk,
            wv=demo_args.wv,
            ws=demo_args.ws,
        )

        if get_rank() == 0:
            print_results(
                (
                    f"Text-to-Bary-to-Video-Subtitle | "
                    f"wv={demo_args.wv}, "
                    f"ws={demo_args.ws} | "
                    f"query: {demo_args.query_input}"
                ),
                results,
            )

    # =====================================================
    # Text -> Video + Audio + Subtitle
    # Keep the original coarse retrieval + ITM reranking.
    # =====================================================
    elif demo_args.demo_mode == "t2vas":
        (
            query_feat,
            query_input_ids,
            query_attention_mask,
        ) = encode_query_text(
            model=model,
            query=demo_args.query,
            return_tokens=True,
        )

        condition_feats_tvas = load_condition_feats(
            index_dir=demo_args.demo_index_dir,
            task="tvas",
            device=device,
        )

        results = search_text_to_video_audio_subtitle(
            model=model,
            query_feat=query_feat,
            query_input_ids=query_input_ids,
            query_attention_mask=query_attention_mask,
            condition_feats_tvas=condition_feats_tvas,
            index=index,
            topk=demo_args.topk,
            rerank_topk=demo_args.rerank_topk,
            small_batch=demo_args.small_batch,
        )

        if get_rank() == 0:
            print_results(
                (
                    f"Text-to-Bary-to-Video-Audio-Subtitle Composite | "
                    f"rerank_topk={demo_args.rerank_topk} | "
                    f"query: {demo_args.query_input}"
                ),
                results,
            )

    # =====================================================
    # Video -> Text
    # =====================================================
    elif demo_args.demo_mode == "v2t":
        check_video_id(
            demo_args.video_id,
            demo_args.demo_mode,
        )

        results = search_video_to_text(
            video_id=demo_args.video_id,
            index=index,
            topk=demo_args.topk,
        )

        if get_rank() == 0:
            print_results(
                f"Video-to-Bary-to-Text | video_id: {demo_args.video_id}",
                results,
            )

    # =====================================================
    # Video -> Audio
    # =====================================================
    elif demo_args.demo_mode == "v2a":
        check_video_id(
            demo_args.video_id,
            demo_args.demo_mode,
        )

        results = search_video_to_audio(
            video_id=demo_args.video_id,
            index=index,
            topk=demo_args.topk,
        )

        if get_rank() == 0:
            print_results(
                f"Video-to-Bary-to-Audio | video_id: {demo_args.video_id}",
                results,
            )

    # =====================================================
    # Video -> Subtitle
    # =====================================================
    elif demo_args.demo_mode == "v2s":
        check_video_id(
            demo_args.video_id,
            demo_args.demo_mode,
        )

        results = search_video_to_subtitle(
            video_id=demo_args.video_id,
            index=index,
            topk=demo_args.topk,
        )

        if get_rank() == 0:
            print_results(
                f"Video-to-Bary-to-Subtitle | video_id: {demo_args.video_id}",
                results,
            )

    # =====================================================
    # Video -> Text + Subtitle
    # =====================================================
    elif demo_args.demo_mode == "v2ts":
        check_video_id(
            demo_args.video_id,
            demo_args.demo_mode,
        )

        results = search_video_to_text_subtitle(
            video_id=demo_args.video_id,
            index=index,
            topk=demo_args.topk,
        )

        if get_rank() == 0:
            print_results(
                f"Video-to-Bary-to-Text | video_id: {demo_args.video_id}",
                results["video_to_text"],
            )

            print_results(
                f"Video-to-Bary-to-Subtitle | video_id: {demo_args.video_id}",
                results["video_to_subtitle"],
            )

    # =====================================================
    # Video -> Text + Audio + Subtitle
    # =====================================================
    elif demo_args.demo_mode == "v2tas":

        check_video_id(
            demo_args.video_id,
            demo_args.demo_mode,
        )

        results = search_video_to_text_audio_subtitle(
            video_id=demo_args.video_id,
            index=index,
            topk=demo_args.topk,
        )

        if get_rank() == 0:

            print_results(
                (
                    f"Video-to-Bary-to-Text-Audio-Subtitle | "
                    f"video_id: {demo_args.video_id}"
                ),
                results,
            )


if __name__ == "__main__":
    main()