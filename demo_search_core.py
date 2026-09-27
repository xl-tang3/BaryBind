import os
import json
import datetime
import torch
from torch.nn import functional as F


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_demo_index(index_dir, device="cpu"):
    index = {}

    text_feat_path = os.path.join(
        index_dir,
        "text_feats.pt",
    )
    video_feat_path = os.path.join(
        index_dir,
        "video_feats.pt",
    )
    audio_feat_path = os.path.join(
        index_dir,
        "audio_feats.pt",
    )
    subtitle_feat_path = os.path.join(
        index_dir,
        "subtitle_feats.pt",
    )

    text_meta_path = os.path.join(
        index_dir,
        "text_meta.json",
    )
    video_meta_path = os.path.join(
        index_dir,
        "video_meta.json",
    )

    if not os.path.exists(text_feat_path):
        raise FileNotFoundError(
            f"Missing {text_feat_path}"
        )

    if not os.path.exists(video_feat_path):
        raise FileNotFoundError(
            f"Missing {video_feat_path}"
        )

    if not os.path.exists(text_meta_path):
        raise FileNotFoundError(
            f"Missing {text_meta_path}"
        )

    if not os.path.exists(video_meta_path):
        raise FileNotFoundError(
            f"Missing {video_meta_path}"
        )

    index["text_feats"] = F.normalize(
        torch.load(
            text_feat_path,
            map_location=device,
        ),
        dim=-1,
    )

    index["video_feats"] = F.normalize(
        torch.load(
            video_feat_path,
            map_location=device,
        ),
        dim=-1,
    )

    index["text_meta"] = load_json(
        text_meta_path
    )

    index["video_meta"] = load_json(
        video_meta_path
    )

    if os.path.exists(audio_feat_path):
        index["audio_feats"] = F.normalize(
            torch.load(
                audio_feat_path,
                map_location=device,
            ),
            dim=-1,
        )
    else:
        index["audio_feats"] = None

    if os.path.exists(subtitle_feat_path):
        index["subtitle_feats"] = F.normalize(
            torch.load(
                subtitle_feat_path,
                map_location=device,
            ),
            dim=-1,
        )
    else:
        index["subtitle_feats"] = None

    return index


def load_condition_feats(
    index_dir,
    task,
    device="cpu",
):
    path = os.path.join(
        index_dir,
        f"condition_feats_{task}.pt",
    )

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing {path}. "
            f"Please rebuild index with "
            f"--demo_task ret%{task}"
        )

    return torch.load(
        path,
        map_location=device,
    )


def topk_from_scores(
    scores,
    topk,
):
    if scores.dim() != 1:
        scores = scores.reshape(-1)

    topk = min(
        int(topk),
        scores.numel(),
    )

    if topk <= 0:
        return [], []

    top_scores, top_indices = torch.topk(
        scores,
        k=topk,
    )

    return (
        top_scores.detach().cpu().tolist(),
        top_indices.detach().cpu().tolist(),
    )


def find_video_index(
    index,
    video_id,
):
    for i, item in enumerate(
        index["video_meta"]
    ):
        current_video_id = item.get(
            "video_id",
            "",
        )

        if str(current_video_id) == str(video_id):
            return i

    return None


def build_caption_lookup(index):
    caption_lookup = {}

    for item in index.get(
        "text_meta",
        [],
    ):
        video_id = str(
            item.get(
                "video_id",
                "",
            )
        )

        caption = item.get(
            "caption",
            "",
        )

        if (
            video_id
            and video_id not in caption_lookup
        ):
            caption_lookup[video_id] = caption

    return caption_lookup


def get_video_result_item(
    index,
    idx,
    rank,
    score,
    caption_lookup=None,
):
    if idx < 0 or idx >= len(
        index["video_meta"]
    ):
        raise IndexError(
            f"Video metadata index out of range: {idx}"
        )

    item = index["video_meta"][idx]

    video_id = item.get(
        "video_id",
        "",
    )

    caption = item.get(
        "caption",
        "",
    )

    if not caption:
        if caption_lookup is None:
            caption_lookup = build_caption_lookup(
                index
            )

        caption = caption_lookup.get(
            str(video_id),
            "",
        )

    return {
        "rank": rank,
        "score": float(score),
        "video_id": video_id,
        "video_path": item.get(
            "video_path",
            "",
        ),
        "caption": caption,
        "subtitle": item.get(
            "subtitle",
            "",
        ),
    }


def format_video_results(
    index,
    top_scores,
    top_indices,
):
    results = []

    caption_lookup = build_caption_lookup(
        index
    )

    for rank, (score, idx) in enumerate(
        zip(
            top_scores,
            top_indices,
        ),
        start=1,
    ):
        results.append(
            get_video_result_item(
                index=index,
                idx=int(idx),
                rank=rank,
                score=score,
                caption_lookup=caption_lookup,
            )
        )

    return results


def format_text_results(
    index,
    top_scores,
    top_indices,
):
    results = []

    for rank, (score, idx) in enumerate(
        zip(
            top_scores,
            top_indices,
        ),
        start=1,
    ):
        if idx < 0 or idx >= len(
            index["text_meta"]
        ):
            raise IndexError(
                f"Text metadata index out of range: {idx}"
            )

        item = index["text_meta"][idx]

        results.append(
            {
                "rank": rank,
                "score": float(score),
                "video_id": item.get(
                    "video_id",
                    "",
                ),
                "video_path": item.get(
                    "video_path",
                    "",
                ),
                "caption": item.get(
                    "caption",
                    "",
                ),
                "subtitle": item.get(
                    "subtitle",
                    "",
                ),
            }
        )

    return results


def prepare_query_feat(
    query_feat,
):
    query_feat = F.normalize(
        query_feat,
        dim=-1,
    )

    if query_feat.dim() == 1:
        query_feat = query_feat.unsqueeze(0)

    return query_feat


def search_text_to_video(
    query_feat,
    index,
    topk=10,
):
    query_feat = prepare_query_feat(
        query_feat
    )

    video_feats = F.normalize(
        index["video_feats"].to(
            query_feat.device
        ),
        dim=-1,
    )

    scores = torch.matmul(
        query_feat,
        video_feats.T,
    ).squeeze(0)

    top_scores, top_indices = topk_from_scores(
        scores,
        topk,
    )

    return format_video_results(
        index,
        top_scores,
        top_indices,
    )


def search_text_to_audio(
    query_feat,
    index,
    topk=10,
):
    if index.get(
        "audio_feats",
        None,
    ) is None:
        raise RuntimeError(
            "audio_feats.pt not found."
        )

    query_feat = prepare_query_feat(
        query_feat
    )

    audio_feats = F.normalize(
        index["audio_feats"].to(
            query_feat.device
        ),
        dim=-1,
    )

    scores = torch.matmul(
        query_feat,
        audio_feats.T,
    ).squeeze(0)

    top_scores, top_indices = topk_from_scores(
        scores,
        topk,
    )

    return format_video_results(
        index,
        top_scores,
        top_indices,
    )


def search_text_to_subtitle(
    query_feat,
    index,
    topk=10,
):
    if index.get(
        "subtitle_feats",
        None,
    ) is None:
        raise RuntimeError(
            "subtitle_feats.pt not found."
        )

    query_feat = prepare_query_feat(
        query_feat
    )

    subtitle_feats = F.normalize(
        index["subtitle_feats"].to(
            query_feat.device
        ),
        dim=-1,
    )

    scores = torch.matmul(
        query_feat,
        subtitle_feats.T,
    ).squeeze(0)

    top_scores, top_indices = topk_from_scores(
        scores,
        topk,
    )

    return format_video_results(
        index,
        top_scores,
        top_indices,
    )


def search_text_to_video_subtitle(
    query_feat,
    index,
    topk=10,
    wv=0.7,
    ws=0.3,
):
    """
    Text -> Video + Subtitle late fusion.

    score = wv * text-video similarity
          + ws * text-subtitle similarity
    """
    if index.get(
        "subtitle_feats",
        None,
    ) is None:
        raise RuntimeError(
            "subtitle_feats.pt not found."
        )

    query_feat = prepare_query_feat(
        query_feat
    )

    video_feats = F.normalize(
        index["video_feats"].to(
            query_feat.device
        ),
        dim=-1,
    )

    subtitle_feats = F.normalize(
        index["subtitle_feats"].to(
            query_feat.device
        ),
        dim=-1,
    )

    if video_feats.shape[0] != subtitle_feats.shape[0]:
        raise ValueError(
            "video_feats and subtitle_feats "
            "must have the same number of samples."
        )

    score_v = torch.matmul(
        query_feat,
        video_feats.T,
    ).squeeze(0)

    score_s = torch.matmul(
        query_feat,
        subtitle_feats.T,
    ).squeeze(0)

    weight_sum = float(wv) + float(ws)

    if weight_sum <= 0:
        raise ValueError(
            "wv + ws must be positive."
        )

    normalized_wv = float(wv) / weight_sum
    normalized_ws = float(ws) / weight_sum

    scores = (
        normalized_wv * score_v
        + normalized_ws * score_s
    )

    top_scores, top_indices = topk_from_scores(
        scores,
        topk,
    )

    return format_video_results(
        index,
        top_scores,
        top_indices,
    )


@torch.no_grad()
def search_text_to_video_audio_subtitle(
    model,
    query_feat,
    query_input_ids,
    query_attention_mask,
    condition_feats_tvas,
    index,
    topk=10,
    rerank_topk=100,
    small_batch=32,
):
    """
    Text -> Video + Audio + Subtitle retrieval.

    Step 1:
        Use text-video, text-audio and text-subtitle
        similarities for coarse candidate selection.

    Step 2:
        Use condition_feats_tvas and
        model.compute_slice_scores() for ITM reranking.

    The original TVAS ranking logic is preserved.
    """
    if index.get(
        "audio_feats",
        None,
    ) is None:
        raise RuntimeError(
            "audio_feats.pt not found."
        )

    if index.get(
        "subtitle_feats",
        None,
    ) is None:
        raise RuntimeError(
            "subtitle_feats.pt not found."
        )

    if small_batch <= 0:
        raise ValueError(
            "small_batch must be positive."
        )

    device = query_feat.device

    query_feat = prepare_query_feat(
        query_feat
    )

    video_feats = F.normalize(
        index["video_feats"].to(device),
        dim=-1,
    )

    audio_feats = F.normalize(
        index["audio_feats"].to(device),
        dim=-1,
    )

    subtitle_feats = F.normalize(
        index["subtitle_feats"].to(device),
        dim=-1,
    )

    num_videos = video_feats.shape[0]

    if audio_feats.shape[0] != num_videos:
        raise ValueError(
            "video_feats and audio_feats "
            "must have the same number of samples."
        )

    if subtitle_feats.shape[0] != num_videos:
        raise ValueError(
            "video_feats and subtitle_feats "
            "must have the same number of samples."
        )

    if condition_feats_tvas.shape[0] != num_videos:
        raise ValueError(
            "condition_feats_tvas and video_feats "
            "must have the same number of samples."
        )

    score_v = torch.matmul(
        query_feat,
        video_feats.T,
    ).squeeze(0)

    score_a = torch.matmul(
        query_feat,
        audio_feats.T,
    ).squeeze(0)

    score_s = torch.matmul(
        query_feat,
        subtitle_feats.T,
    ).squeeze(0)

    coarse_scores = (
        score_v
        + score_a
        + score_s
    )

    candidate_num = min(
        int(rerank_topk),
        coarse_scores.numel(),
    )

    if candidate_num <= 0:
        return []

    _, candidate_indices = torch.topk(
        coarse_scores,
        k=candidate_num,
    )

    candidate_indices = candidate_indices.to(
        device
    )

    condition_feats_tvas = condition_feats_tvas.to(
        device
    )

    candidate_condition_feats = (
        condition_feats_tvas[
            candidate_indices
        ]
    )

    query_input_ids = query_input_ids.to(
        device
    )

    query_attention_mask = (
        query_attention_mask.to(device)
    )

    all_itm_scores = []

    for start in range(
        0,
        candidate_num,
        small_batch,
    ):
        end = min(
            start + small_batch,
            candidate_num,
        )

        current_batch_size = end - start

        cur_condition_feats = (
            candidate_condition_feats[
                start:end
            ]
        )

        cur_input_ids = query_input_ids.expand(
            current_batch_size,
            -1,
        )

        cur_attention_mask = (
            query_attention_mask.expand(
                current_batch_size,
                -1,
            )
        )

        cur_scores = model.compute_slice_scores(
            cur_condition_feats,
            cur_input_ids,
            cur_attention_mask,
        )

        cur_scores = cur_scores.reshape(-1)

        all_itm_scores.append(
            cur_scores.detach()
        )

    all_itm_scores = torch.cat(
        all_itm_scores,
        dim=0,
    )

    final_topk = min(
        int(topk),
        all_itm_scores.numel(),
    )

    if final_topk <= 0:
        return []

    top_scores, local_top_indices = torch.topk(
        all_itm_scores,
        k=final_topk,
    )

    global_top_indices = candidate_indices[
        local_top_indices
    ]

    return format_video_results(
        index=index,
        top_scores=(
            top_scores
            .detach()
            .cpu()
            .tolist()
        ),
        top_indices=(
            global_top_indices
            .detach()
            .cpu()
            .tolist()
        ),
    )


def search_video_to_text(
    video_id,
    index,
    topk=10,
):
    video_idx = find_video_index(
        index,
        video_id,
    )

    if video_idx is None:
        raise ValueError(
            f"Cannot find video_id: {video_id}"
        )

    video_feat = index["video_feats"][
        video_idx:video_idx + 1
    ]

    video_feat = F.normalize(
        video_feat,
        dim=-1,
    )

    text_feats = F.normalize(
        index["text_feats"].to(
            video_feat.device
        ),
        dim=-1,
    )

    scores = torch.matmul(
        video_feat,
        text_feats.T,
    ).squeeze(0)

    top_scores, top_indices = topk_from_scores(
        scores,
        topk,
    )

    return format_text_results(
        index,
        top_scores,
        top_indices,
    )


def search_video_to_audio(
    video_id,
    index,
    topk=10,
):
    if index.get(
        "audio_feats",
        None,
    ) is None:
        raise RuntimeError(
            "audio_feats.pt not found."
        )

    video_idx = find_video_index(
        index,
        video_id,
    )

    if video_idx is None:
        raise ValueError(
            f"Cannot find video_id: {video_id}"
        )

    video_feat = index["video_feats"][
        video_idx:video_idx + 1
    ]

    video_feat = F.normalize(
        video_feat,
        dim=-1,
    )

    audio_feats = F.normalize(
        index["audio_feats"].to(
            video_feat.device
        ),
        dim=-1,
    )

    scores = torch.matmul(
        video_feat,
        audio_feats.T,
    ).squeeze(0)

    top_scores, top_indices = topk_from_scores(
        scores,
        topk,
    )

    return format_video_results(
        index,
        top_scores,
        top_indices,
    )


def search_video_to_subtitle(
    video_id,
    index,
    topk=10,
):
    if index.get(
        "subtitle_feats",
        None,
    ) is None:
        raise RuntimeError(
            "subtitle_feats.pt not found."
        )

    video_idx = find_video_index(
        index,
        video_id,
    )

    if video_idx is None:
        raise ValueError(
            f"Cannot find video_id: {video_id}"
        )

    video_feat = index["video_feats"][
        video_idx:video_idx + 1
    ]

    video_feat = F.normalize(
        video_feat,
        dim=-1,
    )

    subtitle_feats = F.normalize(
        index["subtitle_feats"].to(
            video_feat.device
        ),
        dim=-1,
    )

    scores = torch.matmul(
        video_feat,
        subtitle_feats.T,
    ).squeeze(0)

    top_scores, top_indices = topk_from_scores(
        scores,
        topk,
    )

    return format_video_results(
        index,
        top_scores,
        top_indices,
    )


def search_video_to_text_subtitle(
    video_id,
    index,
    topk=10,
):
    """
    Video -> Text + Subtitle.

    Returns separate text and subtitle retrieval results.
    """
    results_t = search_video_to_text(
        video_id=video_id,
        index=index,
        topk=topk,
    )

    results_s = search_video_to_subtitle(
        video_id=video_id,
        index=index,
        topk=topk,
    )

    return {
        "video_to_text": results_t,
        "video_to_subtitle": results_s,
    }

def search_video_to_text_audio_subtitle(
        video_id,
        index,
        topk=10,
        wt=1.0,
        wa=1.0,
        ws=1.0,
):

    video_idx = find_video_index(
        index,
        video_id
    )

    if video_idx is None:
        raise ValueError(
            f"Cannot find {video_id}"
        )


    video_feat = index["video_feats"][
        video_idx:video_idx+1
    ]

    video_feat = F.normalize(
        video_feat,
        dim=-1
    )


    device = video_feat.device


    # video-text
    text_feats = F.normalize(
        index["text_feats"].to(device),
        dim=-1
    )

    score_t = torch.matmul(
        video_feat,
        text_feats.T
    ).squeeze(0)


    # video-audio
    audio_feats = F.normalize(
        index["audio_feats"].to(device),
        dim=-1
    )

    score_a = torch.matmul(
        video_feat,
        audio_feats.T
    ).squeeze(0)


    # video-subtitle
    subtitle_feats = F.normalize(
        index["subtitle_feats"].to(device),
        dim=-1
    )

    score_s = torch.matmul(
        video_feat,
        subtitle_feats.T
    ).squeeze(0)


    # 关键：
    # 三个 modality 必须对应 video-level index
    # 如果 text 是 caption-level，需要提前mean


    final_scores = (
        wt * score_a +
        ws * score_s
    )


    top_scores, top_indices = topk_from_scores(
        final_scores,
        topk
    )


    return format_video_results(
        index,
        top_scores,
        top_indices
    )


LOG_DIR = "./logs"

os.makedirs(
    LOG_DIR,
    exist_ok=True
)


LOG_FILE = os.path.join(
    LOG_DIR,
    "retrieval.log"
)


def write_log(text):

    with open(
        LOG_FILE,
        "a",
        encoding="utf-8"
    ) as f:

        f.write(text)
        f.write("\n")



def print_results(
    title,
    results,
):

    timestamp = datetime.datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )


    lines = []

    lines.append("")
    lines.append("=" * 100)

    lines.append(
        f"[{timestamp}] {title}"
    )

    lines.append("=" * 100)


    if not results:

        lines.append(
            "No results."
        )

    else:

        for item in results:

            lines.append(
                f"[Rank {item.get('rank')}] "
                f"score={float(item.get('score',0)):.4f}"
            )


            if item.get(
                "video_id",
                ""
            ):

                lines.append(
                    f"video_id   : {item.get('video_id')}"
                )


            if item.get(
                "video_path",
                ""
            ):

                lines.append(
                    f"video_path : {item.get('video_path')}"
                )


            if item.get(
                "caption",
                ""
            ):

                lines.append(
                    f"caption    : {item.get('caption')}"
                )


            if item.get(
                "subtitle",
                ""
            ):

                lines.append(
                    f"subtitle   : {item.get('subtitle')}"
                )


            lines.append(
                "-" * 100
            )


    lines.append("")


    output = "\n".join(lines)


    # terminal
    print(output)


    # file
    write_log(output)