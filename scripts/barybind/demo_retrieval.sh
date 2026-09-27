#!/bin/bash

export HF_ENDPOINT=https://hf-mirror.com

export CUDA_VISIBLE_DEVICES=6
export WANDB_MODE=offline

GPU_ID=6
MASTER_PORT=9834

output_dir=./output/bary/pretrain_bary
INDEX_DIR=./demo_index_tvas

CONFIG=./config/barybind/finetune_cfg/test_image_text_full.json
CKPT=./output/bary/pretrain_bary/downstream/pretrain/ckpt/model_step_1259.pt


# =========================
# INPUT QUERY
# =========================

QUERY_INPUT="a man is cooking in the kitchen"

VIDEO_ID="video8652"


# =========================
# COMMON ARGS
# =========================

COMMON_ARGS="\
--learning_rate 3e-5 \
--checkpointing true \
--first_eval false \
--save_best true \
--config ${CONFIG} \
--pretrain_dir ${output_dir} \
--output_dir ${output_dir}/downstream/retrieval-msrvtt \
--mode testing \
--checkpoint ${CKPT}
"


# =========================================================
# 1. START PERSISTENT TRANSLATOR
# =========================================================

TRANSLATOR_PID=""

start_translator() {
python3 - <<'EOF' &
from transformers import MarianMTModel, MarianTokenizer
import sys

model_name = "Helsinki-NLP/opus-mt-zh-en"

tok = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)


def translate(text):
    inputs = tok(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )

    out = model.generate(**inputs)

    return tok.decode(
        out[0],
        skip_special_tokens=True,
    )


print("Translator ready", flush=True)

for line in sys.stdin:
    text = line.strip()

    if text:
        print(
            translate(text),
            flush=True,
        )
EOF

TRANSLATOR_PID=$!
}


# =========================================================
# 2. CALL TRANSLATOR
# =========================================================

translate() {
    echo "$1" | python3 translate.py | tail -n 1
}


translate_safe() {
python3 - <<EOF
from transformers import MarianMTModel, MarianTokenizer

text = """$1"""

model_name = "Helsinki-NLP/opus-mt-zh-en"

tok = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

inputs = tok(
    text,
    return_tensors="pt",
    padding=True,
    truncation=True,
)

out = model.generate(**inputs)

print(
    tok.decode(
        out[0],
        skip_special_tokens=True,
    )
)
EOF
}


# =========================================================
# 3. LANGUAGE DETECTION
# =========================================================

is_english() {
python3 - <<EOF
text = """$1"""

if len(text.strip()) == 0:
    print("0")
    exit()

ascii_ratio = sum(
    c.isascii()
    for c in text
) / len(text)

print(
    "1"
    if ascii_ratio > 0.9
    else "0"
)
EOF
}


# =========================================================
# 4. BUILD FINAL QUERY
# =========================================================

if [ "$(is_english "$QUERY_INPUT")" == "1" ]; then

    QUERY="$QUERY_INPUT"

else

    QUERY_EN=$(translate_safe "$QUERY_INPUT")

    QUERY="${QUERY_EN} | ${QUERY_INPUT}"

fi


echo "=============================="
echo "Input Query: $QUERY_INPUT"
echo "Final Query: $QUERY"
echo "Video ID: $VIDEO_ID"
echo "=============================="


# =========================================================
# BUILD TV INDEX
# =========================================================

build_tv() {
CUDA_VISIBLE_DEVICES=${GPU_ID} \
WANDB_MODE=offline \
python3 -m torch.distributed.launch \
--nnodes 1 \
--node_rank 0 \
--nproc_per_node 1 \
--master_port ${MASTER_PORT} \
./demo_build_index.py \
--demo_output_dir ${INDEX_DIR} \
--demo_task ret%tv \
${COMMON_ARGS}
}


# =========================================================
# BUILD TVAS INDEX
# =========================================================

build_tvas() {
CUDA_VISIBLE_DEVICES=${GPU_ID} \
WANDB_MODE=offline \
python3 -m torch.distributed.launch \
--nnodes 1 \
--node_rank 0 \
--nproc_per_node 1 \
--master_port ${MASTER_PORT} \
./demo_build_index.py \
--demo_output_dir ${INDEX_DIR} \
--demo_task ret%tvas \
${COMMON_ARGS}
}


# =========================================================
# TEXT -> VIDEO
# =========================================================

t2v() {
CUDA_VISIBLE_DEVICES=${GPU_ID} \
WANDB_MODE=offline \
python3 -m torch.distributed.launch \
--nnodes 1 \
--node_rank 0 \
--nproc_per_node 1 \
--master_port ${MASTER_PORT} \
./demo_cli.py \
--demo_index_dir ${INDEX_DIR} \
--demo_mode t2v \
--query "${QUERY}" \
--query_input "${QUERY_INPUT}" \
--topk 10 \
${COMMON_ARGS}
}


# =========================================================
# TEXT -> AUDIO
# =========================================================

t2a() {
CUDA_VISIBLE_DEVICES=${GPU_ID} \
WANDB_MODE=offline \
python3 -m torch.distributed.launch \
--nnodes 1 \
--node_rank 0 \
--nproc_per_node 1 \
--master_port ${MASTER_PORT} \
./demo_cli.py \
--demo_index_dir ${INDEX_DIR} \
--demo_mode t2a \
--query "${QUERY}" \
--query_input "${QUERY_INPUT}" \
--topk 10 \
${COMMON_ARGS}
}


# =========================================================
# TEXT -> SUBTITLE
# =========================================================

t2s() {
CUDA_VISIBLE_DEVICES=${GPU_ID} \
WANDB_MODE=offline \
python3 -m torch.distributed.launch \
--nnodes 1 \
--node_rank 0 \
--nproc_per_node 1 \
--master_port ${MASTER_PORT} \
./demo_cli.py \
--demo_index_dir ${INDEX_DIR} \
--demo_mode t2s \
--query "${QUERY}" \
--query_input "${QUERY_INPUT}" \
--topk 10 \
${COMMON_ARGS}
}


# =========================================================
# TEXT -> VIDEO + SUBTITLE
# =========================================================

t2vs() {
CUDA_VISIBLE_DEVICES=${GPU_ID} \
WANDB_MODE=offline \
python3 -m torch.distributed.launch \
--nnodes 1 \
--node_rank 0 \
--nproc_per_node 1 \
--master_port ${MASTER_PORT} \
./demo_cli.py \
--demo_index_dir ${INDEX_DIR} \
--demo_mode t2vs \
--query "${QUERY}" \
--query_input "${QUERY_INPUT}" \
--topk 10 \
--wv 0.7 \
--ws 0.3 \
${COMMON_ARGS}
}


# =========================================================
# TEXT -> VIDEO + AUDIO + SUBTITLE
# =========================================================

t2vas() {
CUDA_VISIBLE_DEVICES=${GPU_ID} \
WANDB_MODE=offline \
python3 -m torch.distributed.launch \
--nnodes 1 \
--node_rank 0 \
--nproc_per_node 1 \
--master_port ${MASTER_PORT} \
./demo_cli.py \
--demo_index_dir ${INDEX_DIR} \
--demo_mode t2vas \
--query "${QUERY}" \
--query_input "${QUERY_INPUT}" \
--topk 10 \
--rerank_topk 100 \
--small_batch 32 \
${COMMON_ARGS}
}


# =========================================================
# VIDEO -> TEXT
# =========================================================

v2t() {
CUDA_VISIBLE_DEVICES=${GPU_ID} \
WANDB_MODE=offline \
python3 ./demo_cli.py \
--demo_index_dir ${INDEX_DIR} \
--demo_mode v2t \
--video_id "${VIDEO_ID}" \
--topk 10
}


# =========================================================
# VIDEO -> AUDIO
# =========================================================

v2a() {
CUDA_VISIBLE_DEVICES=${GPU_ID} \
WANDB_MODE=offline \
python3 ./demo_cli.py \
--demo_index_dir ${INDEX_DIR} \
--demo_mode v2a \
--video_id "${VIDEO_ID}" \
--topk 10
}


# =========================================================
# VIDEO -> SUBTITLE
# =========================================================

v2s() {
CUDA_VISIBLE_DEVICES=${GPU_ID} \
WANDB_MODE=offline \
python3 ./demo_cli.py \
--demo_index_dir ${INDEX_DIR} \
--demo_mode v2s \
--video_id "${VIDEO_ID}" \
--topk 10
}


# =========================================================
# VIDEO -> TEXT + SUBTITLE
# =========================================================

v2ts() {
CUDA_VISIBLE_DEVICES=${GPU_ID} \
WANDB_MODE=offline \
python3 ./demo_cli.py \
--demo_index_dir ${INDEX_DIR} \
--demo_mode v2ts \
--video_id "${VIDEO_ID}" \
--topk 10
}


# =========================================================
# VIDEO -> TEXT + AUDIO + SUBTITLE
# =========================================================

v2tas() {
CUDA_VISIBLE_DEVICES=${GPU_ID} \
WANDB_MODE=offline \
python3 ./demo_cli.py \
--demo_index_dir ${INDEX_DIR} \
--demo_mode v2tas \
--video_id "${VIDEO_ID}" \
--topk 10
}


# =========================================================
# ENTRY
# =========================================================

case "$1" in

    build_tv)
        build_tv
        ;;

    build_tvas)
        build_tvas
        ;;

    t2v)
        t2v
        ;;

    t2a)
        t2a
        ;;

    t2s)
        t2s
        ;;

    t2vs)
        t2vs
        ;;

    t2vas)
        t2vas
        ;;

    v2t)
        v2t
        ;;

    v2a)
        v2a
        ;;

    v2s)
        v2s
        ;;

    v2ts)
        v2ts
        ;;

    v2tas)
        v2tas
        ;;

    *)
        echo "Usage:"
        echo "bash demo_retrieval.sh build_tv"
        echo "bash demo_retrieval.sh build_tvas"
        echo "bash demo_retrieval.sh t2v"
        echo "bash demo_retrieval.sh t2a"
        echo "bash demo_retrieval.sh t2s"
        echo "bash demo_retrieval.sh t2vs"
        echo "bash demo_retrieval.sh t2vas"
        echo "bash demo_retrieval.sh v2t"
        echo "bash demo_retrieval.sh v2a"
        echo "bash demo_retrieval.sh v2s"
        echo "bash demo_retrieval.sh v2ts"
        echo "bash demo_retrieval.sh v2tas"
        exit 1
        ;;

esac