config_name='pretrain_bary'
output_dir=./output/bary/$config_name

# config_name='pretrain_bary'
# output_dir=./output/bary/$config_name

# Classification
# WANDB_MODE=offline python3 -m torch.distributed.launch \
# --nnodes 1 \
# --node_rank 0 \
# --nproc_per_node 2 \
# --master_port 9834 \
# ./run_classification.py \
# --learning_rate 2e-5 \
# --checkpointing true \
# --first_eval true \
# --save_best true \
# --config ./config/barybind/finetune_cfg/retrieval-audiovisioncaps-vgg.json \
# --pretrain_dir $output_dir \
# --output_dir $output_dir/downstream/pretrain \
# --mode 'testing' \
# --checkpoint ./output/vast/pretrain_vast/downstream/ckpt/model_step_251.pt

### VIDEO-RET

# pretrain barybind on VAST27M
WANDB_MODE=offline python3 -m torch.distributed.launch \
--nnodes 1 \
--node_rank 0 \
--nproc_per_node 2 \
--master_port 9834 \
./run.py \
--learning_rate 2e-5 \
--checkpointing true \
--first_eval true \
--save_best true \
--config ./config/barybind/finetune_cfg/pretrain-barybind.json \
--pretrain_dir $output_dir \
--output_dir $output_dir/downstream/pretrain \
# --mode 'testing' \
# --checkpoint ./output/vast/pretrain_vast/downstream/ckpt/model_step_251.pt

# #retreival on MSRVTT
# WANDB_MODE=offline python3 -m torch.distributed.launch \
# --nnodes 1 \
# --node_rank 0 \
# --nproc_per_node 2 \
# --master_port 9834 \
# ./run.py \
# --learning_rate 3e-5 \
# --checkpointing true \
# --first_eval false \
# --save_best true \
# --config ./config/barybind/finetune_cfg/retrieval-msrvtt.json \
# --pretrain_dir $output_dir \
# --output_dir $output_dir/downstream/retrieval-msrvtt \
# --mode 'testing' \
# --checkpoint ./output/bary/pretrain_bary/model_step_251.pt


# # #retrieval-vatex
# python3 -m torch.distributed.launch \
# --nnodes 1 \
# --node_rank 0 \
# --nproc_per_node 2 \
# --master_port 9835 \
# ./run.py \
# --learning_rate 3e-5 \
# --checkpointing true \
# --first_eval true \
# --save_best true \
# --config ./config/barybind/finetune_cfg/retrieval-msrvtt.json \
# --pretrain_dir $output_dir \
# --output_dir $output_dir/downstream/retrieval-msrvtt \
# --mode 'testing' \
# --checkpoint $output_dir/downstream/retrieval-msrvtt/ckpt/model_step_9939.pt

# #retrieval-vatex
# python3 -m torch.distributed.launch \
# --nnodes 1 \
# --node_rank 0 \
# --nproc_per_node 8 \
# --master_port 9834 \
# ./run.py \
# --learning_rate 2e-5 \
# --checkpointing true \
# --config ./config/vast/finetune_cfg/retrieval-vatex.json \
# --pretrain_dir $output_dir \
# --save_best true \
# --output_dir $output_dir/downstream/retrieval-vatex\




# #retrieval-youcook
# python3 -m torch.distributed.launch \
# --nnodes 1 \
# --node_rank 0 \
# --nproc_per_node 8 \
# --master_port 9834 \
# ./run.py \
# --learning_rate 3e-5 \
# --checkpointing true \
# --config ./config/vast/finetune_cfg/retrieval-youcook.json \
# --pretrain_dir $output_dir \
# --save_best true \
# --output_dir $output_dir/downstream/retrieval-youcook \



#retrieval-didemo
# python3 -m torch.distributed.launch \
# --nnodes 1 \
# --node_rank 0 \
# --nproc_per_node 2 \
# --master_port 9835 \
# ./run.py \
# --learning_rate 2e-5 \
# --checkpointing true \
# --config ./config/barybind/finetune_cfg/retrieval-didemo.json \
# --pretrain_dir $output_dir \
# --save_best true \
# --output_dir $output_dir/downstream/retrieval-didemo \


# #retrieval-activitynet
# python3 -m torch.distributed.launch \
# --nnodes 1 \
# --node_rank 0 \
# --nproc_per_node 8 \
# --master_port 9834 \
# ./run.py \
# --learning_rate 2e-5 \
# --checkpointing true \
# --config ./config/vast/finetune_cfg/retrieval-activitynet.json \
# --pretrain_dir $output_dir \
# --output_dir $output_dir/downstream/retrieval-activitynet \
# --save_best true \

