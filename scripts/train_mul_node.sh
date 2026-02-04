#!/usr/bin/env bash

set -x

# set dist args
# For multi-node training, set NODE_RANK differently on each machine:
# Machine 1 (172.16.46.140): export NODE_RANK=0
# Machine 2 (172.16.46.142): export NODE_RANK=1
NODE_RANK=${NODE_RANK:-0}  # Default to 0 if not set

# Multi-node configuration
nnodes=2  # Total number of machines
nproc_per_node=1  # Number of GPUs per machine
master_addr=172.16.46.140  # IP of the master node (first machine)
master_port=12345  # Communication port
node_rank=${NODE_RANK}  # This machine's rank (0 or 1)

echo "[nproc_per_node: ${nproc_per_node}]"
echo "[nnodes: ${nnodes}]"
echo "[node_rank: ${node_rank}]"
echo "[master_addr: ${master_addr}]"
echo "[master_port: ${master_port}]"

# set up envs
export OMP_NUM_THREADS=8
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=eno1np0
export NCCL_SOCKET_NTHREADS=4
export NCCL_NSOCKS_PERTHREAD=4


BED=checkpoints
LOCAL_OUT=local_output
mkdir -p $BED
mkdir -p $LOCAL_OUT


export COMPILE_GAN=0
export USE_TIMELINE_SDK=1
export CUDA_TIMER_STREAM_KAFKA_CLUSTER=bmq_data_va
export CUDA_TIMER_STREAM_KAFKA_TOPIC=megatron_cuda_timer_tracing_original_v2
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# wandb offline
exp_name=125M-0.06M-10-overfit-1000ep-multi-node
bed_path=checkpoints/${exp_name}/
data_path='data/FaceID-6M/512_webdataset_10'
video_data_path=''
local_out_path=$LOCAL_OUT/${exp_name}

# rm -rf ${bed_path}
# rm -rf ${local_out_path}

torchrun \
--nproc_per_node=${nproc_per_node} \
--nnodes=${nnodes} \
--node_rank=${node_rank} \
--master_addr=${master_addr} \
--master_port=${master_port} \
train.py \
--ep=1000 \
--opt=adamw \
--cum=3 \
--sche=lin0 \
--fp16=2 \
--ada=0.9_0.97 \
--tini=-1 \
--tclip=5 \
--flash=0 \
--alng=5e-06 \
--saln=1 \
--cos=1 \
--enable_checkpointing=full-block \
--local_out_path ${local_out_path} \
--task_type='t2i' \
--bed=${bed_path} \
--data_path=${data_path} \
--video_data_path=${video_data_path} \
--exp_name=${exp_name} \
--tblr=6e-3 \
--pn 0.25M \
--model=layer12c4 \
--lbs=4 \
--workers=8 \
--short_cap_prob 0.5 \
--online_t5=1 \
--use_streaming_dataset 1 \
--iterable_data_buffersize 30000 \
--Ct5=2048 \
--t5_path=weights/flan-t5-xl \
--vae_type 32 \
--vae_ckpt=weights/infinity_vae_d32reg.pth  \
--wp 0.00000001 \
--wpe=1 \
--dynamic_resolution_across_gpus 1 \
--enable_dynamic_length_prompt 1 \
--reweight_loss_by_scale 1 \
--add_lvl_embeding_only_first_block 1 \
--rope2d_each_sa_layer 1 \
--rope2d_normalized_by_hw 2 \
--use_fsdp_model_ema 0 \
--always_training_scales 100 \
--use_bit_label 1 \
--zero=2 \
--save_model_iters_freq 100 \
--log_freq=50 \
--checkpoint_type='torch' \
--prefetch_factor=16 \
--noise_apply_strength 0.3 \
--noise_apply_layers 13 \
--apply_spatial_patchify 0 \
--use_flex_attn=False \
--pad=128 