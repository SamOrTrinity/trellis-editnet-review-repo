#!/usr/bin/env bash
set -e
cd /workspace/TRELLIS

unset HF_HUB_OFFLINE
export CUDA_HOME=/usr/local/cuda-11.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export ATTN_BACKEND=xformers
export SPCONV_ALGO=native
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128

python -u train_edit_delta_scalable.py 2>&1 | tee "${EDIT_LOG:-/workspace/TRELLIS/edit_train.log}"
