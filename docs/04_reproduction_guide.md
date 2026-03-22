# 04. Reproduction Guide

## Environment

The experiments were run in a GPU-backed RunPod environment.

## Main scripts

- `train_edit_delta_cached.py`
- `train_edit_delta_scalable.py`
- `run_edit_train.sh`
- `eval_rerun_checkpoint.py`

## Example scalable training configuration

A representative safer run configuration was:

```bash
unset HF_HUB_OFFLINE
export CUDA_HOME=/usr/local/cuda-11.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export ATTN_BACKEND=xformers
export SPCONV_ALGO=native
export CUDA_VISIBLE_DEVICES=0
unset PYTORCH_CUDA_ALLOC_CONF
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64

export EDIT_EPOCHS=20
export EDIT_MAX_OBJECTS=40
export EDIT_PROMPTS_PER_OBJECT=2
export EDIT_N_VIEWS=2
export EDIT_RENDER_RES=128
export EDIT_TEXT_GAIN=4.0
export EDIT_TEXT_REPEAT=8
export EDIT_SCALE=0.15
export EDIT_LAMBDA_DELTA=0.02
export EDIT_MAX_VOXELS=22000
export EDIT_LOG=/workspace/TRELLIS/run_medium_safe_tuned.log

./run_edit_train.sh