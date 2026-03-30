# TRELLIS 1 Architecture 2: EditNet Retraining Guide and Technical Report

**Author:** Sambit Hore  
**Institution:** Trinity College Dublin  
**Project:** TRELLIS 1 EditNet — image + edit prompt to edited 3D representation

---

## 1. What this project is trying to do

This repository adapts **TRELLIS 1** from pure image-to-3D generation into a **3D editing system**.

The goal is:

- **input 1:** a source image of an object
- **input 2:** a text edit prompt such as *make it wooden*, *make it metallic*, or *make it red*
- **output:** an edited 3D representation that follows the instruction while preserving the original object as much as possible

Instead of training a full 3D generator from scratch, this approach starts from the latent produced by frozen TRELLIS 1 and learns a **prompt-conditioned residual edit**:

- `z_B = z_A + delta`

where:

- `z_A` is the original TRELLIS latent of the source object
- `delta` is a learned latent residual predicted by **EditNet**
- `z_B` is the edited latent that is decoded and rendered back into a 3D representation

This is therefore a **latent editing** approach, not a fresh text-to-3D generation approach.

---

## 2. Summary

In this work, I adapted TRELLIS 1 into a system that edits a 3D representation from a source image and an edit prompt. The practical training strategy that ultimately worked was not to repeatedly re-encode source images during every training step, but to move to a **cached-latent training pipeline**. In that pipeline, source objects are encoded once, saved to disk, paired with edit prompts, and then used for efficient retraining of two small trainable modules: **EditNet** and **TextProjector**.

The core technical challenge of this project was not the idea of EditNet itself. The EditNet design was reasonable early on. The real challenges were:

- understanding the true TRELLIS latent dimensionality used in the working path
- making the render path differentiable so gradients could reach EditNet
- preventing the training process from collapsing under GPU memory pressure

The final stable direction used:

- frozen TRELLIS 1 backbone components
- a trainable residual editor (`trellis/models/edit_net.py`)
- CLIP-based prompt supervision
- cached TRELLIS latents on disk
- differentiable Gaussian rendering
- a scalable launcher driven by environment variables

This README has two purposes:

1. to show a new user exactly how to retrain the model with the files in this repository and how to change training settings safely
2. to document, in one place, the full technical story of the method, the debugging process, the differentiable rendering requirement, the latent-dimension fix, and the cached-latent solution to the memory bottleneck

---

## 3. Key point and conclusion

### Key point

The central lesson of this project is that the EditNet idea only becomes trainable if the gradient path remains intact from the loss all the way back to the latent residual.

That means the training loop must preserve the following chain:

`CLIP loss -> rendered image tensor -> differentiable Gaussian renderer -> TRELLIS Gaussian decoder -> edited latent -> delta -> EditNet`

If the rendered image is detached into NumPy before the loss is computed, the loop may appear to run, but EditNet will not learn meaningful edits.

### Conclusion

The final outcome of this project is that **TRELLIS 1 latent editing with EditNet is feasible**, but only when:

- the latent dimension is matched to the real TRELLIS tensor path
- differentiable rendering is used directly
- cached latents replace repeated online encoding
- large TRELLIS modules are freed after encoding
- training scale is controlled through safe runtime parameters

The published repository is therefore best understood as a **working retraining and experimentation repo** for prompt-conditioned latent editing on top of TRELLIS 1.

---

## 4. Background and overall goal

The motivation for Architecture 2 was to avoid the heavy pseudo-labeling workflow of the alternative architecture. Instead of creating a large offline pseudo-3D supervision set and then training TRELLIS on those targets, this approach directly edits a TRELLIS latent with a smaller editor network.

The conceptual background is simple:

1. TRELLIS 1 already knows how to convert an image into a 3D latent representation.
2. If that latent can be edited in a controlled way, then the model does not need to regenerate the whole object from scratch.
3. A residual edit is preferable to predicting a full replacement latent because the residual keeps the edit anchored to the source object.

The overall goal was therefore to build a system that could:

- preserve the identity and broad structure of the source object
- obey a text edit prompt
- remain differentiable during training
- stay numerically stable enough to run on the available GPU budget

The hardest constraint came from **TRELLIS 1 itself**: the latent used here is a **coupled SLAT latent**, so appearance and geometry are not cleanly separated. This means the editor cannot assume that a material or color change will never disturb structure. That is why prompt alignment alone is not enough, and why preservation logic became an important part of the method.

---

## 5. Clean repository structure

The repository currently contains the following top-level structure:

```text
trellis-editnet-review-repo/
├─ artifacts/
├─ docs/
│  ├─ figures/
│  ├─ 01_background_and_problem.md
│  ├─ 02_method_and_losses.md
│  ├─ 03_experiments_and_results.md
│  ├─ 04_reproduction_guide.md
│  └─ generated_results.md
├─ scripts/
│  └─ generate_report_assets.py
├─ trellis/
│  └─ models/
│     └─ edit_net.py
├─ eval_rerun_checkpoint.py
├─ run_edit_train.sh
├─ train_edit_delta_cached.py
├─ train_edit_delta_scalable.py
├─ README.md
└─ .gitignore
```

### What each top-level part does

- `README.md` — current public repo overview, problem framing, and file map
- `docs/` — written project documentation, results write-ups, and supporting figures
- `artifacts/` — saved logs, checkpoints, evaluation folders, and workspace metadata from runs
- `scripts/generate_report_assets.py` — helper used to turn logs/results into plots and report assets
- `trellis/models/edit_net.py` — the actual EditNet and TextProjector definitions
- `train_edit_delta_cached.py` — first stable cached-latent trainer
- `train_edit_delta_scalable.py` — scalable trainer driven by environment variables
- `run_edit_train.sh` — launcher that sets the expected runtime environment and starts scalable training
- `eval_rerun_checkpoint.py` — checkpoint inspection and rerendering script

---

## 6. What a user needs before retraining

This guide intentionally does **not** cover TRELLIS deployment. It assumes the user already has:

- a working TRELLIS 1 environment
- a working local TRELLIS model snapshot
- a compatible CUDA / PyTorch / xformers / spconv setup
- access to the directories expected by the training scripts

The expected external data layout for retraining is:

```text
/workspace/edit_training_data/
├─ encoded_latents/
│  ├─ *.pt
│  ├─ *.pt
│  └─ ...
└─ training_pairs.json
```

### Meaning of these files

#### `encoded_latents/*.pt`
Each `.pt` file stores a cached object payload that includes at least:

- the source TRELLIS latent (`slat`)
- the TRELLIS conditioning object (`cond`)
- the image/object name or UID

This is the key to the memory fix: these latents are encoded once and reused, rather than regenerated at the start of every new training experiment.

#### `training_pairs.json`
This file defines which cached object should be paired with which edit prompt.

The scalable trainer reads this file and turns it into the training pair list for each epoch.

A typical row is conceptually of the form:

```json
{
  "uid": "T",
  "edit_prompt": "make it metallic"
}
```

The field `uid` must match either:

- the cached object name inside the payload
- or the stem of the cache filename

---

## 7. Which file the user should run

For most users, the main entry point is:

```bash
bash run_edit_train.sh
```

That launcher performs the expected environment setup and then runs:

```bash
python -u train_edit_delta_scalable.py 2>&1 | tee "${EDIT_LOG:-/workspace/TRELLIS/edit_train.log}"
```

So the practical answer is:

- **use `run_edit_train.sh` for normal retraining**
- **use `train_edit_delta_scalable.py` if you want to inspect or modify the trainer directly**
- **use `train_edit_delta_cached.py` if you want the earlier cached trainer as a reference baseline**
- **use `eval_rerun_checkpoint.py` to inspect a saved checkpoint after training**

---

## 8. Step-by-step retraining guide for a user

This section is the practical handover guide.

### Step 1 — Put the cached latents and pair file in the expected location

Make sure the following exist:

```text
/workspace/edit_training_data/encoded_latents/
/workspace/edit_training_data/training_pairs.json
```

The scalable trainer reads these paths by default.

In code, the default paths are:

```python
PAIRS_JSON = os.getenv("EDIT_PAIRS_JSON", "/workspace/edit_training_data/training_pairs.json")
CACHE_DIR  = os.getenv("EDIT_CACHE_DIR", "/workspace/edit_training_data/encoded_latents")
```

### Step 2 — Check the key runtime parameters

The scalable trainer is controlled by environment variables rather than by hard-coded config files. The most important ones are:

```python
N_EPOCHS           = int(os.getenv("EDIT_EPOCHS", "5"))
LR                 = float(os.getenv("EDIT_LR", "3e-4"))
LAMBDA_PROMPT      = float(os.getenv("EDIT_LAMBDA_PROMPT", "1.0"))
LAMBDA_DELTA       = float(os.getenv("EDIT_LAMBDA_DELTA", "0.05"))
LAMBDA_PRESERVE    = float(os.getenv("EDIT_LAMBDA_PRESERVE", "0.05"))
N_RENDER_VIEWS     = int(os.getenv("EDIT_N_VIEWS", "3"))
RENDER_RES         = int(os.getenv("EDIT_RENDER_RES", "128"))
PROMPTS_PER_OBJECT = int(os.getenv("EDIT_PROMPTS_PER_OBJECT", "2"))
MAX_OBJECTS        = int(os.getenv("EDIT_MAX_OBJECTS", "5"))
TEXT_GAIN          = float(os.getenv("EDIT_TEXT_GAIN", "4.0"))
TEXT_REPEAT        = int(os.getenv("EDIT_TEXT_REPEAT", "8"))
SCALE              = float(os.getenv("EDIT_SCALE", "0.15"))
MAX_VOXELS         = int(os.getenv("EDIT_MAX_VOXELS", "22000"))
PAIRS_JSON         = os.getenv("EDIT_PAIRS_JSON", "/workspace/edit_training_data/training_pairs.json")
CACHE_DIR          = os.getenv("EDIT_CACHE_DIR", "/workspace/edit_training_data/encoded_latents")
CKPT_DIR           = os.getenv("EDIT_CKPT_DIR", "/workspace/edit_checkpoints")
```

### Step 3 — Run the default training job

From the repository root, a normal retraining run is:

```bash
bash run_edit_train.sh
```

The launcher itself is small and important:

```bash
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
```

### Step 4 — Change training settings safely

A user does **not** need to edit Python every time. The intended way to change experiments is to set environment variables before launching training.

#### Example A — A short smoke run

```bash
export EDIT_EPOCHS=2
export EDIT_MAX_OBJECTS=2
export EDIT_PROMPTS_PER_OBJECT=1
export EDIT_N_VIEWS=1
export EDIT_RENDER_RES=128
export EDIT_LOG=/workspace/TRELLIS/smoke_run.log
bash run_edit_train.sh
```

#### Example B — A medium run

```bash
export EDIT_EPOCHS=10
export EDIT_MAX_OBJECTS=20
export EDIT_PROMPTS_PER_OBJECT=2
export EDIT_N_VIEWS=2
export EDIT_RENDER_RES=128
export EDIT_TEXT_GAIN=4.0
export EDIT_TEXT_REPEAT=8
export EDIT_LAMBDA_DELTA=0.02
export EDIT_MAX_VOXELS=22000
export EDIT_LOG=/workspace/TRELLIS/run_medium.log
bash run_edit_train.sh
```

#### Example C — A stronger text-conditioning run

```bash
export EDIT_EPOCHS=20
export EDIT_MAX_OBJECTS=40
export EDIT_PROMPTS_PER_OBJECT=2
export EDIT_N_VIEWS=2
export EDIT_RENDER_RES=128
export EDIT_TEXT_GAIN=5.0
export EDIT_TEXT_REPEAT=10
export EDIT_SCALE=0.15
export EDIT_LAMBDA_DELTA=0.02
export EDIT_LOG=/workspace/TRELLIS/text_boost_retrain.log
bash run_edit_train.sh
```

### Step 5 — Understand what the trainer is doing

At a high level, the scalable trainer does this:

1. load cached latents from `encoded_latents/*.pt`
2. skip oversized objects using `MAX_VOXELS`
3. load the frozen TRELLIS pipeline
4. immediately free the large TRELLIS DiT/image-conditioning modules after latent loading
5. load frozen OpenCLIP
6. instantiate `EditNet` and `TextProjector`
7. read `training_pairs.json`
8. build `(cached object, edit prompt)` pairs
9. decode the edited latent to a Gaussian representation
10. render views differentiably
11. compute losses
12. backpropagate into the trainable modules only
13. save checkpoints to `EDIT_CKPT_DIR`

### Step 6 — Know what “restart training” means in the current repo

The current published trainer **does save checkpoints**, but it does **not expose a full checkpoint-resume mode** through a documented launcher flag.

So for a user working with the repo exactly as published:

- to **start a new run with different settings**, run `bash run_edit_train.sh` again with new environment variables
- to **inspect a saved checkpoint**, run `eval_rerun_checkpoint.py`
- to **continue from a checkpoint as a formal resume**, the current repo does not provide a dedicated public resume script or `optimizer.load_state_dict(...)` path in the trainer

That means the clean documented workflow is:

- rerun training with the desired settings for a new experiment
- use the saved checkpoints for evaluation and comparison

### Step 7 — Evaluate a saved checkpoint

Use `eval_rerun_checkpoint.py` to load a saved checkpoint and compare rerendered edited results.

In the current evaluation script, the key logic includes:

```python
print("Loading checkpoint...")
ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
edit_net = EditNet(latent_dim=8, cond_dim=1024, hidden_dim=256, n_blocks=3, scale=0.1).to(DEVICE)
text_proj = TextProjector(768, 1024).to(DEVICE)
edit_net.load_state_dict(ckpt["edit_net"])
text_proj.load_state_dict(ckpt["text_proj"])
edit_net.eval()
text_proj.eval()
print("Loading cached latent...")
payload = torch.load(CACHE_PATH, weights_only=False)
```

### Step 8 — Generate plots and report assets

After training, the repository’s helper script is:

- `scripts/generate_report_assets.py`

Use it to turn logs and results into assets that can be reused in the report sections under `docs/`.

---

## 9. Recommended parameter scaling order for a new user

A new user should not jump directly to the biggest run.

The recommended scaling order is:

1. **tiny smoke test**
   - 1–2 objects
   - 1 prompt per object
   - 1–2 views
   - 128 resolution
   - 2 epochs

2. **small stable test**
   - 5 objects
   - 2 prompts per object
   - 2 views
   - 128 resolution
   - 5 epochs

3. **medium experiment**
   - 20 objects
   - 2 prompts per object
   - 2 views
   - 128 resolution
   - 10–20 epochs

4. **careful scale-up**
   - 40 objects
   - keep `MAX_VOXELS` enforced
   - increase prompts or views only one axis at a time

This scaling order matters because the real bottleneck is memory and render cost, not the EditNet module itself.

---


## 9A. Canonical retraining path for a new user

This section makes the practical workflow explicit in one place so a new user does not have to infer the intended training path from multiple files.

The **canonical retraining path** in this repository is:

1. prepare cached TRELLIS latents in `/workspace/edit_training_data/encoded_latents/`
2. prepare the pair manifest at `/workspace/edit_training_data/training_pairs.json`
3. set the desired runtime parameters through environment variables
4. launch training with `bash run_edit_train.sh`
5. monitor the log written to `EDIT_LOG`
6. inspect saved checkpoints in `EDIT_CKPT_DIR`
7. evaluate a checkpoint with `eval_rerun_checkpoint.py`

The most important practical rule is this:

- **`run_edit_train.sh` is the normal user-facing entry point**
- **`train_edit_delta_scalable.py` is the main trainer that the launcher calls**
- **`train_edit_delta_cached.py` is the earlier cached trainer kept as a reference baseline**
- **`eval_rerun_checkpoint.py` is the post-training evaluation entry point**

So, for a new user, the repository should be approached as:

`cached latents + pair manifest -> run_edit_train.sh -> train_edit_delta_scalable.py -> checkpoints -> evaluation`

### Canonical commands

#### Minimal smoke test

```bash
export EDIT_EPOCHS=2
export EDIT_MAX_OBJECTS=2
export EDIT_PROMPTS_PER_OBJECT=1
export EDIT_N_VIEWS=1
export EDIT_RENDER_RES=128
export EDIT_LOG=/workspace/TRELLIS/smoke_run.log
bash run_edit_train.sh
```

#### Normal medium retraining run

```bash
export EDIT_EPOCHS=10
export EDIT_MAX_OBJECTS=20
export EDIT_PROMPTS_PER_OBJECT=2
export EDIT_N_VIEWS=2
export EDIT_RENDER_RES=128
export EDIT_TEXT_GAIN=4.0
export EDIT_TEXT_REPEAT=8
export EDIT_LAMBDA_DELTA=0.02
export EDIT_MAX_VOXELS=22000
export EDIT_LOG=/workspace/TRELLIS/run_medium.log
bash run_edit_train.sh
```

#### Stronger text-conditioning retraining run

```bash
export EDIT_EPOCHS=20
export EDIT_MAX_OBJECTS=40
export EDIT_PROMPTS_PER_OBJECT=2
export EDIT_N_VIEWS=2
export EDIT_RENDER_RES=128
export EDIT_TEXT_GAIN=5.0
export EDIT_TEXT_REPEAT=10
export EDIT_SCALE=0.15
export EDIT_LAMBDA_DELTA=0.02
export EDIT_LOG=/workspace/TRELLIS/text_boost_retrain.log
bash run_edit_train.sh
```

## 9B. Precise external data-layout contract

This section makes the expected on-disk input contract fully explicit.

The retraining scripts assume the following external layout:

```text
/workspace/edit_training_data/
├─ encoded_latents/
│  ├─ *.pt
│  ├─ *.pt
│  └─ ...
└─ training_pairs.json
```

### What each cached latent file must contain

Each file in `encoded_latents/*.pt` is expected to be a cached object payload produced from the frozen TRELLIS pipeline. At minimum, the payload should contain:

- `slat` — the TRELLIS source latent that EditNet will modify
- `cond` — the TRELLIS conditioning object needed to reconstruct the image-side conditioning stream
- a stable object identifier such as `name`, `uid`, or a filename stem

In practical terms, the trainer needs to be able to recover:

- the latent features (`slat.feats`)
- the sparse coordinates (`slat.coords`)
- the image conditioning tokens from `cond`
- the object identity used to match prompts from the pair manifest

### What the pair manifest must contain

The pair file is the explicit mapping from cached objects to edit prompts. Conceptually, each row should identify:

- which cached object to use
- which edit prompt to apply

A typical row is of the form:

```json
{
  "uid": "T",
  "edit_prompt": "make it metallic"
}
```

The `uid` must match either:

- the cached object name stored in the payload
- or the stem of the `.pt` cache filename

### Default paths used by the scalable trainer

```python
PAIRS_JSON = os.getenv("EDIT_PAIRS_JSON", "/workspace/edit_training_data/training_pairs.json")
CACHE_DIR  = os.getenv("EDIT_CACHE_DIR", "/workspace/edit_training_data/encoded_latents")
CKPT_DIR   = os.getenv("EDIT_CKPT_DIR", "/workspace/edit_checkpoints")
```

### What a user should verify before launching training

Before running `bash run_edit_train.sh`, the user should confirm:

- the cache directory exists
- the pair file exists
- every `uid` in the pair file maps to a real cached latent
- the cached latent files are loadable by `torch.load(...)`
- the large objects are either intentionally included or controlled with `EDIT_MAX_VOXELS`

This check matters because most training failures in a handoff setting come from path mismatches, missing cache files, or incompatible payload contents rather than from the model itself.

## 9C. Exact restart and relaunch procedure with the current repo

This section makes the restart story explicit using only the files that already exist in the repository.

### What the current repo supports directly

The current published training stack supports:

- launching a fresh run with `bash run_edit_train.sh`
- saving checkpoints during training
- evaluating saved checkpoints with `eval_rerun_checkpoint.py`
- relaunching training with different parameters by changing environment variables

### What the current repo does not expose as a formal public feature

The current repo does **not** expose a dedicated documented checkpoint-resume script that restores the optimizer and continues exactly from the last interrupted step through a one-line public interface.

So, in the repository as it currently exists, **restart training** should be understood in this clean way:

- **restart a new experiment** by relaunching `bash run_edit_train.sh` with new parameters
- **reuse a previous checkpoint for inspection** through `eval_rerun_checkpoint.py`
- **treat checkpoints as experiment outputs and comparison points**, not as a formal auto-resume interface

### Exact relaunch procedure for a new run with different parameters

```bash
unset HF_HUB_OFFLINE
export CUDA_HOME=/usr/local/cuda-11.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export ATTN_BACKEND=xformers
export SPCONV_ALGO=native
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128

export EDIT_EPOCHS=5
export EDIT_MAX_OBJECTS=5
export EDIT_PROMPTS_PER_OBJECT=2
export EDIT_N_VIEWS=2
export EDIT_RENDER_RES=128
export EDIT_LOG=/workspace/TRELLIS/relaunch.log

bash run_edit_train.sh
```

### Exact checkpoint-inspection procedure

```bash
python -u eval_rerun_checkpoint.py
```

The important practical distinction is:

- `run_edit_train.sh` is for launching training runs
- `eval_rerun_checkpoint.py` is for loading and inspecting what training already produced

That distinction should be kept clear for any new user reading the repository.

## 10. The detailed technical report

Everything from this point onward is the technical report section. It explains the method, the architecture, the losses, the debugging process, and how the repository files fit together.

---

## 11. EditNet neural network architecture in detail

The final EditNet used in the working implementation is the **runtime-corrected 8-dimensional version**.

That correction matters because some of the early notes described the TRELLIS latent as 64-dimensional. The actual working tensor path exposed to the editor in this project was 8-dimensional per active voxel. The final implementation was therefore rebuilt around `latent_dim=8`.

### What EditNet receives

EditNet receives two inputs:

1. the source latent that will be edited
2. the joint conditioning tensor that contains both source-image conditioning and text conditioning

So the logic is:

- source image -> TRELLIS -> `z_A` and image conditioning
- edit prompt -> CLIP -> projected text conditioning
- EditNet -> `delta`
- edited latent -> `z_B = z_A + delta`

### Internal architecture

The architecture is a **cross-attention residual editor**.

#### Stage A — Input projection

The source latent token features are projected from 8 dimensions into a hidden space of 256 dimensions.

#### Stage B — Cross-attention blocks

The projected latent tokens attend to the joint conditioning sequence using multi-head cross-attention.

That means the latent tokens are the **queries**, while the conditioning tokens act as **keys and values**.

This is an appropriate design because each latent token can selectively attend to the parts of the image/text conditioning that matter for its own edit.

#### Stage C — Feed-forward refinement

Each block also contains a feed-forward network with GELU and dropout, wrapped in residual connections.

#### Stage D — Output projection

The hidden representation is projected back down to the latent dimensionality.

#### Stage E — Bounded residual

The output is passed through `tanh` and scaled, so the residual stays bounded.

This stabilizes training and prevents very large latent jumps.

#### Stage F — Residual addition

The final edited latent is obtained by adding the bounded residual to the original latent.

### Why zero initialization matters

The output head is zero-initialized.

That means the network starts as an identity mapping:

- `delta ≈ 0`
- `z_B ≈ z_A`

This is desirable because the editor should only move away from the source object when the loss provides a reason to do so.

### Final EditNet code used in the project

```python
class CrossAttentionBlock(nn.Module):
    def __init__(self, hidden_dim=256, cond_dim=1024, n_heads=8, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=n_heads,
            kdim=cond_dim, vdim=cond_dim,
            dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout))

    def forward(self, x, cond):
        x_norm = self.norm1(x)
        attn_out, _ = self.cross_attn(query=x_norm, key=cond, value=cond)
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x

class EditNet(nn.Module):
    def __init__(self, latent_dim=8, cond_dim=1024, hidden_dim=256,
                 n_blocks=3, n_heads=8, scale=0.1, dropout=0.1):
        super().__init__()
        self.scale = scale
        self.latent_dim = latent_dim
        self.input_proj = nn.Sequential(nn.Linear(latent_dim, hidden_dim), nn.GELU())
        self.blocks = nn.ModuleList([
            CrossAttentionBlock(hidden_dim, cond_dim, n_heads, dropout)
            for _ in range(n_blocks)])
        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, latent_dim))
        nn.init.zeros_(self.output_proj[-1].weight)
        nn.init.zeros_(self.output_proj[-1].bias)

    def forward(self, feats_A, e_joint):
        squeeze = False
        if feats_A.dim() == 2:
            feats_A = feats_A.unsqueeze(0)
            squeeze = True
        h = self.input_proj(feats_A)
        for block in self.blocks:
            h = block(h, e_joint)
        delta_raw = self.output_proj(h)
        delta = self.scale * torch.tanh(delta_raw)
        feats_B = feats_A + delta
        if squeeze:
            feats_B = feats_B.squeeze(0)
            delta = delta.squeeze(0)
        return feats_B, delta
```

---

## 12. How training is done in this project

### 12.1 High-level training flow

The final training approach used in this repository is:

1. cache source TRELLIS latents on disk
2. define object–prompt training pairs
3. load cached latents
4. freeze the heavy TRELLIS pipeline components
5. train only `EditNet` and `TextProjector`
6. decode the edited latent into a Gaussian representation
7. render it differentiably
8. score it with CLIP-based supervision and regularization terms
9. save checkpoints and logs

### 12.2 Why cached-latent training is the final working direction

The early pipeline re-encoded source images online and kept too much of TRELLIS alive inside the loop.

The final working direction was:

- encode once
- store the latent and conditioning
- reload it for retraining
- free the heavy image-conditioning / DiT modules
- keep only what the optimization loop really needs

That change was the major solution to the memory chokehold.

### 12.3 Core training pipeline inside the scalable trainer

The scalable trainer does the following in code:

- load cached latents
- skip very large objects by voxel count
- free TRELLIS flow and image-conditioning models
- load frozen CLIP
- build training pairs from `training_pairs.json`
- condition EditNet using image tokens plus repeated/scaled text tokens
- decode the edited latent using the frozen TRELLIS Gaussian decoder
- render the edited Gaussian directly
- compute loss
- backpropagate into EditNet and TextProjector

### 12.4 Key training code paths

#### Cached latent loading

```python
cache_files = sorted(cache_dir.glob("*.pt"))[:MAX_OBJECTS]
encoded = []
for cache_path in cache_files:
    payload = torch.load(cache_path, weights_only=False)
    nvox = payload["slat"].feats.shape[0]
    if nvox > MAX_VOXELS:
        continue
    encoded.append({
        "name": payload.get("image_name", cache_path.name),
        "slat": payload["slat"],
        "cond": payload["cond"],
        "cache_path": str(cache_path),
    })
```

#### Freeing heavy TRELLIS modules after loading

```python
del pipeline.models['sparse_structure_flow_model']
del pipeline.models['slat_flow_model']
del pipeline.models['image_cond_model']
torch.cuda.empty_cache()
gc.collect()
```

#### Reading the pair file

```python
pairs_json = Path(PAIRS_JSON)
with open(pairs_json) as f:
    pair_rows = json.load(f)

encoded_by_name = {}
for obj in encoded:
    encoded_by_name[obj["name"]] = obj
    encoded_by_name[Path(obj["cache_path"]).stem] = obj

pairs = []
for row in pair_rows:
    key = row["uid"]
    if key in encoded_by_name:
        pairs.append((encoded_by_name[key], row["edit_prompt"]))
```

#### Conditioning construction and EditNet call

```python
e_img = obj["cond"]["cond"].detach()
if e_img.dim() == 2:
    e_img = e_img.unsqueeze(0)

text_tokens = clip_tokenizer([prompt]).to(device)
with torch.no_grad():
    e_text_raw = clip_model.encode_text(text_tokens)

e_text_padded = F.pad(e_text_raw.unsqueeze(1), (0, 768 - 512))
e_text_proj_out = text_proj(e_text_padded)
e_text_tokens = e_text_proj_out.repeat(1, TEXT_REPEAT, 1) * TEXT_GAIN
e_joint = torch.cat([e_img, e_text_tokens], dim=1)

feats_A = obj["slat"].feats.detach()
feats_B, delta = edit_net(feats_A, e_joint)
```

#### Edited latent reconstruction and decode

```python
slat_edited = sp.SparseTensor(
    feats=feats_B,
    coords=slat_orig.coords,
    layout=slat_orig.layout if hasattr(slat_orig, 'layout') else None,
)
outputs_B = pipeline.decode_slat(slat_edited, ['gaussian'])
gaussian_B = outputs_B['gaussian'][0]
```

---

## 13. How backpropagation works and why differentiable rendering was needed

### 13.1 Backpropagation path

The training loss is not computed directly on the latent.

It is computed on **rendered views**.

So the actual gradient path is:

`loss -> rendered image tensor -> Gaussian renderer -> TRELLIS decoder -> edited latent -> delta -> EditNet`

Frozen TRELLIS components are still part of the computation graph. They are not updated, but they must pass gradients.

### 13.2 Why differentiable rendering was non-negotiable

This project uses image-level supervision through rendered outputs. That means the renderer must be differentiable.

If the renderer detaches tensors, the training loop becomes visually plausible but mathematically broken.

That is exactly what happened in the early path.

### 13.3 The NumPy problem

The convenience helper path used during early experimentation converted rendered outputs into NumPy arrays. Once that happened, the computation graph was cut.

The visible symptom was that the training loop could run, but the edit magnitude diagnostic `d_max` stayed at zero or near-zero in a way that indicated the editor was not truly receiving useful gradient signal.

### 13.4 The direct differentiable rendering fix

The fix was to stop using the detached helper path and instead call the Gaussian renderer directly through the differentiable route.

This was the critical change:

```python
outputs_B = pipeline.decode_slat(slat_edited, ['gaussian'])
gaussian_B = outputs_B['gaussian'][0]
renderer = render_utils.get_renderer(gaussian_B, resolution=RENDER_RES, bg_color=(0, 0, 0))

color_views = []
for vi in range(N_RENDER_VIEWS):
    res = renderer.render(gaussian_B, ext_list[vi], intr_list[vi])
    color_views.append(res['color'])

rendered = torch.stack(color_views).float().clamp(0, 1)
if rendered.shape[-1] != 224 or rendered.shape[-2] != 224:
    rendered = F.interpolate(rendered, size=(224, 224), mode='bilinear', align_corners=False)
```

This kept the tensors in torch, preserved the graph, and allowed CLIP-based prompt loss to backpropagate properly.

---

## 14. Losses used in the project

The repository documentation and combined report describe three ideas:

1. **prompt loss**
2. **preserve loss**
3. **delta regularization**

### Prompt loss

This encourages the rendered edited result to align with the edit prompt using CLIP similarity.

### Preserve term

This discourages the model from changing too much of the original object while applying the edit.

### Delta regularization

This penalizes overly large latent edits and helps stabilize training.

### d_max diagnostic

`d_max` is not itself a loss term. It is a diagnostic that tracks the maximum absolute latent edit magnitude.

It became one of the most useful debugging statistics in this project.

---

## 15. The latent-dimension mismatch issue and how it was resolved

One of the most important technical corrections in this project was the latent-dimension mismatch.

Early architecture notes assumed a 64-dimensional latent token width. However, runtime inspection of the actual TRELLIS tensor path used by the editor showed that the latent features exposed in the working path were **8-dimensional per active voxel**.

That mismatch affected:

- the input projection layer
- the output projection layer
- the dimensions of the residual update
- the entire parameterization of EditNet

### Resolution

The solution was to rebuild the editor around:

- `latent_dim = 8`
- `hidden_dim = 256`
- cross-attention over 1024-dimensional joint conditioning tokens

The corrected implementation is the current `trellis/models/edit_net.py`.

This issue matters because if the editor is dimensioned for the wrong latent width, even a perfectly written training loop will be operating on the wrong representation.

---

## 16. The VRAM exhaustion problem and how cached latents solved it

This is the second major issue to understand.

### 16.1 What the problem looked like

The system could appear to work for small early steps, but larger runs repeatedly collapsed into out-of-memory failures. This happened because the training loop had to hold:

- frozen TRELLIS models
- cached or live latent tensors
- CLIP
- decoded Gaussian representations
- differentiable rendered views
- intermediate autograd state

all on the same GPU.

### 16.2 Why this was more than “the GPU is too small”

The A40 GPU was not the only issue. The bigger issue was **what remained alive at what stage** of the pipeline.

Repeated encoding plus retained TRELLIS flow/image-conditioning modules created a memory chokehold.

### 16.3 The cached-latent solution

The final working strategy was:

1. cache source latents to disk once
2. load cached latents rather than source images for retraining
3. free heavy TRELLIS flow and image-conditioning modules after loading the latents
4. keep only the decoder path required for differentiable editing
5. enforce voxel-count cutoffs to skip oversized objects
6. use a scalable launcher with conservative runtime settings

### 16.4 Code that implements the memory fix

#### Load cached latents from disk

```python
cache_dir = Path(CACHE_DIR)
cache_files = sorted(cache_dir.glob("*.pt"))[:MAX_OBJECTS]
```

#### Skip large latent objects

```python
nvox = payload["slat"].feats.shape[0]
if nvox > MAX_VOXELS:
    print("SKIP {} too large ({} voxels > {})".format(
        payload.get("image_name", cache_path.name), nvox, MAX_VOXELS), flush=True)
    continue
```

#### Free TRELLIS flow/image-conditioning models

```python
del pipeline.models['sparse_structure_flow_model']
del pipeline.models['slat_flow_model']
del pipeline.models['image_cond_model']
torch.cuda.empty_cache()
gc.collect()
print("Freed ~20GB VRAM.", flush=True)
```

#### Periodic cleanup inside the training loop

```python
if step % 5 == 0:
    torch.cuda.empty_cache()
    gc.collect()
```

### 16.5 Why this solution changed the project

This was the turning point from “interesting but unstable” to “actually retrainable.”

The cached-latent approach is not just a convenience optimization. It is the practical foundation that made the method usable.

---

## 17. What I learned from the debugging process

This section focuses only on the two core lessons you asked to emphasize.

### 17.1 What I learned about differentiable rendering

The main lesson was that a training loop can look correct and still be silently broken.

I learned that:

- a renderer that is perfectly fine for visualization may be unusable for optimization
- converting tensors to NumPy is enough to destroy the learning signal
- checking only whether code runs is not enough
- it is necessary to inspect diagnostics such as `d_max`, gradient presence, and checkpoint behavior

The differentiable renderer was therefore not a polishing improvement. It was the condition that made learning possible.

### 17.2 What I learned about the memory chokehold issue

The memory issue taught me that GPU failure is often a **pipeline design problem**, not just a batch-size problem.

I learned that:

- keeping large frozen models resident during all stages is wasteful
- encoding once and caching is often more valuable than trying to fight the allocator indefinitely
- object size matters, so voxel-count filtering is practical and necessary
- scale should be increased one axis at a time
- a stable smaller run is more informative than a large unstable run

The most important memory lesson is that the training system has to be designed around the lifecycle of tensors and modules, not only around the neural network architecture.

---

## 18. How each file in the repository achieves the objective

This section ties the objective to the actual repository files.

### `README.md`
The public-facing summary of the project. It explains the overall goal, the latent editing idea, the final cached-latent approach, the main files, and the major losses and difficulties.

### `docs/01_background_and_problem.md`
Explains the starting problem: why image + edit prompt to 3D editing is hard and why a latent-editing strategy is appropriate.

### `docs/02_method_and_losses.md`
Explains the method, the residual latent-editing formulation, and the intended loss design.

### `docs/03_experiments_and_results.md`
Records what was tried experimentally and what the final results looked like.

### `docs/04_reproduction_guide.md`
Provides supporting reproduction notes for the experiments.

### `docs/generated_results.md`
Summarizes produced metrics, plots, and generated outputs.

### `docs/figures/`
Stores supporting figures and report-ready visual assets.

### `artifacts/`
Stores the evidence of the experiments:

- logs from smoke runs, fixed runs, reruns, and scale-up runs
- evaluation folders
- saved checkpoints such as `edit_net_epoch_5.pt` and `edit_net_epoch_20.pt`
- final logs such as `run_medium_safe_tuned.log`, `rerun_text_boost.log`, and `tiny_debug_run_2ep.log`

These files matter because they preserve the debugging and training history of the project.

### `scripts/generate_report_assets.py`
Takes logs and outputs and turns them into report-ready assets. This is the bridge from raw training artifacts to the written results under `docs/`.

### `trellis/models/edit_net.py`
Defines:

- `CrossAttentionBlock`
- `EditNet`
- `TextProjector`
- prompt lists/constants used by the training workflow

This is the trainable heart of the method.

### `train_edit_delta_cached.py`
The first stable cached-latent trainer. It is important historically because it established the working direction before the scalable trainer generalized the runtime controls.

### `train_edit_delta_scalable.py`
The main retraining script for practical use. It:

- loads cached latents
- loads the pair file
- configures itself from environment variables
- frees heavy TRELLIS modules
- builds the differentiable rendering loop
- computes losses
- saves checkpoints

This is the file a user should think of as the real training engine.

### `run_edit_train.sh`
The runtime wrapper that sets:

- CUDA toolkit path
- xformers attention backend
- spconv algorithm mode
- allocator settings

and then launches the scalable trainer.

This is the cleanest script for a user to run.

### `eval_rerun_checkpoint.py`
Loads a checkpoint and a cached latent, applies the trained editor, and renders the output again for inspection and comparison. This file closes the loop between training and qualitative evaluation.

---

## 19. Full pipeline with the files used at each stage

The full practical pipeline is:

```text
cached latents on disk + prompt pair JSON
        ↓
train_edit_delta_scalable.py
        ↓
load TRELLIS pipeline and cached latents
        ↓
free TRELLIS flow/image-conditioning modules
        ↓
load OpenCLIP
        ↓
trellis/models/edit_net.py
        ↓
predict latent delta
        ↓
pipeline.decode_slat(..., ['gaussian'])
        ↓
render_utils.get_renderer(...)
        ↓
renderer.render(...)
        ↓
CLIP prompt loss + preserve term + delta regularization
        ↓
backpropagate into EditNet and TextProjector only
        ↓
save checkpoints to artifacts / checkpoint directory
        ↓
eval_rerun_checkpoint.py for rerendering and comparison
        ↓
scripts/generate_report_assets.py for plots and report visuals
```

### Stage-by-stage file mapping

#### Stage 1 — data already prepared
- external cached latents: `/workspace/edit_training_data/encoded_latents/*.pt`
- external pair manifest: `/workspace/edit_training_data/training_pairs.json`

#### Stage 2 — runtime wrapper
- `run_edit_train.sh`

#### Stage 3 — training engine
- `train_edit_delta_scalable.py`
- `train_edit_delta_cached.py`

#### Stage 4 — model definition
- `trellis/models/edit_net.py`

#### Stage 5 — frozen TRELLIS decode and render path
- TRELLIS pipeline and decoders already available in the TRELLIS environment
- called from inside `train_edit_delta_scalable.py`

#### Stage 6 — checkpoint and rerender evaluation
- `eval_rerun_checkpoint.py`

#### Stage 7 — report and plot generation
- `scripts/generate_report_assets.py`
- `docs/`
- `artifacts/`

---

## 20. Final advice to the next user

A new user should not begin by editing the network architecture.

The right order is:

1. verify that cached latents and `training_pairs.json` are valid
2. run a tiny smoke test
3. scale objects, prompts, views, and epochs one axis at a time
4. keep `MAX_VOXELS` active
5. rely on `run_edit_train.sh` and environment variables instead of editing the trainer for every experiment
6. use checkpoints and evaluation scripts to compare runs

The most important practical insight is this:

**the main engineering difficulty is not writing EditNet. It is preserving gradient flow through rendering and preventing the training system from choking on memory.**

Once those two issues were resolved, the method became trainable and reproducible.

---

## 21. One-sentence beginner explanation

This project loads a cached TRELLIS latent for a source object, combines source-image conditioning with an edit prompt, predicts a bounded residual edit with EditNet, decodes the edited latent back into a Gaussian 3D representation, renders it differentiably, and trains the editor using prompt-alignment and regularization losses while keeping the heavy TRELLIS backbone frozen.
