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
- a notebook-driven training workflow on top of the scalable trainer

This README has two purposes:

1. to show a new user exactly how to retrain the model with the files in this repository using the notebook workflow
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
├─ TRELLIS_Colabs.ipynb
├─ README.md
└─ .gitignore
```

### What each top-level part does

- `README.md` — project overview, training workflow, file map, and inference notes
- `TRELLIS_Colab.ipynb` — the main user-facing notebook for cached-latent generation, pair creation, training, log inspection, and inference after training
- `docs/` — written project documentation, results write-ups, and supporting figures
- `artifacts/` — saved logs, checkpoints, evaluation folders, and workspace metadata from runs
- `scripts/generate_report_assets.py` — helper used to turn logs/results into plots and report assets
- `trellis/models/edit_net.py` — the actual EditNet and TextProjector definitions
- `train_edit_delta_cached.py` — earlier cached-latent trainer kept as a reference/baseline
- `train_edit_delta_scalable.py` — the main scalable training engine now used underneath the notebook workflow
- `run_edit_train.sh` — launcher that sets the expected runtime environment and starts scalable training
- `eval_rerun_checkpoint.py` — checkpoint inspection and rerendering script used for inference/evaluation after training

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

For most users, the main entry point is now the notebook:

```text
TRELLIS_Colab_Fresh_Stable_EditNet_Cleaned.ipynb
```

This notebook is the recommended user workflow because it:

- keeps deployment and training clearly separated
- generates cached latents from source images
- builds `training_pairs.json`
- runs smoke, medium, or larger training through the scalable trainer
- shows logs and checkpoints directly in sequence
- provides an inference/evaluation step after training

So the practical answer is:

- **use `TRELLIS_Colab.ipynb` for normal retraining**
- **use `train_edit_delta_scalable.py` if you want to inspect or modify the backend trainer directly**
- **use `train_edit_delta_cached.py` only as an earlier baseline/reference**
- **use `eval_rerun_checkpoint.py` for checkpoint-based rerendering and inference after training**

The notebook is now the easiest way for a new user to understand the whole pipeline end to end.

---

## 8. Notebook-based retraining guide for a user

This section is the practical handover guide.

### Step 1 — Complete the deployment section first

The deployment section of the notebook prepares the TRELLIS environment, installs dependencies, and confirms that the official TRELLIS setup works.

**Do not skip or rewrite that part.**

### Step 2 — Open the EditNet training section of the notebook

After deployment succeeds, continue to the **EditNet** section of the notebook.

That section handles the retraining workflow in this order:

1. overlay the repository files into the TRELLIS checkout
2. create `trellis/utils/edit_losses.py` if needed
3. create `/workspace` path mappings and training directories
4. verify the training scripts and source images
5. encode source images into cached latents
6. build `training_pairs.json`
7. run smoke or medium training
8. inspect logs and checkpoints
9. run inference/evaluation from a saved checkpoint

### Step 3 — Where encoded latents are created

In the notebook, the cached latent generation happens in the cell that:

- reads source images from `/workspace/TRELLIS/assets/example_image/`
- runs the TRELLIS image pipeline on each image
- saves one `.pt` payload per image into:

```text
/workspace/edit_training_data/encoded_latents/
```

Each payload stores at least:

- `slat`
- `cond`
- `image_name`

This is the point in the notebook where the training dataset is actually created for EditNet.

### Step 4 — Where `training_pairs.json` is created

The notebook then creates:

```text
/workspace/edit_training_data/training_pairs.json
```

That cell builds a prompt manifest by matching cached latent file stems to edit prompts.

Examples:

- 1 image × 1 prompt each for a smoke test
- N images × 3 prompts each for a medium run
- 100 images × 1 prompt each for exactly 100 training pairs
- larger prompt-per-image settings for stronger GPUs

### Step 5 — Run smoke or medium training from the notebook

The notebook exposes the main training controls through environment variables such as:

```python
EDIT_EPOCHS
EDIT_MAX_OBJECTS
EDIT_PROMPTS_PER_OBJECT
EDIT_N_VIEWS
EDIT_RENDER_RES
EDIT_TEXT_GAIN
EDIT_TEXT_REPEAT
EDIT_SCALE
EDIT_LAMBDA_DELTA
EDIT_LAMBDA_PRESERVE
EDIT_MAX_VOXELS
EDIT_CACHE_DIR
EDIT_PAIRS_JSON
EDIT_CKPT_DIR
EDIT_LOG
```

The notebook includes cells for:

- a tiny smoke run
- a medium run
- a larger run for stronger GPUs

### Step 6 — Where checkpoints are saved

The training cells write checkpoints into:

```text
/workspace/edit_checkpoints/
```

Typical checkpoint filenames look like:

```text
edit_net_epoch_5.pt
edit_net_epoch_10.pt
edit_net_epoch_20.pt
```

These are the files used later for inference and evaluation.

---

## 9. Inference after training

After training finishes, run the inference cell:


### Minimal inference workflow

```text
checkpoint + cached latent -> EditNet/TextProjector reload -> edited latent -> Gaussian decode -> rerendered output
```

### What inference is for in this repo

Inference here is mainly used to:

- inspect what a checkpoint is actually producing
- compare checkpoints qualitatively
- verify that the trained editor applies the intended prompt-conditioned edit
- generate rerendered outputs for reporting and demonstrations

---

## 10. Recommended scaling order in the notebook

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
   - 1–2 views
   - 128 resolution
   - 5 epochs

3. **medium experiment**
   - 20 objects
   - 3 prompts per object
   - 1–2 views
   - 128 resolution
   - 10–20 epochs

4. **careful scale-up**
   - 40–100 objects
   - keep `MAX_VOXELS` enforced
   - increase prompts or views only one axis at a time

This scaling order matters because the real bottleneck is memory and render cost, not the EditNet module itself.

---

## 11. Canonical workflow for a new user

The **canonical retraining path** in this repository is now:

1. complete the notebook deployment section
2. open the EditNet notebook section
3. place source images in `/workspace/TRELLIS/assets/example_image/`
4. encode them to `/workspace/edit_training_data/encoded_latents/`
5. build `/workspace/edit_training_data/training_pairs.json`
6. run a smoke or medium training cell
7. inspect logs written to `EDIT_LOG`
8. inspect saved checkpoints in `/workspace/edit_checkpoints/`
9. run notebook inference or `eval_rerun_checkpoint.py`

The most important practical rule is this:

- **the notebook is the primary user workflow**
- **`train_edit_delta_scalable.py` is the backend trainer used by the notebook**
- **`eval_rerun_checkpoint.py` is the standalone inference/evaluation path after training**

So, for a new user, the repository should now be approached as:

`deployment notebook section -> EditNet notebook section -> cached latents -> training_pairs.json -> scalable trainer -> checkpoints -> inference`

---

## 12. What the main training files do

### `TRELLIS_Colab_Fresh_Stable_EditNet_Cleaned.ipynb`
This is the main practical notebook for:

- preparing the Colab/TRELLIS runtime
- creating cached latents
- building pair manifests
- launching training
- inspecting logs and checkpoints
- running inference after training

### `trellis/models/edit_net.py`
Defines:

- `CrossAttentionBlock`
- `EditNet`
- `TextProjector`
- the prompt list used by the training workflow

This is the trainable heart of the method.

### `train_edit_delta_scalable.py`
The main backend training engine. It:

- loads cached latents
- loads the pair file
- configures itself from environment variables
- frees heavy TRELLIS modules
- builds the differentiable rendering loop
- computes losses
- saves checkpoints

This is the file the notebook ultimately calls for practical retraining.

### `train_edit_delta_cached.py`
Earlier cached-latent trainer. It is kept as a historical baseline/reference, but the notebook workflow should treat the scalable trainer as the main path.

### `run_edit_train.sh`
Runtime wrapper that sets:

- CUDA toolkit path
- xformers attention backend
- spconv algorithm mode
- allocator settings

and then launches the scalable trainer.

### `eval_rerun_checkpoint.py`
Loads a checkpoint and a cached latent, applies the trained editor, and rerenders the result for qualitative inspection.

### `scripts/generate_report_assets.py`
Takes logs and outputs and turns them into report-ready assets.

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

### 13.3 The direct differentiable rendering fix

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

## 14. The latent-dimension mismatch issue and how it was resolved

One of the most important technical corrections in this project was the latent-dimension mismatch.

Early architecture notes assumed a 64-dimensional latent token width. However, runtime inspection of the actual TRELLIS tensor path used by the editor showed that the latent features exposed in the working path were **8-dimensional per active voxel**.

The solution was to rebuild the editor around:

- `latent_dim = 8`
- `hidden_dim = 256`
- cross-attention over 1024-dimensional joint conditioning tokens

The corrected implementation is the current `trellis/models/edit_net.py`.

---

## 15. The VRAM exhaustion problem and how cached latents solved it

The final working strategy was:

1. encode source images once into cached latents
2. reload cached latents rather than re-encoding source images every time
3. free heavy TRELLIS flow and image-conditioning modules
4. keep only the decoder path required for differentiable editing
5. enforce voxel-count cutoffs to skip oversized objects
6. use conservative runtime settings first, then scale up gradually

This was the turning point from “interesting but unstable” to “actually retrainable.”

---

## 16. Final advice to the next user

A new user should not begin by editing the network architecture.

The right order is:

1. complete deployment
2. use the notebook to create cached latents and `training_pairs.json`
3. run a tiny smoke test
4. scale objects, prompts, views, and epochs one axis at a time
5. keep `MAX_VOXELS` active
6. inspect checkpoints with the notebook or `eval_rerun_checkpoint.py`

The most important practical insight is this:

**the main engineering difficulty is not writing EditNet. It is preserving gradient flow through rendering and preventing the training system from choking on memory.**

Once those two issues were resolved, the method became trainable and reproducible.

---

## 17. One-sentence beginner explanation

This project uses a notebook-driven TRELLIS 1 workflow to encode source images into cached latents, pair them with edit prompts, train a prompt-conditioned residual editor called EditNet, save checkpoints, and rerender edited 3D outputs for inference after training.
