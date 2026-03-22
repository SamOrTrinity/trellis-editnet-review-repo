# TRELLIS 1 EditNet for 2D Image + Edit Prompt to 3D Editing

## Overview

This repository contains my implementation and experimental workflow for editing 3D representations using a single input image and a text edit prompt.

The practical goal is:

- input: a 2D image of an object
- input: a text instruction such as "make it wooden" or "make it glass"
- output: an edited 3D representation that follows the instruction while preserving the original object as much as possible

This work is built on top of TRELLIS 1, which already provides a pretrained pipeline for image-conditioned 3D latent generation.

Instead of training a full text-to-3D model from scratch, this project trains a smaller EditNet that modifies an existing TRELLIS latent using prompt conditioning.

---

## Problem Statement

Editing a 3D object from only:
1. a single image
2. a text edit prompt

is difficult because the model must satisfy several constraints at once:

- the edit should follow the prompt
- the object should still resemble the original object
- the 3D structure should remain coherent
- the result should remain renderable from multiple views
- the latent updates should stay numerically stable

This is harder than ordinary 2D image editing because the system must preserve multi-view 3D consistency, not just one image.

---

## Why TRELLIS 1 Was Used

TRELLIS 1 already provides:

- a pretrained image-to-3D latent pipeline
- structured latent representations
- decoders that turn latents into Gaussian 3D representations
- differentiable rendering support

This made it possible to study latent editing rather than build an entire 3D generation model from zero.

---

## Core Idea

The central idea is to learn a prompt-conditioned latent delta.

If:
- `z_A` = original latent
- `z_B` = edited latent

then the model learns a residual update:
- `z_B = z_A + delta`

where `delta` depends on:

- the original latent
- the image conditioning
- the text conditioning

The trainable module that predicts this delta is called **EditNet**.

---

## Final Working Approach

The final stable approach used in this repository is:

1. Encode images once into cached TRELLIS latents
2. Store those latents on disk
3. Build training pairs using cached object latents and edit prompts
4. Freeze the large TRELLIS backbone components
5. Train only:
   - EditNet
   - TextProjector
6. Decode the edited latent back into a Gaussian representation
7. Render the Gaussian representation differentiably
8. Use CLIP-based supervision and regularization losses

This cached-latent strategy was much more practical and scalable than re-encoding objects during every run.

---

## Main Files

- `train_edit_delta_cached.py`  
  First stable cached-latent trainer.

- `train_edit_delta_scalable.py`  
  Scalable trainer with environment-variable configuration for larger runs.

- `run_edit_train.sh`  
  Launcher script used for training.

- `trellis/models/edit_net.py`  
  Contains the edit model and text projection components.

- `eval_rerun_checkpoint.py`  
  Used to render and compare edited outputs from saved checkpoints.

---

## Losses Used

The training combines three main loss ideas.

### 1. Prompt loss

This encourages the rendered edited result to align with the edit prompt using CLIP similarity.

### 2. Preserve loss

This discourages the model from destroying the original object while applying the edit.

### 3. Delta regularization

This penalizes overly large latent edits so the system stays stable.

### 4. d_max diagnostic

`d_max` tracks the maximum absolute latent edit magnitude and became one of the most useful diagnostics during experimentation.

Interpretation:

- very low `d_max` -> almost no edit
- moderate `d_max` -> visible edits possible
- high `d_max` -> stronger edits but higher risk of drift or instability

---

## What Worked

- cached latent training
- freezing most TRELLIS components
- prompt-conditioned residual editing
- stronger text conditioning
- scaling to more objects and prompts
- saving frequent checkpoints and logs

---

## Main Difficulties

- CLIP prompt supervision is noisy
- total loss does not necessarily decrease smoothly
- stronger edits can distort geometry
- large runs caused CUDA allocator / OOM instability
- oversized latent objects were harder to train safely

---

## Final Experimental State

By the final experiments:

- training completed for a full 20-epoch run
- `d_max` increased substantially relative to early runs
- the model performed meaningful latent updates
- training remained usable, though larger runs still required care for memory stability

The final run is documented through:

- logs in `artifacts/`
- checkpoints in `artifacts/`
- plots in `docs/figures/`

---

## Recommended Reading Order

1. `README.md`
2. `docs/01_background_and_problem.md`
3. `docs/02_method_and_losses.md`
4. `docs/03_experiments_and_results.md`
5. `docs/generated_results.md`
6. `docs/04_reproduction_guide.md`
7. `docs/05_reviewer_guide.md`

---

## Status

This repository is the preserved engineering state of the project at the end of the current experimentation phase.

It is intended so that:

- a reviewer can understand what was implemented
- a future contributor can continue the work
- the project can be resumed later after the RunPod session is terminated