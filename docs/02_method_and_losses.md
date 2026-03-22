# 02. Method and Losses

## Final working method

The final stable method used **cached-latent training**.

Instead of re-running the TRELLIS image encoder inside every training iteration, the pipeline first cached TRELLIS latents to disk and then trained an edit model over those cached representations.

This made the workflow:
- faster
- easier to repeat
- more scalable
- easier to debug

## Training pipeline

### Step 1: cache object latents

Input images were encoded with TRELLIS into structured latents and stored.

### Step 2: build prompt pairs

Each object latent was paired with one or more edit prompts.

### Step 3: freeze large pretrained components

The major TRELLIS backbone components were frozen.
CLIP was also used in frozen mode for supervision.

### Step 4: train only the edit modules

The main trainable modules were:

- **EditNet**
- **TextProjector**

### Step 5: decode edited latent

The edited latent was decoded into a Gaussian representation using TRELLIS decoders.

### Step 6: differentiable rendering

The Gaussian representation was rendered into images that remained differentiable, allowing supervision from image-space losses.

### Step 7: optimize a combined objective

The rendered images and latent deltas were optimized jointly through several losses.

## Conditioning design

The edit model used:
- image-derived conditioning
- text-derived conditioning

In later, improved runs, text conditioning was strengthened by:
- projecting text embeddings into the conditioning space
- repeating projected text tokens
- scaling text token magnitude

This was done because early runs showed that the text signal was too weak.

## Main losses

### 1. Prompt loss

The prompt loss encourages the rendered edited output to match the edit prompt.

Implementation idea:
- encode rendered image with CLIP image encoder
- encode edit prompt with CLIP text encoder
- maximize similarity

Interpretation:
- lower prompt loss / higher similarity indicates stronger prompt alignment

### 2. Preserve loss

The preserve loss discourages the system from drifting too far from the source object.

This is necessary because the model could otherwise produce prompt-aligned outputs that no longer preserve object identity.

Interpretation:
- lower preserve loss indicates stronger retention of the original object

### 3. Delta regularization

Delta regularization penalizes large latent modifications.

This stabilizes training but creates a tradeoff:
- too large a penalty -> edits become too weak
- too small a penalty -> edits can become unstable or destructive

### 4. d_max

`d_max` tracks the maximum absolute latent edit magnitude.

This was one of the most useful diagnostics because it reveals whether the model is actually moving the latent enough to produce visible edits.

Interpretation:
- tiny `d_max` -> edits too weak
- medium `d_max` -> useful editing regime
- saturated `d_max` -> model is using the full allowed edit budget

## Engineering changes that mattered

The following changes were important in practice:

- moving from repeated image re-encoding to cached latents
- increasing text-conditioning strength
- increasing training duration
- scaling from very small experiments to medium runs
- introducing safer memory settings
- skipping oversized cached latents in large runs
- adding periodic cleanup to reduce allocator issues