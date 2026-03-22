# 01. Background and Problem

## Task definition

The target task is **2D image + edit prompt -> edited 3D representation**.

This means:

- start with one image of an object
- reconstruct or represent the object in a 3D latent space
- modify that 3D latent according to a textual edit prompt
- decode the modified latent back into a renderable 3D form

Examples of edit prompts include:

- make it red
- make it wooden
- make it glass
- make it futuristic
- make it rusty

## Why this is difficult

This task is difficult for several reasons.

### 1. Single-view ambiguity

A single 2D image does not fully determine the full 3D geometry of an object. The system must infer hidden structure.

### 2. Prompt ambiguity

Text prompts are high-level and underspecified. For example, “make it wooden” could mean:
- wooden texture only
- darker brown appearance
- polished wood
- rough wood
- structural changes toward a wood-crafted object

### 3. 3D consistency requirement

In ordinary 2D image editing, the system only has to produce one convincing output image.
In 3D editing, the result must stay coherent across viewpoints.

### 4. Appearance and structure are coupled

In TRELLIS 1, the latent space is not perfectly disentangled into “geometry-only” and “appearance-only”.
Therefore, pushing too strongly on appearance may unintentionally distort geometry.

## Why TRELLIS 1 was selected

TRELLIS 1 already provides a strong pretrained backbone for image-conditioned 3D latent generation.

This gives:

- an image-conditioned latent pipeline
- structured latent representations
- decoders to Gaussian 3D representations
- differentiable rendering support

Using TRELLIS 1 made it possible to study **editing of an existing 3D latent**, rather than solving the entire text-to-3D problem from scratch.

## Design intuition

The intended edit process is:

1. obtain an original object latent `z_A`
2. compute a prompt-conditioned delta
3. produce an edited latent `z_B = z_A + delta`
4. decode `z_B` into a 3D representation
5. render and evaluate the result

The entire project revolves around making this delta:
- meaningful
- prompt-sensitive
- stable
- non-destructive