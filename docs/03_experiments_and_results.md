# 03. Experiments and Results

## Early runs

Early runs were mainly for debugging and establishing whether:

- gradients were flowing
- checkpoints were saving
- differentiable rendering worked
- the loss terms could be computed correctly

At this stage, visible edits were weak.

Main observations:
- training technically ran
- losses and gradients existed
- `d_max` was initially too small
- rendered outputs often showed minimal visual change

## Intermediate improvements

The following changes improved behavior:

- switching to cached-latent training
- increasing the amount of prompt conditioning
- repeating/scaling projected text tokens
- running for more epochs
- increasing the allowed edit scale
- slightly relaxing delta regularization

These changes made the model move the latent more strongly.

## Larger scalable runs

A scalable training version was then created with environment-variable control for:

- number of epochs
- number of objects
- prompts per object
- number of rendered views
- render resolution
- text gain
- text repeat
- edit scale
- delta regularization
- voxel cutoff for stability

This made it possible to move from very small debug runs to larger experiments.

## Stability issues

As runs scaled up, memory issues appeared:

- CUDA allocator fragmentation
- internal allocator assertion failures
- occasional OOM events
- failures concentrated around larger latent objects and heavier decode paths

These issues did not always crash the entire run, but they reduced stability and required safer settings.

## Final 20-epoch run

The final experimental stopping point was a successful 20-epoch run.

Key outcomes:
- the run completed
- checkpoints were saved
- `d_max` increased substantially compared with early runs
- the edit model learned much stronger latent movement

Interpretation:
- the model was no longer stuck in a near-zero edit regime
- the latent update magnitude became strong enough to support meaningful edits
- loss remained noisy rather than sharply decreasing, which is consistent with CLIP-based supervision

## What the final run means

The final run demonstrates:
1. the pipeline is operational
2. prompt-conditioned latent editing is feasible in this setup
3. cached-latent training is the correct practical direction
4. scaling is possible, but memory stability becomes a central engineering constraint

## Limitations

The final system is still not the end state.

Current limitations include:
- CLIP prompt supervision remains noisy
- visible edits can still vary in strength depending on object and prompt
- appearance/geometry coupling remains an issue
- preservation can likely be improved further
- larger scale runs still need memory-safe settings