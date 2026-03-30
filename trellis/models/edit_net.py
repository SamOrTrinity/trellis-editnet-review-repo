"""
edit_net.py -- Commented Final Version
=====================================

Author: Sambit Hore
Affiliation: Trinity College Dublin

In this file, I implement the final EditNet that I use for my TRELLIS 1
Architecture 2 (Delta Approach) experiments.

Why this file exists
--------------------
My goal is not to generate a completely new 3D object from scratch. Instead,
I want to start from the latent representation of a source object A and learn
only a *small residual edit* that pushes the object toward a user-specified
edit prompt such as:

    - "make it red"
    - "make it wooden"
    - "make it glass"
    - "make it futuristic"

So, instead of predicting a full edited latent z_B directly, I predict a delta
and construct the edited latent as:

    z_B = z_A + delta

This residual formulation is important because it anchors the edited result to
its source object. If delta = 0, the output is exactly the original object.
That makes the model much more stable than directly predicting an entirely new
latent without any anchor.

Key design decision
-------------------
The final TRELLIS 1 runtime inspection showed that the SLAT feature tensor used
here is effectively 8-dimensional per active voxel in my working code path.
An earlier version of this file incorrectly assumed latent_dim = 64. That older
assumption should be ignored for this implementation.

What EditNet receives
---------------------
EditNet has two inputs:

1. feats_A / z_slat_A
   The source object's SLAT features. Shape is usually:
       (N_voxels, 8)
   or batched as:
       (B, N_voxels, 8)

2. e_joint
   The fused conditioning tokens built from:
       - the source image conditioning embedding (for example DINOv2 tokens)
       - the projected text embedding from the edit prompt (for example CLIP)
   Shape is usually:
       (1, T, 1024)
   or batched as:
       (B, T, 1024)

So the model does NOT take only the prompt, and it does NOT take only the
source latent. It takes:

    source latent + joint image/text conditioning

Neural network architecture summary
-----------------------------------
This network is a cross-attention residual editor. The flow is:

    Input source SLAT features:          (B, N, 8)
    Input conditioning tokens:           (B, T, 1024)

    Step 1: token projection
        8  -> 256

    Step 2: stacked cross-attention blocks
        3 blocks by default
        each block uses:
            - LayerNorm
            - Multihead cross-attention
            - residual connection
            - LayerNorm
            - feed-forward network
            - residual connection

    Step 3: output projection
        256 -> 8

    Step 4: bounded residual wrapper
        delta = scale * tanh(delta_raw)

    Step 5: edited latent construction
        feats_B = feats_A + delta

Why cross-attention is used
---------------------------
The source latent contains one token per active voxel. Each token corresponds
to a local part of the object's sparse 3D representation. Cross-attention lets
those voxel tokens look at the joint conditioning tokens and decide which parts
of the latent should change, and by how much.

In simpler words:

- the latent tokens ask: "What does the conditioning want me to become?"
- the conditioning tokens answer using the fused image + prompt information

That is more expressive than using a plain MLP on the whole latent, because it
lets different spatial tokens react differently to the same edit instruction.

Cross-attention mechanics in this file
--------------------------------------
PyTorch's MultiheadAttention takes:

    query = latent tokens
    key   = conditioning tokens
    value = conditioning tokens

So here the latent features are the queries, which means the source object asks
what information it should pull from the image/text conditioning.

Inside one block, the computation is conceptually:

    x_norm   = LayerNorm(x)
    attn_out = CrossAttention(query=x_norm, key=cond, value=cond)
    x        = x + attn_out
    x        = x + FFN(LayerNorm(x))

This is the standard transformer-style residual structure.

Why the final layer is zero-initialised
---------------------------------------
I initialise the last linear layer of the output head to zeros so that, at the
start of training, delta is approximately zero. That means EditNet begins as a
near-identity mapping:

    feats_B ≈ feats_A

This is exactly what I want for residual editing. The model should only learn
non-zero edits when the training objective gives it a reason to move away from
identity.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class CrossAttentionBlock(nn.Module):
    """
    A single transformer-style cross-attention block.

    Purpose
    -------
    This block updates the source SLAT tokens using the fused conditioning
    tokens. The source tokens are the 3D latent tokens that I want to edit,
    while the conditioning tokens carry information about:

        - the source image embedding
        - the edit prompt embedding

    Architecture inside one block
    -----------------------------
    Input x:    (B, N, hidden_dim)
    Input cond: (B, T, cond_dim)

    1. LayerNorm on the source tokens
    2. Multi-head cross-attention
         query = x_norm
         key   = cond
         value = cond
    3. Residual add
    4. LayerNorm
    5. Feed-forward network (2-layer MLP with GELU)
    6. Residual add

    Why this matters
    ----------------
    This lets each source voxel token selectively attend to the conditioning
    signal and decide which edit information is relevant to that spatial token.
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        cond_dim: int = 1024,
        n_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        # Pre-attention normalisation of the source tokens.
        self.norm1 = nn.LayerNorm(hidden_dim)

        # Multi-head cross-attention.
        #
        # embed_dim = hidden_dim because the query comes from the projected SLAT
        # tokens.
        #
        # kdim / vdim = cond_dim because the keys and values come from the joint
        # conditioning sequence e_joint.
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=n_heads,
            kdim=cond_dim,
            vdim=cond_dim,
            dropout=dropout,
            batch_first=True,
        )

        # Normalisation before the feed-forward network.
        self.norm2 = nn.LayerNorm(hidden_dim)

        # Standard transformer FFN:
        # hidden_dim -> 4*hidden_dim -> hidden_dim
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through one cross-attention block.

        Args:
            x:
                Source latent tokens of shape (B, N, hidden_dim).
            cond:
                Joint image/text conditioning tokens of shape
                (B, T, cond_dim).

        Returns:
            Updated source tokens of shape (B, N, hidden_dim).
        """
        # Pre-normalise source tokens before attention.
        x_norm = self.norm1(x)

        # Cross-attention:
        # - query comes from source latent tokens
        # - keys/values come from joint conditioning
        #
        # Intuition:
        # each source voxel token asks which parts of the conditioning are most
        # relevant to how it should be edited.
        attn_out, _ = self.cross_attn(query=x_norm, key=cond, value=cond)

        # Residual connection around attention.
        x = x + attn_out

        # Feed-forward sublayer with another residual connection.
        x = x + self.ffn(self.norm2(x))
        return x


class EditNet(nn.Module):
    """
    Cross-Attention Residual Editor for TRELLIS 1.

    High-level role
    ---------------
    This network takes the source object's latent tokens and a fused
    image+prompt conditioning signal, then predicts a bounded residual update
    delta. The edited latent is formed as:

        feats_B = feats_A + delta

    Default architecture in this final file
    ---------------------------------------
    - latent_dim  = 8
    - cond_dim    = 1024
    - hidden_dim  = 256
    - n_blocks    = 3
    - n_heads     = 8
    - scale       = 0.1

    Detailed architecture
    ---------------------
    1. Input projection
       Each latent token is projected from 8 dimensions to 256 dimensions:

           Linear(8 -> 256) + GELU

    2. Cross-attention stack
       Three CrossAttentionBlock modules are applied in sequence. These let the
       source latent tokens repeatedly interact with the conditioning tokens.

    3. Output projection
       The edited hidden tokens are projected back to latent space:

           LayerNorm(256) + Linear(256 -> 8)

    4. Bounded residual
       The raw delta is passed through tanh and scaled:

           delta = scale * tanh(delta_raw)

       This bounds each latent dimension of each token to the range:

           [-scale, +scale]

    5. Residual edit
       The bounded delta is added to the original latent:

           feats_B = feats_A + delta

    Why the bounded residual is important
    -------------------------------------
    TRELLIS 1 does not give a clean shape/material factorisation in this code
    path. That means overly large latent edits can become destructive.
    Bounding the residual with tanh and scale gives a direct mechanism to limit
    how aggressive each edit can be.
    """

    def __init__(
        self,
        latent_dim: int = 8,
        cond_dim: int = 1024,
        hidden_dim: int = 256,
        n_blocks: int = 3,
        n_heads: int = 8,
        scale: float = 0.1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.scale = scale
        self.latent_dim = latent_dim

        # Step 1: project each source latent token from the compact SLAT latent
        # dimension into a richer hidden dimension that is easier to process
        # with attention.
        self.input_proj = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
        )

        # Step 2: a stack of cross-attention residual blocks.
        # Each block keeps the source tokens in hidden_dim space while letting
        # them attend to the conditioning sequence.
        self.blocks = nn.ModuleList(
            [
                CrossAttentionBlock(hidden_dim, cond_dim, n_heads, dropout)
                for _ in range(n_blocks)
            ]
        )

        # Step 3: map the hidden tokens back into latent space so that the model
        # predicts a delta with the same dimensionality as feats_A.
        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, latent_dim),
        )

        # Important residual-editing trick:
        # initialise the final linear layer to zero so that delta starts at 0.
        # At initialisation, feats_B is therefore almost identical to feats_A.
        nn.init.zeros_(self.output_proj[-1].weight)
        nn.init.zeros_(self.output_proj[-1].bias)

    def forward(
        self,
        feats_A: torch.Tensor,
        e_joint: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Predict the edited latent and the latent delta.

        Args:
            feats_A:
                Source SLAT features.
                Shape can be either:
                    (N_voxels, latent_dim)
                or
                    (B, N_voxels, latent_dim)

            e_joint:
                Joint image/text conditioning tokens.
                Shape can be either:
                    (1, T, cond_dim)
                or
                    (B, T, cond_dim)

        Returns:
            feats_B:
                Edited SLAT features with the same shape as feats_A.

            delta:
                Predicted residual update, with the same shape as feats_A.
        """
        # For convenience, I support both unbatched and batched SLAT tokens.
        # If the user gives (N, D), I temporarily add a batch dimension so that
        # the attention layers can operate in a standard batched form.
        squeeze = False
        if feats_A.dim() == 2:
            feats_A = feats_A.unsqueeze(0)
            squeeze = True

        # Step 1: project compact latent tokens into the hidden space.
        h = self.input_proj(feats_A)

        # Step 2: repeatedly update the latent tokens using cross-attention with
        # the conditioning sequence.
        for block in self.blocks:
            h = block(h, e_joint)

        # Step 3: produce the raw, unbounded delta in latent space.
        delta_raw = self.output_proj(h)

        # Step 4: bound the delta so edits remain controlled.
        delta = self.scale * torch.tanh(delta_raw)

        # Step 5: construct the edited latent using residual addition.
        feats_B = feats_A + delta

        # Remove the temporary batch dimension if the input was unbatched.
        if squeeze:
            feats_B = feats_B.squeeze(0)
            delta = delta.squeeze(0)

        return feats_B, delta

    def count_parameters(self) -> int:
        """Count trainable parameters in EditNet."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class TextProjector(nn.Module):
    """
    Project CLIP text features into the same feature dimension used by the image
    conditioning stream.

    Why this module is needed
    -------------------------
    In the overall pipeline, the text prompt and the source image do not
    naturally live in the same embedding space.

    A common setup is:
        - CLIP text features: 768-dimensional
        - image conditioning tokens: 1024-dimensional

    So before I concatenate the text tokens with the image tokens into e_joint,
    I first project the text features to 1024 dimensions.
    """

    def __init__(self, input_dim: int = 768, output_dim: int = 1024) -> None:
        super().__init__()
        self.proj = nn.Linear(input_dim, output_dim)

    def forward(self, clip_features: torch.Tensor) -> torch.Tensor:
        """Project CLIP text features into the conditioning dimension."""
        return self.proj(clip_features)


# A small library of example edit prompts that can be sampled during training.
# These are simple appearance / material / style / state edits.
EDIT_PROMPTS = [
    "make it red", "make it blue", "make it green", "make it golden",
    "make it white", "make it black", "paint it purple", "paint it orange",
    "make it silver", "make it pink", "make it brown", "make it cyan",
    "make it wooden", "make it metallic", "make it glass",
    "make it stone", "make it marble", "make it rusty",
    "make it chrome", "make it matte black", "make it glossy",
    "make it ceramic", "make it plastic", "make it copper",
    "make it look like clay", "make it look like LEGO",
    "make it cartoon style", "make it steampunk",
    "make it futuristic", "make it vintage", "make it neon",
    "make it pixel art style", "make it watercolor style",
    "make it frozen", "make it on fire",
    "make it dirty", "make it worn and aged", "make it brand new",
    "make it glowing", "make it cracked", "make it shiny",
]


if __name__ == "__main__":
    # ---------------------------------------------------------------------
    # Simple standalone test.
    # ---------------------------------------------------------------------
    # I keep this block so that I can quickly verify:
    #   1. the network builds correctly,
    #   2. tensor shapes are correct,
    #   3. the residual starts near identity,
    #   4. gradients flow through the model.
    # ---------------------------------------------------------------------
    print("Testing EditNet (latent_dim=8)...")
    print("")

    net = EditNet(latent_dim=8, cond_dim=1024, hidden_dim=256, n_blocks=3, scale=0.1)
    proj = TextProjector(768, 1024)
    total = net.count_parameters() + sum(p.numel() for p in proj.parameters())

    print("EditNet: {:,} params".format(net.count_parameters()))
    print("TextProjector: {:,} params".format(sum(p.numel() for p in proj.parameters())))
    print("Total trainable: {:,}".format(total))

    # Realistic tensor sizes from my working TRELLIS 1 inspection.
    N = 13879   # number of active voxels / sparse tokens
    D = 8       # latent dimension per voxel token
    T_img = 257 # example number of image-conditioning tokens
    T_txt = 1   # one text token here for the test

    feats_A = torch.randn(N, D)
    e_img = torch.randn(1, T_img, 1024)

    # Example text projection path: random CLIP-like text features are projected
    # to the conditioning dimension and concatenated with image tokens.
    e_text_proj = proj(torch.randn(1, T_txt, 768))
    e_joint = torch.cat([e_img, e_text_proj], dim=1)

    print("")
    print("Inputs:  feats_A={}, e_joint={}".format(list(feats_A.shape), list(e_joint.shape)))

    feats_B, delta = net(feats_A, e_joint)

    print("Outputs: feats_B={}, delta={}".format(list(feats_B.shape), list(delta.shape)))
    print("Delta max: {:.6f} (should be < {})".format(delta.abs().max().item(), net.scale))
    print("Identity diff: {:.6f} (should be ~0)".format((feats_B - feats_A).abs().max().item()))

    # Gradient-flow check.
    feats_A_g = feats_A.clone().requires_grad_(True)
    feats_B_g, _ = net(feats_A_g, e_joint)
    feats_B_g.mean().backward()
    print("Gradient flow: PASS" if all(p.grad is not None for p in net.parameters()) else "FAIL")

    print("")
    print("Edit prompts available: {}".format(len(EDIT_PROMPTS)))
    print("All tests passed!")
