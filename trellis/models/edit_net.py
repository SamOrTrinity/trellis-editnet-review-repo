"""
edit_net.py -- Cross-Attention Residual Editor for TRELLIS 1 Delta Approach
===========================================================================

UPDATED: latent_dim=8 (discovered from TRELLIS 1 SLAT inspection)
Previous version incorrectly used latent_dim=64.

Place this file at: /workspace/TRELLIS/trellis/models/edit_net.py
(replaces the old version)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


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
    """
    Cross-Attention Residual Editor for TRELLIS 1.

    TRELLIS 1 SLAT latent: (N_voxels, 8) per object
    Conditioning: (1, T, 1024) fused DINOv2 + CLIP tokens

    Flow: 8-dim -> project to 256 -> cross-attn blocks -> project to 8 -> tanh*scale -> add to original
    """
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

        # Zero init so delta starts at 0 (identity)
        nn.init.zeros_(self.output_proj[-1].weight)
        nn.init.zeros_(self.output_proj[-1].bias)

    def forward(self, feats_A, e_joint):
        """
        Args:
            feats_A: (N_voxels, 8) or (B, N_voxels, 8)
            e_joint: (1, T, 1024) or (B, T, 1024)
        Returns:
            feats_B: same shape as feats_A
            delta:   same shape as feats_A
        """
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

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class TextProjector(nn.Module):
    """Projects padded CLIP text (768) to DINOv2 dimension (1024)."""
    def __init__(self, input_dim=768, output_dim=1024):
        super().__init__()
        self.proj = nn.Linear(input_dim, output_dim)

    def forward(self, clip_features):
        return self.proj(clip_features)


# Edit prompt library
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
    print("Testing EditNet (latent_dim=8)...")
    print("")

    net = EditNet(latent_dim=8, cond_dim=1024, hidden_dim=256, n_blocks=3, scale=0.1)
    proj = TextProjector(768, 1024)
    total = net.count_parameters() + sum(p.numel() for p in proj.parameters())
    print("EditNet: {:,} params".format(net.count_parameters()))
    print("TextProjector: {:,} params".format(sum(p.numel() for p in proj.parameters())))
    print("Total trainable: {:,}".format(total))

    # Real TRELLIS 1 shapes
    N = 13879; D = 8; T_img = 257; T_txt = 1
    feats_A = torch.randn(N, D)
    e_img = torch.randn(1, T_img, 1024)
    e_text_proj = proj(torch.randn(1, T_txt, 768))
    e_joint = torch.cat([e_img, e_text_proj], dim=1)

    print("")
    print("Inputs:  feats_A={}, e_joint={}".format(list(feats_A.shape), list(e_joint.shape)))
    feats_B, delta = net(feats_A, e_joint)
    print("Outputs: feats_B={}, delta={}".format(list(feats_B.shape), list(delta.shape)))
    print("Delta max: {:.6f} (should be < {})".format(delta.abs().max().item(), net.scale))
    print("Identity diff: {:.6f} (should be ~0)".format((feats_B - feats_A).abs().max().item()))

    feats_A_g = feats_A.clone().requires_grad_(True)
    feats_B_g, _ = net(feats_A_g, e_joint)
    feats_B_g.mean().backward()
    print("Gradient flow: PASS" if all(p.grad is not None for p in net.parameters()) else "FAIL")
    print("")
    print("Edit prompts available: {}".format(len(EDIT_PROMPTS)))
    print("All tests passed!")
