"""
train_edit_delta.py -- FINAL WORKING VERSION
=============================================
Confirmed: GaussianRenderer.render() returns tensors with requires_grad=True
Key: use render_utils.get_renderer() and pass raw (4,4) ext and (3,3) intr

Usage:
    cd /workspace/TRELLIS
    conda activate trellis
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True nohup python train_edit_delta.py > /workspace/training.log 2>&1 &
"""
import os, sys, time, random, gc
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from trellis.models.edit_net import EditNet, TextProjector, EDIT_PROMPTS
from trellis.utils.edit_losses import delta_regularisation_loss
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.utils import render_utils
from trellis.modules.sparse import basic as sp


# Config
N_EPOCHS = int(os.getenv("EDIT_EPOCHS", "5"))
LR = float(os.getenv("EDIT_LR", "3e-4"))
LAMBDA_PROMPT = float(os.getenv("EDIT_LAMBDA_PROMPT", "1.0"))
LAMBDA_DELTA = float(os.getenv("EDIT_LAMBDA_DELTA", "0.05"))
LAMBDA_PRESERVE = float(os.getenv("EDIT_LAMBDA_PRESERVE", "0.05"))
N_RENDER_VIEWS = int(os.getenv("EDIT_N_VIEWS", "3"))
RENDER_RES = int(os.getenv("EDIT_RENDER_RES", "128"))
SAVE_EVERY = int(os.getenv("EDIT_SAVE_EVERY", "1"))
PROMPTS_PER_OBJECT = int(os.getenv("EDIT_PROMPTS_PER_OBJECT", "2"))
MAX_OBJECTS = int(os.getenv("EDIT_MAX_OBJECTS", "5"))
TEXT_GAIN = float(os.getenv("EDIT_TEXT_GAIN", "4.0"))
TEXT_REPEAT = int(os.getenv("EDIT_TEXT_REPEAT", "8"))
SCALE = float(os.getenv("EDIT_SCALE", "0.15"))
MAX_VOXELS = int(os.getenv("EDIT_MAX_VOXELS", "22000"))
PAIRS_JSON = os.getenv("EDIT_PAIRS_JSON", "/workspace/edit_training_data/training_pairs.json")
CACHE_DIR = os.getenv("EDIT_CACHE_DIR", "/workspace/edit_training_data/encoded_latents")
CKPT_DIR = os.getenv("EDIT_CKPT_DIR", "/workspace/edit_checkpoints")


def main():
    os.makedirs(CKPT_DIR, exist_ok=True)
    print("=" * 60, flush=True)
    print("  Architecture 2: Delta Training (FINAL)", flush=True)
    print("=" * 60, flush=True)

    device = "cuda"
    output_dir = Path(CKPT_DIR)
    output_dir.mkdir(exist_ok=True)

    # ---- Load pipeline and cached objects ----
    print("\n[1/4] Loading TRELLIS 1 and cached latents...", flush=True)

    pipeline = TrellisImageTo3DPipeline.from_pretrained("microsoft/TRELLIS-image-large")
    pipeline.cuda()

    for _, m in pipeline.models.items():
        m.eval()
        for p in m.parameters():
            p.requires_grad = False

    cache_dir = Path(CACHE_DIR)
    cache_files = sorted(cache_dir.glob("*.pt"))[:MAX_OBJECTS]
    print("  Using {} cached latents (max {})".format(len(cache_files), MAX_OBJECTS), flush=True)

    encoded = []
    for cache_path in cache_files:
        payload = torch.load(cache_path, weights_only=False)
        nvox = payload["slat"].feats.shape[0]
        if nvox > MAX_VOXELS:
            print(
                "    SKIP {} too large ({} voxels > {})".format(
                    payload.get("image_name", cache_path.name), nvox, MAX_VOXELS
                ),
                flush=True,
            )
            continue
        encoded.append(
            {
                "name": payload.get("image_name", cache_path.name),
                "slat": payload["slat"],
                "cond": payload["cond"],
                "cache_path": str(cache_path),
            }
        )
        print(
            "    {} OK feats={} coords={}".format(
                payload.get("image_name", cache_path.name),
                tuple(payload["slat"].feats.shape),
                tuple(payload["slat"].coords.shape),
            ),
            flush=True,
        )

    print("\n[2/4] Freeing DiT models (only need decoders)...", flush=True)
    del pipeline.models["sparse_structure_flow_model"]
    del pipeline.models["slat_flow_model"]
    del pipeline.models["image_cond_model"]
    torch.cuda.empty_cache()
    gc.collect()
    print("  Freed ~20GB VRAM.", flush=True)

    # ---- Load CLIP ----
    print("\n[3/4] Loading CLIP...", flush=True)
    import open_clip

    clip_model, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
    clip_tokenizer = open_clip.get_tokenizer("ViT-B-32")
    clip_model = clip_model.to(device).eval()
    for p in clip_model.parameters():
        p.requires_grad = False
    clip_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device)
    clip_std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device)
    print("  Done.", flush=True)

    # ---- Create EditNet ----
    edit_net = EditNet(latent_dim=8, cond_dim=1024, hidden_dim=256, n_blocks=3, scale=SCALE).to(device)
    text_proj = TextProjector(768, 1024).to(device)
    trainable_params = list(edit_net.parameters()) + list(text_proj.parameters())
    optimizer = optim.AdamW(trainable_params, lr=LR, weight_decay=1e-5)
    print("  EditNet: {:,} params".format(sum(p.numel() for p in trainable_params)), flush=True)

    # ---- Pre-compute cameras ----
    cams = [render_utils.sphere_hammersley_sequence(i, N_RENDER_VIEWS) for i in range(N_RENDER_VIEWS)]
    yaws = [c[0] for c in cams]
    pitchs = [c[1] for c in cams]
    ext_list, intr_list = render_utils.yaw_pitch_r_fov_to_extrinsics_intrinsics(yaws, pitchs, 2, 40)
    ext_list = [e.to(device) for e in ext_list]
    intr_list = [i.to(device) for i in intr_list]

    # ---- Training ----
    print("\n[4/4] Training...", flush=True)

    import json

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

    print("  Loaded {} training pairs from {}".format(len(pairs), PAIRS_JSON), flush=True)
    print("  Differentiable rendering: ON", flush=True)
    print("=" * 60, flush=True)

    for epoch in range(N_EPOCHS):
        edit_net.train()
        text_proj.train()
        epoch_losses = []
        epoch_start = time.time()

        torch.cuda.empty_cache()
        gc.collect()
        random.shuffle(pairs)

        for step, (obj, prompt) in enumerate(pairs):
            if step % 5 == 0:
                torch.cuda.empty_cache()
                gc.collect()
            optimizer.zero_grad(set_to_none=True)

            try:
                # Conditioning
                e_img = obj["cond"]["cond"].detach().to(device)
                if e_img.dim() == 2:
                    e_img = e_img.unsqueeze(0)

                text_tokens = clip_tokenizer([prompt]).to(device)
                with torch.no_grad():
                    e_text_raw = clip_model.encode_text(text_tokens)
                e_text_padded = F.pad(e_text_raw.unsqueeze(1), (0, 768 - 512))
                e_text_proj_out = text_proj(e_text_padded)
                e_text_tokens = e_text_proj_out.repeat(1, TEXT_REPEAT, 1) * TEXT_GAIN
                e_joint = torch.cat([e_img, e_text_tokens], dim=1)

                # EditNet
                feats_A = obj["slat"].feats.detach().to(device)
                feats_B, delta = edit_net(feats_A, e_joint)

                # Build edited SparseTensor
                slat_orig = obj["slat"]
                slat_edited = sp.SparseTensor(
                    feats=feats_B,
                    coords=slat_orig.coords.to(device) if hasattr(slat_orig.coords, "to") else slat_orig.coords,
                    layout=slat_orig.layout if hasattr(slat_orig, "layout") else None,
                )

                # Decode edited latent
                outputs_B = pipeline.decode_slat(slat_edited, ["gaussian"])
                gaussian_B = outputs_B["gaussian"][0]

                # Decode original latent on GPU for preserve reference
                with torch.no_grad():
                    slat_A = sp.SparseTensor(
                        feats=obj["slat"].feats.to(device),
                        coords=obj["slat"].coords.to(device) if hasattr(obj["slat"].coords, "to") else obj["slat"].coords,
                        layout=obj["slat"].layout if hasattr(obj["slat"], "layout") else None,
                    )
                    outputs_A = pipeline.decode_slat(slat_A, ["gaussian"])
                    gaussian_A = outputs_A["gaussian"][0]

                # Renderer
                renderer = render_utils.get_renderer(
                    gaussian_B, resolution=RENDER_RES, bg_color=(0, 0, 0)
                )

                # Differentiable render
                color_views = []
                l_preserve = torch.tensor(0.0, device=delta.device)

                for vi in range(N_RENDER_VIEWS):
                    with torch.no_grad():
                        res_A = renderer.render(gaussian_A, ext_list[vi], intr_list[vi])
                        color_A = res_A["color"].float().clamp(0, 1)
                        mask_A = (color_A.mean(dim=0, keepdim=True) > 1e-3).float()

                    res = renderer.render(gaussian_B, ext_list[vi], intr_list[vi])
                    color_B = res["color"].float().clamp(0, 1)
                    color_views.append(color_B)

                    l_preserve = l_preserve + (
                        ((color_B - color_A).abs() * mask_A).sum()
                        / (mask_A.sum() * 3 + 1e-8)
                    )

                l_preserve = l_preserve / max(1, N_RENDER_VIEWS)

                rendered = torch.stack(color_views).float().clamp(0, 1)

                if rendered.shape[-1] != 224 or rendered.shape[-2] != 224:
                    rendered = F.interpolate(
                        rendered,
                        size=(224, 224),
                        mode="bilinear",
                        align_corners=False,
                    )

                # CLIP normalize
                clip_input = (rendered - clip_mean[None, :, None, None]) / clip_std[None, :, None, None]

                # Prompt loss
                with torch.no_grad():
                    text_embed = clip_model.encode_text(text_tokens)
                    text_embed = text_embed / text_embed.norm(dim=-1, keepdim=True)

                img_embeds = clip_model.encode_image(clip_input)
                img_embeds = img_embeds / img_embeds.norm(dim=-1, keepdim=True)
                sim = (img_embeds * text_embed).sum(dim=-1)
                l_prompt = -sim.mean()

                # Delta loss
                l_delta = delta_regularisation_loss(delta)

                # Total loss
                loss = (
                    LAMBDA_PROMPT * l_prompt
                    + LAMBDA_PRESERVE * l_preserve
                    + LAMBDA_DELTA * l_delta
                )

                # Backprop
                loss.backward()

                has_grad = any(
                    p.grad is not None and p.grad.abs().sum() > 0
                    for p in edit_net.parameters()
                )

                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                optimizer.step()

                loss_dict = {
                    "total": loss.item(),
                    "prompt": l_prompt.item(),
                    "preserve": l_preserve.item(),
                    "delta": l_delta.item(),
                    "d_max": delta.abs().max().item(),
                    "sim": sim.mean().item(),
                    "grad": has_grad,
                }
                epoch_losses.append(loss_dict)

                if (step + 1) % 3 == 0:
                    g = "GRAD" if has_grad else "NO_GRAD"
                    print(
                        "  E{} S{}/{} | loss={:.4f} sim={:.3f} d={:.5f} {} | {}".format(
                            epoch + 1,
                            step + 1,
                            len(pairs),
                            loss_dict["total"],
                            loss_dict["sim"],
                            loss_dict["d_max"],
                            g,
                            prompt,
                        ),
                        flush=True,
                    )

            except Exception as e:
                import traceback

                print(
                    "  S{} FAIL: {}: {}".format(step + 1, type(e).__name__, e),
                    flush=True,
                )
                traceback.print_exc()
                torch.cuda.empty_cache()
                gc.collect()
                optimizer.zero_grad(set_to_none=True)

            # Cleanup
            for v in [
                "slat_edited",
                "slat_A",
                "outputs_A",
                "outputs_B",
                "gaussian_A",
                "gaussian_B",
                "renderer",
                "color_views",
                "rendered",
                "clip_input",
                "img_embeds",
                "loss",
                "l_prompt",
                "l_preserve",
                "l_delta",
                "feats_B",
                "delta",
                "res",
                "res_A",
                "color_A",
                "color_B",
                "mask_A",
            ]:
                try:
                    exec("del {}".format(v))
                except Exception:
                    pass
            torch.cuda.empty_cache()

        # Epoch summary
        t = time.time() - epoch_start
        if epoch_losses:
            avg = {
                k: np.mean([l[k] for l in epoch_losses if isinstance(l[k], (int, float))])
                for k in ["total", "sim", "d_max", "preserve", "delta", "prompt"]
            }
            gc_count = sum(1 for l in epoch_losses if l["grad"])
            print(
                "\n  Epoch {} | {:.0f}s | loss={:.4f} sim={:.3f} preserve={:.4f} prompt={:.4f} delta={:.6f} d_max={:.5f} | {}/{} ok, {}/{} grad".format(
                    epoch + 1,
                    t,
                    avg["total"],
                    avg["sim"],
                    avg["preserve"],
                    avg["prompt"],
                    avg["delta"],
                    avg["d_max"],
                    len(epoch_losses),
                    len(pairs),
                    gc_count,
                    len(epoch_losses),
                ),
                flush=True,
            )
        else:
            print("\n  Epoch {} | {:.0f}s | NO successful steps".format(epoch + 1, t), flush=True)

        if (epoch + 1) % SAVE_EVERY == 0 and epoch_losses:
            p = output_dir / "edit_net_epoch_{}.pt".format(epoch + 1)
            torch.save(
                {
                    "epoch": epoch + 1,
                    "edit_net": edit_net.state_dict(),
                    "text_proj": text_proj.state_dict(),
                    "optimizer": optimizer.state_dict(),
                },
                p,
            )
            print("  Saved: {}".format(p), flush=True)

        torch.cuda.empty_cache()
        gc.collect()

    print("\n" + "=" * 60, flush=True)
    print("  Training complete!", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()
