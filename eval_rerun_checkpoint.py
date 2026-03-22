import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image

from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.utils import render_utils
from trellis.modules.sparse import basic as sp
from trellis.models.edit_net import EditNet, TextProjector
import open_clip

DEVICE = "cuda"
MODEL_PATH = "/workspace/hf_models/TRELLIS-image-large"
CKPT_PATH = "/workspace/frozen_good_state_rerun/edit_net_epoch_5.pt"
CACHE_PATH = "/workspace/edit_training_data/encoded_latents/T.pt"
OUT_DIR = Path("/workspace/edit_eval_rerun")
OUT_DIR.mkdir(parents=True, exist_ok=True)

PROMPTS = [
    "make it red",
    "make it wooden",
    "make it glass",
]

TEXT_GAIN = 4.0
TEXT_REPEAT = 8

print("Loading TRELLIS pipeline...")
pipe = TrellisImageTo3DPipeline.from_pretrained(MODEL_PATH)
pipe.cuda()
for _, m in pipe.models.items():
    m.eval()
    for p in m.parameters():
        p.requires_grad = False

print("Loading CLIP...")
clip_model, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
clip_tokenizer = open_clip.get_tokenizer("ViT-B-32")
clip_model = clip_model.to(DEVICE).eval()
for p in clip_model.parameters():
    p.requires_grad = False

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
slat_loaded = payload["slat"]
cond = payload["cond"]

slat_A = sp.SparseTensor(
    feats=slat_loaded.feats.to(DEVICE),
    coords=slat_loaded.coords.to(DEVICE),
    layout=slat_loaded.layout if hasattr(slat_loaded, "layout") else None,
)

e_img = cond["cond"]
if hasattr(e_img, "to"):
    e_img = e_img.to(DEVICE)
if e_img.dim() == 2:
    e_img = e_img.unsqueeze(0)

with torch.no_grad():
    gaussian_A = pipe.decode_slat(slat_A, ['gaussian'])['gaussian'][0]

ext_list, intr_list = render_utils.yaw_pitch_r_fov_to_extrinsics_intrinsics([0.0], [0.0], 2, 40)
ext = ext_list[0].to(DEVICE)
intr = intr_list[0].to(DEVICE)

renderer = render_utils.get_renderer(gaussian_A, resolution=256, bg_color=(0, 0, 0))

def tensor_to_pil(x):
    x = x.detach().float().clamp(0, 1).cpu()
    x = (x * 255.0).round().byte()
    x = x.permute(1, 2, 0).numpy()
    return Image.fromarray(x)

with torch.no_grad():
    res_A = renderer.render(gaussian_A, ext, intr)
orig_img = tensor_to_pil(res_A["color"])
orig_path = OUT_DIR / "original.png"
orig_img.save(orig_path)
print("Saved:", orig_path)

for prompt in PROMPTS:
    text_tokens = clip_tokenizer([prompt]).to(DEVICE)

    with torch.no_grad():
        e_text_raw = clip_model.encode_text(text_tokens)
        e_text_padded = F.pad(e_text_raw.unsqueeze(1), (0, 768 - 512))
        e_text_proj_out = text_proj(e_text_padded)
        e_text_tokens = e_text_proj_out.repeat(1, TEXT_REPEAT, 1) * TEXT_GAIN
        e_joint = torch.cat([e_img, e_text_tokens], dim=1)

        feats_B, delta = edit_net(slat_A.feats.detach(), e_joint)

        slat_B = sp.SparseTensor(
            feats=feats_B,
            coords=slat_A.coords,
            layout=slat_A.layout if hasattr(slat_A, "layout") else None,
        )

        gaussian_B = pipe.decode_slat(slat_B, ['gaussian'])['gaussian'][0]
        res_B = renderer.render(gaussian_B, ext, intr)
        edit_img = tensor_to_pil(res_B["color"])

    canvas = Image.new("RGB", (orig_img.width * 2, orig_img.height))
    canvas.paste(orig_img, (0, 0))
    canvas.paste(edit_img, (orig_img.width, 0))

    safe = prompt.replace(" ", "_")
    out_path = OUT_DIR / f"compare_{safe}.png"
    canvas.save(out_path)

    print(f"Prompt: {prompt}")
    print(f"  delta_max: {delta.abs().max().item():.6f}")
    print(f"  saved: {out_path}")

print("\nDone.")
