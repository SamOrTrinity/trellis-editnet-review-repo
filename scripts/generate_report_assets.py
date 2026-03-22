from pathlib import Path
import re
import shutil
import pandas as pd
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
ART = REPO / "artifacts"
FIG = REPO / "docs" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

# Find the best final log automatically.
candidates = sorted(ART.glob("*.log"))

best = None
best_epoch = -1
best_mtime = -1

epoch_pat = re.compile(r"Epoch\s+(\d+)\s+\|")

for p in candidates:
    try:
        txt = p.read_text(errors="ignore")
    except Exception:
        continue
    epochs = [int(x) for x in epoch_pat.findall(txt)]
    if not epochs:
        continue
    score = (max(epochs), p.stat().st_mtime)
    if score > (best_epoch, best_mtime):
        best = p
        best_epoch, best_mtime = score

if best is None:
    raise RuntimeError("No training log with epoch summaries found in artifacts/")

text = best.read_text(errors="ignore")

pat = re.compile(
    r"Epoch\s+(\d+)\s+\|\s+([0-9.]+)s\s+\|\s+loss=([-0-9.]+)\s+sim=([-0-9.]+)\s+preserve=([-0-9.]+)\s+prompt=([-0-9.]+)\s+delta=([-0-9.]+)\s+d_max=([-0-9.]+)\s+\|\s+(\d+)/(\d+)\s+ok,\s+(\d+)/(\d+)\s+grad"
)

rows = []
for m in pat.finditer(text):
    rows.append({
        "epoch": int(m.group(1)),
        "seconds": float(m.group(2)),
        "loss": float(m.group(3)),
        "similarity": float(m.group(4)),
        "preserve": float(m.group(5)),
        "prompt_loss": float(m.group(6)),
        "delta_reg": float(m.group(7)),
        "d_max": float(m.group(8)),
        "ok_steps": int(m.group(9)),
        "total_steps": int(m.group(10)),
        "grad_steps": int(m.group(11)),
        "grad_total": int(m.group(12)),
    })

if not rows:
    raise RuntimeError(f"Could not parse epoch summaries from {best}")

df = pd.DataFrame(rows).sort_values("epoch").reset_index(drop=True)
df.to_csv(FIG / "epoch_metrics.csv", index=False)

# Copy chosen log with stable name
shutil.copy2(best, ART / "selected_final_log.log")

# Loss vs epoch
plt.figure(figsize=(7.5, 4.8))
plt.plot(df["epoch"], df["loss"], marker="o")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training loss vs epoch")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(FIG / "loss_vs_epoch.png", dpi=220)
plt.close()

# Similarity vs epoch
plt.figure(figsize=(7.5, 4.8))
plt.plot(df["epoch"], df["similarity"], marker="o")
plt.xlabel("Epoch")
plt.ylabel("CLIP similarity")
plt.title("CLIP similarity vs epoch")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(FIG / "similarity_vs_epoch.png", dpi=220)
plt.close()

# d_max vs epoch
plt.figure(figsize=(7.5, 4.8))
plt.plot(df["epoch"], df["d_max"], marker="o")
plt.xlabel("Epoch")
plt.ylabel("d_max")
plt.title("Maximum latent edit magnitude vs epoch")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(FIG / "dmax_vs_epoch.png", dpi=220)
plt.close()

# Combined figure for presentation
fig, axes = plt.subplots(2, 2, figsize=(10, 7))
axes = axes.ravel()

axes[0].plot(df["epoch"], df["loss"], marker="o")
axes[0].set_title("Loss vs epoch")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].grid(True, alpha=0.3)

axes[1].plot(df["epoch"], df["similarity"], marker="o")
axes[1].set_title("CLIP similarity vs epoch")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Similarity")
axes[1].grid(True, alpha=0.3)

axes[2].plot(df["epoch"], df["delta_reg"], marker="o", label="delta_reg")
axes[2].plot(df["epoch"], df["preserve"], marker="o", label="preserve")
axes[2].set_title("Regularization terms")
axes[2].set_xlabel("Epoch")
axes[2].set_ylabel("Value")
axes[2].legend()
axes[2].grid(True, alpha=0.3)

axes[3].plot(df["epoch"], df["d_max"], marker="o")
axes[3].set_title("d_max vs epoch")
axes[3].set_xlabel("Epoch")
axes[3].set_ylabel("d_max")
axes[3].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(FIG / "training_metrics_summary.png", dpi=220)
plt.close()

final = df.iloc[-1]

out_md = REPO / "docs" / "generated_results.md"
with open(out_md, "w", encoding="utf-8") as f:
    f.write("# Generated Results Summary\n\n")
    f.write(f"Source log: `{best.name}`\n\n")
    f.write("## Final epoch summary\n\n")
    f.write(f"- Final epoch: {int(final['epoch'])}\n")
    f.write(f"- Loss: {final['loss']:.4f}\n")
    f.write(f"- Similarity: {final['similarity']:.4f}\n")
    f.write(f"- Preserve: {final['preserve']:.4f}\n")
    f.write(f"- Prompt loss: {final['prompt_loss']:.4f}\n")
    f.write(f"- Delta regularization: {final['delta_reg']:.6f}\n")
    f.write(f"- d_max: {final['d_max']:.5f}\n")
    f.write(f"- Successful steps: {int(final['ok_steps'])}/{int(final['total_steps'])}\n\n")
    f.write("## Included figures\n\n")
    f.write("- `docs/figures/loss_vs_epoch.png`\n")
    f.write("- `docs/figures/similarity_vs_epoch.png`\n")
    f.write("- `docs/figures/dmax_vs_epoch.png`\n")
    f.write("- `docs/figures/training_metrics_summary.png`\n\n")
    f.write("## Per-epoch metrics\n\n")
    f.write(df.to_markdown(index=False))

print("Selected log:", best)
print("Wrote:", out_md)
print("Figures saved in:", FIG)