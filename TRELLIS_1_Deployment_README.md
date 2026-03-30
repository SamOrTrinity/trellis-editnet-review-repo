# TRELLIS 1 Deployment Guide (RunPod / Linux)

*A detailed README based on the official TRELLIS repository and my successful deployment workflow.*

## Introduction

This README documents the exact workflow I used to deploy TRELLIS 1 correctly on RunPod. It combines the official installation flow from the TRELLIS repository with the practical fixes that were needed to get a clean working environment when the first-pass install failed.

The goal of this guide is simple: help a new user go from an empty RunPod pod to a fully working TRELLIS 1 installation that can run `example.py` and generate `.glb`, `.ply`, and preview video outputs.

This guide is focused on base TRELLIS deployment for inference. Custom editing or training code can be added later, but that is a separate step after the base environment is verified.

## Prerequisites

Before starting, make sure your environment matches the general conditions that the TRELLIS repository expects.

- Operating system: Linux
- GPU: NVIDIA GPU with at least 16 GB VRAM
- CUDA toolkit: needed for compiling native CUDA extensions
- Conda: strongly recommended
- Python: 3.8 or higher

The official repository says the code is currently tested only on Linux, requires an NVIDIA GPU with at least 16 GB of memory, has been verified on A100 and A6000 GPUs, and has been tested with CUDA 11.8 and 12.2. In my RunPod deployment, I used an A40 successfully.

## Recommended RunPod Setup

For a smooth deployment, start with a Linux-based RunPod pod that already includes an NVIDIA GPU and enough storage for the repository, Conda environment, compiled extensions, model weights, and generated outputs.

Recommended starting point:
- Template: a PyTorch / CUDA Linux template
- GPU: A40, L40, RTX 4090, A5000/A6000 class, or better
- Storage: use generous persistent storage so you do not run out of space during extension builds and model downloads

Even if the base pod already contains PyTorch, do not assume that TRELLIS will work out of the box. TRELLIS compiles several CUDA extensions, and those need the CUDA toolkit and Python environment to line up cleanly.

## Step 1 - Install System Packages

Open the terminal in your pod and install the packages needed to build TRELLIS and its native extensions.

```
apt-get update -y && apt-get install -y \
  build-essential cmake ninja-build git wget \
  libgl1-mesa-dev libglib2.0-0
```

## Step 2 - Clone TRELLIS with Submodules

Clone the TRELLIS repository exactly with submodules. The `--recurse-submodules` flag matters because TRELLIS depends on additional repositories during setup.

```
cd /workspace
git clone --recurse-submodules https://github.com/microsoft/TRELLIS.git
cd TRELLIS
```

## Step 3 - Install Miniconda If It Is Missing

On many RunPod templates, `conda` is not available by default. TRELLIS's `--new-env` flow expects Conda, so install Miniconda first if needed.

```
cd /workspace

if ! command -v conda >/dev/null 2>&1; then
  wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
  bash /tmp/miniconda.sh -b -p /workspace/miniconda
  echo 'eval "$(/workspace/miniconda/bin/conda shell.bash hook)"' >> ~/.bashrc
fi

source ~/.bashrc
conda --version
```

## Step 4 - Run the Initial TRELLIS Setup

This is the most important installation step.

Important notes before you run it:
- Use `. ./setup.sh ...`, not `bash setup.sh ...`
- Sourcing the script matters because the script uses `return` and `conda activate trellis`
- The official TRELLIS install command includes both `--xformers` and `--flash-attn`
- In my RunPod deployment, the more stable path was to start with `xformers` and skip `flash-attn` unless I specifically needed it

Recommended first pass for RunPod:

```
. ./setup.sh --new-env --basic --xformers --diffoctreerast --spconv --mipgaussian --kaolin --nvdiffrast
```

```
# Official full command from the TRELLIS README:
. ./setup.sh --new-env --basic --xformers --flash-attn --diffoctreerast --spconv --mipgaussian --kaolin --nvdiffrast
```

## Step 5 - Activate the Environment and Set the Attention Backend

If you installed with `xformers`, make that explicit before running TRELLIS. This prevents TRELLIS from trying to use `flash-attn` when that backend was not installed or is unsupported on your setup.

```
conda activate trellis
export ATTN_BACKEND=xformers
echo 'export ATTN_BACKEND=xformers' >> ~/.bashrc
```

## Step 6 - Verify PyTorch, CUDA, and Core Imports

Do not jump straight to inference. First confirm that the environment is actually healthy.

```
python - <<'PY'
import importlib, torch

print("torch:", torch.__version__)
print("cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))

mods = [
    "trellis",
    "xformers",
    "spconv",
    "kaolin",
    "nvdiffrast.torch",
]

for m in mods:
    try:
        importlib.import_module(m)
        print("[OK]", m)
    except Exception as e:
        print("[FAIL]", m, e)
PY
```

```
python - <<'PY'
try:
    from trellis.pipelines import TrellisImageTo3DPipeline
    print("[OK] TrellisImageTo3DPipeline import")
except Exception as e:
    print("[FAIL] pipeline import:", e)
PY
```

## Step 7 - Run the Smoke Test

If the verification block passes, run the official example script. A successful run of `example.py` is the cleanest signal that base TRELLIS deployment is complete.

```
cd /workspace/TRELLIS
export ATTN_BACKEND=xformers
python example.py
```

## What Success Looks Like

A successful base deployment usually means all of the following are true:

- `torch.cuda.is_available()` returns `True`
- `trellis`, `spconv`, `kaolin`, `xformers`, and `nvdiffrast.torch` all import cleanly
- `from trellis.pipelines import TrellisImageTo3DPipeline` works
- `python example.py` completes without a traceback
- TRELLIS generates output files such as `.glb`, `.ply`, and preview `.mp4` videos

```
find /workspace/TRELLIS -type f \( -name "*.glb" -o -name "*.ply" -o -name "*.mp4" -o -name "*.obj" \) | tail -20
```

## If the First Install Fails: Use This Repair Flow

The rest of this README covers the exact repair path I used when the initial install did not complete cleanly on RunPod. Do not use all of these repairs blindly. Use them only when the verification block shows that something is broken.

The most common problems I ran into were:
- `conda: command not found`
- a broken PyTorch install inside the `trellis` environment
- CUDA version mismatches during extension compilation
- pip build-isolation breaking CUDA extension builds
- missing `xformers` or `kaolin` even after setup finished

## Repair A - Fix a Broken PyTorch Install in the `trellis` Environment

If PyTorch in the `trellis` environment is broken, repair it directly first. In my deployment, replacing the broken install with the official PyTorch 2.4.0 CUDA 11.8 wheels solved the issue and made the rest of the TRELLIS setup recoverable.

```
conda activate trellis

pip uninstall -y torch torchvision torchaudio triton
conda remove -y mkl mkl-service intel-openmp || true

pip install --no-cache-dir --force-reinstall \
  torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 \
  --index-url https://download.pytorch.org/whl/cu118
```

```
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
PY
```

## Repair B - Re-run TRELLIS Dependency Installation Without Creating a New Environment

Once PyTorch is healthy again, re-run the dependency installation without `--new-env`. This lets TRELLIS install the missing pieces into the repaired `trellis` environment instead of creating another new environment.

```
cd /workspace/TRELLIS
. ./setup.sh --basic --xformers --diffoctreerast --spconv --mipgaussian --kaolin --nvdiffrast
```

## Repair C - Make Sure the CUDA Compiler Matches the CUDA Version Used by PyTorch

This was one of the most important fixes in my deployment.

If PyTorch inside the `trellis` environment is `cu118` but your shell is picking up `/usr/local/cuda` from a CUDA 12.x toolkit, native extension builds can fail with a CUDA version mismatch.

The fix is to install the CUDA 11.8 compiler toolchain inside the Conda environment and point the build variables there.

```
conda activate trellis
conda install -y -c nvidia cuda-nvcc=11.8 cuda-cudart-dev=11.8 cuda-libraries-dev=11.8
```

```
unset CUDA_HOME
export CUDA_HOME=$CONDA_PREFIX
export PATH=$CONDA_PREFIX/bin:$PATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$CONDA_PREFIX/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}
export CPATH=$CONDA_PREFIX/include:$CONDA_PREFIX/targets/x86_64-linux/include:${CPATH:-}
export TORCH_CUDA_ARCH_LIST="8.6"

which nvcc
nvcc --version

python - <<'PY'
import torch
from torch.utils.cpp_extension import CUDA_HOME
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("CUDA_HOME:", CUDA_HOME)
PY
```

## Repair D - Install Compiler Tooling for Manual CUDA Builds

If the setup script still leaves native extensions unbuilt, install compiler tooling and build the failed packages manually.

```
conda activate trellis
conda install -y -c conda-forge gcc_linux-64=11 gxx_linux-64=11 cmake ninja make

export CC=$(which x86_64-conda-linux-gnu-cc)
export CXX=$(which x86_64-conda-linux-gnu-c++)
export CUDAHOSTCXX=$CXX
```

## Repair E - Build `nvdiffrast` Manually

`nvdiffrast` is one of the extensions that most often fails during an automated first-pass install. The upstream project explicitly documents installation with `--no-build-isolation`, so this manual step is often much more reliable than repeating the whole TRELLIS setup command.

```
rm -rf /tmp/extensions/nvdiffrast
git clone -b v0.4.0 https://github.com/NVlabs/nvdiffrast.git /tmp/extensions/nvdiffrast
cd /tmp/extensions/nvdiffrast
python setup.py clean --all || true
python setup.py build_ext --inplace -v
pip install --no-build-isolation -e .
```

```
python - <<'PY'
import nvdiffrast.torch as dr
print("nvdiffrast OK")
PY
```

## Repair F - Build `diffoctreerast` Manually

If TRELLIS still cannot import the octree rasterizer, build it directly in the same environment.

```
cd /workspace
rm -rf /tmp/extensions/diffoctreerast
git clone --recurse-submodules https://github.com/JeffreyXiang/diffoctreerast.git /tmp/extensions/diffoctreerast
cd /tmp/extensions/diffoctreerast
pip install --no-build-isolation -v .
```

## Repair G - Build the Gaussian Rasterizer Manually

TRELLIS also depends on the `diff-gaussian-rasterization` submodule from `mip-splatting`. If it fails in the automated setup, build it manually.

```
cd /workspace
rm -rf /tmp/extensions/mip-splatting
git clone https://github.com/autonomousvision/mip-splatting.git /tmp/extensions/mip-splatting
cd /tmp/extensions/mip-splatting/submodules/diff-gaussian-rasterization
pip install --no-build-isolation -v .
```

## Repair H - Install `xformers` and `kaolin` Explicitly If They Are Still Missing

Even after the extension rebuilds, `xformers` or `kaolin` may still be missing. Installing them explicitly was the last step needed in my deployment.

```
conda activate trellis
cd /workspace/TRELLIS

pip install xformers==0.0.27.post2 --index-url https://download.pytorch.org/whl/cu118
pip install kaolin -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.4.0_cu121.html

export ATTN_BACKEND=xformers
echo 'export ATTN_BACKEND=xformers' >> ~/.bashrc
```

## Repair I - Re-verify Everything After the Manual Fixes

After manual repairs, rerun the import checks before you attempt inference again.

```
python - <<'PY'
import importlib

mods = [
    "trellis",
    "xformers",
    "spconv",
    "kaolin",
    "nvdiffrast.torch",
    "diff_gaussian_rasterization",
]

for m in mods:
    try:
        importlib.import_module(m)
        print("[OK]", m)
    except Exception as e:
        print("[FAIL]", m, e)
PY
```

```
cd /workspace/TRELLIS
export ATTN_BACKEND=xformers
python example.py
```

## Run TRELLIS on Your Own Image

Once the base deployment is working, use the script below to run TRELLIS on your own input image and save all major outputs.

```
cd /workspace/TRELLIS
conda activate trellis
export ATTN_BACKEND=xformers

python - <<'PY'
import os
os.environ["SPCONV_ALGO"] = "native"

import imageio
from PIL import Image
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.utils import render_utils, postprocessing_utils

img_path = "/workspace/TRELLIS/my_input.png"   # change this to your image path

pipeline = TrellisImageTo3DPipeline.from_pretrained("microsoft/TRELLIS-image-large")
pipeline.cuda()

image = Image.open(img_path).convert("RGBA")

outputs = pipeline.run(
    image,
    seed=1,
)

video = render_utils.render_video(outputs["gaussian"][0])["color"]
imageio.mimsave("/workspace/TRELLIS/output_gs.mp4", video, fps=30)

video = render_utils.render_video(outputs["radiance_field"][0])["color"]
imageio.mimsave("/workspace/TRELLIS/output_rf.mp4", video, fps=30)

video = render_utils.render_video(outputs["mesh"][0])["normal"]
imageio.mimsave("/workspace/TRELLIS/output_mesh.mp4", video, fps=30)

glb = postprocessing_utils.to_glb(
    outputs["gaussian"][0],
    outputs["mesh"][0],
    simplify=0.95,
    texture_size=1024,
)
glb.export("/workspace/TRELLIS/output.glb")

outputs["gaussian"][0].save_ply("/workspace/TRELLIS/output.ply")

print("Done.")
print("Saved:")
print("/workspace/TRELLIS/output.glb")
print("/workspace/TRELLIS/output.ply")
print("/workspace/TRELLIS/output_gs.mp4")
print("/workspace/TRELLIS/output_rf.mp4")
print("/workspace/TRELLIS/output_mesh.mp4")
PY
```

## Optional - Launch the TRELLIS Web Demo

TRELLIS also ships with a Gradio demo. Install the demo dependencies and then run the app. On RunPod, expose the relevant port in the pod UI and open the address shown in the terminal.

```
cd /workspace/TRELLIS
conda activate trellis
. ./setup.sh --demo
python app.py
```

## Common Mistakes to Avoid

1. Running `bash setup.sh ...` instead of `. ./setup.sh ...`
   The TRELLIS setup script is meant to be sourced so that `return` and `conda activate trellis` work correctly.

2. Mixing CUDA toolkits
   If PyTorch in the `trellis` environment is built for CUDA 11.8, do not compile the extensions with `/usr/local/cuda` from a 12.x toolkit.

3. Re-running the full install again and again without checking imports
   Always verify which package is actually failing before repeating the full setup.

4. Forgetting to set `ATTN_BACKEND=xformers`
   If you installed `xformers` instead of `flash-attn`, make sure TRELLIS knows that before you run inference.

5. Assuming base TRELLIS deployment means custom architecture code is ready
   A working base install only means TRELLIS itself is deployed. Any custom editing, fine-tuning, or additional training pipeline code is a separate layer on top.

## Final Deployment Checklist

Use this checklist before you say the deployment is finished.

- [ ] `conda activate trellis` works
- [ ] `python -c "import torch; print(torch.cuda.is_available())"` returns `True`
- [ ] `import trellis` works
- [ ] `from trellis.pipelines import TrellisImageTo3DPipeline` works
- [ ] `spconv`, `kaolin`, `xformers`, and `nvdiffrast.torch` import successfully
- [ ] `python example.py` completes successfully
- [ ] output files are created
- [ ] your own image can be processed successfully

## References

Official TRELLIS repository:
https://github.com/microsoft/TRELLIS

Official TRELLIS setup script:
https://github.com/microsoft/TRELLIS/blob/main/setup.sh

Official TRELLIS model repo used in the smoke test:
https://huggingface.co/microsoft/TRELLIS-image-large

Upstream nvdiffrast repository:
https://github.com/NVlabs/nvdiffrast
