# MPC-Flow with FLUX.2

This folder contains the FLUX.2 [dev] code path for MPC-Flow image inverse-problem demos.

Run commands from this directory:

```powershell
cd FLUX2
```

The local Python packages live directly in this folder:

- [`flux2/`](flux2/): FLUX.2 loading, sampling, MPC-Flow rewards, and pipeline code.
- [`clip/`](clip/): local CLIP utilities used by the style reward.
- [`cli_mpc_flow.py`](cli_mpc_flow.py): canonical MPC-Flow interface for the paper path.
- [`sweep_mpc.py`](sweep_mpc.py): sweep launcher for configs in [`sweeps/`](sweeps/).

Important: [`cli_mpc_flow.py`](cli_mpc_flow.py) is **not** the upstream FLUX.2 CLI. It is a modified research runner for MPC-Flow built around the FLUX.2 Diffusers loading path. It adds MPC/FlowChef correction modes, reward construction, task-specific image conditioning, prompt-embedding caching, and CSV metrics. Use upstream FLUX/Diffusers documentation only for the base model-loading semantics, not for this script's command-line interface.

## Environment

The tested environment is conda-first:

```powershell
mamba env create -f environment.yml
conda activate mpcflow
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
python -c "import flux2.mpc; print('FLUX2 imports ok')"
python -m ipykernel install --user --name mpcflow --display-name "mpcflow"
```

The environment pins the CUDA-enabled PyTorch stack used here:

```text
torch==2.10.0+cu128
torchvision==0.25.0+cu128
torchaudio==2.10.0+cu128
diffusers==0.37.0
transformers==5.3.0
bitsandbytes==0.49.2
```

Tested GPUs:

- NVIDIA RTX 3090 24GB
- NVIDIA RTX 5090 32GB

CPU-only FLUX.2 execution is not a supported target.

## FLUX.2 Weights

The paper experiments used:

```text
diffusers/FLUX.2-dev-bnb-4bit
```

This is available to download, e.g. from https://huggingface.co/diffusers/FLUX.2-dev-bnb-4bit.

To load from Hugging Face, request access to the gated model and authenticate:

```powershell
hf auth login
```

If you already have a local Diffusers snapshot, pass it as `--repo-id`:

```powershell
python cli_mpc_flow.py --repo-id "C:\path\to\FLUX.2-dev-bnb-4bit\snapshot" --prompt "a cat" --reward none
```

FLUX.2 model use is governed separately by [`model_licenses/LICENSE-FLUX-DEV`](model_licenses/LICENSE-FLUX-DEV).

## Notebooks

The public demo notebooks are:

- [`style_transfer_demo.ipynb`](style_transfer_demo.ipynb)
- [`colorize_luminance_demo.ipynb`](colorize_luminance_demo.ipynb)
- [`superres_mpc_demo.ipynb`](superres_mpc_demo.ipynb)

Open these notebooks from this `FLUX2/` folder and select the `mpcflow` kernel. The top config cell controls the prompt, image, resolution, and MPC/FlowChef hyperparameters.

## CLI Examples

All commands below use the modified MPC-Flow CLI, assume your shell is in `FLUX2/`, and save outputs under `output/`.

### Prompt-only FLUX.2 [dev] 4-bit generation

```powershell
python cli_mpc_flow.py `
  --prompt "a cat sitting on a windowsill at sunset" `
  --reward none `
  --height 448 `
  --width 448 `
  --steps 28 `
  --guidance-scale 4.0 `
  --seed 42 `
  --save-dir output/demos/prompt_only `
  --out prompt_only.png
```

### Style Transfer

```powershell
python cli_mpc_flow.py `
  --prompt "a cat" `
  --reward style `
  --style-image style_images/xingkong.jpg `
  --method mpc `
  --opt-steps 20 `
  --mpc-lr 0.5 `
  --mpc-rho 64.0 `
  --height 448 `
  --width 448 `
  --steps 28 `
  --guidance-scale 4.0 `
  --seed 42 `
  --save-dir output/demos/style_transfer `
  --out style_transfer_mpc.png
```

### Luminance-Constrained Colorization

```powershell
python cli_mpc_flow.py `
  --prompt "colorize this luminance image" `
  --reward luminance `
  --reward-image prompt_images/dog.jpg `
  --image prompt_images/dog.jpg `
  --method mpc `
  --opt-steps 20 `
  --mpc-lr 0.2 `
  --mpc-rho 1e-6 `
  --height 256 `
  --width 256 `
  --steps 28 `
  --guidance-scale 4.0 `
  --seed 42 `
  --save-dir output/demos/luminance `
  --out luminance_mpc.png
```

### Super-Resolution

```powershell
python cli_mpc_flow.py `
  --prompt "produce a very high-quality photorealistic sharp image, that is consistent with this low-resolution image" `
  --reward superres `
  --reward-image prompt_images/house_image.jpg `
  --image prompt_images/house_image.jpg `
  --superres-lr-size 128 200 `
  --method mpc `
  --opt-steps 20 `
  --mpc-lr 0.5 `
  --mpc-rho 2.5e-6 `
  --height 512 `
  --width 800 `
  --steps 28 `
  --guidance-scale 4.0 `
  --seed 42 `
  --save-dir output/demos/superres `
  --out superres_mpc.png
```


## MPC vs FlowChef

[`cli_mpc_flow.py`](cli_mpc_flow.py) exposes two correction modes:

- `--method mpc`: our method.
- `--method flowchef`: the comparison method [FlowChef](https://github.com/FlowChef/flowchef).

To switch a command from MPC to FlowChef, keep the task-specific reward arguments and replace the method and learning rate. Public demo defaults:

- style: `--method flowchef --mpc-lr 0.01`
- luminance: `--method flowchef --mpc-lr 0.2`
- super-resolution: `--method flowchef --mpc-lr 0.5`

FlowChef ignores `--mpc-rho`.

## Sweeps

The public release keeps one paper-scale style-transfer trade-off sweep and its analysis notebook in [`sweeps/`](sweeps/). This is not a demo: the full sweep expands to 900 FLUX.2 runs.

```powershell
python sweep_mpc.py --config sweeps/sweep_mpc_flowchef.json --dry-run
```

Run the real sweep from this `FLUX2/` directory with `--resume`; outputs are written to `output/sweep/`. After the sweep has completed, open [`sweeps/sweep_style_tradeoff_analysis.ipynb`](sweeps/sweep_style_tradeoff_analysis.ipynb) with the `mpcflow` kernel to regenerate the style/content trade-off figures under `output/analysis/style/`. The analysis may download DINOv2 weights on first use if the style/content metrics are not already in the CSVs. See [`sweeps/README.md`](sweeps/README.md).

## Manual Verification

These checks are intentionally manual because full FLUX.2 runs are GPU- and memory-heavy.

First verify the environment and entrypoints:

```powershell
conda activate mpcflow
cd FLUX2
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
python -c "import flux2.mpc; print('FLUX2 imports ok')"
python cli_mpc_flow.py --help
python sweep_mpc.py --config sweeps/sweep_mpc_flowchef.json --dry-run
```

Then run one command from each CLI section above:

- prompt-only generation
- style transfer
- luminance-constrained colorization
- super-resolution

Finally open each public notebook with the `mpcflow` kernel and run it top-to-bottom:

- [`style_transfer_demo.ipynb`](style_transfer_demo.ipynb)
- [`colorize_luminance_demo.ipynb`](colorize_luminance_demo.ipynb)
- [`superres_mpc_demo.ipynb`](superres_mpc_demo.ipynb)

## Provenance

This folder builds on the upstream FLUX open-source release and the Hugging Face Diffusers FLUX.2 loading path, but it is a separate research-code implementation for MPC-Flow. In particular, [`cli_mpc_flow.py`](cli_mpc_flow.py) is modified for MPC-Flow and should not be treated as the upstream FLUX CLI. The repository does not ship FLUX.2 model weights.

Model metadata:

- [`model_cards/FLUX.2-dev.md`](model_cards/FLUX.2-dev.md)
- [`model_licenses/LICENSE-FLUX-DEV`](model_licenses/LICENSE-FLUX-DEV)
