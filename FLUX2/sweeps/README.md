# Style Transfer Sweep And Analysis

This folder keeps one paper-scale style-transfer sweep and its analysis notebook:

- `sweep_mpc_flowchef.json`: style-transfer trade-off sweep over five prompts, nine style images, MPC control regularisation `rho`, and FlowChef learning rate.
- `sweep_style_tradeoff_analysis.ipynb`: analysis notebook for style adherence versus content preservation figures.

This is not a demo. The full sweep expands to 900 FLUX.2 runs and is intended for reproducing/inspecting the paper analysis, and for showing how to do a sweep over hyper-parameters/images/seeds.

The sweep runner calls `cli_mpc_flow.py`, the modified MPC-Flow CLI in the parent `FLUX2/` folder.

## Experiment

The style terminal loss is

```text
Phi(z) = || Style(Dec(z)) - Style(x_ref) ||_2^2
```

where `Style` is computed from Gram matrices of CLIP ViT-B/16 image features and `x_ref` is the reference style image.

The sweep fixes 20 conditioning updates per time step. For MPC-Flow it varies `rho`, the control regularisation weight. For FlowChef it varies the terminal-loss learning rate. The analysis notebook summarises the style/content trade-off across the sweep.

## Run The Sweep

Run from the `FLUX2/` directory:

```powershell
python sweep_mpc.py --config sweeps/sweep_mpc_flowchef.json --dry-run
```

Use `--dry-run` first to inspect the 900 generated commands. A real run writes CSV/image outputs under `output/sweep/`:

```powershell
python sweep_mpc.py --config sweeps/sweep_mpc_flowchef.json --resume
```

`--resume` skips runs whose output image already exists.

If using a local FLUX.2 Diffusers snapshot, add `repo_id` to the config `base_args`:

```json
"repo_id": "C:\\Users\\gw23\\.cache\\huggingface\\hub\\models--diffusers--FLUX.2-dev-bnb-4bit\\snapshots\\<snapshot>"
```

## Run The Analysis

Open `sweeps/sweep_style_tradeoff_analysis.ipynb` from the `FLUX2/` folder with the `mpcflow` kernel after the sweep outputs exist.

The notebook expects:

- the sweep config at `sweeps/sweep_mpc_flowchef.json`
- generated sweep outputs under `output/sweep/`
- source style images under `style_images/`

The notebook computes aggregate style/content metrics and writes figure files to `output/analysis/style/`.

If `dino_style_cosine` or `lpips_to_original` are missing from the sweep CSVs, the notebook computes them from the generated PNGs. This requires DINOv2 via `torch.hub` and LPIPS; DINOv2 weights may be downloaded on first use.
