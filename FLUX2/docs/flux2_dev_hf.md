# FLUX.2 [dev] via Hugging Face Diffusers

The main public entrypoint for this folder is [`cli_mpc_flow.py`](../cli_mpc_flow.py), documented in the FLUX.2 [README](../README.md).

That script is **not** the upstream FLUX.2 CLI. It is a modified MPC-Flow runner that uses the FLUX.2 Diffusers model-loading path and adds MPC/FlowChef control, inverse-problem rewards, image-conditioning conventions, and metrics output. This note is only for users who want to inspect the underlying Hugging Face / Diffusers loading path directly.

## Model Path Used in MPC-Flow

The large-scale FLUX experiments in MPC-Flow used:

```text
diffusers/FLUX.2-dev-bnb-4bit
```

This is the 4-bit quantized Diffusers release of FLUX.2 [dev]. The repository does **not** redistribute weights.

Before use:
1. Request access to FLUX.2 [dev] on Hugging Face.
2. Authenticate locally:

```bash
hf auth login
```

## Minimal Diffusers Load

```python
import torch
from diffusers import AutoModel, Flux2Pipeline
from transformers import Mistral3ForConditionalGeneration

repo_id = "diffusers/FLUX.2-dev-bnb-4bit"
torch_dtype = torch.bfloat16

text_encoder = Mistral3ForConditionalGeneration.from_pretrained(
    repo_id,
    subfolder="text_encoder",
    torch_dtype=torch_dtype,
    device_map="cpu",
)
transformer = AutoModel.from_pretrained(
    repo_id,
    subfolder="transformer",
    torch_dtype=torch_dtype,
    device_map="cpu",
)
pipe = Flux2Pipeline.from_pretrained(
    repo_id,
    text_encoder=text_encoder,
    transformer=transformer,
    torch_dtype=torch_dtype,
)
```

For repository-level MPC-Flow demos, use the modified repository CLI instead:

```bash
python cli_mpc_flow.py --prompt "a cat sitting on a windowsill at sunset" --reward none --save-dir output/demos/prompt_only --out prompt_only.png
```
