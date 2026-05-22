#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import torch
import numpy as np
from diffusers.utils import load_image
from PIL import Image

from flux2.mpc import (
    Flux2PipelineMPC,
    load_pipe,
    load_style_model,
    make_area_ratio_reward,
    make_intensity_range_reward,
    make_luminance_reward,
    make_marbles_count_reward,
    make_red_penalty_reward,
    make_style_reward,
    make_superres_reward,
    make_superres_gtspace_reward,
)
from clip.clip import clip

try:
    import lpips as _lpips
except Exception:
    _lpips = None


_LPIPS_MODELS: Dict[Tuple[str, str], Any] = {}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Flux2 MPC runner.")

    p.add_argument("--repo-id", type=str, default="diffusers/FLUX.2-dev-bnb-4bit")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])

    p.add_argument("--prompt", type=str, required=True)
    p.add_argument("--out", type=str, default="flux2_output.png")

    p.add_argument("--height", type=int, default=448)
    p.add_argument("--width", type=int, default=448)
    p.add_argument("--steps", type=int, default=28)
    p.add_argument("--guidance-scale", type=float, default=4.0)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--image", action="append", default=[], help="Optional conditioning image(s).")
    p.add_argument("--reward-image", type=str, default=None, help="Optional image used to build reward reference.")

    p.add_argument(
        "--reward",
        type=str,
        default="none",
        choices=[
            "none",
            "style",
            "red",
            "range",
            "area_ratio",
            "marbles",
            "luminance",
            "superres",
            "superres_gtspace",
        ],
        help="Reward to use. Extend for custom MPC rewards.",
    )
    p.add_argument(
        "--method",
        type=str,
        default="mpc",
        choices=["mpc", "flowchef"],
        help="Correction method: MPC (control u) or FlowChef (optimize latents).",
    )
    p.add_argument("--opt-steps", type=int, default=10)
    p.add_argument("--mpc-lr", type=float, default=1e-1)
    p.add_argument("--mpc-rho", type=float, default=0.0, help="Control energy weight for MPC.")
    p.add_argument("--style-image", type=str, default=None)

    p.add_argument("--red-threshold", type=float, default=0.6, help="Pixel intensity threshold for red penalty.")
    p.add_argument("--range-low", type=float, default=0.2, help="Lower grayscale intensity bound.")
    p.add_argument("--range-high", type=float, default=0.8, help="Upper grayscale intensity bound.")
    p.add_argument("--area-ratio-target", type=float, default=2.0, help="Target right/left area ratio.")
    p.add_argument("--area-threshold", type=float, default=0.6, help="Color dominance threshold for area ratio.")
    p.add_argument("--marbles-count", type=int, default=37, help="Target marble count.")
    p.add_argument("--marbles-threshold", type=float, default=0.5, help="Peak threshold for marble counting.")
    p.add_argument("--marbles-kernel", type=int, default=9, help="Odd kernel size for peak detection.")
    p.add_argument("--superres-lr-scale", type=int, default=4, help="Downsample factor for super-res LR reference.")
    p.add_argument(
        "--superres-lr-size",
        type=int,
        nargs=2,
        default=None,
        metavar=("H", "W"),
        help="Explicit LR size (overrides --superres-lr-scale).",
    )
    p.add_argument("--save-dir", type=str, default=None, help="Optional directory to save metrics CSV.")

    return p.parse_args()


def _dtype_from_arg(arg: str) -> torch.dtype:
    if arg == "bf16":
        return torch.bfloat16
    if arg == "fp16":
        return torch.float16
    return torch.float32


def _load_images(paths: List[str]):
    if not paths:
        return None
    return [load_image(p) for p in paths]


def _pil_to_tensor_01(img) -> torch.Tensor:
    arr = np.array(img, dtype=np.float32)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    tensor = torch.from_numpy(arr).permute(2, 0, 1) / 255.0
    return tensor


def _pil_to_lpips_tensor(img) -> torch.Tensor:
    tensor = _pil_to_tensor_01(img.convert("RGB")).unsqueeze(0)
    return tensor * 2.0 - 1.0


def _resize_pil(img, height: int, width: int):
    return img.resize((width, height), resample=Image.BICUBIC)


def _get_lpips_model(device: torch.device, net: str = "alex"):
    if _lpips is None:
        return None
    key = (str(device), net)
    model = _LPIPS_MODELS.get(key)
    if model is None:
        model = _lpips.LPIPS(net=net).to(device).eval()
        _LPIPS_MODELS[key] = model
    return model


def _lpips_pil(pred, gt, device: torch.device, net: str = "alex") -> Optional[float]:
    model = _get_lpips_model(device, net=net)
    if model is None:
        return None
    pred_rgb = pred.convert("RGB")
    gt_rgb = gt.convert("RGB")
    if pred_rgb.size != gt_rgb.size:
        pred_rgb = pred_rgb.resize(gt_rgb.size, resample=Image.BICUBIC)
    pred_t = _pil_to_lpips_tensor(pred_rgb).to(device=device, dtype=torch.float32)
    gt_t = _pil_to_lpips_tensor(gt_rgb).to(device=device, dtype=torch.float32)
    with torch.no_grad():
        score = model(pred_t, gt_t)
    return float(score.detach().cpu().item())


def _resolve_output_path(args: argparse.Namespace) -> str:
    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        return os.path.join(args.save_dir, os.path.basename(args.out))
    return args.out


def _resolve_csv_path(output_path: str, save_dir: Optional[str]) -> Optional[str]:
    if save_dir is None:
        return None
    base = os.path.splitext(os.path.basename(output_path))[0]
    return os.path.join(save_dir, f"{base}.csv")


def _compute_clip_score(prompt: str, image, device: torch.device) -> Optional[float]:
    try:
        model, preprocess = clip.load("ViT-B/16", device=device)
        text = clip.tokenize([prompt]).to(device)
        img = preprocess(image).unsqueeze(0).to(device)
        with torch.no_grad():
            image_features = model.encode_image(img)
            text_features = model.encode_text(text)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            score = (image_features * text_features).sum(dim=-1).item()
        return float(score)
    except Exception:
        return None


def _psnr_np(pred: np.ndarray, gt: np.ndarray) -> Optional[float]:
    mse = np.mean((pred - gt) ** 2)
    if mse == 0:
        return float("inf")
    return float(10.0 * np.log10(1.0 / mse))


try:
    from skimage.metrics import structural_similarity as _ssim
except Exception:
    _ssim = None


def _ssim_np(pred: np.ndarray, gt: np.ndarray) -> Optional[float]:
    if _ssim is not None:
        return float(_ssim(pred, gt, channel_axis=2, data_range=1.0))
    x = pred.mean(axis=2)
    y = gt.mean(axis=2)
    mu_x = x.mean()
    mu_y = y.mean()
    var_x = x.var()
    var_y = y.var()
    cov = ((x - mu_x) * (y - mu_y)).mean()
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    return float(((2 * mu_x * mu_y + c1) * (2 * cov + c2)) / ((mu_x ** 2 + mu_y ** 2 + c1) * (var_x + var_y + c2)))


def _prompt_cache_paths(cache_dir: str) -> tuple[str, str]:
    index_path = os.path.join(cache_dir, "index.json")
    data_dir = os.path.join(cache_dir, "data")
    return index_path, data_dir


def _load_prompt_cache(cache_dir: str) -> Dict[str, Any]:
    index_path, _ = _prompt_cache_paths(cache_dir)
    if not os.path.exists(index_path):
        return {}
    with open(index_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_prompt_cache(cache_dir: str, data: Dict[str, Any]) -> None:
    index_path, _ = _prompt_cache_paths(cache_dir)
    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)


def _prompt_cache_key(prompt: str, max_sequence_length: int, text_encoder_out_layers: Tuple[int], num_images_per_prompt: int) -> str:
    payload = json.dumps(
        {
            "prompt": prompt,
            "max_sequence_length": max_sequence_length,
            "text_encoder_out_layers": text_encoder_out_layers,
            "num_images_per_prompt": num_images_per_prompt,
        },
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _load_cached_prompt(
    cache_dir: str,
    prompt: str,
    max_sequence_length: int,
    text_encoder_out_layers: Tuple[int],
    num_images_per_prompt: int,
) -> Optional[Dict[str, Any]]:
    cache = _load_prompt_cache(cache_dir)
    key = _prompt_cache_key(prompt, max_sequence_length, text_encoder_out_layers, num_images_per_prompt)
    entry = cache.get(key)
    if not entry:
        return None
    tensor_path = entry.get("path")
    if not tensor_path or not os.path.exists(tensor_path):
        return None
    return {"path": tensor_path, "key": key}


def _save_cached_prompt(
    cache_dir: str,
    prompt: str,
    max_sequence_length: int,
    text_encoder_out_layers: Tuple[int],
    num_images_per_prompt: int,
    prompt_embeds: torch.Tensor,
    text_ids: torch.Tensor,
) -> None:
    cache = _load_prompt_cache(cache_dir)
    key = _prompt_cache_key(prompt, max_sequence_length, text_encoder_out_layers, num_images_per_prompt)
    _, data_dir = _prompt_cache_paths(cache_dir)
    os.makedirs(data_dir, exist_ok=True)
    tensor_path = os.path.join(data_dir, f"{key}.pt")
    torch.save(
        {
            "prompt_embeds": prompt_embeds.detach().cpu(),
            "text_ids": text_ids.detach().cpu(),
        },
        tensor_path,
    )
    cache[key] = {
        "prompt": prompt,
        "max_sequence_length": max_sequence_length,
        "text_encoder_out_layers": list(text_encoder_out_layers),
        "num_images_per_prompt": num_images_per_prompt,
        "path": tensor_path,
    }
    _save_prompt_cache(cache_dir, cache)


def main() -> None:
    args = parse_args()

    torch_dtype = _dtype_from_arg(args.dtype)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print("Device is ", device)

    pipe: Flux2PipelineMPC = load_pipe(args.repo_id, torch_dtype=torch_dtype)

    reward_dict = None
    mpc_reward_name = None
    style_model = None
    images = _load_images(args.image)
    reward_ref = None
    gray_ref = None
    lr_ref = None
    if args.reward in ("luminance", "superres", "superres_gtspace"):
        if args.reward_image:
            reward_ref = load_image(args.reward_image)
        elif images:
            reward_ref = images[0]
        else:
            raise ValueError("--reward-image or --image is required for this reward.")

    if args.reward == "style":
        if args.style_image is None:
            raise ValueError("--style-image is required when --reward style")
        style_model = load_style_model(device=device)
        reward_fn = make_style_reward(style_model, args.style_image, device=str(device))
        reward_dict = {"name": "style", "value_loss_fn": reward_fn, "style_model": style_model}
        mpc_reward_name = "style"
    elif args.reward == "red":
        reward_fn = make_red_penalty_reward(threshold=args.red_threshold)
        reward_dict = {"name": "red", "value_loss_fn": reward_fn}
        mpc_reward_name = "red"
    elif args.reward == "range":
        reward_fn = make_intensity_range_reward(low=args.range_low, high=args.range_high)
        reward_dict = {"name": "range", "value_loss_fn": reward_fn}
        mpc_reward_name = "range"
    elif args.reward == "area_ratio":
        reward_fn = make_area_ratio_reward(
            target_ratio=args.area_ratio_target,
            threshold=args.area_threshold,
        )
        reward_dict = {"name": "area_ratio", "value_loss_fn": reward_fn}
        mpc_reward_name = "area_ratio"
    elif args.reward == "marbles":
        reward_fn = make_marbles_count_reward(
            target_count=args.marbles_count,
            threshold=args.marbles_threshold,
            kernel_size=args.marbles_kernel,
        )
        reward_dict = {"name": "marbles", "value_loss_fn": reward_fn}
        mpc_reward_name = "marbles"
    elif args.reward == "luminance":
        if args.height and args.width:
            reward_ref = _resize_pil(reward_ref, args.height, args.width)
        gray_ref = reward_ref.convert("L")
        reward_fn = make_luminance_reward(gray_ref, device=str(device))
        reward_dict = {"name": "luminance", "value_loss_fn": reward_fn}
        mpc_reward_name = "luminance"
        images = [gray_ref] + (images[1:] if images else [])
    elif args.reward in ("superres", "superres_gtspace"):
        reward_ref = reward_ref.convert("RGB")
        if args.height and args.width:
            reward_ref = reward_ref.resize((args.width, args.height), resample=Image.BICUBIC)
        if args.superres_lr_size:
            lr_h, lr_w = args.superres_lr_size
        else:
            if args.height and args.width:
                lr_h = max(1, args.height // args.superres_lr_scale)
                lr_w = max(1, args.width // args.superres_lr_scale)
            else:
                lr_w = max(1, reward_ref.size[0] // args.superres_lr_scale)
                lr_h = max(1, reward_ref.size[1] // args.superres_lr_scale)
        lr_ref = reward_ref.resize((lr_w, lr_h), resample=Image.BICUBIC)
        if args.reward == "superres":
            reward_fn = make_superres_reward(lr_ref, device=str(device))
        else:
            reward_fn = make_superres_gtspace_reward(
                lr_ref,
                gt_size=(reward_ref.size[1], reward_ref.size[0]),
                device=str(device),
            )
        reward_dict = {"name": args.reward, "value_loss_fn": reward_fn}
        mpc_reward_name = args.reward
        images = [lr_ref] + (images[1:] if images else [])

    gen = torch.Generator(device=str(device)).manual_seed(args.seed)

    print("Starting inference...")

    prompt_embeds = None
    text_ids = None
    prompt_cache_dir = "cache/prompt_embeddings"
    max_sequence_length = 512
    text_encoder_out_layers = (10, 20, 30)
    num_images_per_prompt = 1

    cached = _load_cached_prompt(
        prompt_cache_dir,
        args.prompt,
        max_sequence_length,
        text_encoder_out_layers,
        num_images_per_prompt,
    )
    if cached is not None:
        data = torch.load(cached["path"], map_location="cpu")
        prompt_embeds = data["prompt_embeds"].to(device)
        text_ids = data["text_ids"].to(device)
        print("Prompt cache hit")
    else:
        pipe.text_encoder.to(device)
        with torch.no_grad():
            prompt_embeds, text_ids = pipe.encode_prompt(
                prompt=args.prompt,
                prompt_embeds=None,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
                text_encoder_out_layers=text_encoder_out_layers,
            )
        pipe.text_encoder.to("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        _save_cached_prompt(
            prompt_cache_dir,
            args.prompt,
            max_sequence_length,
            text_encoder_out_layers,
            num_images_per_prompt,
            prompt_embeds,
            text_ids,
        )

    output_path = _resolve_output_path(args)
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    out = pipe(
        prompt=None,
        height=args.height,
        width=args.width,
        image=images,
        generator=gen,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        mpc_reward_name=mpc_reward_name,
        num_mpc_steps=args.opt_steps if args.reward != "none" else 0,
        reward_dict=reward_dict,
        mpc_opts={"lr": args.mpc_lr, "rho": args.mpc_rho, "inner_its": args.opt_steps},
        mpc_method=args.method,
        prompt_embeds=prompt_embeds,
        text_ids=text_ids,
        max_sequence_length=max_sequence_length,
        text_encoder_out_layers=text_encoder_out_layers,
    ).images[0]

    out.save(output_path)
    print(f"Saved: {output_path}")

    style_loss = None
    if reward_dict is not None and "value_loss_fn" in reward_dict:
        out_tensor = _pil_to_tensor_01(out).to(device)
        with torch.no_grad():
            style_loss = float(reward_dict["value_loss_fn"](decoded_image=out_tensor).detach().cpu())

    clip_score = _compute_clip_score(args.prompt, out, device)

    avg_path_deviation = None
    if hasattr(pipe, "last_metrics"):
        avg_path_deviation = pipe.last_metrics.get("avg_path_deviation")

    psnr_gt = None
    ssim_gt = None
    psnr_luminance = None
    psnr_measurement = None
    psnr_projected_measurement = None
    gt_lpips = None
    if reward_ref is not None:
        gt_rgb = reward_ref.convert("RGB")
        out_rgb = out.convert("RGB")
        if out_rgb.size != gt_rgb.size:
            out_rgb = out_rgb.resize(gt_rgb.size, resample=Image.BICUBIC)
        gt_arr = np.asarray(gt_rgb).astype(np.float32) / 255.0
        out_arr = np.asarray(out_rgb).astype(np.float32) / 255.0
        psnr_gt = _psnr_np(out_arr, gt_arr)
        ssim_gt = _ssim_np(out_arr, gt_arr)
        gt_lpips = _lpips_pil(out_rgb, gt_rgb, device=device)
        if gray_ref is not None:
            gray = gray_ref
            if gray.size != gt_rgb.size:
                gray = gray.resize(gt_rgb.size, resample=Image.BICUBIC)
            gray_arr = np.asarray(gray).astype(np.float32) / 255.0
            lum = 0.299 * out_arr[..., 0] + 0.587 * out_arr[..., 1] + 0.114 * out_arr[..., 2]
            psnr_luminance = _psnr_np(lum, gray_arr)
        if lr_ref is not None:
            out_lr = out_rgb.resize(lr_ref.size, resample=Image.BICUBIC)
            lr_arr = np.asarray(lr_ref.convert("RGB")).astype(np.float32) / 255.0
            out_lr_arr = np.asarray(out_lr).astype(np.float32) / 255.0
            psnr_measurement = _psnr_np(out_lr_arr, lr_arr)
            out_projected = out_lr.resize(gt_rgb.size, resample=Image.BICUBIC)
            ref_projected = lr_ref.resize(gt_rgb.size, resample=Image.BICUBIC)
            out_projected_arr = np.asarray(out_projected).astype(np.float32) / 255.0
            ref_projected_arr = np.asarray(ref_projected).astype(np.float32) / 255.0
            psnr_projected_measurement = _psnr_np(out_projected_arr, ref_projected_arr)
        if gt_lpips is None and _lpips is None:
            print("LPIPS unavailable: install `lpips` to populate gt_lpips.")

    csv_path = _resolve_csv_path(output_path, args.save_dir)
    if csv_path is not None:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "style_loss",
                    "clip_score",
                    "avg_path_deviation",
                    "psnr_gt",
                    "ssim_gt",
                    "gt_lpips",
                    "psnr_measurement",
                    "psnr_projected_measurement",
                    "psnr_luminance",
                ],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "style_loss": "" if style_loss is None else style_loss,
                    "clip_score": "" if clip_score is None else clip_score,
                    "avg_path_deviation": "" if avg_path_deviation is None else avg_path_deviation,
                    "psnr_gt": "" if psnr_gt is None else psnr_gt,
                    "ssim_gt": "" if ssim_gt is None else ssim_gt,
                    "gt_lpips": "" if gt_lpips is None else gt_lpips,
                    "psnr_measurement": "" if psnr_measurement is None else psnr_measurement,
                    "psnr_projected_measurement": "" if psnr_projected_measurement is None else psnr_projected_measurement,
                    "psnr_luminance": "" if psnr_luminance is None else psnr_luminance,
                }
            )
        print(f"Saved metrics: {csv_path}")

    try:
        pipe.to("cpu")
    except Exception:
        pass
    del out, pipe.text_encoder, pipe.transformer, pipe, style_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        raise
    else:
        if os.name == "nt":
            # The Windows notebook/subprocess path can crash in native teardown
            # after successful output writes; exit cleanly once work is done.
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(0)
