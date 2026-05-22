#!/usr/bin/env python3

from __future__ import annotations

import argparse
import itertools
import json
import os
import subprocess
import sys
from typing import Any, Dict, Iterable, List, Tuple


def _load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _to_cli_args(args: Dict[str, Any]) -> List[str]:
    cli_args: List[str] = []
    for key, value in args.items():
        if value is None:
            continue
        if key.startswith("flowchef_") or key == "run_dir_template":
            continue
        flag = f"--{key.replace('_', '-')}"
        if isinstance(value, bool):
            if value:
                cli_args.append(flag)
            continue
        if isinstance(value, list):
            for item in value:
                cli_args.extend([flag, str(item)])
            continue
        cli_args.extend([flag, str(value)])
    return cli_args


def _grid_product(grid: Dict[str, List[Any]], ordered_keys: List[str] | None = None) -> Iterable[Dict[str, Any]]:
    if not grid:
        yield {}
        return
    if ordered_keys is None:
        keys = list(grid.keys())
    else:
        keys = [k for k in ordered_keys if k in grid] + [k for k in grid.keys() if k not in ordered_keys]
    values = [grid[k] for k in keys]
    for combo in itertools.product(*values):
        yield dict(zip(keys, combo))


def _iter_runs(cfg: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    base_args = cfg.get("base_args", {})
    runs = cfg.get("runs", [])
    grid = cfg.get("grid", {})

    def _normalize_opt_steps(run_args: Dict[str, Any]) -> Dict[str, Any]:
        normalized = dict(run_args)
        if "mpc_steps" in normalized and "opt_steps" not in normalized:
            normalized["opt_steps"] = normalized.pop("mpc_steps")
        else:
            normalized.pop("mpc_steps", None)
        return normalized

    if runs:
        for run in runs:
            merged = dict(base_args)
            merged.update(run)
            merged = _normalize_opt_steps(merged)
            yield merged
        return

    seen: set[Tuple[Tuple[str, Any], ...]] = set()
    ordered_keys = ["seed", "style_image", "prompt", "method"]
    for combo in _grid_product(grid, ordered_keys=ordered_keys):
        method = combo.get("method")
        skip_combo = False
        if method == "flowchef":
            for key in [k for k in combo.keys() if k.startswith("mpc_")]:
                if key in grid and combo.get(key) != grid[key][0]:
                    skip_combo = True
                    break
        if method == "mpc":
            for key in [k for k in combo.keys() if k.startswith("flowchef_")]:
                if key in grid and combo.get(key) != grid[key][0]:
                    skip_combo = True
                    break
        if skip_combo:
            continue

        merged = dict(base_args)
        merged.update(combo)
        merged = _normalize_opt_steps(merged)

        if merged.get("method") == "flowchef" and merged.get("opt_steps") == 0:
            continue

        if merged.get("opt_steps") == 0:
            merged.pop("mpc_rho", None)

        if merged.get("method") == "flowchef":
            merged.pop("mpc_rho", None)
            if "flowchef_lr" in merged:
                merged["mpc_lr"] = merged.pop("flowchef_lr")
        elif merged.get("method") == "mpc":
            merged.pop("flowchef_lr", None)

        key = tuple(sorted(merged.items()))
        if key in seen:
            continue
        seen.add(key)
        yield merged


def _prompt_prefix(prompt: str, length: int = 8) -> str:
    cleaned = "".join(ch for ch in prompt.strip() if ch.isalnum() or ch in ("_", "-")).lower()
    return (cleaned or "prompt")[:length]


def _resolve_run_dir(base_dir: str | None, run_idx: int, run_args: Dict[str, Any]) -> str | None:
    run_dir_template = run_args.pop("run_dir_template", None)
    if run_dir_template:
        values = dict(run_args)
        values["run_idx"] = f"{run_idx:04d}"
        if "mpc_rho" not in values:
            values["mpc_rho"] = "na"
        values["effective_lr"] = values.get("flowchef_lr", values.get("mpc_lr"))
        style_image = values.get("style_image")
        if style_image:
            base = os.path.basename(style_image)
            values["style_image_name"] = os.path.splitext(base)[0]
        reward_image = values.get("reward_image")
        if reward_image:
            base = os.path.basename(reward_image)
            values["reward_image_name"] = os.path.splitext(base)[0]
        return run_dir_template.format(**values)
    if base_dir is None:
        return None
    return os.path.join(base_dir, f"run_{run_idx:04d}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sweep runner for cli_mpc_flow.py.")
    p.add_argument("--config", type=str, required=True, help="Path to JSON sweep config.")
    p.add_argument("--python", type=str, default=sys.executable, help="Python interpreter to use.")
    p.add_argument("--cli", type=str, default="cli_mpc_flow.py", help="Path to cli_mpc_flow.py.")
    p.add_argument("--dry-run", action="store_true", help="Print commands without running.")
    p.add_argument("--resume", action="store_true", help="Skip runs with existing outputs.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = _load_config(args.config)
    base_save_dir = cfg.get("base_save_dir")
    output_name_template = cfg.get("output_name_template")
    run_idx = 0

    for run_args in _iter_runs(cfg):
        run_idx += 1
        run_save_dir = _resolve_run_dir(base_save_dir, run_idx, run_args)
        if run_save_dir is not None:
            if not args.dry_run:
                os.makedirs(run_save_dir, exist_ok=True)
            run_args["save_dir"] = run_save_dir

        if output_name_template:
            prompt = run_args.get("prompt", "")
            out_name = output_name_template.format(
                prompt_prefix=_prompt_prefix(prompt),
                run_idx=f"{run_idx:04d}",
                **run_args,
            )
            reward_image = run_args.get("reward_image")
            if reward_image:
                reward_base = os.path.splitext(os.path.basename(reward_image))[0]
                if reward_base and reward_base not in out_name:
                    base, ext = os.path.splitext(out_name)
                    out_name = f"{base}_{reward_base}{ext or '.png'}"
            run_args["out"] = out_name

        if args.resume and run_save_dir is not None and run_args.get("out"):
            output_path = os.path.join(run_save_dir, os.path.basename(run_args["out"]))
            if os.path.exists(output_path):
                print("SKIP", run_idx, ":", output_path)
                continue

        cmd = [args.python, args.cli] + _to_cli_args(run_args)
        # Use Windows-safe quoting so prompts with spaces can be copy/pasted.
        cmd_display = subprocess.list2cmdline(cmd)
        print("RUN", run_idx, ":", cmd_display)
        if args.dry_run:
            continue

        from torch.cuda import empty_cache
        import time, gc
        gc.collect()
        time.sleep(2)
        empty_cache()
        time.sleep(3)

        child_env = os.environ.copy()
        if os.name == "nt":
            child_env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
        subprocess.run(cmd, check=True, env=child_env)


if __name__ == "__main__":
    main()
