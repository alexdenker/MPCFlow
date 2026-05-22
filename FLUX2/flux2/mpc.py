from __future__ import annotations

import inspect
import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import numpy as np
import PIL
import torch.nn.functional as F
from torch.cuda import empty_cache

from diffusers import AutoModel, Flux2Pipeline
from diffusers.pipelines.flux2.pipeline_output import Flux2PipelineOutput
from transformers import Mistral3ForConditionalGeneration

from clip.clip import clip


def compute_empirical_mu(image_seq_len: int, num_steps: int) -> float:
    a1, b1 = 8.73809524e-05, 1.89833333
    a2, b2 = 0.00016927, 0.45666666

    if image_seq_len > 4300:
        mu = a2 * image_seq_len + b2
        return float(mu)

    m_200 = a2 * image_seq_len + b2
    m_10 = a1 * image_seq_len + b1

    a = (m_200 - m_10) / 190.0
    b = m_200 - 200.0 * a
    mu = a * num_steps + b
    return float(mu)


def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed.")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError("Scheduler does not support custom timesteps.")
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError("Scheduler does not support custom sigmas.")
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


class Flux2PipelineMPC(Flux2Pipeline):
    def __call__(
        self,
        image: Optional[Union[List[PIL.Image.Image], PIL.Image.Image]] = None,
        prompt: Union[str, List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        sigmas: Optional[List[float]] = None,
        guidance_scale: Optional[float] = 4.0,
        num_images_per_prompt: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        text_ids: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
        text_encoder_out_layers: Tuple[int] = (10, 20, 30),
        mpc_reward_name: str = None,
        num_mpc_steps: int = 5,
        reward_dict: Optional[Dict[str, Any]] = None,
        mpc_opts: Optional[Dict[str, Any]] = None,
        mpc_method: str = "mpc",
        psnr_target: Optional[Union[PIL.Image.Image, torch.Tensor]] = None,
        offload_transformer_before_decode: Optional[bool] = None,
        offload_models_after_call: Optional[bool] = None,
    ):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        reset_models_before_call = os.environ.get("MPCFLOW_RESET_MODELS_BEFORE_CALL", "0") == "1"
        if offload_transformer_before_decode is None:
            offload_transformer_before_decode = os.environ.get("MPCFLOW_OFFLOAD_TRANSFORMER_BEFORE_DECODE", "0") == "1"
        if offload_models_after_call is None:
            offload_models_after_call = os.environ.get("MPCFLOW_OFFLOAD_MODELS_AFTER_CALL", "0") == "1"

        if reset_models_before_call:
            self.vae.to("cpu")
            self.transformer.to("cpu")
            empty_cache()

        with torch.no_grad():
            self.check_inputs(
                prompt=prompt,
                height=height,
                width=width,
                prompt_embeds=prompt_embeds,
                callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            )
            
            if prompt_embeds is not None and text_ids is None:
                raise ValueError("text_ids must be provided when prompt_embeds is set.")

            use_cached_prompt = prompt_embeds is not None and text_ids is not None
            if not use_cached_prompt:
                self.text_encoder.to(device)
            self._guidance_scale = guidance_scale
            self._attention_kwargs = attention_kwargs
            self._current_timestep = None
            self._interrupt = False

            if prompt is not None and isinstance(prompt, str):
                batch_size = 1
            elif prompt is not None and isinstance(prompt, list):
                batch_size = len(prompt)
            else:
                batch_size = prompt_embeds.shape[0]

            

            if not use_cached_prompt:
                prompt_embeds, text_ids = self.encode_prompt(
                    prompt=prompt,
                    prompt_embeds=prompt_embeds,
                    device=device,
                    num_images_per_prompt=num_images_per_prompt,
                    max_sequence_length=max_sequence_length,
                    text_encoder_out_layers=text_encoder_out_layers,
                )

            if not use_cached_prompt:
                self.text_encoder.to("cpu")
                empty_cache()

            self.vae.to(device)
            self.transformer.to(device)
            empty_cache()

            if image is not None and not isinstance(image, list):
                image = [image]

            condition_images = None
            if image is not None:
                for img in image:
                    self.image_processor.check_image_input(img)

                condition_images = []
                for img in image:
                    image_width, image_height = img.size
                    if image_width * image_height > 1024 * 1024:
                        img = self.image_processor._resize_to_target_area(img, 1024 * 1024)
                        image_width, image_height = img.size

                    multiple_of = self.vae_scale_factor * 2
                    image_width = (image_width // multiple_of) * multiple_of
                    image_height = (image_height // multiple_of) * multiple_of
                    img = self.image_processor.preprocess(img, height=image_height, width=image_width, resize_mode="crop")
                    condition_images.append(img)
                    height = height or image_height
                    width = width or image_width

            height = height or self.default_sample_size * self.vae_scale_factor
            width = width or self.default_sample_size * self.vae_scale_factor

            num_channels_latents = self.transformer.config.in_channels // 4
            latents, latent_ids = self.prepare_latents(
                batch_size=batch_size * num_images_per_prompt,
                num_latents_channels=num_channels_latents,
                height=height,
                width=width,
                dtype=prompt_embeds.dtype,
                device=device,
                generator=generator,
                latents=latents,
            )

            image_latents = None
            image_latent_ids = None
            if condition_images is not None:
                image_latents, image_latent_ids = self.prepare_image_latents(
                    images=condition_images,
                    batch_size=batch_size * num_images_per_prompt,
                    generator=generator,
                    device=device,
                    dtype=self.vae.dtype,
                )

            psnr_ref = None
            if psnr_target is not None:
                if isinstance(psnr_target, PIL.Image.Image):
                    psnr_ref = _pil_to_tensor_01(psnr_target)
                elif torch.is_tensor(psnr_target):
                    psnr_ref = psnr_target
                else:
                    raise TypeError("psnr_target must be a PIL.Image.Image or torch.Tensor.")
                if psnr_ref.ndim == 4:
                    psnr_ref = psnr_ref[0]
                if psnr_ref.ndim != 3 or psnr_ref.shape[0] != 3:
                    raise ValueError("psnr_target must be a 3xHxW tensor or RGB PIL image.")
                psnr_ref = psnr_ref.to(device=device, dtype=torch.float32)

            sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps) if sigmas is None else sigmas
            if hasattr(self.scheduler.config, "use_flow_sigmas") and self.scheduler.config.use_flow_sigmas:
                sigmas = None

            image_seq_len = latents.shape[1]
            mu = compute_empirical_mu(image_seq_len=image_seq_len, num_steps=num_inference_steps)
            timesteps, num_inference_steps = retrieve_timesteps(
                self.scheduler,
                num_inference_steps,
                device,
                sigmas=sigmas,
                mu=mu,
            )

            num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
            self._num_timesteps = len(timesteps)

            guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32)
            guidance = guidance.expand(latents.shape[0])

        self.scheduler.set_begin_index(0)
        path_dev_sum = 0.0
        psnr_by_step = []
        print(f"Begin image generation with {mpc_method}", flush=True)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                self._current_timestep = t
                t_val = float(t.item()) if hasattr(t, "item") else float(t)
                if i + 1 < len(timesteps):
                    next_t = timesteps[i + 1]
                    next_t_val = float(next_t.item()) if hasattr(next_t, "item") else float(next_t)
                else:
                    next_t_val = 0.0
                dt_norm = abs(t_val - next_t_val) / 1000.0
                latents_before = latents.detach()
                timestep = t.expand(latents.shape[0]).to(latents.dtype)

                latent_model_input = latents.to(self.transformer.dtype)
                latent_image_ids = latent_ids

                if image_latents is not None:
                    latent_model_input = torch.cat([latents, image_latents], dim=1).to(self.transformer.dtype)
                    latent_image_ids = torch.cat([latent_ids, image_latent_ids], dim=1)

                with torch.no_grad():
                    self.vae.to("cpu")
                    empty_cache()
                    noise_pred = self.transformer(
                        hidden_states=latent_model_input,
                        timestep=timestep / 1000,
                        guidance=guidance,
                        encoder_hidden_states=prompt_embeds,
                        txt_ids=text_ids,
                        img_ids=latent_image_ids,
                        joint_attention_kwargs=self._attention_kwargs,
                        return_dict=False,
                    )[0]
                    self.vae.to(device)

                noise_pred = noise_pred[:, : latents.size(1) :]
                noise_pred_original = noise_pred.detach()

                if mpc_reward_name is not None and reward_dict is not None:
                    #self.transformer.single_transformer_blocks.to("cpu")
                    empty_cache()
                    #print(f"MPC: correction method={mpc_method}")
                    sigma_t = torch.tensor(self.scheduler.sigmas[i]).unsqueeze(0).to(latents.device)
                    if mpc_method == "flowchef":
                        latents = self.FlowChef(
                            noise_pred=noise_pred,
                            latents=latents,
                            latent_ids=latent_ids,
                            timestep=timestep,
                            sigma=sigma_t,
                            num_mpc_steps=num_mpc_steps,
                            reward_kwargs=reward_dict,
                            mpc_opts=mpc_opts or {"lr": 1e-1, "inner_its": num_mpc_steps},
                        )
                    else:
                        noise_pred = self.MPC(
                            noise_pred=noise_pred,
                            latents=latents,
                            latent_ids=latent_ids,
                            timestep=timestep,
                            sigma=sigma_t,
                            num_mpc_steps=num_mpc_steps,
                            reward_kwargs=reward_dict,
                            mpc_opts=mpc_opts or {"lr": 1e-1, "inner_its": num_mpc_steps},
                        )
                    #self.transformer.single_transformer_blocks.to(device)

                if mpc_reward_name is not None and reward_dict is not None:
                    # Compare correction impact in latent space without advancing scheduler state.
                    sigma_idx = self.scheduler.step_index
                    if sigma_idx is None:
                        sigma_idx = self.scheduler.index_for_timestep(t, self.scheduler.timesteps)
                    dt = (self.scheduler.sigmas[sigma_idx + 1] - self.scheduler.sigmas[sigma_idx]).to(
                        device=latents_before.device, dtype=latents_before.dtype
                    )
                    latents_base = latents_before + dt * noise_pred_original
                    if mpc_method == "flowchef":
                        latents_corr = latents + dt * noise_pred_original
                    else:
                        latents_corr = latents_before + dt * noise_pred
                    diff = (latents_corr - latents_base).abs().mean()
                    path_dev_sum += float((diff * dt_norm).detach().cpu())

                latents_dtype = latents.dtype
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
                if latents.dtype != latents_dtype and torch.backends.mps.is_available():
                    latents = latents.to(latents_dtype)

                if psnr_ref is not None:
                    with torch.no_grad():
                        lat_tmp = self._unpack_latents_with_ids(latents, latent_ids)
                        latents_bn_mean = self.vae.bn.running_mean.view(1, -1, 1, 1).to(
                            lat_tmp.device, lat_tmp.dtype
                        )
                        latents_bn_std = torch.sqrt(
                            self.vae.bn.running_var.view(1, -1, 1, 1) + self.vae.config.batch_norm_eps
                        ).to(lat_tmp.device, lat_tmp.dtype)
                        lat_tmp = lat_tmp * latents_bn_std + latents_bn_mean
                        lat_tmp = self._unpatchify_latents(lat_tmp)
                        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=True):
                            img_step = self.vae.decode(lat_tmp, return_dict=False)[0]
                        img_step = (img_step.clamp(-1, 1) + 1) / 2
                        if img_step.shape[0] == 1:
                            img_step = img_step.squeeze(0)
                        img_step = img_step.to(dtype=torch.float32)
                        mse = F.mse_loss(img_step, psnr_ref)
                        if torch.isfinite(mse) and mse.item() > 0:
                            psnr = 10.0 * torch.log10(1.0 / mse)
                            psnr_by_step.append(float(psnr.detach().cpu()))
                        else:
                            psnr_by_step.append(float("inf"))

                        empty_cache()

                if callback_on_step_end is not None:
                    callback_kwargs = {k: locals()[k] for k in callback_on_step_end_tensor_inputs}
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)
                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

        with torch.no_grad():
            self._current_timestep = None
            if output_type == "latent":
                image_out = latents
            else:
                # The transformer is no longer needed for the final VAE decode.
                # Offloading it first can avoid peak-memory spikes at large image sizes, but
                # it is a slow synchronous transfer for the full FLUX.2 transformer on Windows.
                if offload_transformer_before_decode:
                    self.transformer.to("cpu")
                    empty_cache()

                latents = self._unpack_latents_with_ids(latents, latent_ids)
                latents_bn_mean = self.vae.bn.running_mean.view(1, -1, 1, 1).to(latents.device, latents.dtype)
                latents_bn_std = torch.sqrt(
                    self.vae.bn.running_var.view(1, -1, 1, 1) + self.vae.config.batch_norm_eps
                ).to(latents.device, latents.dtype)
                latents = latents * latents_bn_std + latents_bn_mean
                latents = self._unpatchify_latents(latents)

                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=True):
                    image_out = self.vae.decode(latents, return_dict=False)[0]
                image_out = self.image_processor.postprocess(image_out, output_type=output_type)

            self.maybe_free_model_hooks()
            if not return_dict:
                return (image_out,)

        if offload_models_after_call:
            self.vae.to("cpu")
            self.transformer.to("cpu")
            empty_cache()
        self.last_metrics = {
            "avg_path_deviation": path_dev_sum,
            "psnr_by_step": psnr_by_step if psnr_ref is not None else None,
        }
        return Flux2PipelineOutput(images=image_out)

    def MPC(
        self,
        *,
        noise_pred: torch.Tensor,
        latents: torch.Tensor,
        latent_ids: torch.Tensor,
        timestep: torch.Tensor,
        sigma: torch.Tensor,
        num_mpc_steps: int,
        reward_kwargs: Dict[str, Any],
        mpc_opts: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        if num_mpc_steps is None or num_mpc_steps <= 0:
            return noise_pred
        if reward_kwargs is None:
            return noise_pred

        mpc_opts = mpc_opts or {}
        name = reward_kwargs.get("name", "reward")
        value_loss_fn = reward_kwargs.get("value_loss_fn", None)
        if not callable(value_loss_fn):
            raise ValueError(f"reward_kwargs['value_loss_fn'] must be callable (got {type(value_loss_fn)}).")

        try:
            value_loss_params = inspect.signature(value_loss_fn).parameters
        except (TypeError, ValueError):
            value_loss_params = {}
        use_base_image = "base_image" in value_loss_params

        try:
            value_loss_params = inspect.signature(value_loss_fn).parameters
        except (TypeError, ValueError):
            value_loss_params = {}
        use_base_image = "base_image" in value_loss_params

        requires = reward_kwargs.get("requires", [])
        for k in requires:
            if reward_kwargs.get(k, None) is None:
                raise ValueError(f"reward_kwargs for '{name}' requires key '{k}'.")

        lr = float(mpc_opts.get("lr", 1e-1))
        rho = float(mpc_opts.get("rho", 0.0))
        inner_its = int(mpc_opts.get("inner_its", num_mpc_steps))
        autocast_dtype = mpc_opts.get("autocast_dtype", torch.bfloat16)

        device = latents.device
        use_autocast = (device.type == "cuda")
        t_norm = (timestep.float() / 1000.0).clamp(min=1e-6)

        with torch.no_grad():
            tweedies = (latents + (0.0 - sigma[:, None, None]) * noise_pred).detach()

        base_img_01 = None
        if use_base_image:
            with torch.no_grad():
                lat_tmp = self._unpack_latents_with_ids(tweedies, latent_ids)
                latents_bn_mean = self.vae.bn.running_mean.view(1, -1, 1, 1).to(lat_tmp.device, lat_tmp.dtype)
                latents_bn_std = torch.sqrt(
                    self.vae.bn.running_var.view(1, -1, 1, 1) + self.vae.config.batch_norm_eps
                ).to(lat_tmp.device, lat_tmp.dtype)
                lat_tmp = lat_tmp * latents_bn_std + latents_bn_mean
                lat_tmp = self._unpatchify_latents(lat_tmp)
                with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=use_autocast):
                    img_base = self.vae.decode(lat_tmp, return_dict=False)[0]
                base_img_01 = (img_base.clamp(-1, 1) + 1) / 2
                if base_img_01.shape[0] == 1:
                    base_img_01 = base_img_01.squeeze(0)

        u = torch.zeros_like(tweedies, dtype=tweedies.dtype, device=device, requires_grad=True)
        opt = torch.optim.Adam([u], lr=lr)

        if "style_model" in reward_kwargs and hasattr(reward_kwargs["style_model"], "to"):
            reward_kwargs["style_model"] = reward_kwargs["style_model"].to(device).eval()

        for _ in range(inner_its):
            opt.zero_grad(set_to_none=True)
            tmp = tweedies + u

            lat_tmp = self._unpack_latents_with_ids(tmp, latent_ids)

            latents_bn_mean = self.vae.bn.running_mean.view(1, -1, 1, 1).to(lat_tmp.device, lat_tmp.dtype)
            latents_bn_std = torch.sqrt(
                self.vae.bn.running_var.view(1, -1, 1, 1) + self.vae.config.batch_norm_eps
            ).to(lat_tmp.device, lat_tmp.dtype)

            lat_tmp = lat_tmp * latents_bn_std + latents_bn_mean
            lat_tmp = self._unpatchify_latents(lat_tmp)

            with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=use_autocast):
                img_tmp = self.vae.decode(lat_tmp, return_dict=False)[0]

            img_01 = (img_tmp.clamp(-1, 1) + 1) / 2
            decoded_image = img_01.squeeze(0) if img_01.shape[0] == 1 else img_01
            if use_base_image:
                value_loss = value_loss_fn(decoded_image=decoded_image, base_image=base_img_01)
            else:
                value_loss = value_loss_fn(decoded_image=decoded_image)
            control_loss = torch.sum(u * u)
            weight = (1.0 - t_norm).mean()
            loss = value_loss + rho * weight * control_loss

            loss.backward()
            opt.step()

            del loss, value_loss, control_loss, img_tmp
            #empty_cache()

        with torch.no_grad():
            noise_pred = noise_pred - (u.detach() / sigma[:, None, None])

        return noise_pred.detach()

    def FlowChef(
        self,
        *,
        noise_pred: torch.Tensor,
        latents: torch.Tensor,
        latent_ids: torch.Tensor,
        timestep: torch.Tensor,
        sigma: torch.Tensor,
        num_mpc_steps: int,
        reward_kwargs: Dict[str, Any],
        mpc_opts: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        if num_mpc_steps is None or num_mpc_steps <= 0:
            return noise_pred
        if reward_kwargs is None:
            return noise_pred

        mpc_opts = mpc_opts or {}
        name = reward_kwargs.get("name", "reward")
        value_loss_fn = reward_kwargs.get("value_loss_fn", None)
        if not callable(value_loss_fn):
            raise ValueError(f"reward_kwargs['value_loss_fn'] must be callable (got {type(value_loss_fn)}).")

        try:
            value_loss_params = inspect.signature(value_loss_fn).parameters
        except (TypeError, ValueError):
            value_loss_params = {}
        use_base_image = "base_image" in value_loss_params

        requires = reward_kwargs.get("requires", [])
        for k in requires:
            if reward_kwargs.get(k, None) is None:
                raise ValueError(f"reward_kwargs for '{name}' requires key '{k}'.")

        lr = float(mpc_opts.get("lr", 1e-1))
        inner_its = int(mpc_opts.get("inner_its", num_mpc_steps))
        autocast_dtype = mpc_opts.get("autocast_dtype", torch.bfloat16)

        device = latents.device
        use_autocast = (device.type == "cuda")
        _ = timestep

        noise_pred_detached = noise_pred.detach()
        lat_var = latents.detach().clone().requires_grad_(True)
        opt = torch.optim.Adam([lat_var], lr=lr)

        base_img_01 = None
        if use_base_image:
            with torch.no_grad():
                tweedies = latents + (0.0 - sigma[:, None, None]) * noise_pred_detached
                lat_tmp = self._unpack_latents_with_ids(tweedies, latent_ids)

                latents_bn_mean = self.vae.bn.running_mean.view(1, -1, 1, 1).to(lat_tmp.device, lat_tmp.dtype)
                latents_bn_std = torch.sqrt(
                    self.vae.bn.running_var.view(1, -1, 1, 1) + self.vae.config.batch_norm_eps
                ).to(lat_tmp.device, lat_tmp.dtype)

                lat_tmp = lat_tmp * latents_bn_std + latents_bn_mean
                lat_tmp = self._unpatchify_latents(lat_tmp)

                with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=use_autocast):
                    img_base = self.vae.decode(lat_tmp, return_dict=False)[0]
                base_img_01 = (img_base.clamp(-1, 1) + 1) / 2
                if base_img_01.shape[0] == 1:
                    base_img_01 = base_img_01.squeeze(0)

        if "style_model" in reward_kwargs and hasattr(reward_kwargs["style_model"], "to"):
            reward_kwargs["style_model"] = reward_kwargs["style_model"].to(device).eval()

        for _ in range(inner_its):
            opt.zero_grad(set_to_none=True)

            tweedies = lat_var + (0.0 - sigma[:, None, None]) * noise_pred_detached
            lat_tmp = self._unpack_latents_with_ids(tweedies, latent_ids)

            latents_bn_mean = self.vae.bn.running_mean.view(1, -1, 1, 1).to(lat_tmp.device, lat_tmp.dtype)
            latents_bn_std = torch.sqrt(
                self.vae.bn.running_var.view(1, -1, 1, 1) + self.vae.config.batch_norm_eps
            ).to(lat_tmp.device, lat_tmp.dtype)

            lat_tmp = lat_tmp * latents_bn_std + latents_bn_mean
            lat_tmp = self._unpatchify_latents(lat_tmp)

            with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=use_autocast):
                img_tmp = self.vae.decode(lat_tmp, return_dict=False)[0]
            img_01 = (img_tmp.clamp(-1, 1) + 1) / 2
            decoded_image = img_01.squeeze(0) if img_01.shape[0] == 1 else img_01
            if use_base_image:
                value_loss = value_loss_fn(decoded_image=decoded_image, base_image=base_img_01)
            else:
                value_loss = value_loss_fn(decoded_image=decoded_image)
            value_loss.backward()
            opt.step()

        return lat_var.detach()

def load_style_model(device: str | torch.device = "cuda") -> torch.nn.Module:
    model_name = "ViT-B/16"
    url = clip._MODELS[model_name]
    model_path = clip._download(url)
    clip_model = torch.jit.load(model_path, map_location=device).eval()
    clip_model = clip.build_model(clip_model.state_dict())
    return clip_model


def _transform():
    mean = torch.tensor((0.48145466, 0.4578275, 0.40821073), dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor((0.26862954, 0.26130258, 0.27577711), dtype=torch.float32).view(1, 3, 1, 1)

    def preprocess(batch: torch.Tensor) -> torch.Tensor:
        if batch.ndim != 4 or batch.shape[1] != 3:
            raise ValueError("style image batch must have shape Bx3xHxW.")
        x = batch.to(dtype=torch.float32)
        x = F.interpolate(x, size=(224, 224), mode="bicubic", align_corners=False)
        return (x - mean.to(device=x.device, dtype=x.dtype)) / std.to(device=x.device, dtype=x.dtype)

    return preprocess


def get_style_vector(
    style_model: torch.nn.Module,
    style_image: torch.Tensor,
    device: str | torch.device = "cuda",
) -> torch.Tensor:
    processor = _transform()
    style_model.requires_grad_(False)
    style_model.to(dtype=torch.float32, device=device)

    inputs = processor(style_image.unsqueeze(0))
    _, style_features = style_model.encode_image_with_features(inputs)
    style_features = style_features[2][1:, 0, :]
    gramm_style = torch.matmul(style_features.T, style_features)
    return gramm_style


def make_style_reward(
    style_model: torch.nn.Module,
    style_image_path: str,
    device: str,
) -> Callable[[torch.Tensor], torch.Tensor]:
    if not os.path.exists(style_image_path):
        raise FileNotFoundError(f"Style image not found: {style_image_path}")

    arr = np.array(PIL.Image.open(style_image_path).convert("RGB")) / 255.0
    original_style_image = torch.from_numpy(arr).permute(2, 0, 1).float().to(device)
    original_style_vec = get_style_vector(style_model, original_style_image, device=device)

    def reward_fn(decoded_image: torch.Tensor) -> torch.Tensor:
        new_style_vec = get_style_vector(style_model, decoded_image, device=device)
        value_loss = torch.sum((original_style_vec - new_style_vec) ** 2)
        return value_loss

    return reward_fn


def make_red_penalty_reward(
    threshold: float = 0.6,
    sharpness: float = 50.0,
    preserve_weight: float = 5.0,
) -> Callable[[torch.Tensor], torch.Tensor]:
    def _hue_from_rgb(r: torch.Tensor, g: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        maxc = torch.maximum(r, torch.maximum(g, b))
        minc = torch.minimum(r, torch.minimum(g, b))
        delta = maxc - minc

        eps = 1e-6
        hue = torch.zeros_like(maxc)
        is_r = (maxc == r) & (delta > eps)
        is_g = (maxc == g) & (delta > eps)
        is_b = (maxc == b) & (delta > eps)
        hue = torch.where(is_r, ((g - b) / (delta + eps)) % 6.0, hue)
        hue = torch.where(is_g, ((b - r) / (delta + eps)) + 2.0, hue)
        hue = torch.where(is_b, ((r - g) / (delta + eps)) + 4.0, hue)
        hue = hue * 60.0
        hue = torch.where(hue < 0.0, hue + 360.0, hue)
        return hue

    def smooth_range(x, low, high):
        left = torch.sigmoid(sharpness * (x - low))
        right = torch.sigmoid(sharpness * (high - x))
        return left * right

    def reward_fn(decoded_image: torch.Tensor, base_image: Optional[torch.Tensor] = None) -> torch.Tensor:
        if decoded_image.ndim != 3 or decoded_image.shape[0] != 3:
            raise ValueError("decoded_image must be a 3xHxW tensor in [0, 1].")
        r = decoded_image[0]
        g = decoded_image[1]
        b = decoded_image[2]

        hue = _hue_from_rgb(r, g, b)
        green_mask = smooth_range(hue, 90.0, 150.0)

        if base_image is None:
            return -green_mask.mean()
        if base_image.ndim != 3 or base_image.shape[0] != 3:
            raise ValueError("base_image must be a 3xHxW tensor in [0, 1].")

        br = base_image[0]
        bg = base_image[1]
        bb = base_image[2]
        base_hue = _hue_from_rgb(br, bg, bb)
        base_red_mask = smooth_range(base_hue, 0.0, 15.0) + smooth_range(base_hue, 345.0, 360.0)
        non_red_mask = (1.0 - base_red_mask).clamp(0.0, 1.0)

        diff = (decoded_image - base_image).pow(2).mean(dim=0)
        preserve_penalty = (diff * non_red_mask).mean()
        green_reward = (green_mask * base_red_mask).mean()
        return -green_reward + preserve_weight * preserve_penalty

    return reward_fn


def _pil_to_tensor_01(img: PIL.Image.Image) -> torch.Tensor:
    arr = np.array(img, dtype=np.float32)
    if arr.ndim == 2:
        tensor = torch.from_numpy(arr).unsqueeze(0)
    else:
        tensor = torch.from_numpy(arr).permute(2, 0, 1)
    return tensor / 255.0


def make_luminance_reward(
    reference_gray: PIL.Image.Image,
    device: str | torch.device = "cuda",
) -> Callable[[torch.Tensor], torch.Tensor]:
    ref = reference_gray.convert("L")
    ref_tensor = _pil_to_tensor_01(ref).to(device)

    def reward_fn(decoded_image: torch.Tensor) -> torch.Tensor:
        if decoded_image.ndim != 3 or decoded_image.shape[0] != 3:
            raise ValueError("decoded_image must be a 3xHxW tensor in [0, 1].")
        img = decoded_image.to(device)
        ref_img = ref_tensor.to(device=device, dtype=img.dtype)
        lum = 0.299 * img[0] + 0.587 * img[1] + 0.114 * img[2]
        return F.mse_loss(lum, ref_img[0])

    return reward_fn


def make_superres_reward(
    lr_reference: PIL.Image.Image,
    device: str | torch.device = "cuda",
) -> Callable[[torch.Tensor], torch.Tensor]:
    ref_rgb = lr_reference.convert("RGB")
    ref_tensor = _pil_to_tensor_01(ref_rgb).to(device)
    lr_size = ref_rgb.size[::-1]

    def reward_fn(decoded_image: torch.Tensor) -> torch.Tensor:
        if decoded_image.ndim != 3 or decoded_image.shape[0] != 3:
            raise ValueError("decoded_image must be a 3xHxW tensor in [0, 1].")
        img = decoded_image.to(device).unsqueeze(0)
        img_lr = F.interpolate(img, size=lr_size, mode="bicubic", align_corners=False)[0]
        ref = ref_tensor.to(device=device, dtype=img_lr.dtype)
        return F.mse_loss(img_lr, ref)

    return reward_fn


def make_superres_gtspace_reward(
    lr_reference: PIL.Image.Image,
    gt_size: Tuple[int, int],
    device: str | torch.device = "cuda",
) -> Callable[[torch.Tensor], torch.Tensor]:
    ref_rgb = lr_reference.convert("RGB")
    ref_tensor = _pil_to_tensor_01(ref_rgb).to(device)
    lr_size = ref_rgb.size[::-1]
    projected_ref = F.interpolate(
        ref_tensor.unsqueeze(0),
        size=gt_size,
        mode="bicubic",
        align_corners=False,
    )[0]

    def reward_fn(decoded_image: torch.Tensor) -> torch.Tensor:
        if decoded_image.ndim != 3 or decoded_image.shape[0] != 3:
            raise ValueError("decoded_image must be a 3xHxW tensor in [0, 1].")
        img = decoded_image.to(device).unsqueeze(0)
        img_lr = F.interpolate(img, size=lr_size, mode="bicubic", align_corners=False)
        img_projected = F.interpolate(img_lr, size=gt_size, mode="bicubic", align_corners=False)[0]
        ref = projected_ref.to(device=device, dtype=img_projected.dtype)
        return F.mse_loss(img_projected, ref)

    return reward_fn


def make_intensity_range_reward(
    low: float = 0.2,
    high: float = 0.8,
) -> Callable[[torch.Tensor], torch.Tensor]:
    if low >= high:
        raise ValueError("low must be less than high for intensity range reward.")

    def reward_fn(decoded_image: torch.Tensor) -> torch.Tensor:
        if decoded_image.ndim != 3 or decoded_image.shape[0] != 3:
            raise ValueError("decoded_image must be a 3xHxW tensor in [0, 1].")
        gray = decoded_image.mean(dim=0)
        below = torch.relu(low - gray)
        above = torch.relu(gray - high)
        return (below + above).mean()

    return reward_fn


def make_area_ratio_reward(
    target_ratio: float = 2.0,
    threshold: float = 0.6,
    eps: float = 1e-6,
) -> Callable[[torch.Tensor], torch.Tensor]:
    if target_ratio <= 0:
        raise ValueError("target_ratio must be positive for area ratio reward.")

    def reward_fn(decoded_image: torch.Tensor) -> torch.Tensor:
        if decoded_image.ndim != 3 or decoded_image.shape[0] != 3:
            raise ValueError("decoded_image must be a 3xHxW tensor in [0, 1].")

        width = decoded_image.shape[2]
        mid = width // 2
        left = decoded_image[:, :, :mid]
        right = decoded_image[:, :, mid:]

        r_left = left[0]
        g_left = left[1]
        b_left = left[2]
        red_mask = (r_left > threshold) & (r_left > g_left) & (r_left > b_left)

        r_right = right[0]
        g_right = right[1]
        b_right = right[2]
        blue_mask = (b_right > threshold) & (b_right > r_right) & (b_right > g_right)

        red_area = red_mask.float().mean()
        blue_area = blue_mask.float().mean()
        ratio = blue_area / (red_area + eps)
        return (ratio - target_ratio) ** 2

    return reward_fn


def make_marbles_count_reward(
    target_count: int = 37,
    threshold: float = 0.5,
    kernel_size: int = 9,
) -> Callable[[torch.Tensor], torch.Tensor]:
    if kernel_size % 2 == 0 or kernel_size <= 1:
        raise ValueError("kernel_size must be an odd integer greater than 1.")

    def reward_fn(decoded_image: torch.Tensor) -> torch.Tensor:
        if decoded_image.ndim != 3 or decoded_image.shape[0] != 3:
            raise ValueError("decoded_image must be a 3xHxW tensor in [0, 1].")
        gray = decoded_image.mean(dim=0)
        inv = 1.0 - gray
        inv_4d = inv.unsqueeze(0).unsqueeze(0)
        pooled = F.max_pool2d(inv_4d, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
        peaks = (inv_4d >= (pooled - 1e-6)) & (inv_4d > threshold)
        count = peaks.float().sum()
        return (count - float(target_count)) ** 2

    return reward_fn



def load_pipe(repo_id: str, torch_dtype: torch.dtype = torch.bfloat16) -> Flux2PipelineMPC:
    text_encoder = Mistral3ForConditionalGeneration.from_pretrained(
        repo_id, subfolder="text_encoder", torch_dtype=torch_dtype, device_map="cpu"
    )
    dit = AutoModel.from_pretrained(repo_id, subfolder="transformer", torch_dtype=torch_dtype, device_map="cpu")
    pipe = Flux2PipelineMPC.from_pretrained(repo_id, text_encoder=text_encoder, transformer=dit, torch_dtype=torch_dtype)
    return pipe
