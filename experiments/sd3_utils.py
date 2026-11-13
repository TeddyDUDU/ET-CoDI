"""Shared Stable Diffusion 3 helpers used by exp14."""

from __future__ import annotations

import gc
import hashlib
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.seed import GLOBAL_SEED

LOCAL_MODEL = os.environ.get("SD3_MODEL_DIR", "").strip()
IMAGE_REWARD_ID = "ImageReward-v1.0"


def resolve_model_path(arg: str) -> str:
    if arg:
        return os.path.abspath(arg)
    env = os.environ.get("SD3_MODEL_DIR", "").strip()
    if env:
        return os.path.abspath(env)
    if LOCAL_MODEL and os.path.isdir(LOCAL_MODEL):
        return os.path.abspath(LOCAL_MODEL)
    raise SystemExit("Pass --model-path or set SD3_MODEL_DIR.")


def require_model_tree(model_path: str) -> None:
    required = [
        "model_index.json",
        os.path.join("text_encoder", "config.json"),
        os.path.join("text_encoder", "model.safetensors"),
        os.path.join("tokenizer", "vocab.json"),
        os.path.join("transformer", "config.json"),
        os.path.join("scheduler", "scheduler_config.json"),
        os.path.join("text_encoder_3", "config.json"),
        os.path.join("vae", "config.json"),
    ]
    missing = [rel for rel in required if not os.path.isfile(os.path.join(model_path, rel))]
    if missing:
        raise SystemExit(f"Incomplete SD3 directory: {model_path}\nMissing: {', '.join(missing)}")
    with open(os.path.join(model_path, "model_index.json"), encoding="utf-8") as f:
        index = json.load(f)
    if index.get("_class_name") != "StableDiffusion3Pipeline":
        raise SystemExit(f"model_index.json is not StableDiffusion3Pipeline: {model_path}")


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _text_id(text: str) -> str:
    return "t" + hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def embed_texts(model_path: str, texts: list[str]) -> dict[str, np.ndarray]:
    """SD3 CLIP-L pooled. Encode the original sentence only, not a centroid of three rewrites."""
    import torch
    from transformers import CLIPTextModelWithProjection, CLIPTokenizer

    tokenizer = CLIPTokenizer.from_pretrained(
        os.path.join(model_path, "tokenizer"), local_files_only=True,
    )
    encoder = CLIPTextModelWithProjection.from_pretrained(
        os.path.join(model_path, "text_encoder"),
        torch_dtype=torch.float32,
        local_files_only=True,
        low_cpu_mem_usage=False,
    )
    encoder.eval()
    vectors: dict[str, np.ndarray] = {}

    def encode(text: str) -> np.ndarray:
        max_length = min(int(tokenizer.model_max_length), 77)
        tokens = tokenizer(
            text,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
        )
        out = encoder(input_ids=tokens["input_ids"], attention_mask=tokens["attention_mask"])
        vec = out.text_embeds[0].float()
        vec = vec / vec.norm().clamp_min(1e-8)
        return vec.detach().cpu().numpy().astype(np.float64)

    with torch.inference_mode():
        for text in texts:
            vectors[_text_id(text)] = encode(text)
    del encoder
    gc.collect()
    return vectors


def load_pipeline(model_path: str):
    import torch
    from diffusers import StableDiffusion3Pipeline

    pipe = StableDiffusion3Pipeline.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        local_files_only=True,
    )
    pipe.set_progress_bar_config(disable=True)
    return pipe


def encode_prompts(pipe, texts: list[str], device: str) -> dict[str, dict]:
    import torch

    pipe.text_encoder.to(device)
    pipe.text_encoder_2.to(device)
    pipe.text_encoder_3.to(device)
    encoded: dict[str, dict] = {}
    with torch.inference_mode():
        for text in texts:
            prompt_embeds, neg_embeds, pooled, neg_pooled = pipe.encode_prompt(
                prompt=text,
                prompt_2=text,
                prompt_3=text,
                device=device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=True,
                negative_prompt="",
                negative_prompt_2="",
                negative_prompt_3="",
                max_sequence_length=256,
            )
            encoded[text] = {
                "prompt_embeds": prompt_embeds.detach().to("cpu"),
                "pooled": pooled.detach().to("cpu"),
                "neg_prompt_embeds": neg_embeds.detach().to("cpu"),
                "neg_pooled": neg_pooled.detach().to("cpu"),
            }
    pipe.text_encoder.to("cpu")
    pipe.text_encoder_2.to("cpu")
    pipe.text_encoder_3.to("cpu")
    torch.cuda.empty_cache()
    return encoded


def condition_to_device(cond: dict, device: str) -> dict:
    """Move the condition once per request. Host-to-device copy is outside the per-step denoising timer."""
    import torch

    return {
        key: value.to(device=device, dtype=torch.float16, non_blocking=True)
        for key, value in cond.items()
    }


def step_once(transformer, scheduler, latents, timestep, cond: dict, guidance_scale: float):
    import torch

    prompt_embeds = cond["prompt_embeds"]
    pooled = cond["pooled"]
    neg_embeds = cond["neg_prompt_embeds"]
    neg_pooled = cond["neg_pooled"]
    if guidance_scale > 1.0:
        latent_in = torch.cat([latents, latents], dim=0)
        embeds = torch.cat([neg_embeds, prompt_embeds], dim=0)
        pools = torch.cat([neg_pooled, pooled], dim=0)
    else:
        latent_in = latents
        embeds = prompt_embeds
        pools = pooled
    timestep_batch = timestep.expand(latent_in.shape[0])
    noise_pred = transformer(
        hidden_states=latent_in,
        timestep=timestep_batch,
        encoder_hidden_states=embeds,
        pooled_projections=pools,
        return_dict=False,
    )[0]
    if guidance_scale > 1.0:
        uncond, text = noise_pred.chunk(2)
        noise_pred = uncond + guidance_scale * (text - uncond)
    updated = scheduler.step(noise_pred, timestep, latents, return_dict=False)[0]
    return updated.to(dtype=latents.dtype)


def run_steps(transformer, scheduler, latents, timesteps, cond, guidance_scale: float, start: int, end: int):
    import torch

    # Prefer the public API. Fall back to a private field only on older diffusers.
    if hasattr(scheduler, "set_begin_index"):
        scheduler.set_begin_index(start)
    elif hasattr(scheduler, "_step_index"):
        scheduler._step_index = start
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for index in range(start, end):
        timestep = timesteps[index]
        latents = step_once(transformer, scheduler, latents, timestep, cond, guidance_scale)
    end_event.record()
    torch.cuda.synchronize()
    return latents, float(start_event.elapsed_time(end_event))


def initial_latents(seed: int, height: int, width: int, device: str):
    import torch

    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    return torch.randn(
        (1, 16, height // 8, width // 8),
        generator=generator,
        device=device,
        dtype=torch.float16,
    )


def reset_timesteps(scheduler, steps: int, device: str):
    scheduler.set_timesteps(steps, device=device)
    if hasattr(scheduler, "_step_index"):
        scheduler._step_index = None
    if hasattr(scheduler, "_begin_index"):
        scheduler._begin_index = None
    return scheduler.timesteps


def validate_custom_loop(
    pipe,
    cond: dict,
    guidance_scale: float,
    device: str,
    oracle_steps: int,
) -> dict:
    """Compare the custom scratch latent with the official pipeline output element by element."""
    import torch

    # Make pipeline._execution_device explicitly CUDA. 256 px keeps the oracle activation inside 24 GB.
    text_modules = (pipe.text_encoder, pipe.text_encoder_2, pipe.text_encoder_3)
    for module in text_modules:
        module.to(device)
    height = width = 256
    seed = GLOBAL_SEED - 1
    latents = initial_latents(seed, height, width, device)
    kwargs = {
        "prompt_embeds": cond["prompt_embeds"],
        "negative_prompt_embeds": cond["neg_prompt_embeds"],
        "pooled_prompt_embeds": cond["pooled"],
        "negative_pooled_prompt_embeds": cond["neg_pooled"],
        "num_inference_steps": oracle_steps,
        "guidance_scale": guidance_scale,
        "height": height,
        "width": width,
        "latents": latents.clone(),
        "output_type": "latent",
    }
    try:
        with torch.inference_mode():
            official = pipe(**kwargs).images
            timesteps = reset_timesteps(pipe.scheduler, oracle_steps, device)
            custom, _ = run_steps(
                pipe.transformer, pipe.scheduler, latents.clone(), timesteps,
                cond, guidance_scale, 0, oracle_steps,
            )
    finally:
        for module in text_modules:
            module.to("cpu")
        torch.cuda.empty_cache()
    delta = (official.float() - custom.float()).abs()
    result = {
        "steps": oracle_steps,
        "height": height,
        "width": width,
        "max_abs_error": float(delta.max().item()),
        "mean_abs_error": float(delta.mean().item()),
    }
    if result["max_abs_error"] > 5e-3:
        raise RuntimeError(
            "Custom SD3 loop does not match the official pipeline: "
            f"max_abs_error={result['max_abs_error']:.6g}"
        )
    print(f"Oracle passed: {result}", flush=True)
    return result


def decode_image(vae, latents, transformer=None, device: str = "cuda"):
    """Before a 1024 decode, free the MMDiT and use a tiled VAE so a 24 GB card does not overflow in fp32 decode."""
    import torch
    from PIL import Image

    if transformer is not None:
        transformer.to("cpu")
    if hasattr(vae, "enable_tiling"):
        vae.enable_tiling()
    if hasattr(vae, "enable_slicing"):
        vae.enable_slicing()
    torch.cuda.empty_cache()
    dtype = next(vae.parameters()).dtype
    scaled = latents.to(device=vae.device, dtype=dtype)
    scaled = (scaled / vae.config.scaling_factor) + vae.config.shift_factor
    decoded = vae.decode(scaled, return_dict=False)[0]
    array = ((decoded[0].float().clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8)
    array = array.permute(1, 2, 0).cpu().numpy()
    image = Image.fromarray(array)
    del scaled, decoded
    if transformer is not None:
        transformer.to(device=device, dtype=torch.float16)
        torch.cuda.empty_cache()
    return image


def load_clip_scorer(clip_model: str, device: str):
    import torch
    from transformers import CLIPModel, CLIPProcessor

    local = os.path.isdir(clip_model)
    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    model = CLIPModel.from_pretrained(clip_model, torch_dtype=dtype, local_files_only=local)
    processor = CLIPProcessor.from_pretrained(clip_model, local_files_only=local)
    model.to(device).eval()
    return model, processor


def score_clip(model, processor, image, text: str, device: str) -> float:
    import torch

    inputs = processor(text=[text], images=image, return_tensors="pt", padding=True, truncation=True)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.inference_mode():
        out = model(**inputs)
        image_embeds = out.image_embeds.float()
        text_embeds = out.text_embeds.float()
        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        return float((image_embeds * text_embeds).sum().item())


def load_image_reward(device: str):
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    import ImageReward as reward_module

    print(
        f"Loading ImageReward {IMAGE_REWARD_ID} via {os.environ['HF_ENDPOINT']}",
        flush=True,
    )
    return reward_module.load(IMAGE_REWARD_ID, device=device)


def score_image_reward(model, prompt: str, image) -> float:
    value = model.score(prompt, image)
    if isinstance(value, (list, tuple)):
        value = value[0]
    return float(value)
