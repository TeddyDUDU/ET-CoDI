"""exp14: cache sentence versus the current public prompt.

Every image uses the same schedule. Seed 2026, 28 steps, switch at step 8,
common guidance 2, then the personal prompt at guidance 7. The reference runs
the first 8 steps on that object's own public prompt. Five cache sentences are
chosen by measured CLIP-L cosine to the same public prompt: A >= 0.97,
B 0.93-0.97, C 0.90-0.93, D 0.50-0.90, E < 0.50.

The curve reports ImageReward itself, not a difference. The dashed line is the
mean ImageReward of the public-prompt images. Denoising time for the reference
is all 28 steps; for a cache sentence it is the last 20 steps only.

  python experiments/exp14_controlled.py --model-path "$SD3_MODEL_DIR" --stage all
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from core.paths import EXP14_DIR, ensure_project_cwd, norm_path
from core.seed import GLOBAL_SEED
from experiments.sd3_utils import (
    _text_id,
    condition_to_device,
    decode_image,
    embed_texts,
    encode_prompts,
    file_sha256,
    initial_latents,
    load_clip_scorer,
    load_image_reward,
    load_pipeline,
    require_model_tree,
    reset_timesteps,
    resolve_model_path,
    run_steps,
    score_clip,
    score_image_reward,
    validate_custom_loop,
)

W1 = 2.0
W2 = 7.0
MAIN_SEED = GLOBAL_SEED
K_MAIN = 8
STEPS = 28

# id, personal noun, sibling noun, public noun, place, light
ITEMS: tuple[tuple[str, str, str, str, str, str], ...] = (
    ("apple", "red apple on a marble kitchen table", "red apple on a wooden kitchen table", "red apple on a kitchen table", "", "soft natural light"),
    ("bus", "red double-decker bus", "red city bus", "red bus", "on a wet street", "evening"),
    ("bicycle", "red mountain bicycle", "red city bicycle", "red bicycle", "beside a brick wall", "daylight"),
    ("lighthouse", "white lighthouse on a sandy coast", "white lighthouse on a rocky coast", "white lighthouse on a coast", "", "overcast sky"),
    ("cup", "porcelain coffee cup", "porcelain teacup", "porcelain cup", "on a linen tablecloth", "morning light"),
    ("bread", "baguette", "loaf of bread", "bread", "on a bakery counter", "warm light"),
    ("lake", "mountain lake surrounded by autumn trees", "mountain lake surrounded by pine trees", "mountain lake surrounded by trees", "", "clear day"),
    ("cat", "orange cat lying on a gray sofa", "orange cat sitting on a gray sofa", "orange cat on a gray sofa", "", "indoor daylight"),
    ("chair", "red wooden chair", "blue wooden chair", "wooden chair", "in a sunlit room", "soft daylight"),
    ("lamp", "brass desk lamp", "black desk lamp", "desk lamp", "on a wooden desk", "warm light"),
    ("vase", "blue ceramic vase", "white ceramic vase", "ceramic vase", "on a side table", "morning light"),
    ("bowl", "glass bowl of apples", "glass bowl of oranges", "glass bowl of fruit", "on a kitchen counter", "daylight"),
    ("hat", "straw hat", "felt hat", "hat", "on a wooden bench", "daylight"),
    ("shoe", "red running shoe", "white running shoe", "running shoe", "on a concrete step", "daylight"),
    ("clock", "round wall clock", "square wall clock", "wall clock", "on a white wall", "daylight"),
    ("book", "open hardcover book", "closed hardcover book", "hardcover book", "on a reading desk", "warm light"),
    ("mug", "speckled ceramic mug", "plain white mug", "ceramic mug", "on a cafe table", "morning light"),
    ("plant", "flowering plant", "leafy green plant", "potted plant", "on a windowsill", "daylight"),
    ("bottle", "green glass bottle", "clear glass bottle", "glass bottle", "on a wooden shelf", "soft light"),
    ("camera", "black film camera", "silver film camera", "film camera", "on a table", "studio light"),
    ("guitar", "acoustic guitar", "electric guitar", "guitar", "on a stand in a room", "warm light"),
    ("violin", "brown violin", "black violin", "violin", "on a chair", "soft daylight"),
    ("piano", "black grand piano", "white grand piano", "grand piano", "in a living room", "daylight"),
    ("drum", "red snare drum", "blue snare drum", "snare drum", "on a wooden floor", "studio light"),
    ("boat", "white sailboat", "blue fishing boat", "boat", "on a calm lake", "daylight"),
    ("train", "red passenger train", "blue passenger train", "passenger train", "at a station platform", "daylight"),
    ("car", "red sedan", "blue sedan", "sedan", "on a city street", "daylight"),
    ("truck", "yellow truck", "white truck", "truck", "on a highway", "daylight"),
    ("plane", "white passenger plane", "blue passenger plane", "passenger plane", "on an airport runway", "daylight"),
    ("bridge", "stone arch bridge", "wooden footbridge", "bridge", "over a river", "daylight"),
    ("house", "brick house", "white wooden house", "house", "behind a garden", "daylight"),
    ("barn", "red barn", "white barn", "barn", "in a green field", "daylight"),
    ("tower", "stone clock tower", "brick clock tower", "clock tower", "in a town square", "daylight"),
    ("gate", "iron garden gate", "wooden garden gate", "garden gate", "in a hedge", "daylight"),
    ("fountain", "stone fountain", "bronze fountain", "fountain", "in a plaza", "daylight"),
    ("bench", "wooden park bench", "metal park bench", "park bench", "under a tree", "daylight"),
    ("umbrella", "red patio umbrella", "blue patio umbrella", "patio umbrella", "over a table", "daylight"),
    ("tent", "green camping tent", "orange camping tent", "camping tent", "in a meadow", "daylight"),
    ("balloon", "red hot-air balloon", "yellow hot-air balloon", "hot-air balloon", "above a field", "clear day"),
    ("kite", "red kite", "blue kite", "kite", "in a windy sky", "daylight"),
    ("flower", "red rose in a vase", "white rose in a vase", "rose in a vase", "on a table", "soft light"),
    ("tree", "blooming cherry tree", "green oak tree", "tree", "in a park", "daylight"),
    ("mountain", "snowy mountain peak", "rocky mountain peak", "mountain peak", "under a blue sky", "daylight"),
    ("waterfall", "wide waterfall", "narrow waterfall", "waterfall", "in a forest", "daylight"),
    ("beach", "sandy tropical beach", "pebble beach", "beach", "beside the ocean", "daylight"),
    ("island", "green island", "rocky island", "island", "in a blue sea", "daylight"),
    ("river", "wide calm river", "narrow forest river", "river", "through a valley", "daylight"),
    ("field", "wheat field", "lavender field", "field of crops", "under a blue sky", "daylight"),
    ("street", "cobblestone street", "asphalt street", "street", "in an old town", "evening"),
    ("market", "outdoor fruit market", "outdoor flower market", "outdoor market", "on a town square", "daylight"),
    ("cafe", "corner cafe with awning", "corner cafe with umbrellas", "corner cafe", "on a sidewalk", "daylight"),
    ("library", "wood-paneled library", "modern glass library", "library interior", "with tall shelves", "warm light"),
    ("kitchen", "white modern kitchen", "wooden rustic kitchen", "kitchen", "with a central table", "daylight"),
    ("bedroom", "blue bedroom", "green bedroom", "bedroom", "with a made bed", "morning light"),
    ("office", "glass-wall office", "wood-desk office", "office", "with a desk and chair", "daylight"),
    ("dog", "brown dog lying on a rug", "brown dog sitting on a rug", "brown dog on a rug", "", "indoor daylight"),
    ("bird", "red parrot on a perch", "blue parrot on a perch", "parrot on a perch", "", "daylight"),
    ("horse", "white horse", "brown horse", "horse", "in a grassy field", "daylight"),
    ("fish", "goldfish in a bowl", "blue fish in a bowl", "fish in a bowl", "on a table", "soft light"),
    ("butterfly", "orange butterfly", "blue butterfly", "butterfly", "on a flower", "daylight"),
)

MECHANISM_IDS = ("apple", "bus", "bicycle", "lighthouse", "cup", "bread", "lake", "cat")
SHOWCASE_IDS = ("apple", "bus", "bicycle", "lighthouse")
UNRELATED = (
    "a medieval stone castle on a cliff above the ocean, storm clouds, photorealistic",
    "a close-up of a mechanical wristwatch gear, macro photography",
    "a bowl of ramen with chopsticks, restaurant lighting, photorealistic",
    "an astronaut riding a horse on mars, cinematic, photorealistic",
    "a snowy pine forest at night, moonlight, photorealistic",
    "a stack of gold coins on black velvet, studio light, photorealistic",
)

BINS = (
    ("A", 0.97, 1.01, 0.985),
    ("B", 0.93, 0.97, 0.95),
    ("C", 0.90, 0.93, 0.915),
    ("D", 0.50, 0.90, 0.70),
    ("E", -1.0, 0.50, 0.0),
)

CSV_FIELDS = [
    "item_id", "arm", "level", "seed", "condition", "target_text", "cache_text",
    "sim_pub", "guidance_common", "switch_step", "gap", "denoise_ms",
    "clip_score", "image_reward", "image_path",
]


def photo(noun: str, place: str, light: str) -> str:
    text = f"a photorealistic {noun}"
    if place:
        text += f" {place}"
    if light:
        text += f", {light}"
    return text


def item_record(row: tuple[str, str, str, str, str, str]) -> dict:
    item_id, personal_noun, sibling_noun, public_noun, place, light = row
    return {
        "item_id": item_id,
        "personal": photo(personal_noun, place, light),
        "sibling": photo(sibling_noun, place, light),
        "public": photo(public_noun, place, light),
        "public_noun": public_noun,
        "mechanism": item_id in MECHANISM_IDS,
        "showcase": item_id in SHOWCASE_IDS,
    }


def cache_candidates(item: dict) -> list[str]:
    public = item["public"]
    noun = item["public_noun"]
    head, _, tail = public.partition(", ")
    clipped = head if tail else public
    head_noun = noun.split()[-1]
    return [
        public,
        public.replace("a photorealistic ", "a photorealistic picture of ", 1),
        public.replace("a photorealistic ", "a realistic ", 1),
        public.replace("a photorealistic ", "a simple ", 1),
        public.replace("a photorealistic ", "an ordinary ", 1),
        public.replace("a photorealistic ", "a plain ", 1),
        public.replace("a photorealistic ", "a ", 1),
        public + ", plain background",
        public + ", simple background",
        clipped,
        clipped + ", simple background",
        f"a photorealistic {noun}",
        f"a photorealistic {noun}, plain background",
        f"a photorealistic {head_noun}",
        f"a photorealistic {head_noun}, plain background",
        f"a realistic {noun}",
        f"a photograph of {noun}",
        f"a close view of {noun}",
        f"a {noun}",
        f"a small {noun} in the distance",
        f"a blurry photo of {noun}",
        f"a drawing of {noun}",
        *UNRELATED,
    ]


def assign_levels(item: dict, vectors: dict[str, np.ndarray]) -> list[dict]:
    public_vec = vectors[_text_id(item["public"])]
    chosen = []
    used: set[str] = set()
    scored = []
    for text in dict.fromkeys(cache_candidates(item)):
        scored.append((text, float(np.dot(public_vec, vectors[_text_id(text)]))))
    for name, lo, hi, center in BINS:
        pool = [(text, sim) for text, sim in scored if lo <= sim < hi and text not in used]
        if not pool:
            return []
        text, sim = min(pool, key=lambda pair: (abs(pair[1] - center), pair[0]))
        used.add(text)
        chosen.append({"level": name, "cache_text": text, "sim_pub": round(sim, 6)})
    return chosen


def build_manifest(model_path: str, out_path: str) -> dict:
    items = [item_record(row) for row in ITEMS]
    texts = []
    for item in items:
        texts.extend([item["personal"], item["sibling"], item["public"], *cache_candidates(item)])
    texts = list(dict.fromkeys(texts))
    print(f"Embedding {len(texts)} prompts", flush=True)
    vectors = embed_texts(model_path, texts)
    selected = []
    missing = []
    for item in items:
        levels = assign_levels(item, vectors)
        if len(levels) != len(BINS):
            missing.append(item["item_id"])
            continue
        row = dict(item)
        row["levels"] = levels
        selected.append(row)
    kept = {item["item_id"] for item in selected}
    if "apple" not in kept or "bus" not in kept or len(selected) < 40:
        raise SystemExit(
            f"Too few complete items ({len(selected)}). Missing bins: {', '.join(missing)}."
        )
    if missing:
        print(f"Skipped items with an empty bin: {', '.join(missing)}", flush=True)
    weight = os.path.join(model_path, "text_encoder", "model.safetensors")
    payload = {
        "embedding": "sd3-clip-l-pooled",
        "text_encoder_sha256": file_sha256(weight) if os.path.isfile(weight) else "",
        "guidance_common": W1,
        "guidance_personal": W2,
        "main_seed": MAIN_SEED,
        "k": K_MAIN,
        "steps": STEPS,
        "n_items": len(selected),
        "items": selected,
    }
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    print(f"Manifest written: {len(selected)} items, 5 cache levels each", flush=True)
    for item in selected:
        if item["item_id"] not in SHOWCASE_IDS:
            continue
        brief = " ".join(f"{level['level']}={level['sim_pub']:.3f}" for level in item["levels"])
        print(f"  {item['item_id']}: {brief}", flush=True)
    return payload


def _jobs(payload: dict) -> list[dict]:
    jobs = []
    seen: set[tuple] = set()

    def add(job: dict) -> None:
        key = (
            job["personal"], job["cache_text"], job["seed"], job["w1"],
            job["switch_step"], job["arm"], job["level"],
        )
        if key in seen:
            return
        seen.add(key)
        jobs.append(job)

    for item in payload["items"]:
        personal = item["personal"]
        public = item["public"]
        add({
            "item_id": item["item_id"], "arm": "main", "level": "public",
            "seed": MAIN_SEED, "personal": personal, "cache_text": public,
            "sim_pub": 1.0, "w1": W1, "switch_step": K_MAIN,
            "condition": "public", "showcase": item["showcase"],
        })
        for level in item["levels"]:
            add({
                "item_id": item["item_id"], "arm": "main", "level": level["level"],
                "seed": MAIN_SEED, "personal": personal, "cache_text": level["cache_text"],
                "sim_pub": level["sim_pub"], "w1": W1, "switch_step": K_MAIN,
                "condition": "reuse", "showcase": item["showcase"],
            })
    return jobs


def _row(job: dict, denoise_ms: float, image_path: str) -> dict:
    switch = int(job["switch_step"])
    return {
        "item_id": job["item_id"],
        "arm": job["arm"],
        "level": job["level"],
        "seed": job["seed"],
        "condition": job["condition"],
        "target_text": job["personal"],
        "cache_text": job.get("cache_text", ""),
        "sim_pub": "" if job.get("sim_pub") is None else f"{float(job['sim_pub']):.6f}",
        "guidance_common": job["w1"],
        "switch_step": switch,
        "gap": 0,
        "denoise_ms": f"{denoise_ms:.3f}",
        "clip_score": "",
        "image_reward": "",
        "image_path": image_path,
    }


def append_row(path: str, row: dict) -> None:
    new_file = not os.path.isfile(path)
    with open(path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if new_file:
            writer.writeheader()
        writer.writerow(row)
        f.flush()


def load_rows(path: str) -> list[dict]:
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_rows(path: str, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _run_segment(pipe, latents, cond, guidance: float, start: int, end: int, steps: int, device: str):
    timesteps = reset_timesteps(pipe.scheduler, steps, device)
    return run_steps(
        pipe.transformer, pipe.scheduler, latents, timesteps, cond, guidance, start, end,
    )


def run_generation(model_path: str, payload: dict, out_dir: str, args: argparse.Namespace) -> None:
    import torch

    if not torch.cuda.is_available():
        raise SystemExit("Full generation requires CUDA. Use --dry-run without a GPU.")
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    jobs = _jobs(payload)
    if args.limit is not None:
        jobs = jobs[: args.limit]
    device = "cuda"
    image_dir = os.path.join(out_dir, "images")
    os.makedirs(image_dir, exist_ok=True)
    metrics_path = os.path.join(out_dir, "metrics.csv")
    if args.stage == "score":
        score_saved_images(metrics_path, args.clip_model)
        plot_and_sheets(out_dir, load_rows(metrics_path))
        return
    if os.path.isfile(metrics_path) and not args.append:
        os.remove(metrics_path)

    print(f"Loading SD3 from {model_path}", flush=True)
    pipe = load_pipeline(model_path)
    texts = sorted({job["personal"] for job in jobs} | {job["cache_text"] for job in jobs if job["cache_text"]})
    print(f"Encoding {len(texts)} prompts", flush=True)
    conditions = encode_prompts(pipe, texts, device)
    pipe.transformer.to(device=device, dtype=torch.float16)
    if getattr(pipe.vae.config, "force_upcast", False):
        pipe.vae.to(device=device, dtype=torch.float32)
    else:
        pipe.vae.to(device=device, dtype=torch.float16)
    torch.cuda.empty_cache()
    warmup = condition_to_device(next(iter(conditions.values())), device)
    oracle = validate_custom_loop(pipe, warmup, W2, device, 4)
    del warmup
    for module in (pipe.text_encoder, pipe.text_encoder_2, pipe.text_encoder_3):
        module.to("cpu")
    torch.cuda.empty_cache()
    print(f"Running {len(jobs)} images", flush=True)

    def save_image(latents, filename: str) -> str:
        image = decode_image(pipe.vae, latents, pipe.transformer, device)
        path = os.path.join(image_dir, filename)
        image.save(path)
        return path

    for job in jobs:
        personal = condition_to_device(conditions[job["personal"]], device)
        seed = int(job["seed"])
        latents0 = initial_latents(seed, args.height, args.width, device)
        cache = condition_to_device(conditions[job["cache_text"]], device)
        with torch.inference_mode():
            latent, common_ms = _run_segment(
                pipe, latents0.clone(), cache, float(job["w1"]), 0, K_MAIN, STEPS, device,
            )
            latent, personal_ms = _run_segment(
                pipe, latent, personal, W2, K_MAIN, STEPS, STEPS, device,
            )
        # Reference counts all 28 steps. A cache sentence counts the last 20.
        denoise_ms = common_ms + personal_ms if job["condition"] == "public" else personal_ms
        path = save_image(
            latent,
            f"{job['item_id']}_s{seed}_{job['arm']}_{job['level']}.png",
        )
        append_row(metrics_path, _row(job, denoise_ms, path))
        print(f"  {job['item_id']} {job['arm']} {job['level']} seed={seed}", flush=True)
    meta = {
        "model_path": model_path,
        "steps": STEPS,
        "k": K_MAIN,
        "guidance_common": W1,
        "guidance_personal": W2,
        "n_jobs": len(jobs),
        "oracle": oracle,
        "clip_model": args.clip_model,
        "hf_endpoint": os.environ.get("HF_ENDPOINT", ""),
    }
    with open(os.path.join(out_dir, "run_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
        f.write("\n")
    del conditions, pipe
    gc.collect()
    torch.cuda.empty_cache()
    if args.stage == "all":
        score_saved_images(metrics_path, args.clip_model)
        plot_and_sheets(out_dir, load_rows(metrics_path))


def score_saved_images(metrics_path: str, clip_model_name: str) -> None:
    import torch
    from PIL import Image

    if not clip_model_name or not os.path.isdir(clip_model_name):
        raise SystemExit("Set --clip-model or CLIP_MODEL_DIR to a local CLIP model path.")
    rows = load_rows(metrics_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Scoring CLIP on {device} from {clip_model_name}", flush=True)
    clip_model, processor = load_clip_scorer(clip_model_name, device)
    for row in rows:
        with Image.open(row["image_path"]) as image:
            row["clip_score"] = f"{score_clip(clip_model, processor, image.convert('RGB'), row['target_text'], device):.6f}"
    del clip_model, processor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("Scoring ImageReward on cuda", flush=True)
    reward_model = load_image_reward(device)
    for row in rows:
        row["image_reward"] = f"{score_image_reward(reward_model, row['target_text'], row['image_path']):.6f}"
    del reward_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    write_rows(metrics_path, rows)


def _stats(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    std = float(array.std(ddof=1)) if len(array) > 1 else 0.0
    ci = 1.96 * std / np.sqrt(len(array)) if len(array) > 1 else 0.0
    return float(array.mean()), ci


def _main_rows(rows: list[dict], level: str) -> list[dict]:
    return [
        row for row in rows
        if row["arm"] == "main" and int(row["seed"]) == MAIN_SEED and row["level"] == level
        and row.get("image_reward")
    ]


def plot_and_sheets(out_dir: str, rows: list[dict]) -> None:
    import matplotlib.pyplot as plt
    from PIL import Image, ImageDraw

    from experiments.ieee_plot import apply_ieee_style, save_fig

    public_vals = [float(row["image_reward"]) for row in _main_rows(rows, "public")]
    by_level: dict[str, list[float]] = {name: [] for name, *_ in BINS}
    for name, *_ in BINS:
        by_level[name] = [float(row["image_reward"]) for row in _main_rows(rows, name)]
    apply_ieee_style()
    # Included at 0.38\textwidth. The width is twice that printed width, the same
    # scale as the single-column figures, so the English prints at the same size.
    # The aspect matches the example strip at 0.60\textwidth.
    fig, ax = plt.subplots(figsize=(5.29, 2.98))
    xs = list(range(len(BINS)))
    means, cis = [], []
    for name, *_ in BINS:
        mean, ci = _stats(by_level[name]) if by_level[name] else (float("nan"), 0.0)
        means.append(mean)
        cis.append(ci)
    public_mean, public_ci = _stats(public_vals) if public_vals else (float("nan"), 0.0)
    if public_vals:
        ax.axhline(public_mean, color="0.45", linestyle="--", linewidth=1.0, label="public", zorder=1)
    ax.plot(xs, means, color="0.35", linewidth=1.2, zorder=2)
    styles = (
        ("high", "#009E73"),
        ("mid", "#0072B2"),
        ("near", "#E69F00"),
        ("rejected", "#D55E00"),
        ("unrelated", "#CC79A7"),
    )
    for x, mean, ci, (label, color) in zip(xs, means, cis, styles):
        ax.errorbar(
            [x], [mean], yerr=[ci], marker="o", capsize=3, color=color,
            linestyle="none", label=label, zorder=3,
        )
    ax.set_xticks(xs)
    ax.set_xticklabels([
        r"$\geq 0.97$",
        r"$0.93$–$0.97$",
        r"$0.90$–$0.93$",
        r"$0.50$–$0.90$",
        r"$<0.50$",
    ])
    ax.set_xlabel("Public-prompt similarity")
    ax.set_ylabel("ImageReward")
    span = [mean - ci for mean, ci in zip(means, cis) if mean == mean]
    span += [mean + ci for mean, ci in zip(means, cis) if mean == mean]
    if public_vals:
        span.extend((public_mean - public_ci, public_mean + public_ci))
    if span:
        pad = 0.2
        ax.set_ylim(min(span) - pad, max(span) + pad)
    ax.legend(
        loc="lower left",
        ncol=2,
        frameon=False,
        handlelength=1.4,
        columnspacing=0.8,
        labelspacing=0.25,
        borderaxespad=0.2,
    )
    fig.tight_layout(pad=0.25)
    # figure not written; not used in the paper (image_reward_by_bin.png)
    save_fig(fig, os.path.join(out_dir, "exp14_reward_delta.png"))

    sheet_dir = os.path.join(out_dir, "sheets")
    os.makedirs(sheet_dir, exist_ok=True)
    order = ["public", "A", "B", "C", "D", "E"]
    for item_id in SHOWCASE_IDS:
        paths = []
        labels = []
        for level in order:
            match = [
                row for row in rows
                if row["item_id"] == item_id and int(row["seed"]) == MAIN_SEED
                and row["arm"] == "main" and row["level"] == level
            ]
            if not match or not os.path.isfile(match[0]["image_path"]):
                paths = []
                break
            paths.append(match[0]["image_path"])
            sim = match[0].get("sim_pub") or ""
            labels.append("public" if level == "public" else (f"{level} {float(sim):.2f}" if sim else level))
        if len(paths) != len(order):
            continue
        side = 320
        header = 28
        canvas = Image.new("RGB", (side * len(paths), side + header), "white")
        draw = ImageDraw.Draw(canvas)
        for index, (path, label) in enumerate(zip(paths, labels)):
            with Image.open(path) as image:
                canvas.paste(image.convert("RGB").resize((side, side)), (index * side, header))
            draw.text((index * side + 8, 6), label, fill="black")
        canvas.save(os.path.join(sheet_dir, f"{item_id}_ladder.png"))
        canvas.close()
    print(f"Display sheets written to {sheet_dir}", flush=True)
    _write_summary(out_dir, rows, public_vals, by_level)
    write_paper_examples(out_dir, rows)


def _write_summary(out_dir: str, rows: list[dict], public_vals: list[float], by_level: dict[str, list[float]]) -> None:
    lines = ["exp14 ImageReward of the generated image.", ""]

    def emit(name: str, values: list[float], level: str) -> None:
        if not values:
            lines.append(f"{name}: n=0")
            return
        mean, ci = _stats(values)
        timed = [
            float(row["denoise_ms"]) for row in _main_rows(rows, level) if row.get("denoise_ms")
        ]
        ms = _stats(timed)[0] if timed else float("nan")
        lines.append(f"{name}: n={len(values)} ImageReward {mean:.3f} +/- {ci:.3f} denoise_ms {ms:.1f}")

    emit("public", public_vals, "public")
    for name, *_ in BINS:
        emit(name, by_level[name], name)
    text = "\n".join(lines) + "\n"
    with open(os.path.join(out_dir, "summary.txt"), "w", encoding="utf-8") as handle:
        handle.write(text)
    print(text, flush=True)


def _load_paper_font(size: int, bold: bool = False):
    from PIL import ImageFont

    candidates = (
        (
            r"C:\Windows\Fonts\timesbd.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSerif-Bold.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf",
        )
        if bold
        else (
            r"C:\Windows\Fonts\times.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
        )
    )
    for path in candidates:
        if os.path.isfile(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def write_paper_examples(out_dir: str, rows: list[dict]) -> None:
    """Two rows: public prompt, then the five cache sentences."""
    from PIL import Image, ImageDraw

    def caption(item_id: str, level: str) -> str:
        if level == "public":
            return "public"
        match = [
            row for row in rows
            if row["item_id"] == item_id and row["arm"] == "main" and row["level"] == level
            and int(row["seed"]) == MAIN_SEED and row.get("sim_pub")
        ]
        if not match:
            return level
        return f"{float(match[0]['sim_pub']):.2f}"

    picks = tuple(
        (name, (0, 1, 2, 3, 4, 5), tuple(caption(name, level) for level in ("public", "A", "B", "C", "D", "E")))
        for name in ("apple", "bus")
    )
    side = 320
    header = 28
    crops = []
    labels = []
    for name, indexes, captions in picks:
        path = os.path.join(out_dir, "sheets", f"{name}_ladder.png")
        if not os.path.isfile(path):
            print(f"Missing showcase sheet: {path}", flush=True)
            return
        with Image.open(path) as sheet:
            row = []
            for index in indexes:
                row.append(sheet.crop((index * side, header, (index + 1) * side, header + side)))
            crops.append(row)
            labels.append(captions)
    gap = 8
    # 34 px prints at about 5 pt under 0.60\textwidth, matching the axis labels
    # of the single-column figures.
    font = _load_paper_font(34, bold=False)
    font_b = _load_paper_font(34, bold=True)
    row_names = ("Apple", "Bus")
    label_w = max(font_b.getbbox(name)[2] for name in row_names) + 16
    caption_h = font.getbbox("public")[3] + 16
    n_cols = 6
    width = label_w + n_cols * side + (n_cols - 1) * gap
    height = 2 * (side + caption_h) + gap
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    for r, (row, captions, row_name) in enumerate(zip(crops, labels, row_names)):
        y0 = r * (side + caption_h + gap)
        name_box = font_b.getbbox(row_name)
        name_h = name_box[3] - name_box[1]
        draw.text((8, y0 + (side - name_h) / 2 - name_box[1]), row_name, fill="black", font=font_b)
        for c, (image, caption) in enumerate(zip(row, captions)):
            x0 = label_w + c * (side + gap)
            canvas.paste(image, (x0, y0))
            cap_box = font.getbbox(caption)
            cap_w = cap_box[2] - cap_box[0]
            draw.text(
                (x0 + (side - cap_w) / 2, y0 + side + 8 - cap_box[1]),
                caption,
                fill="black",
                font=font,
            )
    out_path = os.path.join(out_dir, "exp14_examples.png")
    canvas.save(out_path, dpi=(300, 300))
    canvas.close()
    print(f"Paper example figure written to {out_path}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="exp14 controlled cache-prompt ladder")
    parser.add_argument("--model-path", type=str, default="")
    parser.add_argument("--manifest", type=str, default="")
    parser.add_argument("--out-dir", type=str, default=EXP14_DIR)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--stage", choices=("all", "generate", "score"), default="all")
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--append", action="store_true")
    parser.add_argument(
        "--clip-model",
        type=str,
        default=os.environ.get("CLIP_MODEL_DIR", "").strip()
        or os.environ.get("CLIP_MODEL", "").strip(),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_project_cwd()
    out_dir = norm_path(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    manifest_path = norm_path(args.manifest) if args.manifest else os.path.join(out_dir, "manifest.json")
    if args.plot_only:
        rows = load_rows(os.path.join(out_dir, "metrics.csv"))
        plot_and_sheets(out_dir, rows)
        return
    if args.dry_run:
        model_path = resolve_model_path(args.model_path)
        require_model_tree(model_path)
        build_manifest(model_path, manifest_path)
        return
    model_path = resolve_model_path(args.model_path)
    require_model_tree(model_path)
    if not os.path.isfile(manifest_path):
        payload = build_manifest(model_path, manifest_path)
    else:
        with open(manifest_path, encoding="utf-8") as f:
            payload = json.load(f)
        print(f"Using manifest {manifest_path}", flush=True)
    run_generation(model_path, payload, out_dir, args)


if __name__ == "__main__":
    main()
