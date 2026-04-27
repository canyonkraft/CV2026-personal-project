"""
Generative Dog Images Pipeline
================================
Fine-tunes a Latent Diffusion Model (Stable Diffusion) with LoRA on dog breed datasets,
with structural guidance via ControlNet and evaluation via FID + CLIP scores.

Usage:
    python pipeline.py --stage preprocess
    python pipeline.py --stage caption
    python pipeline.py --stage train
    python pipeline.py --stage generate --breed "Golden Retriever" --num_images 4
    python pipeline.py --stage evaluate
"""

import argparse
import os
import csv
import json
import random
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

# ── Core ML / vision ──────────────────────────────────────────────────────────
import torch
import numpy as np
from PIL import Image
from torchvision import transforms

# ── Plotting ──────────────────────────────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")  # headless-safe backend
import matplotlib.pyplot as plt

# ── HuggingFace ecosystem ─────────────────────────────────────────────────────
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionControlNetPipeline,
    ControlNetModel,
    DDPMScheduler,
    AutoencoderKL,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler
from transformers import (
    CLIPTextModel,
    CLIPTokenizer,
    CLIPProcessor,
    CLIPModel,
    BlipProcessor,
    BlipForConditionalGeneration,
)
from peft import LoraConfig, get_peft_model  # pip install peft
from accelerate import Accelerator           # pip install accelerate

# ── Metrics ───────────────────────────────────────────────────────────────────
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.multimodal import CLIPScore

# ── Datasets ──────────────────────────────────────────────────────────────────
from torch.utils.data import Dataset, DataLoader
import torchvision.datasets as tvdatasets

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Config:
    # ── Paths ─────────────────────────────────────────────────────────────────
    data_root: str = "data"
    stanford_dir: str = "data/stanford_dogs"
    oxford_dir: str = "data/oxford_pets"
    # StanfordExtra references Stanford Dogs images by relative path,
    # so we point pose_dir at the Stanford images and just drop the JSON
    # alongside it as `stanford_extra.json`.
    pose_dir: str = "data/stanford_dogs"
    pose_annotations_file: str = "data/stanford_dogs/stanford_extra.json"
    processed_dir: str = "data/processed"
    captions_file: str = "data/captions.json"
    output_dir: str = "outputs"
    charts_dir: str = "outputs/charts"
    training_log: str = "outputs/training_log.csv"
    model_dir: str = "models/lora_checkpoint"

    # ── Base model ────────────────────────────────────────────────────────────
    base_model_id: str = "runwayml/stable-diffusion-v1-5"
    controlnet_model_id: str = "lllyasviel/sd-controlnet-openpose"

    # ── Image settings ────────────────────────────────────────────────────────
    image_size: int = 512
    center_crop: bool = True

    # ── LoRA hyperparameters ──────────────────────────────────────────────────
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list = field(
        default_factory=lambda: ["to_q", "to_v", "to_k", "to_out.0"]
    )

    # ── Training hyperparameters ──────────────────────────────────────────────
    learning_rate: float = 1e-4
    train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    max_train_steps: int = 5000
    save_steps: int = 500
    mixed_precision: str = "fp16"   # "no" | "fp16" | "bf16"
    seed: int = 42

    # ── Generation ────────────────────────────────────────────────────────────
    guidance_scale: float = 7.5
    num_inference_steps: int = 50
    num_images_per_prompt: int = 4

    # ── Captioning ────────────────────────────────────────────────────────────
    # BLIP-Base is ~2x faster than BLIP-Large with marginal quality drop for
    # this domain. Batch inference is the biggest win — 16-32 fits on most GPUs.
    caption_model_id: str = "Salesforce/blip-image-captioning-base"
    caption_batch_size: int = 32
    caption_num_workers: int = 4   # parallel image loading via DataLoader
    caption_save_every: int = 500  # checkpoint captions every N images

    # ── Evaluation ────────────────────────────────────────────────────────────
    fid_num_samples: int = 1000


CFG = Config()


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 1 — DATA PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────

class DogImagePreprocessor:
    """Resize, center-crop, and normalize images from all three datasets."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.transform = transforms.Compose([
            transforms.Resize(cfg.image_size, interpolation=transforms.InterpolationMode.LANCZOS),
            transforms.CenterCrop(cfg.image_size) if cfg.center_crop else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),   # → [-1, 1] for diffusion
        ])
        Path(cfg.processed_dir).mkdir(parents=True, exist_ok=True)

    def _process_stanford(self):
        """Stanford Dogs: folder-per-breed layout (ImageFolder compatible)."""
        print("[1/3] Processing Stanford Dogs Dataset…")
        dataset = tvdatasets.ImageFolder(
            root=os.path.join(self.cfg.stanford_dir, "Images"),
        )
        records = []
        for idx, (img_path, class_idx) in enumerate(dataset.imgs):
            breed = dataset.classes[class_idx].split("-", 1)[-1].replace("_", " ").title()
            records.append({"path": img_path, "breed": breed, "source": "stanford"})
        print(f"   → {len(records)} images across {len(dataset.classes)} breeds")
        return records

    def _process_oxford(self):
        """Oxford-IIIT: includes segmentation masks and bounding boxes."""
        print("[2/3] Processing Oxford-IIIT Pet Dataset…")
        img_dir = Path(self.cfg.oxford_dir) / "images"
        records = []
        for img_path in img_dir.glob("*.jpg"):
            # Oxford naming: Breed_Name_001.jpg  (dogs are title-cased)
            parts = img_path.stem.rsplit("_", 1)
            breed = parts[0].replace("_", " ").title()
            records.append({"path": str(img_path), "breed": breed, "source": "oxford"})
        print(f"   → {len(records)} images found")
        return records

    def _process_pose(self):
        """
        StanfordExtra: extends Stanford Dogs with 20-keypoint annotations.

        Format: a JSON file that is a flat list of dicts. Each entry contains:
          - img_path: relative path within Stanford Dogs Images/ folder
                      (e.g. "n02085620-Chihuahua/n02085620_10074.jpg")
          - joints:   list of 20 keypoints, each [x, y, visibility]
          - seg:      RLE-encoded silhouette mask (optional)
          - img_bbox: [x, y, w, h] bounding box around the dog

        The breed is encoded in the directory portion of img_path:
          "n02085620-Chihuahua" → "Chihuahua"
        """
        print("[3/3] Processing StanfordExtra (pose) annotations…")
        ann_file = Path(self.cfg.pose_annotations_file)
        if not ann_file.exists():
            print(f"   ⚠ {ann_file} not found — skipping pose annotations")
            return []

        with open(ann_file) as f:
            anns = json.load(f)

        # StanfordExtra ships as a flat list. Some forks wrap it under "data".
        if isinstance(anns, dict):
            anns = anns.get("data") or anns.get("images") or []

        # Stanford Dogs images live in <stanford_dir>/Images/<class>/<file>.jpg
        img_root = Path(self.cfg.stanford_dir) / "Images"

        records, skipped = [], 0
        for item in anns:
            rel_path = item.get("img_path") or item.get("file_name")
            if not rel_path:
                skipped += 1
                continue

            full_path = img_root / rel_path
            if not full_path.exists():
                skipped += 1
                continue

            # Breed name from "n02085620-Chihuahua" → "Chihuahua"
            class_dir = rel_path.split("/", 1)[0]
            breed = class_dir.split("-", 1)[-1].replace("_", " ").title()

            records.append({
                "path": str(full_path),
                "breed": breed,
                "keypoints": item.get("joints", []),  # 20 × [x, y, visibility]
                "bbox": item.get("img_bbox"),
                "source": "pose",
            })

        if skipped:
            print(f"   ⚠ Skipped {skipped} entries (missing img_path or file)")
        print(f"   → {len(records)} pose-annotated images")
        return records

    def run(self):
        all_records = []
        all_records += self._process_stanford()
        all_records += self._process_oxford()
        all_records += self._process_pose()

        # Deduplicate by file path. When the same image appears in multiple
        # sources (e.g. Stanford + StanfordExtra), merge so we keep the keypoints.
        merged = {}
        for r in all_records:
            path = r["path"]
            if path not in merged:
                merged[path] = r
            else:
                # Merge new fields (e.g. 'keypoints', 'bbox') into the existing record
                # without overwriting non-null values.
                for k, v in r.items():
                    if k not in merged[path] or merged[path][k] in (None, [], "Unknown"):
                        merged[path][k] = v
        unique = list(merged.values())

        manifest_path = Path(self.cfg.processed_dir) / "manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(unique, f, indent=2)

        n_with_keypoints = sum(1 for r in unique if r.get("keypoints"))
        print(f"\n✓ Preprocessing complete. {len(unique)} total images "
              f"({n_with_keypoints} with keypoints) → {manifest_path}")
        return unique


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 2 — AUTO-CAPTIONING (BLIP-2)
# ─────────────────────────────────────────────────────────────────────────────

class _CaptionImageDataset(Dataset):
    """Lightweight dataset for parallel image loading during captioning."""

    def __init__(self, records: list, processor, target_size: int = 384):
        self.records = records
        self.processor = processor
        self.target_size = target_size

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        try:
            img = Image.open(record["path"]).convert("RGB")
            # Pre-resize to BLIP's expected input (~384px) before preprocessing
            img.thumbnail((self.target_size * 2, self.target_size * 2), Image.LANCZOS)
            return {
                "image": img,
                "path": record["path"],
                "breed": record.get("breed", "dog"),
                "ok": True,
            }
        except Exception:
            return {"image": None, "path": record["path"],
                    "breed": record.get("breed", "dog"), "ok": False}


def _caption_collate(batch):
    """Collate function: keeps PIL images as a list, gathers metadata."""
    return {
        "images": [b["image"] for b in batch if b["ok"]],
        "paths":  [b["path"]  for b in batch if b["ok"]],
        "breeds": [b["breed"] for b in batch if b["ok"]],
        "failed_paths":  [b["path"]  for b in batch if not b["ok"]],
        "failed_breeds": [b["breed"] for b in batch if not b["ok"]],
    }


class DogCaptioner:
    """
    Batched BLIP captioning with parallel image loading and incremental saves.

    Speed wins (vs original single-image loop):
      • BLIP-Base instead of BLIP-Large: ~2x faster, near-identical quality here
      • Batched generation (default 32 per batch): the big GPU win, ~10x speedup
      • DataLoader with num_workers: image decoding overlaps with GPU work
      • Resume support: skips already-captioned paths from previous runs
      • Incremental saves: progress is checkpointed every N images so an
        interrupted run doesn't lose hours of work

    Combined effect on a modern GPU: ~7 hours → ~25-40 minutes for 20k images.
    """

    def __init__(self, cfg: Config, device: str = "cuda"):
        self.cfg = cfg
        self.device = device
        print(f"Loading captioning model ({cfg.caption_model_id})…")
        self.processor = BlipProcessor.from_pretrained(cfg.caption_model_id)
        self.model = BlipForConditionalGeneration.from_pretrained(
            cfg.caption_model_id,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        ).to(device)
        self.model.eval()

    @staticmethod
    def _enrich(raw_caption: str, breed: str) -> str:
        """Inject breed name if BLIP didn't already mention it."""
        if breed.lower() in raw_caption.lower():
            return raw_caption
        return f"A photo of a {breed} dog, {raw_caption}"

    def _load_existing_captions(self) -> dict:
        """Resume support: load any captions from a previous interrupted run."""
        path = Path(self.cfg.captions_file)
        if path.exists():
            try:
                with open(path) as f:
                    existing = json.load(f)
                print(f"   Found {len(existing)} existing captions — resuming from there")
                return existing
            except json.JSONDecodeError:
                print("   ⚠ Existing captions file is corrupt — starting fresh")
        return {}

    def _save(self, captions: dict):
        """Atomic write: save to temp file then rename."""
        path = Path(self.cfg.captions_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump(captions, f, indent=2)
        tmp.replace(path)

    @torch.no_grad()
    def _caption_batch(self, images: list, breeds: list) -> list:
        """Run BLIP on a batch of PIL images and return enriched captions."""
        if not images:
            return []
        inputs = self.processor(images=images, return_tensors="pt", padding=True).to(self.device)
        if self.device == "cuda":
            inputs = {k: (v.half() if v.dtype == torch.float32 else v) for k, v in inputs.items()}
        outputs = self.model.generate(**inputs, max_new_tokens=40, num_beams=1)
        raw_captions = self.processor.batch_decode(outputs, skip_special_tokens=True)
        return [self._enrich(c, b) for c, b in zip(raw_captions, breeds)]

    def run(self, manifest: list, limit: Optional[int] = None) -> dict:
        cfg = self.cfg
        captions = self._load_existing_captions()

        # Filter out already-captioned records
        records = manifest[:limit] if limit else manifest
        todo = [r for r in records if r["path"] not in captions]
        print(f"   {len(records)} total records, {len(captions)} already done, "
              f"{len(todo)} remaining")
        if not todo:
            print("✓ Nothing to caption.")
            return captions

        ds = _CaptionImageDataset(todo, self.processor)
        loader = DataLoader(
            ds,
            batch_size=cfg.caption_batch_size,
            num_workers=cfg.caption_num_workers,
            collate_fn=_caption_collate,
            shuffle=False,
            pin_memory=False,  # we hold PIL images, not tensors
        )

        import time
        t0 = time.time()
        seen = 0
        for batch_idx, batch in enumerate(loader):
            # Captions for successfully-loaded images
            new_captions = self._caption_batch(batch["images"], batch["breeds"])
            for path, caption in zip(batch["paths"], new_captions):
                captions[path] = caption

            # Fallback for files that failed to load
            for path, breed in zip(batch["failed_paths"], batch["failed_breeds"]):
                captions[path] = f"A photo of a {breed} dog"

            seen += len(batch["paths"]) + len(batch["failed_paths"])

            # Periodic checkpoint save + progress print
            if seen % cfg.caption_save_every < cfg.caption_batch_size or batch_idx == len(loader) - 1:
                self._save(captions)
                elapsed = time.time() - t0
                rate = seen / elapsed if elapsed > 0 else 0
                eta = (len(todo) - seen) / rate if rate > 0 else 0
                print(f"   {seen:>5}/{len(todo)}  "
                      f"({rate:.1f} img/s, ETA {eta/60:.1f} min)")

        self._save(captions)
        print(f"✓ Captions saved → {cfg.captions_file}  "
              f"({len(captions)} total in {(time.time()-t0)/60:.1f} min)")
        return captions


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 3 — DATASET CLASS FOR TRAINING
# ─────────────────────────────────────────────────────────────────────────────

class DogBreedDataset(Dataset):
    """
    Torch Dataset that returns (pixel_values, input_ids) pairs for
    DreamBooth / LoRA fine-tuning with a Stable Diffusion pipeline.
    """

    def __init__(self, manifest: list, captions: dict, tokenizer, cfg: Config):
        self.records = [r for r in manifest if r["path"] in captions]
        self.captions = captions
        self.tokenizer = tokenizer
        self.image_transforms = transforms.Compose([
            transforms.Resize(cfg.image_size, interpolation=transforms.InterpolationMode.LANCZOS),
            transforms.CenterCrop(cfg.image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        image = Image.open(record["path"]).convert("RGB")
        pixel_values = self.image_transforms(image)

        caption = self.captions[record["path"]]
        tokens = self.tokenizer(
            caption,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        )
        return {"pixel_values": pixel_values, "input_ids": tokens.input_ids.squeeze(0)}


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 4 — LoRA FINE-TUNING
# ─────────────────────────────────────────────────────────────────────────────

class LoRATrainer:
    """
    Fine-tunes the SD UNet with LoRA adapters using HuggingFace PEFT + Accelerate.
    Only the LoRA weights (~few MB) are trained; the base model is frozen.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.accelerator = Accelerator(
            gradient_accumulation_steps=cfg.gradient_accumulation_steps,
            mixed_precision=cfg.mixed_precision,
        )
        torch.manual_seed(cfg.seed)

    def _build_lora_unet(self, unet):
        lora_config = LoraConfig(
            r=self.cfg.lora_rank,
            lora_alpha=self.cfg.lora_alpha,
            target_modules=self.cfg.lora_target_modules,
            lora_dropout=self.cfg.lora_dropout,
            bias="none",
        )
        return get_peft_model(unet, lora_config)

    def train(self, dataset: DogBreedDataset):
        cfg = self.cfg
        print(f"\n{'─'*60}")
        print("Loading base SD model components…")

        tokenizer = CLIPTokenizer.from_pretrained(cfg.base_model_id, subfolder="tokenizer")
        text_encoder = CLIPTextModel.from_pretrained(cfg.base_model_id, subfolder="text_encoder")
        vae = AutoencoderKL.from_pretrained(cfg.base_model_id, subfolder="vae")
        unet = UNet2DConditionModel.from_pretrained(cfg.base_model_id, subfolder="unet")
        noise_scheduler = DDPMScheduler.from_pretrained(cfg.base_model_id, subfolder="scheduler")

        # Freeze everything except LoRA
        vae.requires_grad_(False)
        text_encoder.requires_grad_(False)
        unet = self._build_lora_unet(unet)
        unet.print_trainable_parameters()

        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, unet.parameters()),
            lr=cfg.learning_rate,
        )
        dataloader = DataLoader(dataset, batch_size=cfg.train_batch_size, shuffle=True)
        lr_scheduler = get_scheduler(
            "cosine",
            optimizer=optimizer,
            num_warmup_steps=100,
            num_training_steps=cfg.max_train_steps,
        )

        unet, optimizer, dataloader, lr_scheduler = self.accelerator.prepare(
            unet, optimizer, dataloader, lr_scheduler
        )

        # ── Dtype handling for frozen modules ─────────────────────────────────
        # Accelerate handles autocast for the *trainable* unet, but the VAE and
        # text encoder are frozen and live outside that autocast. We cast them
        # explicitly so all conv/linear ops match the input dtype.
        if cfg.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif cfg.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16
        else:
            weight_dtype = torch.float32

        vae.to(self.accelerator.device, dtype=weight_dtype)
        text_encoder.to(self.accelerator.device, dtype=weight_dtype)

        global_step = 0
        print(f"Starting LoRA training for {cfg.max_train_steps} steps "
              f"(weight_dtype={weight_dtype})…\n")

        # ── Training log: per-step loss + lr written to CSV for plotting ──────
        log_path = Path(cfg.training_log)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_file = open(log_path, "w", newline="")
        log_writer = csv.writer(log_file)
        log_writer.writerow(["step", "loss", "lr"])

        for epoch in range(9999):  # breaks on max_train_steps
            for batch in dataloader:
                with self.accelerator.accumulate(unet):
                    # Encode images → latent space (cast pixels to match VAE dtype)
                    pixel_values = batch["pixel_values"].to(
                        device=self.accelerator.device, dtype=weight_dtype
                    )
                    latents = vae.encode(pixel_values).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor

                    # Sample noise (matches latents' dtype automatically)
                    noise = torch.randn_like(latents)
                    bsz = latents.shape[0]
                    timesteps = torch.randint(
                        0, noise_scheduler.config.num_train_timesteps,
                        (bsz,), device=latents.device,
                    ).long()
                    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                    # Encode text (input_ids are integer; output picks up text_encoder's dtype)
                    encoder_hidden_states = text_encoder(
                        batch["input_ids"].to(self.accelerator.device)
                    )[0]

                    # Predict noise — accelerator's autocast keeps unet ops in the right precision
                    noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

                    # MSE loss in fp32 for numerical stability
                    loss = torch.nn.functional.mse_loss(noise_pred.float(), noise.float())
                    self.accelerator.backward(loss)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                global_step += 1

                # Log every step to CSV (cheap), print every 50
                log_writer.writerow([
                    global_step, float(loss.item()), float(lr_scheduler.get_last_lr()[0])
                ])
                if global_step % 50 == 0:
                    log_file.flush()  # ensure data hits disk in case of crash
                    print(f"  step {global_step:>5} / {cfg.max_train_steps}  loss={loss.item():.4f}")

                if global_step % cfg.save_steps == 0:
                    save_path = Path(cfg.model_dir) / f"checkpoint-{global_step}"
                    self.accelerator.unwrap_model(unet).save_pretrained(str(save_path))
                    print(f"  💾 Checkpoint saved → {save_path}")

                if global_step >= cfg.max_train_steps:
                    break
            if global_step >= cfg.max_train_steps:
                break

        # Final save
        Path(cfg.model_dir).mkdir(parents=True, exist_ok=True)
        self.accelerator.unwrap_model(unet).save_pretrained(cfg.model_dir)
        log_file.close()
        print(f"\n✓ LoRA training complete. Weights → {cfg.model_dir}")
        print(f"  Training log → {log_path}")


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 5 — GENERATION WITH CONTROLNET (STRUCTURAL GUIDANCE)
# ─────────────────────────────────────────────────────────────────────────────

class DogImageGenerator:
    """
    Generates dog images using the LoRA-fine-tuned SD model with optional
    ControlNet (OpenPose) for structural guidance from the Dog-Pose dataset.
    """

    PROMPT_TEMPLATE = (
        "A highly detailed, photorealistic photograph of a {breed}, "
        "full body portrait, accurate anatomy, natural pose, studio lighting, "
        "sharp focus, 8k resolution, professional wildlife photography"
    )
    NEGATIVE_PROMPT = (
        "blurry, deformed, extra limbs, missing limbs, bad anatomy, "
        "cartoon, illustration, low quality, watermark, text"
    )

    def __init__(self, cfg: Config, use_controlnet: bool = False):
        self.cfg = cfg
        device = "cuda" if torch.cuda.is_available() else "cpu"

        if use_controlnet:
            print("Loading ControlNet pipeline…")
            controlnet = ControlNetModel.from_pretrained(
                cfg.controlnet_model_id, torch_dtype=torch.float16
            )
            self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
                cfg.base_model_id,
                controlnet=controlnet,
                torch_dtype=torch.float16,
            ).to(device)
        else:
            print("Loading base SD pipeline with LoRA…")
            self.pipe = StableDiffusionPipeline.from_pretrained(
                cfg.base_model_id, torch_dtype=torch.float16
            ).to(device)

        # Load LoRA weights if available.
        # The trainer saves with PEFT (adapter_model.safetensors + adapter_config.json),
        # so we load the same way rather than via the legacy load_attn_procs() path
        # which would expect pytorch_lora_weights.bin.
        lora_path = Path(cfg.model_dir)
        adapter_file = lora_path / "adapter_model.safetensors"
        legacy_file = lora_path / "pytorch_lora_weights.safetensors"

        if adapter_file.exists():
            from peft import PeftModel
            self.pipe.unet = PeftModel.from_pretrained(
                self.pipe.unet, str(lora_path)
            ).to(device)
            print(f"  ✓ LoRA adapter (PEFT) loaded from {lora_path}")
        elif legacy_file.exists():
            # Fallback for checkpoints saved in the older diffusers format
            self.pipe.load_lora_weights(str(lora_path))
            print(f"  ✓ LoRA weights (legacy) loaded from {lora_path}")
        else:
            print(f"  ⚠ No LoRA checkpoint found in {lora_path} — using base model")
            print(f"     (looked for adapter_model.safetensors or pytorch_lora_weights.safetensors)")

        self.pipe.enable_attention_slicing()
        self.use_controlnet = use_controlnet

    def generate(
        self,
        breed: str,
        num_images: int = 4,
        pose_image: Optional[Image.Image] = None,
        seed: Optional[int] = None,
    ) -> list[Image.Image]:
        prompt = self.PROMPT_TEMPLATE.format(breed=breed)
        generator = torch.Generator().manual_seed(seed or self.cfg.seed)

        kwargs = dict(
            prompt=prompt,
            negative_prompt=self.NEGATIVE_PROMPT,
            num_images_per_prompt=num_images,
            num_inference_steps=self.cfg.num_inference_steps,
            guidance_scale=self.cfg.guidance_scale,
            generator=generator,
        )
        if self.use_controlnet and pose_image is not None:
            kwargs["image"] = pose_image

        result = self.pipe(**kwargs)
        return result.images

    def save_images(self, images: list[Image.Image], breed: str, out_dir: str = None) -> list[str]:
        out_dir = Path(out_dir or self.cfg.output_dir) / breed.replace(" ", "_")
        out_dir.mkdir(parents=True, exist_ok=True)
        paths = []
        for i, img in enumerate(images):
            path = out_dir / f"gen_{i:03d}.png"
            img.save(path)
            paths.append(str(path))
        print(f"✓ {len(images)} images saved → {out_dir}")
        return paths


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 6 — EVALUATION (FID + CLIP)
# ─────────────────────────────────────────────────────────────────────────────

class DogEvaluator:
    """
    Computes FID (realism vs. dataset) and CLIP score (text-image alignment).
    Both scores complement human preference evaluation.
    """

    def __init__(self, cfg: Config, device: str = "cuda"):
        self.cfg = cfg
        self.device = device
        self.fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
        self.clip_scorer = CLIPScore(model_name_or_path="openai/clip-vit-base-patch32").to(device)
        self.transform = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
        ])

    def _load_tensor(self, img_path: str) -> torch.Tensor:
        img = Image.open(img_path).convert("RGB")
        return self.transform(img).unsqueeze(0).to(self.device)

    def compute_fid(self, real_paths: list[str], generated_paths: list[str]) -> float:
        print(f"Computing FID ({len(real_paths)} real, {len(generated_paths)} generated)…")
        for path in real_paths[:self.cfg.fid_num_samples]:
            t = (self._load_tensor(path) * 255).byte()
            self.fid.update(t, real=True)
        for path in generated_paths:
            t = (self._load_tensor(path) * 255).byte()
            self.fid.update(t, real=False)
        score = self.fid.compute().item()
        print(f"  FID = {score:.2f}  (lower is better; < 30 is strong)")
        return score

    def compute_clip_score(self, generated_paths: list[str], prompts: list[str]) -> float:
        print("Computing CLIP score…")
        scores = []
        for path, prompt in zip(generated_paths, prompts):
            img = Image.open(path).convert("RGB")
            img_t = transforms.ToTensor()(img).unsqueeze(0).to(self.device)
            s = self.clip_scorer(img_t, prompt).item()
            scores.append(s)
        avg = np.mean(scores)
        print(f"  CLIP score = {avg:.3f}  (higher is better; > 0.28 is strong)")
        return float(avg)

    def run(self, real_paths: list[str], generated_paths: list[str], breed: str) -> dict:
        prompt = DogImageGenerator.PROMPT_TEMPLATE.format(breed=breed)
        prompts = [prompt] * len(generated_paths)
        fid = self.compute_fid(real_paths, generated_paths)
        clip = self.compute_clip_score(generated_paths, prompts)
        results = {"breed": breed, "fid": fid, "clip_score": clip}
        results_path = Path(self.cfg.output_dir) / "eval_results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n✓ Evaluation results → {results_path}")
        return results


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 7 — CHART GENERATION
# ─────────────────────────────────────────────────────────────────────────────

class ChartGenerator:
    """
    Produces three PNG outputs for the project report:

      1. loss_curve.png        — training loss vs. step (with smoothed overlay)
      2. per_breed_table.png   — sorted table of FID/CLIP per breed
      3. baseline_vs_finetuned.png — bar chart comparing base SD vs fine-tuned

    All charts are saved under cfg.charts_dir.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        Path(cfg.charts_dir).mkdir(parents=True, exist_ok=True)
        # Consistent typography across all charts
        plt.rcParams.update({
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 120,
        })

    # ── 1. Training loss curve ────────────────────────────────────────────────
    def plot_loss_curve(self, log_path: Optional[str] = None) -> str:
        log_path = Path(log_path or self.cfg.training_log)
        if not log_path.exists():
            print(f"⚠ {log_path} not found — run training first")
            return ""

        steps, losses, lrs = [], [], []
        with open(log_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                steps.append(int(row["step"]))
                losses.append(float(row["loss"]))
                lrs.append(float(row["lr"]))

        if not steps:
            print(f"⚠ {log_path} is empty")
            return ""

        # Exponential moving average for the smoothed line
        ema, alpha = [], 0.05
        running = losses[0]
        for v in losses:
            running = alpha * v + (1 - alpha) * running
            ema.append(running)

        fig, ax1 = plt.subplots(figsize=(9, 5))
        ax1.plot(steps, losses, color="#9bb7d4", linewidth=0.6, alpha=0.5,
                 label="raw loss")
        ax1.plot(steps, ema, color="#1f4e79", linewidth=2.0,
                 label="loss (EMA, α=0.05)")
        ax1.set_xlabel("Training step")
        ax1.set_ylabel("MSE loss (noise prediction)")
        ax1.set_title("LoRA fine-tuning — training loss")
        ax1.grid(True, linestyle=":", alpha=0.4)

        # Secondary axis for learning rate
        ax2 = ax1.twinx()
        ax2.plot(steps, lrs, color="#c47b3a", linewidth=1.3,
                 linestyle="--", label="learning rate")
        ax2.set_ylabel("Learning rate", color="#c47b3a")
        ax2.tick_params(axis="y", labelcolor="#c47b3a")
        ax2.spines["top"].set_visible(False)

        # Combined legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

        out = Path(self.cfg.charts_dir) / "loss_curve.png"
        fig.tight_layout()
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        print(f"✓ Loss curve → {out}")
        return str(out)

    # ── 2. Per-breed evaluation table ─────────────────────────────────────────
    def plot_per_breed_table(self, per_breed_results: list[dict],
                             top_n: int = 25) -> str:
        """
        Renders a sorted table (best → worst FID) as a PNG.

        per_breed_results: list of dicts with keys
            'breed', 'fid', 'clip_score', 'n_samples'
        """
        if not per_breed_results:
            print("⚠ No per-breed results to plot")
            return ""

        # Sort by FID ascending (lower = better realism)
        rows = sorted(per_breed_results, key=lambda r: r.get("fid", float("inf")))
        if top_n:
            rows = rows[:top_n]

        col_labels = ["Rank", "Breed", "FID ↓", "CLIP ↑", "n samples"]
        cell_text = [
            [str(i + 1),
             r["breed"],
             f"{r.get('fid', 0):.2f}",
             f"{r.get('clip_score', 0):.3f}",
             str(r.get("n_samples", "—"))]
            for i, r in enumerate(rows)
        ]

        # Sized to comfortably show top_n rows
        fig_height = max(3, 0.32 * (len(rows) + 2))
        fig, ax = plt.subplots(figsize=(8, fig_height))
        ax.axis("off")

        tbl = ax.table(
            cellText=cell_text,
            colLabels=col_labels,
            loc="center",
            cellLoc="center",
            colWidths=[0.08, 0.42, 0.13, 0.15, 0.15],
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(9)
        tbl.scale(1, 1.35)

        # Header styling
        for i, _ in enumerate(col_labels):
            cell = tbl[(0, i)]
            cell.set_facecolor("#1f4e79")
            cell.set_text_props(color="white", weight="bold")

        # Alternating row shading
        for r in range(1, len(rows) + 1):
            color = "#f4f6f9" if r % 2 == 0 else "white"
            for c in range(len(col_labels)):
                tbl[(r, c)].set_facecolor(color)

        ax.set_title(f"Per-breed evaluation (top {len(rows)} by FID)",
                     pad=14, weight="bold")

        out = Path(self.cfg.charts_dir) / "per_breed_table.png"
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        print(f"✓ Per-breed table → {out}")

        # Also save a sortable bar chart of FID per breed (more visual)
        self._plot_per_breed_bars(rows)
        return str(out)

    def _plot_per_breed_bars(self, rows: list[dict]) -> str:
        breeds = [r["breed"] for r in rows]
        fids = [r.get("fid", 0) for r in rows]
        clips = [r.get("clip_score", 0) for r in rows]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, max(5, 0.25 * len(rows))))

        # FID bars
        ax1.barh(breeds, fids, color="#1f4e79")
        ax1.set_xlabel("FID (lower is better)")
        ax1.set_title("FID per breed")
        ax1.invert_yaxis()
        ax1.grid(True, axis="x", linestyle=":", alpha=0.4)

        # CLIP bars
        ax2.barh(breeds, clips, color="#c47b3a")
        ax2.set_xlabel("CLIP score (higher is better)")
        ax2.set_title("CLIP score per breed")
        ax2.invert_yaxis()
        ax2.grid(True, axis="x", linestyle=":", alpha=0.4)

        out = Path(self.cfg.charts_dir) / "per_breed_bars.png"
        fig.tight_layout()
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        print(f"✓ Per-breed bar chart → {out}")
        return str(out)

    # ── 3. Baseline vs fine-tuned comparison ──────────────────────────────────
    def plot_baseline_vs_finetuned(self, comparison: dict) -> str:
        """
        comparison dict shape:
          {
            "breeds": ["Golden Retriever", "Husky", ...],
            "baseline":  {"fid": [...], "clip": [...]},
            "finetuned": {"fid": [...], "clip": [...]},
          }
        """
        breeds = comparison["breeds"]
        x = np.arange(len(breeds))
        width = 0.38

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

        # FID comparison (lower better)
        ax1.bar(x - width/2, comparison["baseline"]["fid"], width,
                label="Base SD v1.5", color="#9bb7d4")
        ax1.bar(x + width/2, comparison["finetuned"]["fid"], width,
                label="LoRA fine-tuned", color="#1f4e79")
        ax1.set_xticks(x)
        ax1.set_xticklabels(breeds, rotation=35, ha="right")
        ax1.set_ylabel("FID (lower is better)")
        ax1.set_title("FID: baseline vs fine-tuned")
        ax1.legend()
        ax1.grid(True, axis="y", linestyle=":", alpha=0.4)

        # CLIP comparison (higher better)
        ax2.bar(x - width/2, comparison["baseline"]["clip"], width,
                label="Base SD v1.5", color="#e6c79c")
        ax2.bar(x + width/2, comparison["finetuned"]["clip"], width,
                label="LoRA fine-tuned", color="#c47b3a")
        ax2.set_xticks(x)
        ax2.set_xticklabels(breeds, rotation=35, ha="right")
        ax2.set_ylabel("CLIP score (higher is better)")
        ax2.set_title("CLIP score: baseline vs fine-tuned")
        ax2.legend()
        ax2.grid(True, axis="y", linestyle=":", alpha=0.4)

        out = Path(self.cfg.charts_dir) / "baseline_vs_finetuned.png"
        fig.tight_layout()
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        print(f"✓ Baseline comparison → {out}")
        return str(out)


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 8 — MULTI-BREED EVALUATION + BASELINE COMPARISON
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_multiple_breeds(cfg: Config, breeds: list, manifest: list,
                              num_per_breed: int = 8,
                              compare_baseline: bool = False) -> dict:
    """
    For each breed in `breeds`:
      • generate `num_per_breed` images with the LoRA-fine-tuned model
      • optionally also generate from the base model (no LoRA)
      • compute FID + CLIP against real images of that breed

    Returns a dict with per-breed results and (optionally) comparison data
    suitable for ChartGenerator.plot_baseline_vs_finetuned.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    evaluator = DogEvaluator(cfg, device=device)

    # Build a breed → list of real image paths index from the manifest
    breed_to_paths = {}
    for r in manifest:
        breed_to_paths.setdefault(r["breed"], []).append(r["path"])

    finetuned_results = []
    baseline_results = []

    # ── Fine-tuned generation pass ────────────────────────────────────────────
    print("\n=== Generating from LoRA fine-tuned model ===")
    ft_generator = DogImageGenerator(cfg, use_controlnet=False)
    for breed in breeds:
        if breed not in breed_to_paths:
            print(f"  ⚠ {breed} not in manifest — skipping")
            continue
        print(f"  → {breed}")
        images = ft_generator.generate(breed=breed, num_images=num_per_breed)
        out_dir = Path(cfg.output_dir) / "eval" / "finetuned" / breed.replace(" ", "_")
        out_dir.mkdir(parents=True, exist_ok=True)
        gen_paths = []
        for i, img in enumerate(images):
            p = out_dir / f"gen_{i:03d}.png"
            img.save(p)
            gen_paths.append(str(p))

        # Reset FID counters between breeds — torchmetrics accumulates state
        evaluator.fid.reset()
        real_paths = breed_to_paths[breed][:cfg.fid_num_samples]
        fid = evaluator.compute_fid(real_paths, gen_paths)
        prompt = DogImageGenerator.PROMPT_TEMPLATE.format(breed=breed)
        clip = evaluator.compute_clip_score(gen_paths, [prompt] * len(gen_paths))
        finetuned_results.append({
            "breed": breed, "fid": fid, "clip_score": clip,
            "n_samples": len(gen_paths),
        })

    del ft_generator
    if device == "cuda":
        torch.cuda.empty_cache()

    # ── Baseline generation pass (skip LoRA loading entirely) ─────────────────
    if compare_baseline:
        print("\n=== Generating from base SD v1.5 (no LoRA) ===")
        # Temporarily point model_dir at a non-existent path so __init__ skips loading
        original_model_dir = cfg.model_dir
        cfg.model_dir = "/tmp/__no_such_lora_dir__"
        base_generator = DogImageGenerator(cfg, use_controlnet=False)
        cfg.model_dir = original_model_dir

        for breed in breeds:
            if breed not in breed_to_paths:
                continue
            print(f"  → {breed}")
            images = base_generator.generate(breed=breed, num_images=num_per_breed)
            out_dir = Path(cfg.output_dir) / "eval" / "baseline" / breed.replace(" ", "_")
            out_dir.mkdir(parents=True, exist_ok=True)
            gen_paths = []
            for i, img in enumerate(images):
                p = out_dir / f"gen_{i:03d}.png"
                img.save(p)
                gen_paths.append(str(p))

            evaluator.fid.reset()
            real_paths = breed_to_paths[breed][:cfg.fid_num_samples]
            fid = evaluator.compute_fid(real_paths, gen_paths)
            prompt = DogImageGenerator.PROMPT_TEMPLATE.format(breed=breed)
            clip = evaluator.compute_clip_score(gen_paths, [prompt] * len(gen_paths))
            baseline_results.append({
                "breed": breed, "fid": fid, "clip_score": clip,
                "n_samples": len(gen_paths),
            })

        del base_generator
        if device == "cuda":
            torch.cuda.empty_cache()

    # ── Persist + return ──────────────────────────────────────────────────────
    output = {
        "finetuned": finetuned_results,
        "baseline": baseline_results,
    }
    results_path = Path(cfg.output_dir) / "multi_breed_results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n✓ Multi-breed results → {results_path}")
    return output


# ─────────────────────────────────────────────────────────────────────────────
# CLI ENTRYPOINT
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Generative Dog Images Pipeline")
    p.add_argument("--stage", choices=[
        "preprocess", "caption", "train", "generate", "evaluate",
        "compare", "charts",
    ], required=True)
    p.add_argument("--breed", type=str, default="Golden Retriever",
                   help="Breed for generation/evaluation")
    p.add_argument("--breeds", type=str, nargs="+",
                   default=["Golden Retriever", "Siberian Husky",
                            "Chihuahua", "Beagle", "Border Collie"],
                   help="List of breeds for compare/charts stages")
    p.add_argument("--num_images", type=int, default=4)
    p.add_argument("--num_per_breed", type=int, default=8,
                   help="Images per breed during multi-breed evaluation")
    p.add_argument("--use_controlnet", action="store_true")
    p.add_argument("--caption_limit", type=int, default=None,
                   help="Limit captioning to N images (for testing)")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = CFG
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    if args.stage == "preprocess":
        preprocessor = DogImagePreprocessor(cfg)
        preprocessor.run()

    elif args.stage == "caption":
        manifest_path = Path(cfg.processed_dir) / "manifest.json"
        with open(manifest_path) as f:
            manifest = json.load(f)
        captioner = DogCaptioner(cfg, device=device)
        captioner.run(manifest, limit=args.caption_limit)

    elif args.stage == "train":
        manifest_path = Path(cfg.processed_dir) / "manifest.json"
        with open(manifest_path) as f:
            manifest = json.load(f)
        with open(cfg.captions_file) as f:
            captions = json.load(f)
        tokenizer = CLIPTokenizer.from_pretrained(cfg.base_model_id, subfolder="tokenizer")
        dataset = DogBreedDataset(manifest, captions, tokenizer, cfg)
        print(f"Training dataset: {len(dataset)} samples")
        trainer = LoRATrainer(cfg)
        trainer.train(dataset)

    elif args.stage == "generate":
        generator = DogImageGenerator(cfg, use_controlnet=args.use_controlnet)
        images = generator.generate(breed=args.breed, num_images=args.num_images)
        generator.save_images(images, breed=args.breed)

    elif args.stage == "evaluate":
        # Load some real images for FID
        manifest_path = Path(cfg.processed_dir) / "manifest.json"
        with open(manifest_path) as f:
            manifest = json.load(f)
        breed_records = [r for r in manifest if r.get("breed", "").lower() == args.breed.lower()]
        real_paths = [r["path"] for r in breed_records[:cfg.fid_num_samples]]

        # Collect generated images
        gen_dir = Path(cfg.output_dir) / args.breed.replace(" ", "_")
        generated_paths = list(gen_dir.glob("*.png"))
        if not generated_paths:
            print(f"No generated images found in {gen_dir}. Run --stage generate first.")
            return

        evaluator = DogEvaluator(cfg, device=device)
        results = evaluator.run(
            real_paths=real_paths,
            generated_paths=[str(p) for p in generated_paths],
            breed=args.breed,
        )
        print("\nFinal Results:")
        print(json.dumps(results, indent=2))

    elif args.stage == "compare":
        # Run multi-breed evaluation against both fine-tuned and baseline models
        manifest_path = Path(cfg.processed_dir) / "manifest.json"
        with open(manifest_path) as f:
            manifest = json.load(f)
        evaluate_multiple_breeds(
            cfg, breeds=args.breeds, manifest=manifest,
            num_per_breed=args.num_per_breed, compare_baseline=True,
        )
        print("\nNext step: run `--stage charts` to render the PNG outputs.")

    elif args.stage == "charts":
        charter = ChartGenerator(cfg)

        # 1. Loss curve from training_log.csv
        charter.plot_loss_curve()

        # 2 & 3 — depend on multi_breed_results.json from --stage compare
        results_path = Path(cfg.output_dir) / "multi_breed_results.json"
        if results_path.exists():
            with open(results_path) as f:
                data = json.load(f)
            ft = data.get("finetuned", [])
            base = data.get("baseline", [])

            if ft:
                charter.plot_per_breed_table(ft, top_n=25)

            if ft and base:
                # Align breeds present in both runs
                ft_map = {r["breed"]: r for r in ft}
                base_map = {r["breed"]: r for r in base}
                shared = [b for b in ft_map if b in base_map]
                comparison = {
                    "breeds": shared,
                    "baseline":  {"fid":  [base_map[b]["fid"] for b in shared],
                                   "clip": [base_map[b]["clip_score"] for b in shared]},
                    "finetuned": {"fid":  [ft_map[b]["fid"] for b in shared],
                                   "clip": [ft_map[b]["clip_score"] for b in shared]},
                }
                charter.plot_baseline_vs_finetuned(comparison)
        else:
            print(f"⚠ {results_path} not found — run `--stage compare` first "
                  "to populate multi-breed evaluation data.")

        print(f"\n✓ All charts saved under {cfg.charts_dir}/")


if __name__ == "__main__":
    main()