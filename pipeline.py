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
    pose_dir: str = "data/dog_pose"
    processed_dir: str = "data/processed"
    captions_file: str = "data/captions.json"
    output_dir: str = "outputs"
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
        """Dog-Pose Dataset: keypoint annotations stored in JSON."""
        print("[3/3] Processing Dog-Pose Dataset…")
        ann_file = Path(self.cfg.pose_dir) / "annotations.json"
        records = []
        if ann_file.exists():
            with open(ann_file) as f:
                anns = json.load(f)
            for item in anns.get("images", []):
                records.append({
                    "path": str(Path(self.cfg.pose_dir) / "images" / item["file_name"]),
                    "breed": item.get("breed", "Unknown"),
                    "keypoints": item.get("keypoints", []),
                    "source": "pose",
                })
        print(f"   → {len(records)} pose-annotated images")
        return records

    def run(self):
        all_records = []
        all_records += self._process_stanford()
        all_records += self._process_oxford()
        all_records += self._process_pose()

        # Deduplicate by file path
        seen, unique = set(), []
        for r in all_records:
            if r["path"] not in seen:
                seen.add(r["path"])
                unique.append(r)

        manifest_path = Path(self.cfg.processed_dir) / "manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(unique, f, indent=2)

        print(f"\n✓ Preprocessing complete. {len(unique)} total images → {manifest_path}")
        return unique


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 2 — AUTO-CAPTIONING (BLIP-2)
# ─────────────────────────────────────────────────────────────────────────────

class DogCaptioner:
    """
    Generates natural-language captions using BLIP, then enriches them
    with breed name for structured prompts like:
      "A photo of a Golden Retriever dog with fluffy golden coat, sitting on grass"
    """

    def __init__(self, cfg: Config, device: str = "cuda"):
        self.cfg = cfg
        self.device = device
        print("Loading BLIP captioning model…")
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
        self.model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-large",
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        ).to(device)

    def caption_image(self, img_path: str, breed: str) -> str:
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            return f"A photo of a {breed} dog"

        inputs = self.processor(image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=50)
        raw_caption = self.processor.decode(out[0], skip_special_tokens=True)

        # Inject breed name if not already present
        breed_lower = breed.lower()
        if breed_lower not in raw_caption.lower():
            caption = f"A photo of a {breed} dog, {raw_caption}"
        else:
            caption = raw_caption
        return caption

    def run(self, manifest: list, limit: Optional[int] = None) -> dict:
        captions = {}
        records = manifest[:limit] if limit else manifest
        for i, record in enumerate(records):
            path = record["path"]
            breed = record.get("breed", "dog")
            captions[path] = self.caption_image(path, breed)
            if (i + 1) % 100 == 0:
                print(f"   Captioned {i + 1}/{len(records)}")

        with open(self.cfg.captions_file, "w") as f:
            json.dump(captions, f, indent=2)
        print(f"✓ Captions saved → {self.cfg.captions_file}")
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
                if global_step % 50 == 0:
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
        print(f"\n✓ LoRA training complete. Weights → {cfg.model_dir}")


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

        # Load LoRA weights if available
        lora_path = Path(cfg.model_dir)
        if lora_path.exists():
            self.pipe.unet.load_attn_procs(str(lora_path))
            print(f"  LoRA weights loaded from {lora_path}")
        else:
            print("  ⚠ No LoRA checkpoint found — using base model")

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
# CLI ENTRYPOINT
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Generative Dog Images Pipeline")
    p.add_argument("--stage", choices=["preprocess", "caption", "train", "generate", "evaluate"], required=True)
    p.add_argument("--breed", type=str, default="Golden Retriever", help="Breed for generation/evaluation")
    p.add_argument("--num_images", type=int, default=4)
    p.add_argument("--use_controlnet", action="store_true")
    p.add_argument("--caption_limit", type=int, default=None, help="Limit captioning to N images (for testing)")
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


if __name__ == "__main__":
    main()