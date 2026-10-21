"""train.py

CLIP fine-tuning pipeline for E-Commerce Image Retrieval.

Features:
  - Reads all hyperparameters from `config.yaml` (or `--config <path>`).
  - Supports 3 training modes:
      1. "lora"            : LoRA adapters on Vision & Text attention + Projections
      2. "projection_only" : Trains only visual_projection, text_projection & logit_scale
      3. "full"            : Full fine-tuning of all parameters
  - Multi-positive symmetric contrastive loss (InfoNCE) that avoids penalizing
    duplicate queries or products within the same mini-batch.
  - End-of-epoch Validation Loss (`val_loss`) to monitor overfitting/underfitting.
  - End-of-epoch Validation Global Vector Retrieval (`Recall@1/5/10`, `NDCG@10`, `MRR`).
  - Descriptive checkpoint naming including hyperparameters and validation metrics,
    saved at the end of every epoch plus `best_model/`.
  - Seamless AWS SageMaker support (automatically pulls `images_bundle.tar` from S3
    and syncs epoch checkpoints back to S3).

Usage:
    python train.py
    python train.py --config config.yaml
"""

import argparse
import hashlib
import json
import math
import os
import random
import re
import tarfile
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

from dataset import (
    CLIPDataset,
    ProductImageDataset,
    QueryTextDataset,
    image_collate,
    pair_collate,
    text_collate,
)
from eval import evaluate_candidate_reranking, evaluate_global_retrieval

PROJECT_ROOT = Path(__file__).resolve().parent
IS_SAGEMAKER = "SM_CHANNEL_SPLITS" in os.environ

SPLITS_DIR = Path(
    os.environ.get("SM_CHANNEL_SPLITS", PROJECT_ROOT / "data" / "processed" / "splits")
)
TRAIN_PARQUET = SPLITS_DIR / "train.parquet"
VAL_PARQUET = SPLITS_DIR / "val.parquet"
TEST_PARQUET = SPLITS_DIR / "test.parquet"

LOCAL_IMAGES_DIR = PROJECT_ROOT / "data" / "images"
IMAGES_DIR = Path("/tmp/images") if IS_SAGEMAKER else LOCAL_IMAGES_DIR

CHECKPOINTS_ROOT = Path(os.environ.get("SM_MODEL_DIR", PROJECT_ROOT / "checkpoints"))
OUT_DIR = Path(os.environ.get("SM_OUTPUT_DATA_DIR", PROJECT_ROOT / "analysis"))


# =====================================================================
# 1. LoRA Implementation for CLIP Vision & Text Attention
# =====================================================================
class LoRALinear(nn.Module):
    """Low-Rank Adaptation wrapper around an existing nn.Linear layer."""

    def __init__(self, base_layer: nn.Linear, r: int = 16, alpha: int = 32, dropout: float = 0.05):
        super().__init__()
        self.base_layer = base_layer
        self.base_layer.weight.requires_grad = False
        if self.base_layer.bias is not None:
            self.base_layer.bias.requires_grad = False

        self.r = r
        self.alpha = alpha
        self.scaling = alpha / max(1, r)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        in_features = base_layer.in_features
        out_features = base_layer.out_features
        dtype = base_layer.weight.dtype
        device = base_layer.weight.device

        self.lora_A = nn.Parameter(torch.empty((r, in_features), dtype=dtype, device=device))
        self.lora_B = nn.Parameter(torch.zeros((out_features, r), dtype=dtype, device=device))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base_layer(x)
        lora_out = F.linear(F.linear(self.dropout(x), self.lora_A), self.lora_B) * self.scaling
        return base_out + lora_out


def inject_lora(model: nn.Module, target_modules, r: int, alpha: int, dropout: float):
    """Replaces target nn.Linear layers inside CLIP with LoRALinear."""
    target_set = set(target_modules)
    for _name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            if child_name in target_set and isinstance(child, nn.Linear):
                setattr(
                    module,
                    child_name,
                    LoRALinear(child, r=r, alpha=alpha, dropout=dropout),
                )


def configure_trainable_parameters(model: CLIPModel, cfg: dict):
    """Configures model parameters according to `training_mode` in config.yaml."""
    mode = cfg.get("training_mode", "lora").lower()

    if mode == "full":
        for p in model.parameters():
            p.requires_grad = True

    elif mode == "projection_only":
        for p in model.parameters():
            p.requires_grad = False
        for p in model.visual_projection.parameters():
            p.requires_grad = True
        for p in model.text_projection.parameters():
            p.requires_grad = True
        model.logit_scale.requires_grad = True

    elif mode == "lora":
        for p in model.parameters():
            p.requires_grad = False

        inject_lora(
            model,
            target_modules=cfg.get(
                "lora_target_modules", ["q_proj", "k_proj", "v_proj", "out_proj"]
            ),
            r=int(cfg.get("lora_r", 16)),
            alpha=int(cfg.get("lora_alpha", 32)),
            dropout=float(cfg.get("lora_dropout", 0.05)),
        )
        # Also train the projection heads and logit_scale
        for p in model.visual_projection.parameters():
            p.requires_grad = True
        for p in model.text_projection.parameters():
            p.requires_grad = True
        model.logit_scale.requires_grad = True
    else:
        raise ValueError(f"Unknown training_mode: {mode}")

    if bool(cfg.get("gradient_checkpointing", True)) and hasattr(
        model, "gradient_checkpointing_enable"
    ):
        model.gradient_checkpointing_enable()

        def _make_inputs_require_grad(module, args, output):
            if isinstance(output, torch.Tensor):
                output.requires_grad_(True)
            return output

        model.vision_model.embeddings.register_forward_hook(_make_inputs_require_grad)
        model.text_model.embeddings.register_forward_hook(_make_inputs_require_grad)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(
        f"Training mode: '{mode}' | "
        f"Trainable params: {trainable:,} / {total:,} ({100.0 * trainable / total:.2f}%)",
        flush=True,
    )
    return trainable, total


def _config_hash(cfg: dict) -> str:
    """SHA-256 hash of the config (sorted keys, for reproducibility checks)."""
    canonical = yaml.safe_dump(cfg, sort_keys=True, default_flow_style=False)
    return "sha256:" + hashlib.sha256(canonical.encode()).hexdigest()[:16]


def save_checkpoint(model: CLIPModel, cfg: dict, metrics: dict, save_dir: Path):
    """Saves trainable weights, config, metrics, and version metadata to `save_dir`."""
    save_dir.mkdir(parents=True, exist_ok=True)
    mode = cfg.get("training_mode", "lora").lower()

    if mode in ("lora", "projection_only"):
        state = {
            k: v.detach().cpu()
            for k, v in model.state_dict().items()
            if ("lora_" in k)
            or ("visual_projection" in k)
            or ("text_projection" in k)
            or ("logit_scale" in k)
        }
    else:
        state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    # Model version (from env or config)
    model_version = os.environ.get("MODEL_VERSION", cfg.get("model_version", "v0.0.0"))

    payload = {
        "model_name": cfg["model_name"],
        "training_mode": mode,
        "model_version": model_version,
        "config_hash": _config_hash(cfg),
        "config": cfg,
        "metrics": metrics,
        "state_dict": state,
    }
    ckpt_file = save_dir / "checkpoint.pt"
    torch.save(payload, ckpt_file)

    with open(save_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    with open(save_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)

    # Metadata.json (for the CT registry)
    metadata = {
        "version": model_version,
        "git_sha": os.environ.get("GITHUB_SHA", os.environ.get("GIT_COMMIT", "local"))[:7],
        "git_branch": os.environ.get("GITHUB_REF_NAME", os.environ.get("GIT_BRANCH", "local")),
        "trained_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "trained_by": os.environ.get("GITHUB_ACTOR", os.environ.get("USER", "local")),
        "config_hash": _config_hash(cfg),
        "sagemaker_job": os.environ.get("SM_TRAINING_JOB_NAME", None),
        "instance_type": os.environ.get("SM_INSTANCE_TYPE", None),
        "is_sagemaker": IS_SAGEMAKER,
    }
    with open(save_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4, ensure_ascii=False)

    return ckpt_file


def load_trained_clip_model(checkpoint_path: str | Path, device: str = "cpu"):
    """Utility to reconstruct a fine-tuned CLIPModel from a saved checkpoint.pt."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    model_name = ckpt["model_name"]

    model = CLIPModel.from_pretrained(model_name)
    configure_trainable_parameters(model, cfg)
    model.load_state_dict(ckpt["state_dict"], strict=False)
    model.to(device).eval()
    processor = CLIPProcessor.from_pretrained(model_name)
    return model, processor, ckpt


# =====================================================================
# 2. Loss Function (Multi-Positive Symmetric Contrastive Loss)
# =====================================================================
def as_tensor(emb):
    if hasattr(emb, "pooler_output"):
        return emb.pooler_output
    return emb


def compute_contrastive_loss(model: CLIPModel, batch: dict, device: str):
    """Computes symmetric InfoNCE loss with multi-positive masking for duplicate qids/pids."""
    input_ids = batch["input_ids"].to(device, non_blocking=True)
    attention_mask = batch["attention_mask"].to(device, non_blocking=True)
    pixel_values = batch["pixel_values"].to(device, non_blocking=True)

    text_embeds = as_tensor(
        model.get_text_features(input_ids=input_ids, attention_mask=attention_mask)
    )
    image_embeds = as_tensor(model.get_image_features(pixel_values=pixel_values))

    text_embeds = F.normalize(text_embeds, dim=-1)
    image_embeds = F.normalize(image_embeds, dim=-1)

    # Clamp logit_scale to max ln(100) = 4.6052 as in original CLIP paper
    logit_scale = model.logit_scale.clamp(max=4.6052).exp()
    logits_per_text = logit_scale * (text_embeds @ image_embeds.t())
    logits_per_image = logits_per_text.t()

    # Build positive target mask: (i, j) is positive if same query_id or same product_id
    qids = batch["query_ids"]
    pids = batch["product_ids"]
    bsz = len(qids)

    pos_mask = torch.eye(bsz, device=device, dtype=torch.float32)
    for i in range(bsz):
        for j in range(i + 1, bsz):
            if qids[i] == qids[j] or pids[i] == pids[j]:
                pos_mask[i, j] = 1.0
                pos_mask[j, i] = 1.0

    # Normalize target probabilities across all valid positives in the row
    target_probs_t2i = pos_mask / pos_mask.sum(dim=-1, keepdim=True)
    target_probs_i2t = pos_mask.t() / pos_mask.t().sum(dim=-1, keepdim=True)

    loss_t2i = -(target_probs_t2i * F.log_softmax(logits_per_text, dim=-1)).sum(dim=-1).mean()
    loss_i2t = -(target_probs_i2t * F.log_softmax(logits_per_image, dim=-1)).sum(dim=-1).mean()

    return 0.5 * (loss_t2i + loss_i2t)


# =====================================================================
# 3. Validation Loss & Validation Retrieval Evaluation
# =====================================================================
@torch.no_grad()
def evaluate_validation_loss(model: CLIPModel, val_loader: DataLoader, device: str, use_amp: bool):
    model.eval()
    total_loss = 0.0
    total_samples = 0

    for batch in tqdm(val_loader, desc="Computing Val Loss", unit="batch"):
        bsz = len(batch["query_ids"])
        with torch.cuda.amp.autocast(enabled=use_amp):
            loss = compute_contrastive_loss(model, batch, device)
        total_loss += float(loss.item()) * bsz
        total_samples += bsz

    return total_loss / max(1, total_samples)


@torch.no_grad()
def evaluate_validation_retrieval(
    model: CLIPModel,
    processor: CLIPProcessor,
    val_parquet: Path,
    images_dir: Path,
    *,
    device: str,
    batch_size: int,
    num_workers: int,
    max_samples=None,
):
    model.eval()
    val_df = pd.read_parquet(val_parquet)
    if max_samples is not None and max_samples > 0:
        available_pids = {p.stem for p in Path(images_dir).glob("*.jpg")}
        val_df = val_df[val_df["product_id"].astype(str).isin(available_pids)]
        val_df = val_df.iloc[: int(max_samples)].reset_index(drop=True)
        temp_parquet = (
            Path("/tmp/val_subset.parquet")
            if IS_SAGEMAKER
            else (PROJECT_ROOT / "data" / "processed" / "splits" / "_val_dryrun.parquet")
        )
        val_df.to_parquet(temp_parquet, index=False)
        parquet_to_use = temp_parquet
    else:
        parquet_to_use = val_parquet

    # Encode validation queries
    q_ds = QueryTextDataset(parquet_to_use)
    q_loader = DataLoader(
        q_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=text_collate(processor),
    )
    query_emb_map = {}
    for qids, batch in tqdm(q_loader, desc="Val: Encoding queries", unit="batch"):
        inputs = {k: v.to(device) for k, v in batch.items()}
        embs = F.normalize(as_tensor(model.get_text_features(**inputs)), dim=-1).cpu().numpy()
        for qid, e in zip(qids, embs, strict=True):
            query_emb_map[int(qid)] = e.astype(np.float32)

    # Encode validation images
    img_ds = ProductImageDataset(parquet_to_use, images_dir)
    img_loader = DataLoader(
        img_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=image_collate(processor),
    )
    img_emb_map = {}
    for pids, valid, batch in tqdm(img_loader, desc="Val: Encoding images", unit="batch"):
        inputs = {k: v.to(device) for k, v in batch.items()}
        embs = F.normalize(as_tensor(model.get_image_features(**inputs)), dim=-1).cpu().numpy()
        for pid, ok, e in zip(pids, valid, embs, strict=True):
            if ok:
                img_emb_map[str(pid)] = e.astype(np.float32)

    if max_samples is not None and max_samples > 0 and parquet_to_use.exists():
        parquet_to_use.unlink(missing_ok=True)

    if not img_emb_map:
        return {}

    # Filter val_df to products that have valid images on disk
    val_df_valid = val_df[val_df["product_id"].astype(str).isin(img_emb_map.keys())].reset_index(
        drop=True
    )
    global_metrics = evaluate_global_retrieval(val_df_valid, query_emb_map, img_emb_map)
    rerank_metrics = evaluate_candidate_reranking(val_df_valid, query_emb_map, img_emb_map)

    return {
        "global_retrieval": global_metrics,
        "candidate_reranking": rerank_metrics,
    }


# =====================================================================
# 4. Cloud Image Bundle Unpacking & S3 Checkpoint Sync
# =====================================================================
def ensure_images_ready(cfg: dict):
    """On SageMaker, downloads and extracts `images_bundle.tar` from S3 into `/tmp/images`."""
    if not IS_SAGEMAKER:
        return

    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    existing = sum(1 for _ in IMAGES_DIR.glob("*.jpg"))
    if existing > 10000:
        print(f"Found {existing:,} images already in {IMAGES_DIR}", flush=True)
        return

    # boto3 only ships in the SageMaker DLC image, so import it lazily.
    import boto3  # noqa: PLC0415

    bucket = cfg["s3_bucket"]
    tar_key = cfg.get("s3_images_tar_key", "data/images_bundle.tar")
    local_tar = Path("/tmp/images_bundle.tar")

    print(f"Downloading s3://{bucket}/{tar_key} to {local_tar} ...", flush=True)
    t0 = time.time()
    s3 = boto3.client("s3")
    s3.download_file(bucket, tar_key, str(local_tar))
    print(f"Downloaded in {time.time() - t0:.1f}s. Extracting to {IMAGES_DIR} ...", flush=True)

    t1 = time.time()
    with tarfile.open(local_tar, "r") as tar:
        tar.extractall(path=IMAGES_DIR)
    local_tar.unlink(missing_ok=True)
    total_extracted = sum(1 for _ in IMAGES_DIR.glob("*.jpg"))
    print(f"Extracted {total_extracted:,} images in {time.time() - t1:.1f}s!", flush=True)


def sync_directory_to_s3(local_dir: Path, s3_bucket: str, s3_prefix: str):
    """Uploads all files in `local_dir` to `s3://<s3_bucket>/<s3_prefix>/`."""
    if not IS_SAGEMAKER:
        return
    # boto3 only ships in the SageMaker DLC image, so import it lazily.
    import boto3  # noqa: PLC0415

    s3 = boto3.client("s3")
    for file_path in local_dir.rglob("*"):
        if file_path.is_file():
            rel = file_path.relative_to(local_dir).as_posix()
            s3_key = f"{s3_prefix.rstrip('/')}/{rel}"
            s3.upload_file(str(file_path), s3_bucket, s3_key)


# =====================================================================
# 5. Main Training Loop
# =====================================================================
def build_run_name(cfg: dict) -> str:
    model_short = (
        cfg["model_name"].split("/")[-1].replace("clip-vit-", "vit-").replace("patch", "p")
    )
    mode = cfg.get("training_mode", "lora")
    bs = cfg["batch_size"]
    lr = cfg["learning_rate"]
    ep = cfg["epochs"]
    if mode == "lora":
        r = cfg.get("lora_r", 16)
        a = cfg.get("lora_alpha", 32)
        return f"{model_short}_lora-r{r}-a{a}_bs{bs}_lr{lr:.0e}_ep{ep}"
    return f"{model_short}_{mode}_bs{bs}_lr{lr:.0e}_ep{ep}"


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_env_placeholders(cfg: dict) -> dict:
    """Replace ${VAR} placeholders in config values with os.environ[VAR].

    Allows keeping secrets (e.g. S3_BUCKET) out of version-controlled YAML.
    Missing env vars raise a clear error so misconfiguration is loud, not silent.
    """
    pattern = re.compile(r"\$\{([A-Z_][A-Z0-9_]*)\}")

    def _walk(node):
        if isinstance(node, dict):
            return {k: _walk(v) for k, v in node.items()}
        if isinstance(node, list):
            return [_walk(v) for v in node]
        if isinstance(node, str):
            missing = []

            def repl(m):
                var = m.group(1)
                val = os.environ.get(var)
                if val is None:
                    missing.append(var)
                return val if val is not None else m.group(0)

            resolved = pattern.sub(repl, node)
            if missing:
                raise RuntimeError(
                    f"Config references env var(s) {missing} but they are not set. "
                    "Set them in your shell or .env (which is gitignored)."
                )
            return resolved
        return node

    return _walk(cfg)


def main():  # noqa: PLR0912, PLR0915  (linear training orchestration)
    parser = argparse.ArgumentParser(description="Fine-tune CLIP for Image Retrieval")
    parser.add_argument("--config", type=str, default=str(PROJECT_ROOT / "config.yaml"))
    args, _ = parser.parse_known_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Resolve ${ENV_VAR} placeholders (e.g. s3_bucket) from os.environ.
    cfg = _resolve_env_placeholders(cfg)

    set_seed(int(cfg.get("seed", 42)))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = bool(cfg.get("mixed_precision", True)) and (device == "cuda")
    num_workers = int(cfg.get("num_workers", 4)) if device == "cuda" else 0

    run_name = build_run_name(cfg)
    run_dir = CHECKPOINTS_ROOT / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 65, flush=True)
    print(f"  Experiment Run : {run_name}", flush=True)
    print(f"  Model          : {cfg['model_name']}", flush=True)
    print(f"  Device         : {device} (AMP={use_amp})", flush=True)
    print(f"  Checkpoints Dir: {run_dir}", flush=True)
    print("=" * 65, flush=True)

    # Save exact config at run root
    with open(run_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    # Ensure images are unpacked on cloud
    ensure_images_ready(cfg)

    # Load model & processor
    model = CLIPModel.from_pretrained(cfg["model_name"])
    processor = CLIPProcessor.from_pretrained(cfg["model_name"])
    configure_trainable_parameters(model, cfg)
    model.to(device)

    # Build Datasets & DataLoaders
    train_ds = CLIPDataset(
        TRAIN_PARQUET,
        IMAGES_DIR,
        positive_only=bool(cfg.get("positive_only", True)),
        require_existing_images=True,
        max_samples=cfg.get("max_train_samples"),
    )
    val_loss_ds = CLIPDataset(
        VAL_PARQUET,
        IMAGES_DIR,
        positive_only=True,
        require_existing_images=True,
        max_samples=cfg.get("max_val_samples"),
    )
    print(f"Train samples (existing images): {len(train_ds):,}", flush=True)
    print(f"Val loss samples (existing images): {len(val_loss_ds):,}", flush=True)

    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg["batch_size"]),
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
        drop_last=(len(train_ds) > int(cfg["batch_size"])),
        collate_fn=pair_collate(processor),
    )
    val_loss_loader = DataLoader(
        val_loss_ds,
        batch_size=int(cfg.get("eval_batch_size", cfg["batch_size"])),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
        collate_fn=pair_collate(processor),
    )

    epochs = int(cfg["epochs"])
    total_steps = max(1, len(train_loader) * epochs)
    warmup_steps = int(total_steps * float(cfg.get("warmup_ratio", 0.1)))

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=float(cfg["learning_rate"]),
        weight_decay=float(cfg.get("weight_decay", 0.01)),
    )

    def lr_lambda(current_step: int):
        if current_step < warmup_steps:
            return float(current_step + 1) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_metric_name = cfg.get("best_metric", "val_loss")
    best_metric_val = float("inf") if "loss" in best_metric_name else float("-inf")
    history = []

    global_step = 0
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_start = time.time()
        running_loss = 0.0
        n_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [Train]", unit="batch")
        for batch in pbar:
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                loss = compute_contrastive_loss(model, batch, device)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable_params, float(cfg.get("max_grad_norm", 1.0)))
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            global_step += 1
            running_loss += float(loss.item())
            n_batches += 1
            pbar.set_postfix(
                loss=f"{running_loss / n_batches:.4f}",
                lr=f"{scheduler.get_last_lr()[0]:.2e}",
            )

        train_loss = running_loss / max(1, n_batches)

        # 1. Compute Validation Loss
        val_loss = None
        if cfg.get("compute_val_loss_every_epoch", True) and len(val_loss_ds) > 0:
            val_loss = evaluate_validation_loss(model, val_loss_loader, device, use_amp)

        # 2. Compute Validation Retrieval Metrics
        val_ret = {}
        if cfg.get("compute_val_retrieval_every_epoch", True):
            val_ret = evaluate_validation_retrieval(
                model=model,
                processor=processor,
                val_parquet=VAL_PARQUET,
                images_dir=IMAGES_DIR,
                device=device,
                batch_size=int(cfg.get("eval_batch_size", cfg["batch_size"])),
                num_workers=num_workers,
                max_samples=cfg.get("max_val_samples"),
            )

        val_global = val_ret.get("global_retrieval", {})
        val_rec10 = val_global.get("Recall@10", 0.0)
        val_ndcg10 = val_global.get("NDCG@10", 0.0)
        val_mrr = val_global.get("MRR", 0.0)

        epoch_record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "overfit_gap": (val_loss - train_loss) if val_loss is not None else None,
            "learning_rate": scheduler.get_last_lr()[0],
            "elapsed_sec": round(time.time() - epoch_start, 2),
            "val_global_retrieval": val_global,
            "val_candidate_reranking": val_ret.get("candidate_reranking", {}),
        }
        history.append(epoch_record)

        print(
            f"\n[Epoch {epoch}/{epochs}] "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss if val_loss is not None else float('nan'):.4f} | "
            f"val_Recall@10={val_rec10:.4f} | "
            f"val_NDCG@10={val_ndcg10:.4f} | "
            f"val_MRR={val_mrr:.4f}",
            flush=True,
        )

        # 3. Save Epoch Checkpoint with Descriptive Name
        if cfg.get("save_every_epoch", True):
            vloss_str = f"{val_loss:.4f}" if val_loss is not None else "na"
            ep_folder_name = (
                f"epoch_{epoch:02d}_{cfg['training_mode']}_"
                f"bs{cfg['batch_size']}_lr{cfg['learning_rate']:.0e}_"
                f"tloss{train_loss:.4f}_vloss{vloss_str}_rec10-{val_rec10:.4f}"
            )
            ep_dir = run_dir / ep_folder_name
            ckpt_path = save_checkpoint(model, cfg, epoch_record, ep_dir)
            print(f"Saved epoch checkpoint -> {ckpt_path}", flush=True)

        # 4. Track & Save Best Model
        if best_metric_name == "val_loss" and val_loss is not None:
            current_val = val_loss
            is_better = current_val < best_metric_val
        elif best_metric_name == "val_Recall@10":
            current_val = val_rec10
            is_better = current_val > best_metric_val
        else:
            current_val = val_ndcg10
            is_better = current_val > best_metric_val

        if is_better:
            best_metric_val = current_val
            best_dir = run_dir / "best_model"
            save_checkpoint(model, cfg, epoch_record, best_dir)
            print(
                f"*** New best model ({best_metric_name}={best_metric_val:.4f}) saved to {best_dir} ***",
                flush=True,
            )

        # 5. Save Training History JSON & Sync to S3
        history_payload = {
            "run_name": run_name,
            "config": cfg,
            "best_metric": best_metric_name,
            "best_metric_value": best_metric_val,
            "epochs": history,
        }
        (run_dir / "training_history.json").write_text(json.dumps(history_payload, indent=4))
        (OUT_DIR / "training_history.json").write_text(json.dumps(history_payload, indent=4))

        if IS_SAGEMAKER:
            # Versioned S3 layout:
            #   s3://{bucket}/checkpoints/{run_name}/  -> every epoch (debug)
            #   s3://{bucket}/models/{version}/        -> best_model only (registry)
            s3_prefix = f"{cfg.get('s3_checkpoints_prefix', 'checkpoints')}/{run_name}"
            sync_directory_to_s3(run_dir, cfg["s3_bucket"], s3_prefix)
            print(f"Synced checkpoints to s3://{cfg['s3_bucket']}/{s3_prefix}/", flush=True)

            # Copy the best model under the versioned models/ prefix
            model_version = os.environ.get("MODEL_VERSION", cfg.get("model_version", "v0.0.0"))
            if model_version != "v0.0.0" and best_dir.exists():
                models_prefix = f"models/{model_version}"
                sync_directory_to_s3(best_dir, cfg["s3_bucket"], models_prefix)
                print(f"Best model synced to s3://{cfg['s3_bucket']}/{models_prefix}/", flush=True)

    # 6. Final Evaluation of Best Model on Test Split
    best_ckpt = run_dir / "best_model" / "checkpoint.pt"
    if best_ckpt.exists() and TEST_PARQUET.exists() and cfg.get("max_train_samples") is None:
        print("\nEvaluating best_model on TEST split ...", flush=True)
        ckpt_data = torch.load(best_ckpt, map_location="cpu")
        model.load_state_dict(ckpt_data["state_dict"], strict=False)
        model.eval()
        test_ret = evaluate_validation_retrieval(
            model=model,
            processor=processor,
            val_parquet=TEST_PARQUET,
            images_dir=IMAGES_DIR,
            device=device,
            batch_size=int(cfg.get("eval_batch_size", cfg["batch_size"])),
            num_workers=num_workers,
            max_samples=None,
        )
        test_results = {
            "run_name": run_name,
            "best_epoch": ckpt_data["metrics"]["epoch"],
            "model": cfg["model_name"],
            "global_vector_retrieval": test_ret.get("global_retrieval", {}),
            "candidate_reranking": test_ret.get("candidate_reranking", {}),
        }
        (run_dir / "finetuned_test_results.json").write_text(json.dumps(test_results, indent=4))
        (OUT_DIR / "finetuned_test_results.json").write_text(json.dumps(test_results, indent=4))
        print("\n=== FINAL TEST SPLIT (GLOBAL RETRIEVAL) ===", flush=True)
        for k, v in test_results["global_vector_retrieval"].items():
            print(f"{k:>28}: {v:.4f}" if isinstance(v, float) else f"{k:>28}: {v}", flush=True)
        if IS_SAGEMAKER:
            s3_prefix = f"{cfg.get('s3_checkpoints_prefix', 'checkpoints')}/{run_name}"
            sync_directory_to_s3(run_dir, cfg["s3_bucket"], s3_prefix)

    print("\nTraining completed successfully!", flush=True)


if __name__ == "__main__":
    main()
