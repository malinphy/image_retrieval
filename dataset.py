"""dataset.py

PyTorch Datasets and collate helpers shared by evaluation and training.

- CLIPDataset           : row-level (query, product image, label) pairs
- ProductImageDataset   : unique products only (image encoding)
- QueryTextDataset      : unique queries only (text encoding)
"""

from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


def load_image(images_dir, product_id):
    path = Path(images_dir) / f"{product_id}.jpg"
    if not path.exists():
        return None
    try:
        return Image.open(path).convert("RGB")
    except Exception:
        return None


class CLIPDataset(Dataset):
    """Row-level (query, product) pairs. Used for training / fine-tuning."""

    def __init__(
        self,
        parquet_path,
        images_dir,
        positive_only=True,
        require_existing_images=True,
        max_samples=None,
    ):
        df = pd.read_parquet(parquet_path)
        if positive_only:
            df = df[df["esci_label"] == "E"]

        images_dir = Path(images_dir)
        if require_existing_images and images_dir.exists():
            available_pids = {p.stem for p in images_dir.glob("*.jpg")}
            df = df[df["product_id"].astype(str).isin(available_pids)]

        df = df.reset_index(drop=True)
        if max_samples is not None and max_samples > 0:
            df = df.iloc[: int(max_samples)].reset_index(drop=True)

        self.df = df
        self.images_dir = images_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        return {
            "query_id": int(row["query_id"]),
            "query": str(row["query"]),
            "product_id": str(row["product_id"]),
            "label": str(row["esci_label"]),
            "image": load_image(self.images_dir, row["product_id"]),
        }


class ProductImageDataset(Dataset):
    """Unique products only (for encoding images once during evaluation)."""

    def __init__(self, parquet_path, images_dir):
        self.df = (
            pd.read_parquet(parquet_path, columns=["product_id"])
            .drop_duplicates()
            .reset_index(drop=True)
        )
        self.images_dir = images_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        pid = self.df.iloc[idx]["product_id"]
        return pid, load_image(self.images_dir, pid)


class QueryTextDataset(Dataset):
    """Unique queries only (for encoding texts once during evaluation)."""

    def __init__(self, parquet_path):
        self.df = (
            pd.read_parquet(parquet_path, columns=["query_id", "query"])
            .groupby("query_id")["query"]
            .first()
            .reset_index()
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        return int(row["query_id"]), row["query"]


def _blank():
    return Image.new("RGB", (224, 224))


def text_collate(processor):
    def collate(batch):
        qids, texts = zip(*batch, strict=False)
        inputs = processor(text=list(texts), return_tensors="pt", padding=True, truncation=True)
        return list(qids), inputs

    return collate


def image_collate(processor):
    def collate(batch):
        pids = [b[0] for b in batch]
        valid = [b[1] is not None for b in batch]
        images = [b[1] if b[1] is not None else _blank() for b in batch]
        inputs = processor(images=images, return_tensors="pt")
        return pids, valid, inputs

    return collate


def pair_collate(processor):
    """Collate helper for training/validation (query, image) positive pairs."""

    def collate(batch):
        valid_items = [b for b in batch if b["image"] is not None]
        if not valid_items:
            valid_items = batch
            images = [_blank() for _ in valid_items]
        else:
            images = [b["image"] for b in valid_items]

        texts = [b["query"] for b in valid_items]
        qids = [b["query_id"] for b in valid_items]
        pids = [b["product_id"] for b in valid_items]

        text_inputs = processor(text=texts, return_tensors="pt", padding=True, truncation=True)
        image_inputs = processor(images=images, return_tensors="pt")

        return {
            "query_ids": qids,
            "product_ids": pids,
            "input_ids": text_inputs["input_ids"],
            "attention_mask": text_inputs["attention_mask"],
            "pixel_values": image_inputs["pixel_values"],
        }

    return collate
