"""test_smoke.py

End-to-end smoke tests using the smallest available CLIP model.

These tests verify that the full pipeline (model load -> forward -> loss ->
backward -> optimizer step) works on CPU with a tiny model. They are slow
compared to pure unit tests but fast enough for CI (~10-30s).

Marked with @pytest.mark.smoke so they can be skipped with `-m "not smoke"`.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from train import LoRALinear, compute_contrastive_loss, inject_lora


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _random_pil(rng: np.random.Generator, size=(32, 32)) -> Image.Image:
    arr = rng.integers(0, 255, size=(size[0], size[1], 3), dtype=np.uint8)
    return Image.fromarray(arr, mode="RGB")


# ---------------------------------------------------------------------------
# Tiny CLIP load + forward
# ---------------------------------------------------------------------------
@pytest.mark.smoke
def test_tiny_clip_loads_and_forward_passes(cpu_device):
    """Load the smallest CLIP variant and run a forward pass on CPU."""
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model.eval()

    rng = np.random.default_rng(0)
    img = _random_pil(rng)
    inputs = processor(
        text=["a photo of a cat", "a photo of a dog"],
        images=[img, img],
        return_tensors="pt",
        padding=True,
    )

    with torch.no_grad():
        outputs = model(**inputs)

    # logits_per_image shape: (num_images, num_texts)
    assert outputs.logits_per_image.shape == (2, 2)
    assert torch.isfinite(outputs.logits_per_image).all()


@pytest.mark.smoke
def test_tiny_clip_text_image_similarity_is_finite(cpu_device):
    """Text and image embeddings must be L2-normalizable and finite."""
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model.eval()

    rng = np.random.default_rng(1)
    img = _random_pil(rng)
    inputs = processor(
        text=["hello world"],
        images=[img],
        return_tensors="pt",
        padding=True,
    )

    with torch.no_grad():
        text_emb = model.get_text_features(input_ids=inputs["input_ids"])
        img_emb = model.get_image_features(pixel_values=inputs["pixel_values"])

    text_emb = F.normalize(text_emb, dim=-1)
    img_emb = F.normalize(img_emb, dim=-1)

    assert torch.isfinite(text_emb).all()
    assert torch.isfinite(img_emb).all()
    # Cosine similarity in [-1, 1].
    sim = (text_emb @ img_emb.t()).item()
    assert -1.0 <= sim <= 1.0


# ---------------------------------------------------------------------------
# LoRA injection on real CLIP
# ---------------------------------------------------------------------------
@pytest.mark.smoke
def test_lora_injection_on_tiny_clip(cpu_device):
    """inject_lora must work on a real (tiny) CLIP model without errors."""
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

    # Inject LoRA into the standard CLIP attention projections.
    inject_lora(
        model,
        target_modules=["q_proj", "v_proj"],
        r=4,
        alpha=8,
        dropout=0.0,
    )

    # Count how many LoRALinear wrappers we created.
    n_lora = sum(1 for _m in model.modules() if isinstance(_m, LoRALinear))
    assert n_lora > 0, "inject_lora should have wrapped at least one layer"

    # Forward pass must still work.
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    rng = np.random.default_rng(2)
    img = _random_pil(rng)
    inputs = processor(
        text=["a test sentence"],
        images=[img],
        return_tensors="pt",
        padding=True,
    )

    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
    assert outputs.logits_per_image.shape == (1, 1)


# ---------------------------------------------------------------------------
# Full training step (forward + backward + optimizer)
# ---------------------------------------------------------------------------
@pytest.mark.smoke
def test_one_training_step_runs(cpu_device):
    """A single forward+backward+optimizer step must complete without error."""
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    inject_lora(
        model,
        target_modules=["q_proj", "v_proj"],
        r=4,
        alpha=8,
        dropout=0.0,
    )

    # Freeze everything except LoRA params.
    for p in model.parameters():
        p.requires_grad = False
    for m in model.modules():
        if isinstance(m, LoRALinear):
            m.lora_A.requires_grad = True
            m.lora_B.requires_grad = True

    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=1e-3)

    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    rng = np.random.default_rng(3)
    img = _random_pil(rng)
    inputs = processor(
        text=["red shoes", "blue hat", "green bag"],
        images=[img, img, img],
        return_tensors="pt",
        padding=True,
    )
    batch = {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "pixel_values": inputs["pixel_values"],
        "query_ids": [1, 2, 3],
        "product_ids": ["p1", "p2", "p3"],
    }

    model.train()
    optimizer.zero_grad()
    loss = compute_contrastive_loss(model, batch, cpu_device)
    assert torch.isfinite(loss)
    loss.backward()

    # At least one LoRA param must have a non-None grad.
    grads = [m.lora_A.grad for m in model.modules() if isinstance(m, LoRALinear)]
    assert any(g is not None and torch.isfinite(g).all() for g in grads), (
        "LoRA params should receive gradients"
    )

    optimizer.step()
