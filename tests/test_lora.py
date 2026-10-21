"""test_lora.py

Unit tests for the LoRA implementation in train.py.

We verify the invariants that make LoRA correct and safe:
  1. At initialization, lora_B == 0 so the wrapped layer behaves identically
     to the base layer (no perturbation at step 0).
  2. Base layer weights are frozen (requires_grad=False).
  3. LoRA parameters (lora_A, lora_B) are trainable (requires_grad=True).
  4. inject_lora replaces ONLY the target modules, leaving others untouched.
  5. Forward pass output shape matches the base layer.
  6. Round-trip: applying LoRA twice with different r/alpha gives different
     parameter counts but same forward shape.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from train import LoRALinear, inject_lora


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_linear(in_f=8, out_f=4):
    layer = nn.Linear(in_f, out_f, bias=True)
    # Deterministic weights for reproducibility.
    torch.manual_seed(0)
    layer.weight.data = torch.randn(out_f, in_f)
    if layer.bias is not None:
        layer.bias.data = torch.randn(out_f)
    return layer


# ---------------------------------------------------------------------------
# Initialization invariants
# ---------------------------------------------------------------------------
def test_lora_B_initialized_to_zero():
    """lora_B must be all zeros at init so the adapter is a no-op."""
    base = _make_linear()
    lora = LoRALinear(base, r=4, alpha=8, dropout=0.0)

    assert torch.allclose(lora.lora_B, torch.zeros_like(lora.lora_B)), (
        "lora_B must be initialized to zeros"
    )


def test_lora_A_initialized_kaiming():
    """lora_A must be non-zero (Kaiming init) so gradients can flow."""
    base = _make_linear()
    lora = LoRALinear(base, r=4, alpha=8, dropout=0.0)

    assert lora.lora_A.abs().sum() > 0, "lora_A must be non-zero"


def test_lora_at_init_matches_base_layer():
    """At init (lora_B=0), LoRALinear(x) == base_layer(x) exactly."""
    base = _make_linear()
    lora = LoRALinear(base, r=4, alpha=8, dropout=0.0)

    x = torch.randn(3, 8)
    base_out = base(x)
    lora_out = lora(x)
    assert torch.allclose(base_out, lora_out, atol=1e-6), (
        "LoRA at init must produce identical output to base layer"
    )


# ---------------------------------------------------------------------------
# Parameter freezing
# ---------------------------------------------------------------------------
def test_base_layer_weights_frozen():
    """Base layer weights and bias must have requires_grad=False."""
    base = _make_linear()
    LoRALinear(base, r=4, alpha=8, dropout=0.0)

    assert base.weight.requires_grad is False
    if base.bias is not None:
        assert base.bias.requires_grad is False


def test_lora_params_trainable():
    """lora_A and lora_B must have requires_grad=True."""
    base = _make_linear()
    lora = LoRALinear(base, r=4, alpha=8, dropout=0.0)

    assert lora.lora_A.requires_grad is True
    assert lora.lora_B.requires_grad is True


def test_scaling_is_alpha_over_r():
    """scaling = alpha / r (standard LoRA convention)."""
    base = _make_linear()
    lora = LoRALinear(base, r=8, alpha=32, dropout=0.0)
    assert lora.scaling == pytest.approx(32 / 8)


# ---------------------------------------------------------------------------
# inject_lora behavior
# ---------------------------------------------------------------------------
def test_inject_lora_replaces_only_targets():
    """inject_lora must replace ONLY modules whose name is in target_modules."""
    # Build a tiny model with named children.
    model = nn.ModuleDict(
        {
            "q_proj": nn.Linear(8, 8),
            "k_proj": nn.Linear(8, 8),
            "v_proj": nn.Linear(8, 8),
            "out_proj": nn.Linear(8, 8),
            "mlp": nn.Linear(8, 8),  # NOT a target
        }
    )

    inject_lora(model, target_modules=["q_proj", "v_proj"], r=4, alpha=8, dropout=0.0)

    assert isinstance(model["q_proj"], LoRALinear)
    assert isinstance(model["v_proj"], LoRALinear)
    # Untouched modules remain plain Linear.
    assert isinstance(model["k_proj"], nn.Linear) and not isinstance(model["k_proj"], LoRALinear)
    assert isinstance(model["out_proj"], nn.Linear) and not isinstance(
        model["out_proj"], LoRALinear
    )
    assert isinstance(model["mlp"], nn.Linear) and not isinstance(model["mlp"], LoRALinear)


def test_inject_lora_handles_nested_modules():
    """inject_lora must recurse into nested children (e.g. CLIP attention blocks)."""
    block = nn.Module()
    block.q_proj = nn.Linear(8, 8)
    block.k_proj = nn.Linear(8, 8)
    block.other = nn.Linear(8, 8)

    model = nn.Module()
    model.block = block

    inject_lora(model, target_modules=["q_proj"], r=4, alpha=8, dropout=0.0)

    assert isinstance(model.block.q_proj, LoRALinear)
    assert isinstance(model.block.k_proj, nn.Linear) and not isinstance(
        model.block.k_proj, LoRALinear
    )
    assert isinstance(model.block.other, nn.Linear) and not isinstance(
        model.block.other, LoRALinear
    )


# ---------------------------------------------------------------------------
# Forward pass
# ---------------------------------------------------------------------------
def test_forward_shape_matches_base():
    """LoRALinear forward output shape must match base layer output shape."""
    base = _make_linear(in_f=16, out_f=8)
    lora = LoRALinear(base, r=4, alpha=8, dropout=0.0)

    x = torch.randn(5, 16)
    out = lora(x)
    assert out.shape == (5, 8)


def test_forward_with_dropout_changes_output():
    """With dropout > 0 and lora_B != 0, output differs from base layer."""
    base = _make_linear()
    lora = LoRALinear(base, r=4, alpha=8, dropout=0.5)

    # Force lora_B to be non-zero to make the adapter active.
    with torch.no_grad():
        lora.lora_B.data = torch.randn_like(lora.lora_B.data)

    x = torch.randn(3, 8)
    base_out = base(x)
    lora_out = lora(x)
    # With dropout=0.5 and non-zero lora_B, outputs should differ.
    assert not torch.allclose(base_out, lora_out, atol=1e-6)


# ---------------------------------------------------------------------------
# Round-trip / parameter count
# ---------------------------------------------------------------------------
def test_lora_parameter_count_scales_with_r():
    """Doubling r should roughly double the LoRA parameter count."""
    base_small = _make_linear(in_f=64, out_f=64)
    base_large = _make_linear(in_f=64, out_f=64)

    lora_small = LoRALinear(base_small, r=4, alpha=8, dropout=0.0)
    lora_large = LoRALinear(base_large, r=8, alpha=16, dropout=0.0)

    n_small = lora_small.lora_A.numel() + lora_small.lora_B.numel()
    n_large = lora_large.lora_A.numel() + lora_large.lora_B.numel()

    # r=4: A=(4,64)=256, B=(64,4)=256 -> 512
    # r=8: A=(8,64)=512, B=(64,8)=512 -> 1024
    assert n_small == 512
    assert n_large == 1024
    assert n_large == 2 * n_small
