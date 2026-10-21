"""Unit tests for scripts/bump_version.py — semver logic + config diff detection."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

# scripts/ dizinini path'e ekle
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from bump_version import (
    bump,
    config_hash,
    detect_bump_type,
    format_version,
    parse_version,
)


# =====================================================================
# parse_version + format_version
# =====================================================================
class TestParseVersion:
    def test_valid(self):
        assert parse_version("v1.2.3") == (1, 2, 3)
        assert parse_version("v0.0.0") == (0, 0, 0)
        assert parse_version("v10.20.30") == (10, 20, 30)

    def test_without_v_prefix(self):
        assert parse_version("1.2.3") == (1, 2, 3)

    def test_invalid_format(self):
        with pytest.raises(ValueError, match="Invalid version format"):
            parse_version("1.2")
        with pytest.raises(ValueError, match="Invalid version format"):
            parse_version("v1.2.3.4")

    def test_invalid_components(self):
        with pytest.raises(ValueError, match="Invalid version components"):
            parse_version("v1.2.x")


class TestFormatVersion:
    def test_basic(self):
        assert format_version(1, 2, 3) == "v1.2.3"
        assert format_version(0, 0, 0) == "v0.0.0"


# =====================================================================
# bump
# =====================================================================
class TestBump:
    def test_major_resets_minor_patch(self):
        assert bump("v1.2.3", "major") == "v2.0.0"

    def test_minor_resets_patch(self):
        assert bump("v1.2.3", "minor") == "v1.3.0"

    def test_patch_increments(self):
        assert bump("v1.2.3", "patch") == "v1.2.4"

    def test_unknown_type_raises(self):
        with pytest.raises(ValueError, match="Unknown bump type"):
            bump("v1.2.3", "invalid")


# =====================================================================
# detect_bump_type
# =====================================================================
class TestDetectBumpType:
    BASE_CFG: dict = {
        "model_name": "openai/clip-vit-large-patch14",
        "training_mode": "lora",
        "loss_type": "infonce",
        "learning_rate": 1e-4,
        "batch_size": 64,
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "num_epochs": 5,
        "seed": 42,
    }

    def test_no_change_returns_patch(self):
        assert detect_bump_type(self.BASE_CFG, dict(self.BASE_CFG)) == "patch"

    def test_model_name_change_is_major(self):
        new_cfg = dict(self.BASE_CFG)
        new_cfg["model_name"] = "openai/clip-vit-base-patch32"
        assert detect_bump_type(self.BASE_CFG, new_cfg) == "major"

    def test_training_mode_change_is_major(self):
        new_cfg = dict(self.BASE_CFG)
        new_cfg["training_mode"] = "full"
        assert detect_bump_type(self.BASE_CFG, new_cfg) == "major"

    def test_loss_type_change_is_major(self):
        new_cfg = dict(self.BASE_CFG)
        new_cfg["loss_type"] = "triplet"
        assert detect_bump_type(self.BASE_CFG, new_cfg) == "major"

    def test_learning_rate_change_is_minor(self):
        new_cfg = dict(self.BASE_CFG)
        new_cfg["learning_rate"] = 1e-5
        assert detect_bump_type(self.BASE_CFG, new_cfg) == "minor"

    def test_batch_size_change_is_minor(self):
        new_cfg = dict(self.BASE_CFG)
        new_cfg["batch_size"] = 32
        assert detect_bump_type(self.BASE_CFG, new_cfg) == "minor"

    def test_lora_r_change_is_minor(self):
        new_cfg = dict(self.BASE_CFG)
        new_cfg["lora_r"] = 32
        assert detect_bump_type(self.BASE_CFG, new_cfg) == "minor"

    def test_lora_alpha_change_is_minor(self):
        new_cfg = dict(self.BASE_CFG)
        new_cfg["lora_alpha"] = 64
        assert detect_bump_type(self.BASE_CFG, new_cfg) == "minor"

    def test_num_epochs_change_is_minor(self):
        new_cfg = dict(self.BASE_CFG)
        new_cfg["num_epochs"] = 10
        assert detect_bump_type(self.BASE_CFG, new_cfg) == "minor"

    def test_seed_change_is_patch(self):
        new_cfg = dict(self.BASE_CFG)
        new_cfg["seed"] = 123
        assert detect_bump_type(self.BASE_CFG, new_cfg) == "patch"

    def test_major_takes_precedence_over_minor(self):
        # Hem model_name hem lr değişirse → major
        new_cfg = dict(self.BASE_CFG)
        new_cfg["model_name"] = "different"
        new_cfg["learning_rate"] = 1e-5
        assert detect_bump_type(self.BASE_CFG, new_cfg) == "major"


# =====================================================================
# config_hash
# =====================================================================
class TestConfigHash:
    def test_same_config_same_hash(self):
        cfg = {"a": 1, "b": 2}
        assert config_hash(cfg) == config_hash(cfg)

    def test_different_config_different_hash(self):
        assert config_hash({"a": 1}) != config_hash({"a": 2})

    def test_key_order_doesnt_matter(self):
        # sort_keys=True olduğu için sıra farketmez
        assert config_hash({"a": 1, "b": 2}) == config_hash({"b": 2, "a": 1})

    def test_hash_format(self):
        h = config_hash({"a": 1})
        assert h.startswith("sha256:")
        assert len(h) == len("sha256:") + 16  # 16 hex chars


# =====================================================================
# CLI integration (subprocess)
# =====================================================================
class TestCLI:
    PROJECT_ROOT = Path(__file__).resolve().parent.parent

    def _run(self, *args, registry_path):
        """Run bump_version.py with a custom --registry path."""
        return subprocess.run(
            [sys.executable, "scripts/bump_version.py", *args, "--registry", str(registry_path)],
            capture_output=True,
            text=True,
            cwd=self.PROJECT_ROOT,
            check=False,
        )

    def test_current_no_registry(self, tmp_path):
        """--current returns v0.0.0 when no registry exists."""
        fake_registry = tmp_path / "registry.json"
        result = self._run("--current", registry_path=fake_registry)
        assert result.returncode == 0
        assert result.stdout.strip() == "v0.0.0"

    def test_auto_first_model_is_v1_0_0(self, tmp_path):
        """--auto on empty registry returns v1.0.0 (initial)."""
        fake_registry = tmp_path / "registry.json"
        result = self._run("--auto", registry_path=fake_registry)
        assert result.returncode == 0, result.stderr
        assert "version=v1.0.0" in result.stdout
        assert "bump_type=initial" in result.stdout

    def test_bump_patch_increments(self, tmp_path):
        """--bump patch increments patch number."""
        fake_registry = tmp_path / "registry.json"
        fake_registry.write_text(
            json.dumps(
                {
                    "schema_version": "1.0",
                    "models": [{"version": "v1.0.0", "config_snapshot": {}}],
                    "production": None,
                }
            )
        )
        result = self._run("--bump", "patch", registry_path=fake_registry)
        assert result.returncode == 0, result.stderr
        assert "version=v1.0.1" in result.stdout
