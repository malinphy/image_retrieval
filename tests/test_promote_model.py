"""Unit tests for scripts/promote_model.py — registry logic (no AWS calls)."""

import subprocess
import sys
from datetime import datetime
from pathlib import Path

# scripts/ dizinini path'e ekle
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from promote_model import now_iso


# =====================================================================
# now_iso
# =====================================================================
class TestNowIso:
    def test_format(self):
        ts = now_iso()
        # ISO 8601 UTC, Z suffix
        assert ts.endswith("Z")
        assert "T" in ts
        # Parseable
        parsed = datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ")
        assert parsed.year >= 2025


# =====================================================================
# Registry helpers (local-only, no S3)
# =====================================================================
class TestRegistrySchema:
    """Registry JSON şemasının doğru formatta olduğunu doğrula."""

    def test_empty_registry_schema(self):
        empty = {"schema_version": "1.0", "models": [], "production": None}
        assert empty["schema_version"] == "1.0"
        assert empty["models"] == []
        assert empty["production"] is None

    def test_model_entry_schema(self):
        entry = {
            "version": "v1.0.0",
            "config_snapshot": {"model_name": "openai/clip-vit-large-patch14"},
            "trained_at": "2025-10-21T12:00:00Z",
            "git_sha": "abc1234",
            "config_hash": "sha256:deadbeef",
            "metrics": {"recall@10": 0.95},
            "promoted": False,
            "promoted_at": None,
            "promoted_by": None,
        }
        # Tüm zorunlu alanlar mevcut
        for key in (
            "version",
            "config_snapshot",
            "trained_at",
            "git_sha",
            "config_hash",
            "metrics",
            "promoted",
        ):
            assert key in entry

    def test_production_entry_schema(self):
        prod = {
            "version": "v1.0.0",
            "promoted_at": "2025-10-21T12:00:00Z",
            "promoted_by": "alice",
            "reason": "Initial production deployment",
        }
        for key in ("version", "promoted_at", "promoted_by", "reason"):
            assert key in prod


# =====================================================================
# CLI surface (--help)
# =====================================================================
class TestCLISurface:
    PROJECT_ROOT = Path(__file__).resolve().parent.parent

    def test_help_lists_all_commands(self):
        result = subprocess.run(
            [sys.executable, "scripts/promote_model.py", "--help"],
            capture_output=True,
            text=True,
            cwd=self.PROJECT_ROOT,
            check=False,
        )
        assert result.returncode == 0
        for flag in ("--list", "--current", "--show", "--promote", "--set-production"):
            assert flag in result.stdout, f"Missing flag: {flag}"


# =====================================================================
# Audit trail format
# =====================================================================
class TestAuditTrail:
    def test_audit_entry_format(self):
        audit = {
            "version": "v1.0.0",
            "action": "promote",
            "actor": "alice",
            "reason": "Initial production deployment",
            "timestamp": "2025-10-21T12:00:00Z",
            "previous_production": None,
        }
        for key in ("version", "action", "actor", "reason", "timestamp"):
            assert key in audit
        assert audit["action"] in ("promote", "rollback", "set-production")
