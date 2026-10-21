"""bump_version.py — Model version management (semver).

Reads the latest version from the registry, then bumps it based on the config
diff or assigns an explicit version.

Usage:
    # Automatic bump (based on the config diff)
    python scripts/bump_version.py --auto

    # Manual bump
    python scripts/bump_version.py --bump patch
    python scripts/bump_version.py --bump minor
    python scripts/bump_version.py --bump major

    # Explicit set
    python scripts/bump_version.py --set v2.0.0

    # Show the current version
    python scripts/bump_version.py --current

Output: GitHub Actions output format (key=value)
    version=v1.2.4
    bump_type=patch
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REGISTRY_PATH = PROJECT_ROOT / "models" / "registry.json"
S3_REGISTRY_KEY = "models/registry.json"


def now_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


# Key-name differences between config.yaml and older registry entries
KEY_ALIASES = {"epochs": "num_epochs"}


def normalize_cfg(cfg: dict | None) -> dict:
    """Normalise config key names (`epochs` -> `num_epochs`)."""
    out = dict(cfg or {})
    for alias, canonical in KEY_ALIASES.items():
        if alias in out and canonical not in out:
            out[canonical] = out.pop(alias)
    return out


def config_hash(cfg: dict) -> str:
    """SHA-256 hash of the config (sorted keys, no comments)."""
    canonical = yaml.safe_dump(cfg, sort_keys=True, default_flow_style=False)
    return "sha256:" + hashlib.sha256(canonical.encode()).hexdigest()[:16]


def parse_version(version: str) -> tuple[int, int, int]:
    """v1.2.3 -> (1, 2, 3). Raises ValueError on invalid format."""
    v = version.strip().lstrip("v")
    parts = v.split(".")
    if len(parts) != 3:
        raise ValueError(f"Invalid version format: {version} (expected vMAJOR.MINOR.PATCH)")
    try:
        return int(parts[0]), int(parts[1]), int(parts[2])
    except ValueError as e:
        raise ValueError(f"Invalid version components: {version}") from e


def format_version(major: int, minor: int, patch: int) -> str:
    return f"v{major}.{minor}.{patch}"


def bump(version: str, bump_type: str) -> str:
    """Bump version according to semver rules."""
    major, minor, patch = parse_version(version)
    if bump_type == "major":
        return format_version(major + 1, 0, 0)
    if bump_type == "minor":
        return format_version(major, minor + 1, 0)
    if bump_type == "patch":
        return format_version(major, minor, patch + 1)
    raise ValueError(f"Unknown bump type: {bump_type}")


def detect_bump_type(current_cfg: dict, new_cfg: dict) -> str:
    """Suggest a bump type based on the config diff.

    Rules:
      - model_name changed -> major
      - training_mode, loss_type changed -> major
      - learning_rate, batch_size, lora_r, lora_alpha changed -> minor
      - seed changed -> patch
      - nothing changed -> patch (reproducibility re-run)
    """
    current_cfg = normalize_cfg(current_cfg)
    new_cfg = normalize_cfg(new_cfg)

    def changed(key: str) -> bool:
        # If the key is missing on either side there is nothing to compare -> not a change.
        if key not in current_cfg or key not in new_cfg:
            return False
        return current_cfg[key] != new_cfg[key]

    # Major triggers
    major_keys = ["model_name", "training_mode", "loss_type"]
    for key in major_keys:
        if changed(key):
            return "major"

    # Minor triggers
    minor_keys = [
        "learning_rate",
        "batch_size",
        "lora_r",
        "lora_alpha",
        "lora_dropout",
        "num_epochs",
        "epochs",
        "max_train_samples",
        "data_version",
    ]
    for key in minor_keys:
        if changed(key):
            return "minor"

    # Patch triggers (seed changed, or nothing changed at all)
    return "patch"


def load_registry_local(path: Path) -> dict:
    """Load registry from local file. Returns empty registry if not found."""
    if not path.exists():
        return {"schema_version": "1.0", "models": [], "production": None}
    return json.loads(path.read_text(encoding="utf-8"))


def save_registry_local(path: Path, registry: dict) -> None:
    """Write registry to disk atomically (tmp file + replace)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(registry, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def load_registry_s3(bucket: str, region: str) -> dict:
    """Load registry from S3. Returns empty registry if the object is missing."""
    import boto3
    from botocore.exceptions import ClientError

    client = boto3.client("s3", region_name=region)
    try:
        body = client.get_object(Bucket=bucket, Key=S3_REGISTRY_KEY)["Body"].read()
    except ClientError as e:
        if e.response["Error"]["Code"] in ("NoSuchKey", "404"):
            return {"schema_version": "1.0", "models": [], "production": None}
        raise
    return json.loads(body.decode("utf-8"))


def save_registry_s3(bucket: str, region: str, registry: dict) -> None:
    """Upload registry to S3 so `promote_model.py` can read it."""
    import boto3

    client = boto3.client("s3", region_name=region)
    client.put_object(
        Bucket=bucket,
        Key=S3_REGISTRY_KEY,
        Body=(json.dumps(registry, indent=2) + "\n").encode("utf-8"),
        ContentType="application/json",
    )


def get_latest_version(registry: dict) -> str | None:
    """Return the latest version in the registry."""
    if not registry.get("models"):
        return None
    # Parse the versions and pick the highest one
    versions = [m["version"] for m in registry["models"]]
    return max(versions, key=parse_version)


def get_previous_config(registry: dict) -> dict | None:
    """Return the config of the newest model in the registry."""
    if not registry.get("models"):
        return None
    latest = max(registry["models"], key=lambda m: parse_version(m["version"]))
    return latest.get("config_snapshot")


def is_promoted(model: dict) -> bool:
    """Has the model been promoted to production? (promote_model.py writes `promoted`.)"""
    return bool(model.get("promoted")) or model.get("status") == "promoted"


def find_reusable_version(registry: dict, cfg_hash: str) -> str | None:
    """Return the existing version if the same config was already registered.

    This stops `--auto` from incrementing the version on every run for the
    same config (idempotency).
    """
    if not registry.get("models"):
        return None
    latest = max(registry["models"], key=lambda m: parse_version(m["version"]))
    if latest.get("config_hash") != cfg_hash:
        return None
    if is_promoted(latest):
        return None
    return latest["version"]


def upsert_model(registry: dict, version: str, cfg: dict, cfg_hash: str, bump_type: str) -> dict:
    """Insert or update the version in the registry."""
    registry.setdefault("schema_version", "1.0")
    registry.setdefault("models", [])
    registry.setdefault("production", None)

    entry = next((m for m in registry["models"] if m.get("version") == version), None)
    if entry is None:
        entry = {"version": version, "created_at": now_iso()}
        registry["models"].append(entry)
        registry["models"].sort(key=lambda m: parse_version(m["version"]))

    entry["config_snapshot"] = cfg
    entry["config_hash"] = cfg_hash
    entry["bump_type"] = bump_type
    entry["updated_at"] = now_iso()
    # Do not downgrade the status if we are revisiting a promoted model.
    if not is_promoted(entry):
        entry["status"] = "pending"

    return registry


def main():
    parser = argparse.ArgumentParser(description="Model version bumping (semver)")
    parser.add_argument(
        "--auto", action="store_true", help="Bump automatically from the config diff"
    )
    parser.add_argument("--bump", choices=["major", "minor", "patch"], help="Manual bump type")
    parser.add_argument("--set", type=str, help="Assign an explicit version (e.g. v2.0.0)")
    parser.add_argument("--current", action="store_true", help="Print the current version and exit")
    parser.add_argument("--config", type=str, default="config.yaml", help="New config file")
    parser.add_argument(
        "--registry", type=str, default=str(DEFAULT_REGISTRY_PATH), help="Registry file"
    )
    parser.add_argument(
        "--s3-bucket",
        type=str,
        default=os.environ.get("S3_BUCKET") or None,
        help="If set, the registry is read from and written back to S3 (for CI)",
    )
    parser.add_argument(
        "--region",
        type=str,
        default=os.environ.get("AWS_REGION", "us-west-2"),
        help="S3 region",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute the version but do not write it to the registry",
    )
    args = parser.parse_args()

    registry_path = Path(args.registry)

    # Registry source: S3 when provided (source of truth), otherwise the local file.
    registry = {"schema_version": "1.0", "models": [], "production": None}
    if args.s3_bucket:
        registry = load_registry_s3(args.s3_bucket, args.region)
    if not registry.get("models"):
        # S3 is empty (first run) — bootstrap from the registry in the repo.
        local = load_registry_local(registry_path)
        if local.get("models"):
            registry = local

    latest = get_latest_version(registry)

    if args.current:
        print(latest or "v0.0.0")
        return

    # Load the new config
    config_path = PROJECT_ROOT / args.config
    if not config_path.exists():
        print(f"::error::Config not found: {config_path}", file=sys.stderr)
        sys.exit(1)
    new_cfg = normalize_cfg(yaml.safe_load(config_path.read_text(encoding="utf-8")))
    new_hash = config_hash(new_cfg)

    # Determine the bump type
    reusable = None if args.set else find_reusable_version(registry, new_hash)
    if reusable:
        # The same config is already registered — do not bump, return the existing version.
        new_version = reusable
        bump_type = "unchanged"
    elif args.set:
        new_version = args.set
        bump_type = "explicit"
    elif args.bump:
        if not latest:
            print("::error::No previous version to bump from", file=sys.stderr)
            sys.exit(1)
        new_version = bump(latest, args.bump)
        bump_type = args.bump
    elif args.auto:
        prev_cfg = get_previous_config(registry)
        if prev_cfg is None:
            # First model — v1.0.0
            new_version = "v1.0.0"
            bump_type = "initial"
        else:
            bump_type = detect_bump_type(prev_cfg, new_cfg)
            base = latest or "v0.0.0"
            new_version = bump(base, bump_type)
    else:
        print("::error::No action specified (--auto, --bump, --set, or --current)", file=sys.stderr)
        sys.exit(1)

    # Persist to the registry (the actual write step)
    if bump_type != "unchanged" and not args.dry_run:
        registry = upsert_model(registry, new_version, new_cfg, new_hash, bump_type)
        save_registry_local(registry_path, registry)
        if args.s3_bucket:
            save_registry_s3(args.s3_bucket, args.region, registry)

    # GitHub Actions output
    if os.environ.get("GITHUB_OUTPUT"):
        with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as f:
            f.write(f"version={new_version}\n")
            f.write(f"bump_type={bump_type}\n")
            f.write(f"config_hash={new_hash}\n")

    print(f"version={new_version}")
    print(f"bump_type={bump_type}")
    print(f"config_hash={new_hash}")


if __name__ == "__main__":
    main()
