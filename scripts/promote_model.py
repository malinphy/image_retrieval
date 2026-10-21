"""promote_model.py — Model registry management and promotion.

Reads/writes the registry from S3, lists model versions, switches the
production model, and writes an audit trail.

Usage:
    # List all models
    python scripts/promote_model.py --list

    # Show the production model
    python scripts/promote_model.py --current

    # Details for a specific model
    python scripts/promote_model.py --show v1.2.3

    # Promotion (manual approval)
    python scripts/promote_model.py --promote v1.2.3 --reason "Recall@10 +2.1%"

    # Rollback / set production
    python scripts/promote_model.py --set-production v1.2.2 --reason "v1.2.3 regression"

Environment:
    S3_BUCKET: bucket name (required for S3 mode)
    AWS_REGION: AWS region (default: us-west-2)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
S3_REGISTRY_KEY = "models/registry.json"
S3_PRODUCTION_KEY = "models/production.json"
S3_DEPLOYMENTS_PREFIX = "deployments/"


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_registry_s3(bucket: str, region: str) -> dict:
    """Download the registry from S3. Returns an empty registry if not found."""
    import boto3

    s3 = boto3.client("s3", region_name=region)
    try:
        resp = s3.get_object(Bucket=bucket, Key=S3_REGISTRY_KEY)
        return json.loads(resp["Body"].read().decode("utf-8"))
    except s3.exceptions.NoSuchKey:
        return {"schema_version": "1.0", "models": [], "production": None}


def save_registry_s3(bucket: str, region: str, registry: dict) -> None:
    """Write the registry to S3."""
    import boto3

    s3 = boto3.client("s3", region_name=region)
    registry["last_updated"] = now_iso()
    body = json.dumps(registry, indent=2, ensure_ascii=False)
    s3.put_object(
        Bucket=bucket,
        Key=S3_REGISTRY_KEY,
        Body=body.encode("utf-8"),
        ContentType="application/json",
    )


def write_audit_s3(bucket: str, region: str, version: str, action: str, reason: str) -> None:
    """Write an audit trail object: deployments/promoted-{version}.json"""
    import boto3

    s3 = boto3.client("s3", region_name=region)
    audit = {
        "version": version,
        "action": action,
        "reason": reason,
        "actor": os.environ.get("GITHUB_ACTOR", "local"),
        "at": now_iso(),
        "run_id": os.environ.get("GITHUB_RUN_ID", "local"),
    }
    key = f"{S3_DEPLOYMENTS_PREFIX}promoted-{version}.json"
    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=json.dumps(audit, indent=2).encode("utf-8"),
        ContentType="application/json",
    )


def cmd_list(registry: dict) -> None:
    """Print all models as a table."""
    models = registry.get("models", [])
    if not models:
        print("Registry is empty.")
        return

    production = registry.get("production")
    print(f"{'VERSION':<10} {'PROMOTED':<10} {'RECALL@10':<12} {'TRAINED_AT':<22} {'SHA':<10}")
    print("-" * 70)
    for m in sorted(models, key=lambda x: x["version"], reverse=True):
        promoted = "★" if m["version"] == production else ""
        recall = m.get("metrics", {}).get("test_Recall@10", "N/A")
        recall_str = f"{recall:.4f}" if isinstance(recall, (int, float)) else str(recall)
        print(
            f"{m['version']:<10} {promoted:<10} {recall_str:<12} "
            f"{m.get('trained_at', 'N/A'):<22} {m.get('git_sha', 'N/A'):<10}"
        )


def cmd_current(registry: dict) -> None:
    """Show the production model."""
    prod = registry.get("production")
    if not prod:
        print("No production model set.")
        return
    model = next((m for m in registry["models"] if m["version"] == prod), None)
    if not model:
        print(f"Production points to {prod} but model not found in registry.")
        return
    print(f"Production: {prod}")
    print(f"  Trained at: {model.get('trained_at')}")
    print(f"  Git SHA: {model.get('git_sha')}")
    print(f"  Metrics: {json.dumps(model.get('metrics', {}), indent=2)}")


def cmd_show(registry: dict, version: str) -> None:
    """Show details for a specific model."""
    model = next((m for m in registry["models"] if m["version"] == version), None)
    if not model:
        print(f"Model {version} not found in registry.")
        sys.exit(1)
    print(json.dumps(model, indent=2, ensure_ascii=False))


def cmd_promote(registry: dict, bucket: str, region: str, version: str, reason: str) -> None:
    """Mark the model as promoted and make it the production model."""
    model = next((m for m in registry["models"] if m["version"] == version), None)
    if not model:
        print(f"::error::Model {version} not found in registry", file=sys.stderr)
        sys.exit(1)

    # Mark the previous production model as superseded
    prev_prod = registry.get("production")
    if prev_prod and prev_prod != version:
        prev_model = next((m for m in registry["models"] if m["version"] == prev_prod), None)
        if prev_model:
            prev_model["superseded_by"] = version

    # Mark the new model as promoted
    model["promoted"] = True
    model["promoted_at"] = now_iso()
    registry["production"] = version

    save_registry_s3(bucket, region, registry)
    write_audit_s3(bucket, region, version, "promote", reason)

    # Also write production.json (fast lookup)
    import boto3

    s3 = boto3.client("s3", region_name=region)
    s3.put_object(
        Bucket=bucket,
        Key=S3_PRODUCTION_KEY,
        Body=json.dumps({"version": version, "promoted_at": now_iso()}, indent=2).encode("utf-8"),
        ContentType="application/json",
    )

    print(f"✓ {version} promoted to production (reason: {reason})")


def cmd_set_production(registry: dict, bucket: str, region: str, version: str, reason: str) -> None:
    """Set production to a specific version (including rollback)."""
    model = next((m for m in registry["models"] if m["version"] == version), None)
    if not model:
        print(f"::error::Model {version} not found in registry", file=sys.stderr)
        sys.exit(1)

    prev_prod = registry.get("production")
    registry["production"] = version
    model["promoted"] = True
    model["promoted_at"] = now_iso()

    save_registry_s3(bucket, region, registry)
    write_audit_s3(bucket, region, version, f"set-production (was {prev_prod})", reason)

    import boto3

    s3 = boto3.client("s3", region_name=region)
    s3.put_object(
        Bucket=bucket,
        Key=S3_PRODUCTION_KEY,
        Body=json.dumps({"version": version, "promoted_at": now_iso()}, indent=2).encode("utf-8"),
        ContentType="application/json",
    )

    print(f"✓ Production set to {version} (was {prev_prod}, reason: {reason})")


def main():
    parser = argparse.ArgumentParser(description="Model registry & promotion")
    parser.add_argument("--list", action="store_true", help="List all models")
    parser.add_argument("--current", action="store_true", help="Show the production model")
    parser.add_argument("--show", type=str, help="Details for a specific model (e.g. v1.2.3)")
    parser.add_argument("--promote", type=str, help="Promote a model (make it production)")
    parser.add_argument("--set-production", type=str, help="Set production (including rollback)")
    parser.add_argument("--reason", type=str, default="", help="Promotion/set reason (for audit)")
    parser.add_argument("--bucket", type=str, default=os.environ.get("S3_BUCKET"), help="S3 bucket")
    parser.add_argument("--region", type=str, default=os.environ.get("AWS_REGION", "us-west-2"))
    args = parser.parse_args()

    # Read-only commands do not require S3 (the registry can be read from a local cache)
    needs_s3 = any([args.promote, args.set_production])

    if needs_s3 and not args.bucket:
        print(
            "::error::--bucket or S3_BUCKET env var required for write operations", file=sys.stderr
        )
        sys.exit(1)

    if args.list or args.current or args.show or needs_s3:
        registry = (
            load_registry_s3(args.bucket, args.region)
            if args.bucket
            else {"schema_version": "1.0", "models": [], "production": None}
        )

    if args.list:
        cmd_list(registry)
    elif args.current:
        cmd_current(registry)
    elif args.show:
        cmd_show(registry, args.show)
    elif args.promote:
        if not args.reason:
            print("::error::--reason is required for --promote", file=sys.stderr)
            sys.exit(1)
        cmd_promote(registry, args.bucket, args.region, args.promote, args.reason)
    elif args.set_production:
        if not args.reason:
            print("::error::--reason is required for --set-production", file=sys.stderr)
            sys.exit(1)
        cmd_set_production(registry, args.bucket, args.region, args.set_production, args.reason)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
