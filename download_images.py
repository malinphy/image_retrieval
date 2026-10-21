"""download_images.py

Downloads product images referenced in the dataset to disk.

Each image is saved as data/images/<product_id>.jpg. Existing files are
skipped, so re-running resumes where it left off. Unreachable/broken URLs
are skipped and logged to data/images/_failed.csv.

When running on AWS SageMaker (SM_CHANNEL_SPLITS is set), it downloads all
required images for:
  - train split (exact matches 'E' for contrastive training)
  - val split (all labels for validation retrieval)
  - test split (all labels for test retrieval)
and packages them into a single tar archive in S3:
  s3://<S3_BUCKET>/data/images_bundle.tar
for fast loading during GPU training jobs.

Usage:
    python download_images.py
"""

import asyncio
import json
import os
import shutil
import tarfile
import time
from pathlib import Path

import aiohttp
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
SPLITS_DIR = Path(
    os.environ.get("SM_CHANNEL_SPLITS", PROJECT_ROOT / "data" / "processed" / "splits")
)
JOINED = PROJECT_ROOT / "data" / "processed" / "joined_dataset.parquet"
OUT_DIR = Path(os.environ.get("SM_OUTPUT_DATA_DIR", PROJECT_ROOT / "analysis"))

IS_SAGEMAKER = "SM_CHANNEL_SPLITS" in os.environ
EXISTING_IMAGES_DIR = Path(os.environ.get("SM_CHANNEL_IMAGES", PROJECT_ROOT / "data" / "images"))
IMAGES_DIR = Path("/tmp/images") if IS_SAGEMAKER else EXISTING_IMAGES_DIR

# Set via the S3_BUCKET env variable; there is no hardcoded fallback (security).
# To run download_images.py locally:
#   $env:S3_BUCKET="your-bucket-name"
S3_BUCKET = os.environ.get("S3_BUCKET")
if S3_BUCKET is None:
    raise RuntimeError(
        "S3_BUCKET environment variable is not set. Locally: $env:S3_BUCKET='your-bucket-name'"
    )
S3_TAR_KEY = os.environ.get("S3_TAR_KEY", "data/images_bundle.tar")

# Options:
#   "train_e_val_test" -> train (only 'E') + val (all) + test (all)
#   "test" / "val" / "train" -> single split
#   None -> all rows from joined_dataset.parquet
SPLIT = "train_e_val_test"

CONCURRENCY = 128 if IS_SAGEMAKER else 32
TIMEOUT = 25


async def fetch(session, sem, pid, url, *, results, progress):
    path = IMAGES_DIR / f"{pid}.jpg"
    if path.exists():
        progress["done"] += 1
        return
    async with sem:
        try:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=TIMEOUT)) as resp:
                if resp.status != 200:
                    results.append((pid, url, resp.status))
                else:
                    data = await resp.read()
                    path.write_bytes(data)
                    progress["downloaded"] += 1
        except Exception as e:
            results.append((pid, url, f"ERR:{type(e).__name__}"))

        progress["done"] += 1
        done = progress["done"]
        total = progress["total"]
        if done % 2500 == 0 or done == total:
            elapsed = max(0.1, time.time() - progress["start"])
            rate = progress["downloaded"] / elapsed
            print(
                f"Progress: {done:,}/{total:,} ({100.0 * done / total:.1f}%) | "
                f"New downloaded: {progress['downloaded']:,} | "
                f"Failed: {len(results):,} | Speed: {rate:.1f} img/s",
                flush=True,
            )


async def run(tasks):
    sem = asyncio.Semaphore(CONCURRENCY)
    results = []
    progress = {
        "done": 0,
        "downloaded": 0,
        "total": len(tasks),
        "start": time.time(),
    }
    connector = aiohttp.TCPConnector(limit=CONCURRENCY, ttl_dns_cache=300)
    headers = {"User-Agent": "Mozilla/5.0 (compatible; ImageRetrievalBot/1.0)"}
    async with aiohttp.ClientSession(connector=connector, headers=headers) as session:
        await asyncio.gather(
            *(
                fetch(session, sem, pid, url, results=results, progress=progress)
                for pid, url in tasks
            )
        )
    return results, progress["downloaded"]


def collect_target_pairs():
    if SPLIT == "train_e_val_test":
        train_df = pd.read_parquet(SPLITS_DIR / "train.parquet")
        train_e = train_df[train_df["esci_label"] == "E"][["product_id", "image_url"]]
        val_df = pd.read_parquet(SPLITS_DIR / "val.parquet")[["product_id", "image_url"]]
        test_df = pd.read_parquet(SPLITS_DIR / "test.parquet")[["product_id", "image_url"]]
        df = pd.concat([train_e, val_df, test_df], ignore_index=True)
    elif SPLIT:
        src = SPLITS_DIR / f"{SPLIT}.parquet"
        df = pd.read_parquet(src)[["product_id", "image_url"]]
    else:
        df = pd.read_parquet(JOINED)[["product_id", "image_url"]]

    pairs = df.dropna(subset=["image_url"]).drop_duplicates("product_id").reset_index(drop=True)
    return pairs


def main():
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    # If running on SageMaker and existing images were mounted, copy them first
    if IS_SAGEMAKER and EXISTING_IMAGES_DIR.exists() and EXISTING_IMAGES_DIR != IMAGES_DIR:
        existing_files = list(EXISTING_IMAGES_DIR.glob("*.jpg"))
        print(
            f"Copying {len(existing_files):,} existing images from {EXISTING_IMAGES_DIR} ...",
            flush=True,
        )
        for src_file in existing_files:
            dst_file = IMAGES_DIR / src_file.name
            if not dst_file.exists():
                shutil.copy2(src_file, dst_file)

    pairs = collect_target_pairs()
    print(f"{len(pairs):,} unique products targeted (mode={SPLIT or 'all'})", flush=True)

    already_set = {p.stem for p in IMAGES_DIR.glob("*.jpg")}
    print(f"{len(already_set):,} images already present (will be skipped)", flush=True)

    missing_pairs = pairs[~pairs["product_id"].astype(str).isin(already_set)]
    print(f"{len(missing_pairs):,} new images to download ...", flush=True)

    tasks = list(missing_pairs.itertuples(index=False, name=None))
    failures, newly_downloaded = asyncio.run(run(tasks)) if tasks else ([], 0)

    total_on_disk = sum(1 for _ in IMAGES_DIR.glob("*.jpg"))
    print(f"\nDownloaded {newly_downloaded:,} new images.", flush=True)
    print(f"Total images on disk: {total_on_disk:,}", flush=True)
    print(f"Failed: {len(failures):,}", flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if failures:
        fail_df = pd.DataFrame(failures, columns=["product_id", "image_url", "reason"])
        fail_df.to_csv(OUT_DIR / "_failed_images.csv", index=False)
        if not IS_SAGEMAKER:
            fail_df.to_csv(IMAGES_DIR / "_failed.csv", index=False)

    summary = {
        "mode": SPLIT,
        "targeted_unique_products": len(pairs),
        "already_present": len(already_set),
        "newly_downloaded": int(newly_downloaded),
        "total_images_on_disk": int(total_on_disk),
        "failed": len(failures),
    }
    (OUT_DIR / "download_summary.json").write_text(json.dumps(summary, indent=4))

    if IS_SAGEMAKER:
        # boto3 only ships in the SageMaker DLC image, so import it lazily.
        import boto3  # noqa: PLC0415

        tar_path = Path("/tmp/images_bundle.tar")
        print(f"Packing {total_on_disk:,} images into {tar_path} ...", flush=True)
        with tarfile.open(tar_path, "w") as tar:
            for jpg in IMAGES_DIR.glob("*.jpg"):
                tar.add(jpg, arcname=jpg.name)
        size_mb = tar_path.stat().st_size / (1024 * 1024)
        print(
            f"Created {tar_path} ({size_mb:.1f} MB). Uploading to s3://{S3_BUCKET}/{S3_TAR_KEY} ...",
            flush=True,
        )
        s3 = boto3.client("s3")
        s3.upload_file(str(tar_path), S3_BUCKET, S3_TAR_KEY)
        s3.upload_file(
            str(OUT_DIR / "download_summary.json"),
            S3_BUCKET,
            "data/download_summary.json",
        )
        print(f"Successfully uploaded s3://{S3_BUCKET}/{S3_TAR_KEY}!", flush=True)


if __name__ == "__main__":
    main()
