"""prepare_data.py

Builds the connected dataset used for CLIP fine-tuning:

    query (text) + product (title) + image (url) + relevance label

It joins three sources on `product_id`:

  1. Amazon ESCI examples  -> query text, product_id, esci_label (E/S/C/I)
  2. Amazon ESCI products  -> product_title
  3. SQID image urls       -> image_url

Images only exist for the SQID subset of ESCI
(small_version=1, split="test", product_locale="us"), so the script filters
to that scope. The result is a single self-contained table that you later
re-split into train / validation / test (see split_dataset.py).

Output: data/processed/joined_dataset.parquet
"""

from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download

PROJECT_ROOT = Path(__file__).resolve().parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

ESCI_REPO = "khanmu2003/amazon-shopping-queries-dataset"
ESCI_EXAMPLES_FILE = "shopping_queries_dataset_examples.parquet"
ESCI_PRODUCTS_FILE = "shopping_queries_dataset_products.parquet"

SQID_REPO = "crossingminds/shopping-queries-image-dataset"
SQID_IMAGE_URLS_FILE = "data/product_image_urls.parquet"

# SQID was built from this exact subset of ESCI.
SQID_SCOPE = {
    "small_version": 1,
    "split": "test",
    "product_locale": "us",
}

# Amazon's generic "no image available" placeholder, not a real product image.
PLACEHOLDER_IMAGE = (
    "https://m.media-amazon.com/images/G/01/digital/video/web/"
    "Default_Background_Art_LTR._SX1080_FMjpg_.jpg"
)


def download(repo_id: str, filename: str) -> Path:
    """Download a single file from the Hub into data/raw/ (cached)."""
    print(f"  downloading {repo_id}/{filename} ...")
    return Path(
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="dataset",
            local_dir=RAW_DIR,
        )
    )


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Download raw sources -------------------------------------------------
    print("[1/4] Downloading raw sources ...")
    examples_path = download(ESCI_REPO, ESCI_EXAMPLES_FILE)
    products_path = download(ESCI_REPO, ESCI_PRODUCTS_FILE)
    image_urls_path = download(SQID_REPO, SQID_IMAGE_URLS_FILE)

    # 2. Load examples (queries + labels) and filter to SQID scope ------------
    print("[2/4] Loading examples and filtering to SQID scope ...")
    examples = pd.read_parquet(examples_path)
    print(f"  total examples: {len(examples):,}")
    mask = pd.Series(True, index=examples.index)
    for col, val in SQID_SCOPE.items():
        mask &= examples[col] == val
    examples = examples.loc[mask].copy()
    print(f"  SQID-scope examples: {len(examples):,}")

    # 3. Load products (titles) ----------------------------------------------
    print("[3/4] Loading product titles ...")
    products = pd.read_parquet(products_path, columns=["product_id", "product_title"])
    products = products.dropna(subset=["product_title"]).drop_duplicates("product_id")
    print(f"  products with titles: {len(products):,}")

    # 4. Load image urls and join everything ----------------------------------
    print("[4/4] Loading image urls and joining ...")
    image_urls = pd.read_parquet(image_urls_path)
    image_urls = image_urls.dropna(subset=["image_url"]).drop_duplicates("product_id")
    print(f"  products with image urls: {len(image_urls):,}")

    joined = examples.merge(products, on="product_id", how="left")
    joined = joined.merge(image_urls, on="product_id", how="left")

    # Drop rows with no usable image (missing URL or Amazon placeholder).
    before = len(joined)
    joined = joined[joined["image_url"].notna()]
    joined = joined[joined["image_url"] != PLACEHOLDER_IMAGE]
    print(f"  dropped {before - len(joined):,} rows with no usable image")
    print(f"  final rows: {len(joined):,}")

    # Keep a clean, explicit column order.
    joined = joined[
        [
            "query_id",
            "query",
            "product_id",
            "product_title",
            "image_url",
            "esci_label",
        ]
    ].reset_index(drop=True)

    # Save --------------------------------------------------------------------
    out_path = PROCESSED_DIR / "joined_dataset.parquet"
    joined.to_parquet(out_path, index=False)
    print(f"\nSaved {len(joined):,} rows -> {out_path}")

    print("\nSample:")
    print(joined.head().to_string(max_colwidth=60))


if __name__ == "__main__":
    main()
