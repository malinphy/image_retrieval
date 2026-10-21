"""split_dataset.py

Splits data/processed/joined_dataset.parquet into train / validation / test.

The split is done at the QUERY level (all rows sharing a query_id go to the
same split) to prevent information leakage between splits.

Default ratios: 80% train, 10% validation, 10% test.

Outputs (in data/processed/splits/):
  train.parquet, val.parquet, test.parquet
"""

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
SPLITS_DIR = PROCESSED_DIR / "splits"

TRAIN_RATIO = 0.9
VAL_RATIO = 0.05
TEST_RATIO = 0.05
SEED = 42


def main():
    src = PROCESSED_DIR / "joined_dataset.parquet"
    if not src.exists():
        raise FileNotFoundError(f"{src} not found. Run prepare_data.py first.")

    df = pd.read_parquet(src)
    print(f"Loaded {len(df):,} rows, {df['query_id'].nunique():,} unique queries")

    query_ids = df["query_id"].unique()
    rng = np.random.default_rng(SEED)
    rng.shuffle(query_ids)

    n = len(query_ids)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)
    # test gets the remainder (keeps exact totals)

    train_ids = set(query_ids[:n_train])
    val_ids = set(query_ids[n_train : n_train + n_val])
    # The remainder lands in test (the fallback branch in assign())

    def assign(qid):
        if qid in train_ids:
            return "train"
        if qid in val_ids:
            return "val"
        return "test"

    df["split"] = df["query_id"].map(assign)

    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    for name in ("train", "val", "test"):
        part = df[df["split"] == name].drop(columns="split")
        part.to_parquet(SPLITS_DIR / f"{name}.parquet", index=False)
        print(f"  {name:5s}: {len(part):>7,} rows, {part['query_id'].nunique():>6,} queries")

    print(f"\nSplits written to {SPLITS_DIR}")


if __name__ == "__main__":
    main()
