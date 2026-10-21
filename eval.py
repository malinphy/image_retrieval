"""eval.py

Zero-shot CLIP evaluation on the test split (before any fine-tuning).

Evaluates two retrieval paradigms:
  1. Global Vector Retrieval (Full-Corpus Similarity Search):
     Indexes all unique product images in the test corpus (8,725 images) and
     retrieves top-K products for each query via cosine similarity search
     (exact inner product over L2-normalized vectors).
     Reports True Recall@K (1, 5, 10, 20, 50, 100), HitRate@K (1, 5, 10, 20, 50),
     Global NDCG@10/20, and Global MRR.

  2. Candidate Re-ranking (ESCI Benchmark):
     Ranks only the ~20 pre-judged candidate products per query and reports
     NDCG@10/20, True Recall@1/5/10, HitRate@1/5/10, and MRR.

Usage:
    python eval.py
"""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

from dataset import (
    ProductImageDataset,
    QueryTextDataset,
    image_collate,
    text_collate,
)

PROJECT_ROOT = Path(__file__).resolve().parent
SPLITS_DIR = Path(
    os.environ.get("SM_CHANNEL_SPLITS", PROJECT_ROOT / "data" / "processed" / "splits")
)
TEST_PARQUET = SPLITS_DIR / "test.parquet"
IMAGES_DIR = Path(os.environ.get("SM_CHANNEL_IMAGES", PROJECT_ROOT / "data" / "images"))
OUT_DIR = Path(os.environ.get("SM_OUTPUT_DATA_DIR", PROJECT_ROOT / "analysis"))
OUT_PATH = OUT_DIR / "zero_shot_results.json"
EMB_CACHE_PATH = OUT_DIR / "zero_shot_embeddings.npz"

MODEL_NAME = "openai/clip-vit-large-patch14"
DEVICE = None  # None -> auto (cuda if available, else cpu)

TEXT_BATCH_SIZE = 128
IMAGE_BATCH_SIZE = 128 if torch.cuda.is_available() else 32
NUM_WORKERS = 4 if torch.cuda.is_available() else 0

GAINS = {"E": 1.0, "S": 0.1, "C": 0.01, "I": 0.0}
RELEVANT = {"E", "S"}  # used for MRR


def load_model(model_name, device):
    model = CLIPModel.from_pretrained(model_name).to(device).eval()
    processor = CLIPProcessor.from_pretrained(model_name)
    return model, processor


def normalize(x):
    return x / x.norm(dim=-1, keepdim=True)


def as_tensor(emb):
    if hasattr(emb, "pooler_output"):
        return emb.pooler_output
    return emb


def dcg(gains, k):
    gains = np.asarray(gains[:k], dtype=float)
    if len(gains) == 0:
        return 0.0
    denom = np.log2(np.arange(2, len(gains) + 2))
    return float((gains / denom).sum())


def ndcg(pred_gains, ideal_gains, k):
    d = dcg(pred_gains, k)
    i = dcg(ideal_gains, k)
    return d / i if i > 0 else 0.0


def hit_rate_at_k(ranked_labels, k):
    return 1.0 if "E" in ranked_labels[:k] else 0.0


def true_recall_at_k(ranked_labels, total_exact, k):
    if total_exact == 0:
        return 0.0
    found_exact = sum(1 for lbl in ranked_labels[:k] if lbl == "E")
    return found_exact / total_exact


def reciprocal_rank(ranked_labels):
    for i, lbl in enumerate(ranked_labels):
        if lbl in RELEVANT:
            return 1.0 / (i + 1)
    return 0.0


@torch.no_grad()
def encode_queries(model, processor, device):
    ds = QueryTextDataset(TEST_PARQUET)
    loader = DataLoader(
        ds,
        batch_size=TEXT_BATCH_SIZE,
        num_workers=NUM_WORKERS,
        collate_fn=text_collate(processor),
    )
    emb_map = {}
    for qids, batch in tqdm(loader, desc="Encoding queries", unit="batch"):
        inputs = {k: v.to(device) for k, v in batch.items()}
        embs = normalize(as_tensor(model.get_text_features(**inputs))).cpu().numpy()
        for qid, e in zip(qids, embs, strict=True):
            emb_map[int(qid)] = e.astype(np.float32)
    return emb_map


@torch.no_grad()
def encode_images(model, processor, device):
    ds = ProductImageDataset(TEST_PARQUET, IMAGES_DIR)
    loader = DataLoader(
        ds,
        batch_size=IMAGE_BATCH_SIZE,
        num_workers=NUM_WORKERS,
        collate_fn=image_collate(processor),
    )
    emb_map = {}
    n_missing = 0
    for pids, valid, batch in tqdm(loader, desc="Encoding images", unit="batch"):
        inputs = {k: v.to(device) for k, v in batch.items()}
        embs = normalize(as_tensor(model.get_image_features(**inputs))).cpu().numpy()
        for pid, ok, e in zip(pids, valid, embs, strict=True):
            if ok:
                emb_map[str(pid)] = e.astype(np.float32)
            else:
                n_missing += 1
    return emb_map, n_missing


def evaluate_global_retrieval(df, query_emb_map, img_emb_map):
    """Full-corpus vector similarity search across all encoded product images."""
    gallery_pids = list(img_emb_map.keys())
    gallery_matrix = np.stack([img_emb_map[pid] for pid in gallery_pids], axis=0)  # (N_img, D)
    pid_to_idx = {pid: idx for idx, pid in enumerate(gallery_pids)}

    ks_recall = [1, 5, 10, 20, 50, 100]
    ks_hit = [1, 5, 10, 20, 50]

    recalls = {k: 0.0 for k in ks_recall}
    hits = {k: 0.0 for k in ks_hit}
    m_ndcg10 = 0.0
    m_ndcg20 = 0.0
    m_mrr = 0.0

    n_queries_all = 0
    n_queries_with_exact = 0

    # Prepare query batch for fast matrix similarity search
    grouped = list(df.groupby("query_id"))
    qids = [int(qid) for qid, _ in grouped]
    query_matrix = np.stack([query_emb_map[qid] for qid in qids], axis=0)  # (N_q, D)

    # Vector similarity search: (N_q, D) @ (D, N_img) -> (N_q, N_img)
    sim_matrix = query_matrix @ gallery_matrix.T

    max_k = max(*ks_recall, *ks_hit, 20)

    for row_idx, (_qid, g) in enumerate(tqdm(grouped, desc="Global vector retrieval")):
        scores = sim_matrix[row_idx]
        # Partial sort for top-max_k, and full rank of ground-truth items for exact MRR
        top_indices = np.argpartition(-scores, kth=min(max_k, len(scores) - 1))[:max_k]
        top_indices = top_indices[np.argsort(-scores[top_indices])]
        top_pids = [gallery_pids[i] for i in top_indices]

        pid_label_map = dict(zip(g["product_id"].astype(str), g["esci_label"], strict=False))
        ranked_top_labels = [pid_label_map.get(pid, "I") for pid in top_pids]
        ranked_top_gains = [GAINS[lbl] for lbl in ranked_top_labels]

        ideal_gains = sorted([GAINS[lbl] for lbl in g["esci_label"]], reverse=True)
        total_exact = sum(1 for lbl in g["esci_label"] if lbl == "E")

        m_ndcg10 += ndcg(ranked_top_gains, ideal_gains, 10)
        m_ndcg20 += ndcg(ranked_top_gains, ideal_gains, 20)

        # Compute exact global MRR across the entire corpus
        rel_Gallery_indices = [
            pid_to_idx[pid]
            for pid, lbl in pid_label_map.items()
            if lbl in RELEVANT and pid in pid_to_idx
        ]
        if rel_Gallery_indices:
            best_rel_score = max(scores[i] for i in rel_Gallery_indices)
            # Rank is 1 + number of products in corpus with strictly higher score
            rank = 1 + int(np.sum(scores > best_rel_score))
            m_mrr += 1.0 / rank

        n_queries_all += 1

        if total_exact > 0:
            n_queries_with_exact += 1
            for k in ks_recall:
                recalls[k] += true_recall_at_k(ranked_top_labels, total_exact, k)
            for k in ks_hit:
                hits[k] += hit_rate_at_k(ranked_top_labels, k)

    metrics = {
        "corpus_size": len(gallery_pids),
        "queries_evaluated": n_queries_all,
        "queries_with_exact_match": n_queries_with_exact,
        "NDCG@10": m_ndcg10 / n_queries_all,
        "NDCG@20": m_ndcg20 / n_queries_all,
        "MRR": m_mrr / n_queries_all,
    }
    for k in ks_recall:
        metrics[f"Recall@{k}"] = recalls[k] / max(1, n_queries_with_exact)
    for k in ks_hit:
        metrics[f"HitRate@{k}"] = hits[k] / max(1, n_queries_with_exact)

    return metrics


def evaluate_candidate_reranking(df, query_emb_map, img_emb_map):
    """Per-query candidate re-ranking over the ~20 ESCI judged products."""
    m_ndcg10 = m_ndcg20 = m_mrr = 0.0
    m_rec1 = m_rec5 = m_rec10 = 0.0
    m_hit1 = m_hit5 = m_hit10 = 0.0
    n_queries = 0
    n_queries_with_exact = 0

    for qid, g in tqdm(
        df.groupby("query_id"),
        total=df["query_id"].nunique(),
        desc="Candidate re-ranking",
    ):
        pids = g["product_id"].astype(str).tolist()
        labels = g["esci_label"].tolist()
        qe = query_emb_map[int(qid)]
        scores = np.full(len(pids), -np.inf)
        for i, pid in enumerate(pids):
            e = img_emb_map.get(pid)
            if e is not None:
                scores[i] = float(qe @ e)
        order = np.argsort(-scores)
        ranked_labels = [labels[i] for i in order]
        ranked_gains = [GAINS[lbl] for lbl in ranked_labels]
        ideal_gains = sorted(ranked_gains, reverse=True)
        total_exact = sum(1 for lbl in labels if lbl == "E")

        m_ndcg10 += ndcg(ranked_gains, ideal_gains, 10)
        m_ndcg20 += ndcg(ranked_gains, ideal_gains, 20)
        m_mrr += reciprocal_rank(ranked_labels)
        n_queries += 1

        if total_exact > 0:
            n_queries_with_exact += 1
            m_rec1 += true_recall_at_k(ranked_labels, total_exact, 1)
            m_rec5 += true_recall_at_k(ranked_labels, total_exact, 5)
            m_rec10 += true_recall_at_k(ranked_labels, total_exact, 10)
            m_hit1 += hit_rate_at_k(ranked_labels, 1)
            m_hit5 += hit_rate_at_k(ranked_labels, 5)
            m_hit10 += hit_rate_at_k(ranked_labels, 10)

    return {
        "queries_evaluated": n_queries,
        "NDCG@10": m_ndcg10 / n_queries,
        "NDCG@20": m_ndcg20 / n_queries,
        "Recall@1": m_rec1 / max(1, n_queries_with_exact),
        "Recall@5": m_rec5 / max(1, n_queries_with_exact),
        "Recall@10": m_rec10 / max(1, n_queries_with_exact),
        "HitRate@1": m_hit1 / max(1, n_queries_with_exact),
        "HitRate@5": m_hit5 / max(1, n_queries_with_exact),
        "HitRate@10": m_hit10 / max(1, n_queries_with_exact),
        "MRR": m_mrr / n_queries,
    }


def main():
    device = DEVICE or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Model: {MODEL_NAME}")
    print(f"Device: {device}")

    df = pd.read_parquet(TEST_PARQUET)
    print(f"Test rows: {len(df):,}, queries: {df['query_id'].nunique():,}")

    if EMB_CACHE_PATH.exists():
        print(f"Loading cached embeddings from {EMB_CACHE_PATH} ...")
        cache = np.load(EMB_CACHE_PATH, allow_pickle=True)
        query_emb_map = {int(k): v for k, v in zip(cache["qids"], cache["q_embs"], strict=True)}
        img_emb_map = {str(k): v for k, v in zip(cache["pids"], cache["img_embs"], strict=True)}
        n_missing = int(cache["n_missing"])
    else:
        model, processor = load_model(MODEL_NAME, device)
        query_emb_map = encode_queries(model, processor, device)
        img_emb_map, n_missing = encode_images(model, processor, device)
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            EMB_CACHE_PATH,
            qids=np.array(list(query_emb_map.keys()), dtype=np.int64),
            q_embs=np.stack(list(query_emb_map.values()), axis=0),
            pids=np.array(list(img_emb_map.keys())),
            img_embs=np.stack(list(img_emb_map.values()), axis=0),
            n_missing=np.array(n_missing, dtype=np.int64),
        )
        print(f"Saved embeddings -> {EMB_CACHE_PATH}")

    print(f"Images encoded: {len(img_emb_map):,}, missing on disk: {n_missing:,}")

    global_metrics = evaluate_global_retrieval(df, query_emb_map, img_emb_map)
    rerank_metrics = evaluate_candidate_reranking(df, query_emb_map, img_emb_map)

    results = {
        "model": MODEL_NAME,
        "queries": int(df["query_id"].nunique()),
        "corpus_images": len(img_emb_map),
        "products_missing_image": n_missing,
        "global_vector_retrieval": global_metrics,
        "candidate_reranking": rerank_metrics,
    }

    print("\n" + "=" * 55)
    print("  1. GLOBAL VECTOR RETRIEVAL (All 8,725 Images)")
    print("=" * 55)
    for k, v in global_metrics.items():
        if isinstance(v, float):
            print(f"{k:>28}: {v:.4f}")
        else:
            print(f"{k:>28}: {v}")

    print("\n" + "=" * 55)
    print("  2. CANDIDATE RE-RANKING (~20 Candidates per Query)")
    print("=" * 55)
    for k, v in rerank_metrics.items():
        if isinstance(v, float):
            print(f"{k:>28}: {v:.4f}")
        else:
            print(f"{k:>28}: {v}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(results, indent=4))
    print(f"\nSaved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
