# SQID & Amazon ESCI Dataset Analysis Report

This document provides a comprehensive exploratory data analysis (EDA) of the **Shopping Queries Image Dataset (SQID)** and **Amazon ESCI** datasets, covering dataset scale, relevance label distributions, query characteristics, and strategic recommendations for training, fine-tuning, and evaluating multimodal image retrieval models.

---

## 1. Dataset Scale & Coverage

| Metric | Full Amazon ESCI Dataset | SQID Test Benchmark Subset |
| :--- | :--- | :--- |
| **Total Judgments (`<query, item>` pairs)** | 2,621,288 | 181,701 |
| **Unique Queries** | 130,652 | 8,956 |
| **Unique Products** | 1,802,772 | 164,900 |
| **Products with Image URLs** | - | 156,545 (~95% coverage) |
| **CLIP Embedding Dimension** | - | **768** (CLIP ViT-L/14 or equivalent) |

---

## 2. Relevance Label Distribution (`esci_label`)

In the SQID test subset (US Locale, Task 1 Small Version), query-product relevance is categorized into four classes:

| Label | Description | Sample Count | Percentage | Standard Ranking Gain Weight |
| :---: | :--- | :---: | :---: | :---: |
| **`E`** | **Exact Match** | 79,708 | **43.87%** | `1.0` |
| **`S`** | **Substitute** | 63,563 | **34.98%** | `0.1` |
| **`C`** | **Complement** | 8,099 | **4.46%** | `0.01` |
| **`I`** | **Irrelevant** | 30,331 | **16.69%** | `0.0` |

### Key Observations & Implications:
* **Positive Matches:** Combined relevant pairs (`E` + `S`) account for **78.85%** of the dataset.
* **Hard Negatives:** The **16.69%** of candidates labeled as `I` (Irrelevant) originate from actual search engine candidate lists. They serve as ideal **Hard Negatives** for contrastive learning (InfoNCE / Triplet loss) rather than relying solely on random in-batch negatives.

---

## 3. Query Depth & Candidate Pool Analysis

Number of candidate products evaluated per user search query:

* **Average Query Depth:** `20.29` products / query
* **Median Depth:** `16` products
* **Min / Max Depth:** `8` / `95` products
* **Queries with $\ge$ 1 Exact Match (`E`):** `8,955` (**100.0%**)
* **Queries with 0 Exact Matches:** Only `1` (**0.0%**)

> [!NOTE]
> Having at least one Exact Match across nearly 100% of queries ensures that retrieval metrics like **Recall@K (K=1, 5, 10)** and **MRR** remain well-defined and stable across all evaluated test queries.

---

## 4. Query Text Statistics & CLIP Tokenizer Compatibility

* **Average Word Count:** `3.89` words (Min: `1`, Max: `29` words)
* **Average Character Length:** `22.97` characters (Max: `151` characters)

### Compatibility with CLIP:
* Standard OpenAI and OpenCLIP architectures (e.g., `ViT-B/32`, `ViT-L/14`) enforce a strict context limit of **77 tokens**.
* Since the maximum query length is only 29 words (~35-40 sub-word tokens), **100% of queries fit comfortably inside CLIP's context window** without any truncation or information loss.

---

## 5. Model Training & Evaluation Strategies

### A. Training & Fine-Tuning Recommendations:
1. **Multiple Negatives Ranking Loss (InfoNCE):**
   * **Positive Pair:** `(Query, Product_E)`
   * **Hard Negatives:** Candidate products labeled as `I` within the same query result list.
   * **In-Batch Negatives:** Products paired with other queries in the same minibatch.
2. **Triplet / Margin Ranking Loss:**
   * Triplet format: `(Query, Product_E, Product_I)` enforcing:
     $$\text{sim}(Q, E) - \text{sim}(Q, I) > \text{margin (e.g., 0.2)}$$
3. **Handling Annotation Noise:**
   * Given the inherent crowd-worker label noise (~5-10%), apply **Label Smoothing (0.1)** or tune the softmax temperature ($\tau \approx 0.05 - 0.07$) to avoid overfitting to ambiguous or mislabeled samples.

### B. Evaluation Metrics:
1. **NDCG@10 / NDCG@20 (Normalized Discounted Cumulative Gain):**
   * Official ESCI benchmark metric using standard gain weights:
     $$\text{Gain} = \{'E': 1.0, 'S': 0.1, 'C': 0.01, 'I': 0.0\}$$
2. **Recall@K (K=1, 5, 10):**
   * Measures whether at least one Exact (`E`) product appears within the top-K retrieved results.
3. **MRR (Mean Reciprocal Rank):**
   * Computes the reciprocal rank of the first relevant (`E` or `S`) product.
