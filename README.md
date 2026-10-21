# E-Commerce Image Retrieval (CLIP Fine-Tuning)

An image-retrieval system that matches search queries to e-commerce product
images by fine-tuning OpenAI's CLIP model. The dataset joins Amazon ESCI
(queries + relevance labels) with SQID (product image URLs).

---

## Datasets

| Source | Role | Key fields |
| :--- | :--- | :--- |
| Amazon ESCI (`khanmu2003/amazon-shopping-queries-dataset`) | queries, products, labels | `query`, `product_id`, `product_title`, `esci_label` |
| SQID (`crossingminds/shopping-queries-image-dataset`) | product images | `product_id`, `image_url` |

The two join on `product_id`. Images only exist for the SQID subset of ESCI
(`small_version=1`, `split="test"`, `product_locale="us"`), so the whole
dataset is built from that subset (~172k rows) and then re-split for training.

Relevance labels (`esci_label`): `E` = exact, `S` = substitute, `C` = complement,
`I` = irrelevant.

---

## Project structure

Data pipeline:

| File | Purpose |
| :--- | :--- |
| `prepare_data.py` | Download ESCI + SQID, join into one table, save to parquet |
| `split_dataset.py` | Query-level train / val / test split |
| `download_images.py` | Download product images to disk |
| `dataset.py` | PyTorch `Dataset` classes and collate helpers |

Training and evaluation:

| File | Purpose |
| :--- | :--- |
| `train.py` | CLIP fine-tuning (LoRA / projection-only / full), SageMaker aware |
| `eval.py` | Zero-shot and fine-tuned retrieval evaluation |
| `config.yaml` | All training hyperparameters and S3 settings |
| `analysis/analyze_dataset.py` | Exploratory data analysis, writes `analysis/analysis_report.md` |
| `analysis/analysis_report.md` | Dataset statistics (label distribution, query depth, etc.) |
| `analysis/baseline_results.json` | Reference zero-shot metrics from precomputed SQID embeddings |
| `analysis/zero_shot_results.json` | Zero-shot metrics produced locally by `eval.py` |
| `analysis/finetuned_test_results*.json` | Test-set metrics for fine-tuned checkpoints |
| `analysis/training_history*.json` | Per-epoch training/validation curves |
| `create_dataset.py` | Legacy one-off CSV merge (superseded by `prepare_data.py`) |

Automation and operations:

| File | Purpose |
| :--- | :--- |
| `scripts/launch_sagemaker.py` | Package code, launch a SageMaker job, stream logs |
| `scripts/bump_version.py` | Semantic version bump for the model registry |
| `scripts/promote_model.py` | Registry management, promotion, rollback |
| `.github/workflows/ci.yml` | CI: ruff lint/format, pytest, gitleaks |
| `.github/workflows/deploy.yml` | CD: SageMaker job launch on merge to `main` |
| `.github/workflows/promote.yml` | CT: model promotion workflow |
| `notes/DEPLOYMENT.md` | Deployment runbook |
| `notes/VERSIONING.md` | Model versioning and registry rules |

Documentation and exploration:

| File | Purpose |
| :--- | :--- |
| `tests/` | Test suite (44 unit + 4 smoke) |
| `notes/` | Background notes, guides, and experiment write-ups |
| `datacheck.ipynb` | Ad-hoc dataset exploration notebook |

---

## Pipeline

Run from the project root. Activate the venv first if applicable
(`venv\Scripts\activate` on Windows).

### 1. Build the joined dataset

```bash
python prepare_data.py
```

Downloads the raw parquet sources into `data/raw/`, joins them, and writes
`data/processed/joined_dataset.parquet` with columns `query_id`, `query`,
`product_id`, `product_title`, `image_url`, `esci_label`.

### 2. Split into train / val / test

```bash
python split_dataset.py
```

Splits at the **query level** (all rows sharing a query stay together) to avoid
leakage. Default ratios are **90% train / 5% val / 5% test** (edit
`TRAIN_RATIO`, `VAL_RATIO`, `TEST_RATIO`, `SEED` at the top of the file to change).

Writes `data/processed/splits/{train,val,test}.parquet`.

### 3. Download images

```bash
python download_images.py
```

Downloads each image to `data/images/<product_id>.jpg`. Existing files are
skipped (resumable). Failures are logged to `data/images/_failed.csv`.

Set `SPLIT` at the top of the file to download only one split
(`None` = all, or `"train"` / `"val"` / `"test"`).

### 4. Fine-tune

```bash
python train.py
python train.py --config config.yaml
```

`train.py` reads every hyperparameter from `config.yaml`. Three training modes
are supported via `training_mode`:

| Mode | What is trained |
| :--- | :--- |
| `lora` | LoRA adapters on attention + projections (recommended) |
| `projection_only` | Only `visual_projection`, `text_projection`, `logit_scale` |
| `full` | All parameters |

Training uses a multi-positive symmetric InfoNCE loss. Every epoch writes a
checkpoint with the hyperparameters and validation metrics in its name, plus a
`best_model/` directory selected by `best_metric`. On SageMaker, checkpoints and
logs are synced to `s3://{S3_BUCKET}/checkpoints/{run_name}/`.

### 5. Evaluate

```bash
python eval.py
```

Encodes query texts and product images with CLIP, ranks candidates by cosine
similarity, and reports NDCG@10/20, Recall@1/5/10 and MRR. Results are printed
and saved to `analysis/zero_shot_results.json`.

---

## Cloud runs (SageMaker)

```bash
# Train on a SageMaker GPU (default instance selected from the mode)
python scripts/launch_sagemaker.py --mode train --follow

# Evaluate an existing checkpoint
python scripts/launch_sagemaker.py --mode eval
```

Required environment variables: `S3_BUCKET`, `SAGEMAKER_ROLE_ARN`, optionally
`AWS_REGION` (default `us-west-2`). See [notes/DEPLOYMENT.md](notes/DEPLOYMENT.md) for the
full runbook and [notes/gpu_interactive_guide.md](notes/gpu_interactive_guide.md)
for the interactive-SSH alternative.

---

## CI / CD / CT

| Layer | Trigger | What happens |
| :--- | :--- | :--- |
| CI | PR / push to `main`, `develop` | ruff lint + format check, pytest (smoke skipped), gitleaks secret scan |
| CD | push to `main` (code paths) or manual `workflow_dispatch` | preflight -> SageMaker job -> log collection -> notify, rollback marker on failure |
| CT | manual `workflow_dispatch` | version bump, model promotion, registry update |

Local pre-PR validation:

```bash
ruff check pyproject.toml tests/ .github/
ruff format --check pyproject.toml tests/
pytest -m "not slow" --cov=. --cov-fail-under=60
```

See [notes/VERSIONING.md](notes/VERSIONING.md) for the model registry and promotion rules.

---

## Configuration

`config.yaml` (training) - key settings:

| Key | Default | Meaning |
| :--- | :--- | :--- |
| `model_name` | `openai/clip-vit-large-patch14` | CLIP checkpoint (768-dim, matches SQID) |
| `training_mode` | `lora` | `lora` / `projection_only` / `full` |
| `lora_r` / `lora_alpha` / `lora_dropout` | `16` / `32` / `0.05` | LoRA adapter size and dropout |
| `epochs` / `batch_size` | `5` / `64` | Training length and batch size |
| `learning_rate` / `weight_decay` | `1e-4` / `0.01` | Optimizer settings |
| `mixed_precision` / `gradient_checkpointing` | `true` / `true` | Memory-saving switches |
| `best_metric` | `val_loss` | `val_loss` (min), `val_NDCG@10` or `val_Recall@10` (max) |
| `max_train_samples` / `max_val_samples` | `null` | Set small values for a fast CPU dry run |

`eval.py` and `download_images.py` are configured by editing constants at the
top of the file (no command-line args).

`launch_sagemaker.py` arguments:

| Argument | Default | Meaning |
| :--- | :--- | :--- |
| `--mode` | `train` | `train`, `eval`, `download` |
| `--bucket` / `--role-arn` / `--region` | env vars / `us-west-2` | AWS target |
| `--instance-type` | auto per mode | Override the SageMaker instance type |
| `--job-name` | auto | Custom job name |
| `--follow` | off | Stream CloudWatch logs (local use only, blocks in CI) |

---

## Notes

- `eval.py` downloads CLIP weights (~1.7 GB) on first run from Hugging Face.
- The local environment ships CPU-only torch; ViT-L/14 encoding is slow on CPU.
  Use a GPU (e.g. SageMaker or Colab) for the real baseline, or temporarily set
  `model_name: "openai/clip-vit-base-patch32"` in `config.yaml` for a fast local
  sanity check.
- Images are required on disk before training or evaluation (run
  `download_images.py` first).
- Model weights (`*.pt`, `*.pth`, `*.safetensors`) are git-ignored; they live in
  `checkpoints/` locally and in S3 for shared runs.
- `notes/` holds background material that is not part of the runnable pipeline.
