# Model Versioning & Registry — IMAGE_RETRIEVAL

Bu belge, eğitilen CLIP modellerinin nasıl versiyonlanacağını, S3'te nasıl
saklanacağını ve "production" modelinin nasıl seçileceğini tanımlar.

---

## 1. Versiyon Şeması (Semantic Versioning)

Format: **`v{MAJOR}.{MINOR}.{PATCH}`** (örn: `v1.2.3`)

### Bump Kuralları

| Bump | Ne zaman | Örnek |
|---|---|---|
| **MAJOR** | Model mimarisi değişti (CLIP-base → CLIP-large), veya loss fonksiyonu değişti | `v1.x.x` → `v2.0.0` |
| **MINOR** | Hiperparametre değişti (lr, batch_size, LoRA r/alpha), veya yeni veri seti eklendi | `v1.2.x` → `v1.3.0` |
| **PATCH** | Sadece seed değişti, veya aynı config ile yeniden eğitildi (reproducibility check) | `v1.2.3` → `v1.2.4` |

### Otomatik Bump (CI/CD tarafından)

`scripts/bump_version.py` mevcut `latest_version`'ı okur, değişiklik tipine göre bump yapar:

```bash
python scripts/bump_version.py --bump patch   # v1.2.3 → v1.2.4
python scripts/bump_version.py --bump minor   # v1.2.3 → v1.3.0
python scripts/bump_version.py --bump major   # v1.2.3 → v2.0.0
```

### Manuel Override

Acil durumlarda (rollback, hotfix) explicit versiyon verilebilir:

```bash
python scripts/bump_version.py --set v1.2.5
```

---

## 2. S3 Layout (Versiyonlanmış)

```
s3://{bucket}/
├── models/
│   ├── registry.json              # tüm modellerin index'i
│   ├── v1.0.0/
│   │   ├── checkpoint.pt          # model state_dict + config
│   │   ├── config.yaml            # eğitim config snapshot
│   │   ├── metrics.json           # test set metrikleri
│   │   ├── training_history.json  # tüm epoch'lar
│   │   └── metadata.json          # sha, timestamp, git_commit, config_hash
│   ├── v1.1.0/
│   │   └── ...
│   ├── v1.2.3/
│   │   └── ...
│   └── production.json            # aktif production model pointer
├── deployments/
│   ├── failed-{sha}.json          # rollback marker (CD'den)
│   └── promoted-{version}.json    # promotion audit trail
└── sagemaker-output/              # SageMaker job output (ham)
    └── {job_name}/
        └── output/
            └── model.tar.gz
```

### `registry.json` Şeması

```json
{
  "schema_version": "1.0",
  "last_updated": "2025-10-20T14:30:22Z",
  "models": [
    {
      "version": "v1.2.3",
      "s3_path": "s3://bucket/models/v1.2.3/",
      "git_sha": "a3f9c2d",
      "git_branch": "main",
      "trained_at": "2025-10-19T14:30:22Z",
      "trained_by": "github-actions",
      "config_hash": "sha256:abc123...",
      "metrics": {
        "test_Recall@10": 0.912,
        "test_NDCG@10": 0.847,
        "test_MRR": 0.723
      },
      "promoted": false,
      "promoted_at": null,
      "superseded_by": null
    }
  ],
  "production": "v1.2.3"
}
```

### `metadata.json` (her model için)

```json
{
  "version": "v1.2.3",
  "git_sha": "a3f9c2d",
  "git_branch": "main",
  "trained_at": "2025-10-19T14:30:22Z",
  "trained_by": "github-actions",
  "config_hash": "sha256:abc123...",
  "config_diff_from_previous": {
    "learning_rate": {"old": "1e-4", "new": "1e-5"},
    "batch_size": {"old": 64, "new": 32}
  },
  "sagemaker_job": "clip-train-a3f9c2d-20251019-143022",
  "instance_type": "ml.g5.xlarge",
  "training_duration_seconds": 18432
}
```

---

## 3. Promotion (Manuel)

Promotion **otomatik değil** — `workflow_dispatch` ile manuel tetiklenir.

### Neden Manuel?

- Recall@10 iyileşmesi her zaman "daha iyi model" demek değil
- Edge case'ler (uzun kuyruklu sorgular, niche kategori) gözden kaçabilir
- İş kararı: "bu model production'a çıksın mı?" → insan gözü gerekir

### Promotion Adımları

1. **Actions UI → Promote workflow → Run workflow**
2. **version** seç (dropdown: registry'deki tüm modeller)
3. **reason** yaz (örn: "Recall@10 +2.1%, latency OK")
4. Workflow:
   - Registry'de `promoted: true` işaretler
   - `production.json`'ı günceller
   - `deployments/promoted-{version}.json` audit dosyası yazar
   - Slack bildirimi gönderir (opsiyonel)

### Demotion / Rollback

```bash
# Production'ı önceki versiyona geri al
python scripts/promote_model.py --set-production v1.2.2 --reason "v1.2.3 latency regression"
```

---

## 4. Config Hash (Değişiklik Tespiti)

Her eğitim config'i SHA-256 ile hash'lenir. Bu sayede:

- Aynı config + aynı veri + aynı seed → aynı hash → **reproducibility check**
- Config değişti mi otomatik tespit → bump tipi önerilir (minor)

```python
import hashlib
import yaml


def config_hash(cfg: dict) -> str:
    """Config'in SHA-256 hash'i (sorted keys, no comments)."""
    canonical = yaml.safe_dump(cfg, sort_keys=True, default_flow_style=False)
    return "sha256:" + hashlib.sha256(canonical.encode()).hexdigest()[:16]
```

---

## 5. CLI Kullanımı

### Versiyon bump

`bump_version.py` sürümü hesaplar **ve registry'ye yazar** (`models/registry.json`).
`--s3-bucket` verilirse registry S3'ten okunur (tek doğruluk kaynağı) ve oraya da yazılır.

```bash
# Otomatik bump (config diff'e göre) — local registry'ye yazar
python scripts/bump_version.py --auto

# CI'da: S3 registry'sini kullan (versiyonlar birikimli artsın)
S3_BUCKET=my-bucket python scripts/bump_version.py --auto --s3-bucket "$S3_BUCKET"

# Manuel bump
python scripts/bump_version.py --bump minor

# Explicit set
python scripts/bump_version.py --set v2.0.0

# Sadece hesapla, registry'ye yazma
python scripts/bump_version.py --auto --dry-run
```

Idempotency: aynı config hash'i registry'deki son (henüz promote edilmemiş) modelle
eşleşirse sürüm artırılmaz, `bump_type=unchanged` döner.

### Registry sorgulama

```bash
# Tüm modelleri listele
python scripts/promote_model.py --list

# Production modeli göster
python scripts/promote_model.py --current

# Belirli modelin metrikleri
python scripts/promote_model.py --show v1.2.3
```

### Promotion

```bash
# Manuel promotion (CI'dan)
python scripts/promote_model.py --promote v1.2.3 --reason "Recall@10 +2.1%"

# Rollback
python scripts/promote_model.py --set-production v1.2.2 --reason "v1.2.3 regression"
```

---

## 6. CI/CD Entegrasyonu

### `deploy.yml` (mevcut) — uygulanmış

`preflight` job: `short_sha` + `job_name` üretir (secret/config kontrolleri).

`deploy` job (AWS creds burada):
1. `scripts/bump_version.py --auto --s3-bucket "$S3_BUCKET"` → versiyon S3 registry'sine yazılır (`status: pending`)
2. `scripts/launch_sagemaker.py` → SageMaker Training Job (`MODEL_VERSION` = adım 1'in çıktısı)
3. Job bitene kadar polling + CloudWatch logları `ci-artifacts/` altına
4. `s3://{bucket}/sagemaker-output/{job}/` altında artifact doğrulaması
5. Artifact'lar GitHub Actions'a yüklenir (30 gün)

`notify` job: failure durumunda `deployments/failed-{sha}.json` rollback marker + Slack.

Not: `train.py` her run için `metadata.json` (version, git_sha, config_hash, sagemaker_job)
yazar; registry entry'sini `metadata.json` ile zenginleştirmek ileride eklenebilir.

### `promote.yml` (yeni) — workflow_dispatch

Manuel promotion job'ı:
1. Input: `version` (dropdown), `reason` (text)
2. Registry'de `promoted: true` işaretler
3. `production.json`'ı günceller
4. Audit dosyası yazar
5. Slack bildirimi

---

## 7. Güvenlik

- `registry.json` ve `production.json` S3'te public değil (bucket policy)
- Promotion audit trail **silinmez** (compliance)
- Config hash ile "hangi config ile eğitildi?" sorusu her zaman yanıtlanabilir
