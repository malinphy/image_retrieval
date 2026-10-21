# Deployment Guide — IMAGE_RETRIEVAL

Bu belge, **Continuous Deployment (CD)** katmanının nasıl çalıştığını, hangi
secret'ların gerektiğini ve manuel müdahale senaryolarını açıklar.

---

## 1. Mimari Genel Bakış

```
main branch'e merge
       │
       ▼
┌──────────────────┐
│  preflight job   │  ← secrets doğrula, config.yaml temiz mi?
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│   deploy job     │  ← AWS auth → launch_sagemaker.py → SageMaker job
│   (8 saat max)   │     → log toplama → artifact doğrulama
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│   notify job     │  ← başarılı: summary yaz
│                  │     başarısız: rollback marker + Slack
└──────────────────┘
```

### Tetikleme Koşulları

`deploy.yml` aşağıdaki durumlarda tetiklenir:

| Tetikleyici | Koşul |
|---|---|
| `push` to `main` | Aşağıdaki dosyalardan biri değişti: `train.py`, `eval.py`, `dataset.py`, `download_images.py`, `config.yaml`, `scripts/launch_sagemaker.py`, `requirements.txt`, `.github/workflows/deploy.yml` |
| `workflow_dispatch` | Manuel tetikleme (Actions UI → Run workflow → mode seç) |

> **Not:** Docs-only commit'ler (`.md`, `LICENSE`, vb.) deploy tetiklemez.

---

## 2. Gerekli GitHub Secrets

Repo → Settings → Secrets and variables → Actions → New repository secret

| Secret | Açıklama | Örnek |
|---|---|---|
| `AWS_ACCESS_KEY_ID` | IAM user access key | `AKIA...` |
| `AWS_SECRET_ACCESS_KEY` | IAM user secret key | `wJalr...` |
| `S3_BUCKET` | SageMaker'ın okuyup yazdığı bucket | `image-retrieval-myorg-us-west-2` |
| `SAGEMAKER_ROLE_ARN` | SageMaker execution role ARN | `arn:aws:iam::123456789012:role/SageMakerExecutionRole` |

### Opsiyonel Secrets

| Secret | Varsayılan | Açıklama |
|---|---|---|
| `AWS_REGION` | `us-west-2` | AWS bölgesi |
| `SLACK_WEBHOOK_URL` | (yok) | Failure bildirimi için Slack incoming webhook |

### IAM Policy (minimum)

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "S3ReadWrite",
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:PutObject",
        "s3:ListBucket",
        "s3:DeleteObject"
      ],
      "Resource": [
        "arn:aws:s3:::${S3_BUCKET}",
        "arn:aws:s3:::${S3_BUCKET}/*"
      ]
    },
    {
      "Sid": "SageMakerJobs",
      "Effect": "Allow",
      "Action": [
        "sagemaker:CreateTrainingJob",
        "sagemaker:DescribeTrainingJob",
        "sagemaker:StopTrainingJob",
        "sagemaker:ListTrainingJobs"
      ],
      "Resource": "*"
    },
    {
      "Sid": "CloudWatchLogs",
      "Effect": "Allow",
      "Action": [
        "logs:GetLogEvents",
        "logs:FilterLogEvents",
        "logs:DescribeLogStreams"
      ],
      "Resource": "arn:aws:logs:*:*:log-group:/aws/sagemaker/TrainingJobs:*"
    }
  ]
}
```

> **Güvenlik notu:** Access key yerine **OIDC** (IAM Role + GitHub OIDC trust)
> kullanmak daha güvenlidir. İleride `id-token: write` permission zaten
> workflow'da tanımlı; OIDC'ye geçiş tek PR'da yapılabilir.

---

## 3. Deploy Akışı (adım adım)

### 3.1. Pre-flight (`preflight` job)

1. `git log` erişimi için `fetch-depth: 0` ile checkout
2. Job adı üret: `clip-{mode}-{short_sha}-{timestamp}`
3. `S3_BUCKET` ve `SAGEMAKER_ROLE_ARN` secret'ları boş mu kontrolü
4. `config.yaml` içinde AWS account ID (12 haneli sayı) sızıntısı var mı?

### 3.2. Deploy (`deploy` job)

1. AWS credentials configure (`aws-actions/configure-aws-credentials@v4`)
2. `aws sts get-caller-identity` ile auth doğrula
3. `python scripts/launch_sagemaker.py --mode {mode} --job-name {name}` çağır
   - `--follow` KULLANILMAZ (CI blocking olur, log'lar aşağıda toplanır)
4. `boto3` ile job tamamlanana kadar poll (30s interval, 8 saat timeout)
5. CloudWatch logları `ci-artifacts/logs/{job_name}.log` olarak indirilir
6. `s3://{bucket}/sagemaker-output/{job_name}/` altında model artifact doğrulanır
7. Tüm artifact'ler GitHub Actions artifact olarak 30 gün saklanır

### 3.3. Notify (`notify` job)

- **Başarılı:** GitHub Step Summary'ye job bilgileri yazılır
- **Başarısız:**
  - `s3://{bucket}/deployments/failed-{sha}.json` rollback marker yazılır
  - `SLACK_WEBHOOK_URL` tanımlıysa Slack bildirimi gönderilir

---

## 4. SageMaker Job Modları

`workflow_dispatch` ile manuel tetiklerken mode seçilebilir:

| Mode | Entry script | Default instance | Max runtime | Disk |
|---|---|---|---|---|
| `train` | `train.py` | `ml.g5.xlarge` (A10G 24GB) | 8 saat | 100 GB |
| `eval` | `eval.py` | `ml.g4dn.xlarge` (T4 16GB) | 1 saat | 50 GB |
| `download` | `download_images.py` | `ml.m5.xlarge` (CPU) | 2 saat | 50 GB |

> **Otomatik tetikleme (main merge):** her zaman `train` modu çalışır.
> Eval/download için manuel tetikleme gerekir.

---

## 5. Manuel Müdahale Senaryoları

### 5.1. Job hâlâ çalışıyorken iptal etmek

```bash
aws sagemaker stop-training-job \
  --training-job-name clip-train-abc1234-20251019-143022
```

### 5.2. Başarısız deploy'u rollback etmek

```bash
# 1. Hangi SHA'nın başarısız olduğunu gör
aws s3 ls s3://$S3_BUCKET/deployments/

# 2. main'i önceki başarılı commit'e revert et
git revert <failed_merge_commit_sha>
git push origin main

# 3. (Opsiyonel) SageMaker'da yarım kalan job'ları temizle
aws sagemaker list-training-jobs --status-equals InProgress
```

### 5.3. SageMaker job loglarını canlı izlemek (lokal)

```bash
python scripts/launch_sagemaker.py --mode train --follow
```

### 5.4. Sadece eval çalıştırmak (deploy tetiklemeden)

```bash
# Actions UI → Deploy workflow → Run workflow → mode: eval
```

---

## 6. Maliyet Kontrolleri

| Kontrol | Nerede |
|---|---|
| `MaxRuntimeInSeconds` (8 saat) | `launch_sagemaker.py` `default_instances["train"]` |
| `VolumeSizeInGB` (100 GB) | Aynı yer |
| `StoppingCondition` | SageMaker API parametresi |
| GitHub Actions timeout (540 dk) | `deploy.yml` `timeout-minutes` |
| `concurrency.cancel-in-progress: false` | Yeni push geldiğinde mevcut deploy iptal **edilmez** |

> **SageMaker maliyet tahmini** (us-west-2, Ekim 2026):
> - `ml.g5.xlarge`: ~$1.50/saat → 8 saatlik job = ~$12
> - `ml.g4dn.xlarge`: ~$0.70/saat → 1 saatlik eval = ~$0.70
> - `ml.m5.xlarge`: ~$0.25/saat → 2 saatlik download = ~$0.50

---

## 7. Troubleshooting

| Sorun | Çözüm |
|---|---|
| `::error::S3_BUCKET secret is missing` | Repo Settings → Secrets → `S3_BUCKET` ekle |
| `::error::config.yaml içinde AWS account ID tespit edildi!` | `config.yaml`'dan account ID'yi kaldır, `${S3_BUCKET}` placeholder kullan |
| `FailureReason: CapacityError` | SageMaker'da o instance tipinde kota yok. `service-quotas` ile kota artırımı iste |
| Job 8 saatte timeout | `launch_sagemaker.py` `max_run_seconds` artır (veya job'u böl) |
| `aws sts get-caller-identity` başarısız | IAM key expire olmuş, yeni key üret + secret güncelle |
| Slack bildirimi gelmiyor | `SLACK_WEBHOOK_URL` secret'ı boş → workflow sessizce geçer (by design) |

---

## 8. Güvenlik Kontrol Listesi

- [x] AWS account ID `config.yaml`'dan kaldırıldı (`${S3_BUCKET}` placeholder)
- [x] `download_images.py` `S3_BUCKET` env var zorunlu
- [x] `train.py` `_resolve_env_placeholders()` ile runtime substitution
- [x] Pre-flight job config.yaml'da account ID sızıntısı kontrol ediyor
- [x] `gitleaks` CI job secret tarama yapıyor
- [x] Access key GitHub Secrets'ta (encrypted at rest)
- [ ] **TODO:** OIDC'ye geçiş (access key yerine IAM Role)
- [ ] **TODO:** `concurrency.cancel-in-progress: true` (deploy'lar arası sıralı)
