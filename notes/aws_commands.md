# AWS CLI Komutları ve Kullanım Rehberi (`aws_commands.md`)

Bu belge, projedeki verileri S3'e yüklemek, AWS kota/yetki kontrollerini yapmak ve **Zero-Shot CLIP Evaluation (`eval.py`)** işlemini **AWS SageMaker GPU** üzerinde çalıştırmak için kullanılan tüm komutları ve açıklamalarını içerir.

---

## 1. Kimlik, Bölge ve Bucket Kontrol Komutları

### 1.1. AWS CLI Çıktı Formatını Düzeltme
```powershell
aws configure set output json
```
* **Ne işe yarar?** AWS CLI'ın komut çıktılarını standart `json` formatında vermesini sağlar.
* **Neden kullanıldı?** İlk kurulumda `output = no` olarak kaldığı için AWS CLI `Unknown output type: no` hatası veriyordu; bu komutla kalıcı olarak düzeltildi.

### 1.2. Aktif AWS Kullanıcısını ve Hesap Numarasını Görme
```powershell
aws sts get-caller-identity
```
* **Ne işe yarar?** Terminalde hangi IAM kullanıcısı (`ml_school`) ve hangi AWS Hesap ID'si (`<YOUR_ACCOUNT_ID>`) ile işlem yapıldığını doğrular.
* **Neden kullanıldı?** Rol ARN'lerini (`arn:aws:iam::<YOUR_ACCOUNT_ID>:role/...`) doğru oluşturmak ve bağlantıyı teyit etmek için kullanıldı.

### 1.3. S3 Bucket Listeleme ve Bölge Kontrolü
```powershell
aws s3 ls
aws s3api get-bucket-location --bucket <YOUR_S3_BUCKET>
```
* **Ne işe yarar?** Hesaptaki S3 bucket'larını listeler ve bucket'ın hangi AWS bölgesinde (`us-west-2` - Oregon) olduğunu gösterir.
* **Neden kullanıldı?** Veri transfer ücreti ödememek ve yüksek hız almak için GPU makinesini bucket ile **aynı bölgede (`us-west-2`)** açmamız gerekiyordu.

---

## 2. Veri Yükleme (Data Sync) Komutları

### 2.1. Yerel Veriyi S3'e Senkronize Etme
```powershell
aws s3 sync E:\PERSONAL_PROJ\IMAGE_RETRIEVAL\data s3://<YOUR_S3_BUCKET>/data/
```
* **Ne işe yarar?** Yerel `data/` klasöründeki tüm alt klasörleri (`images/`, `processed/`, `raw/` - toplam ~8.737 dosya, 1.5 GB) paralel (multithreaded) olarak S3'e yükler.
* **Neden kullanıldı?** Tek tek yüklemeye göre çok daha hızlıdır ve bağlantı kopsa bile tekrar çalıştırıldığında sadece eksik kalan dosyaları yükler.

### 2.2. S3'teki Klasörleri Doğrulama
```powershell
aws s3 ls s3://<YOUR_S3_BUCKET>/data/
```
* **Ne işe yarar?** S3 içindeki `data/` dizininde `images/`, `processed/` ve `raw/` klasörlerinin eksiksiz oluştuğunu kontrol eder.

---

## 3. Kota (Service Quotas) ve IAM Yetki Komutları

### 3.1. EC2 ve SageMaker GPU Kotalarını Sorgulama
```powershell
# EC2 GPU kotalarını kontrol etme
aws service-quotas list-service-quotas --service-code ec2 --region us-west-2 --query "Quotas[?contains(QuotaName, 'G and VT')].[QuotaName,Value]"

# SageMaker GPU kotalarını kontrol etme
aws service-quotas list-service-quotas --service-code sagemaker --region us-west-2 --query "Quotas[?contains(QuotaName, 'g4dn') || contains(QuotaName, 'g5')].[QuotaName,Value]"
```
* **Ne işe yarar?** Hesabınızda hangi GPU tiplerine (`g4dn`, `g5`) izin verildiğini (kota değerini) listeler.
* **Neden kullanıldı?** EC2 GPU kotasının `0.0` (kapalı), ancak **SageMaker Training Job GPU kotasının `1.0` (açık)** olduğunu bu komut sayesinde tespit ettik ve beklemeden SageMaker üzerinden ilerledik.

### 3.2. Mevcut IAM Rollerini Listeleme ve Yetki Ekleme
```powershell
# İlgili IAM rollerini bulma
aws iam list-roles --query "Roles[?contains(RoleName, 'SageMaker')].[RoleName,Arn]"

# SageMaker rolüne S3 okuma/yazma yetkisi verme
aws iam attach-role-policy --role-name SageMakerExecutionRole --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess
```
* **Ne işe yarar?** SageMaker'ın çalışırken kullandığı `SageMakerExecutionRole` rolüne S3 erişim izni (`AmazonS3FullAccess`) bağlar.
* **Neden kullanıldı?** Varsayılan SageMaker rolü yalnızca adında `sagemaker` geçen bucket'lara erişebilir. Bizim bucket'ımız `image-retrieval-...` olduğu için GPU makinesinin verileri okuyup sonuçları yazabilmesi adına bu yetki eklendi.

---

## 4. Kodu Paketleme ve SageMaker GPU İşini Başlatma

### 4.1. Çalıştırılacak Kodları Paketleyip (`sourcedir.tar.gz`) S3'e Yükleme
```powershell
.\venv\Scripts\python.exe -c "
import io, tarfile
with tarfile.open('sourcedir.tar.gz', 'w:gz') as tar:
    tar.add('eval.py', arcname='eval.py')
    tar.add('dataset.py', arcname='dataset.py')
    req = ('transformers==4.41.2' + chr(10) + 'pyarrow' + chr(10)).encode('utf-8')
    info = tarfile.TarInfo(name='requirements.txt')
    info.size = len(req)
    tar.addfile(info, io.BytesIO(req))
"
aws s3 cp sourcedir.tar.gz s3://<YOUR_S3_BUCKET>/code/sourcedir.tar.gz
```
* **Ne işe yarar?** `eval.py`, `dataset.py` ve minimal `requirements.txt` dosyasını `.tar.gz` arşivi yapıp S3'e yükler.
* **Neden kullanıldı?**
  1. Yerel `requirements.txt` içinde Windows paketleri ve CPU-PyTorch olduğu için, container'ın GPU-PyTorch sürümünü bozmamak adına sadece gerekli kütüphaneler eklendi.
  2. **Sürüm Sabitleme (`transformers==4.41.2`):** Sürüm belirtilmezse en güncel `transformers` (5.x) kurulur ve SageMaker'daki PyTorch 2.1.0 imajıyla sürüm uyumsuzluğu (`CLIPModel requires the PyTorch library`) hatası verir. `4.41.2` sürümü PyTorch 2.1 ile tam uyumludur.

### 4.2. SageMaker GPU İşini (`create-training-job`) Başlatma
```powershell
aws sagemaker create-training-job --region us-west-2 --cli-input-json <JSON_PAYLOAD>
```
Kullanılan JSON konfigürasyonu:
```json
{
  "TrainingJobName": "clip-zero-shot-eval-03",
  "AlgorithmSpecification": {
    "TrainingImage": "763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-training:2.1.0-gpu-py310-cu121-ubuntu20.04-sagemaker",
    "TrainingInputMode": "File"
  },
  "RoleArn": "arn:aws:iam::<YOUR_ACCOUNT_ID>:role/SageMakerExecutionRole",
  "InputDataConfig": [
    {
      "ChannelName": "splits",
      "DataSource": {
        "S3DataSource": {
          "S3DataType": "S3Prefix",
          "S3Uri": "s3://<YOUR_S3_BUCKET>/data/processed/splits/",
          "S3DataDistributionType": "FullyReplicated"
        }
      }
    },
    {
      "ChannelName": "images",
      "DataSource": {
        "S3DataSource": {
          "S3DataType": "S3Prefix",
          "S3Uri": "s3://<YOUR_S3_BUCKET>/data/images/",
          "S3DataDistributionType": "FullyReplicated"
        }
      }
    }
  ],
  "OutputDataConfig": {
    "S3OutputPath": "s3://<YOUR_S3_BUCKET>/sagemaker-output/"
  },
  "ResourceConfig": {
    "InstanceType": "ml.g4dn.xlarge",
    "InstanceCount": 1,
    "VolumeSizeInGB": 50
  },
  "StoppingCondition": {
    "MaxRuntimeInSeconds": 3600
  },
  "HyperParameters": {
    "sagemaker_program": "\"eval.py\"",
    "sagemaker_submit_directory": "\"s3://<YOUR_S3_BUCKET>/code/sourcedir.tar.gz\""
  }
}
```
* **Ne işe yarar?**
  * `ml.g4dn.xlarge` (NVIDIA T4 16GB GPU) makinesini başlatır.
  * AWS'nin resmi CUDA 12.1 + PyTorch 2.1.0 Docker imajını kullanır.
  * Sadece `splits` (~15 MB) ve `images` (~360 MB) klasörlerini bağlar (1.1 GB'lık `raw` klasörünü boşuna indirmez).
  * `eval.py` bittiğinde üretilen `zero_shot_results.json` dosyasını `sagemaker-output/` altına kaydedip **makineyi otomatik kapatır**.

---

## 5. İş Durumunu, Canlı Logları Takip Etme ve Sonuçları İndirme

### 5.1. İşin Durumunu ve Faturalanan Süreyi Sorgulama
```powershell
aws sagemaker describe-training-job --region us-west-2 --training-job-name clip-zero-shot-eval-03 --query "[TrainingJobStatus,SecondaryStatus,BillableTimeInSeconds,FailureReason]"
```
* **Ne işe yarar?** İşin hangi aşamada olduğunu (`Starting`, `Downloading`, `Training`, `Uploading`, `Completed`), kaç saniye GPU kullanıldığını (`BillableTimeInSeconds`: 615 sn) ve olası hata mesajlarını (`FailureReason`) gösterir.

### 5.2. CloudWatch Canlı Loglarını ve `tqdm` İlerleme Çubuğunu Okuma
```powershell
$env:PYTHONUTF8="1"
aws logs filter-log-events --region us-west-2 --log-group-name /aws/sagemaker/TrainingJobs --log-stream-name-prefix clip-zero-shot-eval-03 --query "events[-20:].message"
```
* **Ne işe yarar?** SageMaker üzerinde çalışan scriptin `print` çıktılarını ve `tqdm` ilerleme çubuğunu (`Encoding images: 77%|█████...`) canlı olarak terminale getirir.
* **Neden `$env:PYTHONUTF8="1"` kullanıldı?** Windows PowerShell varsayılan olarak `cp1252` karakter kodlaması kullandığı için `tqdm` içerisindeki Unicode blok karakterleri (`█`) `charmap codec can't encode characters` hatasına yol açar; `PYTHONUTF8="1"` bu hatayı önler.

### 5.3. Sonuç Dosyasını S3'ten İndirme ve JSON Olarak Yedekleme
```powershell
# 1. SageMaker'ın ürettiği output.tar.gz arşivini indirip analysis/ klasörüne çıkarma
aws s3 cp s3://<YOUR_S3_BUCKET>/sagemaker-output/clip-zero-shot-eval-03/output/output.tar.gz ./output.tar.gz
.\venv\Scripts\python.exe -c "import tarfile; t=tarfile.open('output.tar.gz'); t.extractall('analysis')"
Remove-Item output.tar.gz

# 2. Çıkan JSON dosyasını doğrudan okunabilir şekilde S3 results/ klasörüne yedekleme
aws s3 cp analysis\zero_shot_results.json s3://<YOUR_S3_BUCKET>/results/zero_shot_results.json
```
* **Ne işe yarar?** SageMaker'ın sıkıştırılmış olarak kaydettiği çıktıyı yerel `analysis/zero_shot_results.json` konumuna açar ve ayrıca S3 üzerinde sıkıştırılmamış saf `.json` olarak `results/zero_shot_results.json` altında saklar.

---

## 6. Görsel İndirme (`clip-download-images-01`) ve 5-Epoch LoRA Fine-Tuning (`clip-lora-finetune-5ep-03`) Komutları

### 6.1. Tüm Görselleri (`81,054` Adet) Bulutta İndirip Tek Arşiv (`images_bundle.tar`) Yapma
```powershell
aws sagemaker create-training-job --region us-west-2 --cli-input-json file://job_download.json
```
* **Ne işe yarar?** `ml.m5.xlarge` (CPU) makinesi üzerinde `download_images.py` scriptini çalıştırarak Train (`E`), Validation ve Test setlerindeki eksik 72,329 görseli saniyede ~475 görsel hızla indirir ve toplam **81,054 görseli** tek bir `s3://<YOUR_S3_BUCKET>/data/images_bundle.tar` (3.3 GiB) dosyası olarak paketler.
* **Neden tek `.tar` dosyası yapıldı?** S3'ten 81,000 küçük `.jpg` dosyasını tek tek kopyalamak ~25 dakika sürerken, tek bir `.tar` arşivini indirip diskte açmak sadece **~25 saniye** sürer.

### 6.2. 5-Epoch LoRA Fine-Tuning İşini (`ml.g5.xlarge` - NVIDIA A10G 24GB GPU) Başlatma
```json
{
  "TrainingJobName": "clip-lora-finetune-5ep-03",
  "AlgorithmSpecification": {
    "TrainingImage": "763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-training:2.1.0-gpu-py310-cu121-ubuntu20.04-sagemaker",
    "TrainingInputMode": "File"
  },
  "RoleArn": "arn:aws:iam::<YOUR_ACCOUNT_ID>:role/SageMakerExecutionRole",
  "InputDataConfig": [
    {
      "ChannelName": "splits",
      "DataSource": {
        "S3DataSource": {
          "S3DataType": "S3Prefix",
          "S3Uri": "s3://<YOUR_S3_BUCKET>/data/processed/splits/",
          "S3DataDistributionType": "FullyReplicated"
        }
      }
    },
    {
      "ChannelName": "images_bundle",
      "DataSource": {
        "S3DataSource": {
          "S3DataType": "S3Prefix",
          "S3Uri": "s3://<YOUR_S3_BUCKET>/data/images_bundle.tar",
          "S3DataDistributionType": "FullyReplicated"
        }
      }
    }
  ],
  "OutputDataConfig": {
    "S3OutputPath": "s3://<YOUR_S3_BUCKET>/sagemaker-output/"
  },
  "ResourceConfig": {
    "InstanceType": "ml.g5.xlarge",
    "InstanceCount": 1,
    "VolumeSizeInGB": 100
  },
  "StoppingCondition": {
    "MaxRuntimeInSeconds": 28800
  },
  "Environment": {
    "S3_BUCKET": "<YOUR_S3_BUCKET>",
    "CONFIG_PATH": "config.yaml"
  },
  "HyperParameters": {
    "sagemaker_program": "\"train.py\"",
    "sagemaker_submit_directory": "\"s3://<YOUR_S3_BUCKET>/code/sourcedir_train.tar.gz\""
  }
}
```

### 6.3. Eğitim Sonuçlarını ve `best_model` Ağırlıklarını S3'ten Yerel Bilgisayara İndirme
```powershell
# Eğitim geçmişi ve Test seti sonuçlarını indirme
aws s3 cp s3://<YOUR_S3_BUCKET>/checkpoints/vit-large-p14_lora-r16-a32_bs64_lr1e-04_ep5/training_history.json E:\PERSONAL_PROJ\IMAGE_RETRIEVAL\analysis\training_history.json
aws s3 cp s3://<YOUR_S3_BUCKET>/checkpoints/vit-large-p14_lora-r16-a32_bs64_lr1e-04_ep5/finetuned_test_results.json E:\PERSONAL_PROJ\IMAGE_RETRIEVAL\analysis\finetuned_test_results.json

# En iyi modelin (best_model) ağırlıklarını ve metriklerini indirme
aws s3 sync s3://<YOUR_S3_BUCKET>/checkpoints/vit-large-p14_lora-r16-a32_bs64_lr1e-04_ep5/best_model/ E:\PERSONAL_PROJ\IMAGE_RETRIEVAL\checkpoints\vit-large-p14_lora-r16-a32_bs64_lr1e-04_ep5\best_model\
```

---

## 7. Resmi AWS CLI Dokümantasyon ve Referans Linkleri

Bu projede kullandığımız tüm komutların resmi AWS CLI v2 referans sayfaları aşağıdadır:

1. **Genel AWS CLI v2 Komut Referansı (Ana Sayfa):**
   * [AWS CLI v2 Command Reference](https://awscli.amazonaws.com/v2/documentation/api/latest/index.html)
   * [AWS CLI `--query` (JMESPath) Filtreleme Rehberi](https://docs.aws.amazon.com/cli/latest/userguide/cli-usage-filter.html)

2. **SageMaker Eğitim ve İzleme Komutları (`aws sagemaker`):**
   * [`aws sagemaker create-training-job` (GPU/CPU Eğitim İşi Başlatma ve JSON Parametreleri)](https://awscli.amazonaws.com/v2/documentation/api/latest/reference/sagemaker/create-training-job.html)
   * [`aws sagemaker describe-training-job` (İş Durumu, Faturalanan Süre ve Hata Sorgulama)](https://awscli.amazonaws.com/v2/documentation/api/latest/reference/sagemaker/describe-training-job.html)
   * [`aws sagemaker list-training-jobs` (Çalışan veya Tamamlanan İşleri Listeleme)](https://awscli.amazonaws.com/v2/documentation/api/latest/reference/sagemaker/list-training-jobs.html)
   * [`aws sagemaker stop-training-job` (Çalışan Bir İşi Manuel Durdurma)](https://awscli.amazonaws.com/v2/documentation/api/latest/reference/sagemaker/stop-training-job.html)
   * [AWS Deep Learning Containers (PyTorch GPU Docker İmaj Listesi)](https://github.com/aws/deep-learning-containers/blob/master/available_images.md)

3. **S3 Dosya Transfer Komutları (`aws s3` & `aws s3api`):**
   * [`aws s3 sync` (Klasörleri Paralel Senkronize Etme)](https://awscli.amazonaws.com/v2/documentation/api/latest/reference/s3/sync.html)
   * [`aws s3 cp` (Tekil Dosya Kopyalama / İndirme / Yükleme)](https://awscli.amazonaws.com/v2/documentation/api/latest/reference/s3/cp.html)
   * [`aws s3 ls` (Bucket ve Klasör İçeriğini Listeleme)](https://awscli.amazonaws.com/v2/documentation/api/latest/reference/s3/ls.html)
   * [`aws s3api get-bucket-location` (Bucket Bölgesini Öğrenme)](https://awscli.amazonaws.com/v2/documentation/api/latest/reference/s3api/get-bucket-location.html)

4. **CloudWatch Canlı Log Okuma Komutları (`aws logs`):**
   * [`aws logs filter-log-events` (SageMaker Container Loglarını Terminalden Okuma)](https://awscli.amazonaws.com/v2/documentation/api/latest/reference/logs/filter-log-events.html)

5. **Kota, Kimlik ve IAM Yetki Komutları (`service-quotas`, `sts`, `iam`):**
   * [`aws service-quotas list-service-quotas` (EC2 ve SageMaker GPU Kotalarını Görme)](https://awscli.amazonaws.com/v2/documentation/api/latest/reference/service-quotas/list-service-quotas.html)
   * [`aws sts get-caller-identity` (Aktif Kullanıcı ve Hesap ID Kontrolü)](https://awscli.amazonaws.com/v2/documentation/api/latest/reference/sts/get-caller-identity.html)
   * [`aws iam list-roles` (IAM Rollerini Listeleme)](https://awscli.amazonaws.com/v2/documentation/api/latest/reference/iam/list-roles.html)
   * [`aws iam attach-role-policy` (Role S3 Erişim Politikası Ekleme)](https://awscli.amazonaws.com/v2/documentation/api/latest/reference/iam/attach-role-policy.html)
