# Doğrudan Bulut GPU Makinesine Bağlanarak Çalışma Rehberi (`gpu_interactive_guide.md`)

Bu rehber, kodun içine ekstra AWS/SageMaker paketleme kodları yazmadan, **doğrudan buluttaki bir GPU makinesinin terminaline bağlanıp** kendi bilgisayarındaymış gibi `python train.py` ve `python eval.py` çalıştırma yöntemini adım adım anlatır.

---

## 1. Bu Yöntemin Avantajları ve Farkı Nedir?

| Özellik | Training Job Yöntemi *(Önceki Yöntem)* | Doğrudan GPU Makinesi *(Bu Rehberdeki Yöntem)* |
| :--- | :--- | :--- |
| **Kullanım Şekli** | Kodu paketle (`.tar.gz`) $\rightarrow$ S3'e at $\rightarrow$ Job başlat $\rightarrow$ Log bekle. | Makineye bağlan $\rightarrow$ Terminalde doğrudan `python train.py` yaz. |
| **Gecikme Süresi** | Her denemede makine açılması için 2-3 dk beklenir. | **0 saniye!** Kodu değiştirdiğin an saniyesinde GPU'da çalışır. |
| **Ekstra AWS Kodu** | JSON dosyaları veya `launch_sagemaker.py` gerekir. | **0 satır ekstra kod!** Sadece saf PyTorch kodun çalışır. |
| **Sabit Disk (Veriler)** | Her iş bittiğinde sıfırlanır. | **Kalıcıdır (`/home/ec2-user/SageMaker`).** Veriyi 1 kez indirirsin, makineyi kapatıp açsan da silinmez. |
| **Kapanma Şekli** | Eğitim bitince **otomatik** kapanır. | İşin bitince senin **Stop (Durdur)** komutu vermen gerekir. |

---

## 2. Hazır Açık Olan `ml.g5.2xlarge` GPU Makinesini Kullanma (Adım Adım)

Hesabında **`ml.g5.2xlarge for notebook instance usage` (1x NVIDIA A10G 24GB GPU, 8 vCPU, 32 GB RAM)** kotası halihazırda **açık (`1.0`)** durumdadır.

### Adım 1: GPU Makinesini Oluşturma *(Sadece İlk Seferde 1 Kez Yapılır)*

Kendi bilgisayarındaki PowerShell terminalinde şu komutu çalıştırarak **100 GB kalıcı diske** sahip `ml.g5.2xlarge` makinesini oluştur ve başlat:

```powershell
aws sagemaker create-notebook-instance `
  --notebook-instance-name clip-gpu-workstation `
  --instance-type ml.g5.2xlarge `
  --role-arn arn:aws:iam::<YOUR_ACCOUNT_ID>:role/SageMakerExecutionRole `
  --volume-size-in-gb 100 `
  --region us-west-2
```

Makinenin açılıp hazır (`InService`) hale gelmesi yaklaşık **2-3 dakika** sürer. Durumunu kontrol etmek için:

```powershell
aws sagemaker describe-notebook-instance `
  --notebook-instance-name clip-gpu-workstation `
  --region us-west-2 `
  --query "[NotebookInstanceStatus,Url]"
```
*(Çıktıda `"InService"` gördüğünde makine hazırdır).*

---

### Adım 2: Makineye Tek Tıkla Bağlanma (Tarayıcı veya Terminal)

Makine `InService` olduğunda kendi bilgisayarında şu komutu çalıştırarak **doğrudan giriş linki** alabilirsin:

```powershell
aws sagemaker create-presigned-notebook-instance-url `
  --notebook-instance-name clip-gpu-workstation `
  --region us-west-2 `
  --query "AuthorizedUrl" --output text
```
1. Bu komutun verdiği `https://...` linkini kopyalayıp tarayıcında aç (veya AWS Console $\rightarrow$ SageMaker $\rightarrow$ Notebook instances $\rightarrow$ **Open JupyterLab**'e tıkla).
2. Açılan ekranda **File $\rightarrow$ New $\rightarrow$ Terminal** seçeneğine tıkla.
3. Artık doğrudan 24 GB GPU'lu Linux makinenin içindesin! Kontrol etmek için terminale şunu yaz:
   ```bash
   nvidia-smi
   ```

---

### Adım 3: Verileri ve Kodu Makineye Çekme *(Kalıcı Disk — Sadece 1 Kez Yapılır)*

Makine içindeki **`/home/ec2-user/SageMaker`** klasörü senin **kalıcı sabit diskindir** (makineyi `Stop` yapsan bile buradaki dosyalar asla silinmez).

Açtığın GPU terminalinde sırasıyla şu komutları çalıştır:

```bash
# 1. Kalıcı disk klasörüne git ve proje klasörünü oluştur
cd /home/ec2-user/SageMaker
mkdir -p IMAGE_RETRIEVAL && cd IMAGE_RETRIEVAL

# 2. (Eğer kodu GitHub'a attıysan git clone yapabilir veya S3'ten çekebilirsin)
# S3'teki hazır veri setini (splits) indir (~5 saniye):
mkdir -p data/processed/splits data/images
aws s3 sync s3://<YOUR_S3_BUCKET>/data/processed/splits/ data/processed/splits/

# 3. S3'teki 81,054 görselli hazır arşivi indirip aç (~25 saniye):
aws s3 cp s3://<YOUR_S3_BUCKET>/data/images_bundle.tar ./images_bundle.tar
tar -xf images_bundle.tar -C data/images/
rm images_bundle.tar
```

Ardından makine içindeki hazır PyTorch GPU ortamını aktif edip kütüphaneleri kur:

```bash
# Hazır Conda PyTorch ortamını aktif et
conda activate pytorch_p310

# Gerekli kütüphaneleri kur
pip install transformers==4.41.2 peft==0.11.1 accelerate==0.30.1 pyarrow pyyaml aiohttp
```

---

### Adım 4: Kodları Doğrudan Çalıştırma (0 Satır Ekstra AWS Kodu!)

Artık kendi bilgisayarında çalışıyormuş gibi doğrudan komutları çalıştırabilirsin:

```bash
# Zero-shot veya Fine-tuned model değerlendirmesi:
python eval.py

# Veya doğrudan eğitime başlama:
python train.py
```
* Sol taraftaki dosya gezgininden `config.yaml` veya `train.py` dosyasını çift tıklayıp değiştirdiğin an, terminalde tekrar `python train.py` yazarak **beklemeden (0 saniye gecikmeyle)** yeni denemeni başlatabilirsin!

---

## 3. Çok Önemli: İşin Bittiğinde Makineyi Durdurma / Açma / Silme Komutları

> [!WARNING]
> **Maliyet Uyarısı:** Doğrudan GPU makinesi yönteminde makine kendi kendine kapanmaz (`InService` kaldığı sürece saatte **~$1.51** yazar). Çalışman bittiğinde makineyi **mutlaka durdurmalısın (`stop`)**.

### A. Çalışman Bittiğinde Makineyi Durdurma (`Stop` — GPU Ücreti Anında Durur)
Kendi bilgisayarındaki PowerShell terminalinden:
```powershell
aws sagemaker stop-notebook-instance `
  --notebook-instance-name clip-gpu-workstation `
  --region us-west-2
```
* **Ne olur?** GPU ve CPU tamamen kapanır, saatlik $1.51 ücret **anında kesilir**.
* **Dosyaların silinir mi?** **Hayır!** `/home/ec2-user/SageMaker/IMAGE_RETRIEVAL` içindeki kodların ve 81,054 görselin kalıcı diskte durmaya devam eder.

### B. Ertesi Gün Tekrar Devam Etmek İstediğinde Başlatma (`Start`)
Daha önce `create-notebook-instance` yaptığın için bir daha oluşturmana gerek yoktur; sadece **Start** dersin:
```powershell
aws sagemaker start-notebook-instance `
  --notebook-instance-name clip-gpu-workstation `
  --region us-west-2
```
*(2 dakika içinde tüm dosyaların ve 81,054 görselin bıraktığın yerden hazır şekilde açılır).*

### C. Proje Tamamen Bittiğinde Makineyi ve Diski Silme (`Delete` — Tamamen $0.00)
Proje tamamen bittiğinde ve artık kalıcı diske de ihtiyacın kalmadığında (önce `stop` yaptıktan sonra):
```powershell
aws sagemaker delete-notebook-instance `
  --notebook-instance-name clip-gpu-workstation `
  --region us-west-2
```

---

## 4. Bonus: EC2 GPU Kotası (`L-DB2E81BA`) Onaylandığında VS Code ile Bağlanma

Başvuruda bulunduğumuz EC2 GPU kotası (`CaseId: 179236316900610`, 4 vCPU) onaylandığında, aynı işlemi **Amazon EC2** üzerinden daha düşük saatlik maliyetle (`g4dn.xlarge`: **~$0.52/saat**, `g5.xlarge`: **~$1.00/saat**) şu şekilde yapabilirsin:

1. **Kota Durumunu Kontrol Etme:**
   ```powershell
   aws service-quotas get-requested-service-quota-change --request-id d8b7492fafac4bf6a1a7852e60056843ZApZK86S --region us-west-2
   ```
2. **EC2 GPU Makinesi Açma:**
   * AWS Console $\rightarrow$ **EC2** $\rightarrow$ **Launch Instance**
   * **AMI (İşletim Sistemi):** Arama kutusuna `Deep Learning OSS Nvidia Driver AMI GPU PyTorch` (Ubuntu) yazıp seç (NVIDIA sürücüleri ve PyTorch hazır kurulu gelir).
   * **Instance Type:** `g4dn.xlarge` (T4 16GB) veya `g5.xlarge` (A10G 24GB).
   * **Key Pair:** Bir `.pem` SSH anahtarı oluşturup bilgisayarınıza indirin.
3. **VS Code / Antigravity İçinden Doğrudan Bağlanma (`Remote-SSH`):**
   * Kendi bilgisayarındaki editörde **Remote-SSH** eklentisini açıp:
     ```bash
     ssh -i C:\path\to\key.pem ubuntu@<EC2-PUBLIC-IP>
     ```
     adresiyle bağlandığında kendi editörün doğrudan buluttaki GPU makinesinin içinde çalışır!
