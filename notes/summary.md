# Cloud Üzerinde CLIP Fine-Tuning Stratejisi ve Çalışma Rehberi

Bu belge, yerel donanımın yetersiz olduğu senaryolarda CLIP modelinin veri indirme, ön işleme, fine-tuning ve değerlendirme adımlarını bulutta (özellikle AWS ekosisteminde) en verimli ve ekonomik şekilde yürütmek için hazırlanan kılavuzdur.

---

## 1. Temel Felsefe: Yerel Geliştirme + Bulutta Çalıştırma (Decoupled Architecture)

* **Kod Geliştirme (Local):** Kodu kendi bilgisayarınızda yazarsınız. 5-10 örnek veriyle CPU üzerinde hızlı bir test (`dry-run`) yaparak import, tensör boyutları ve döngü hatalarını ücretsiz/hızlıca çözersiniz.
* **Ağır İşler (Cloud):** Veri indirme (yüksek bant genişliği), veri işleme, GPU eğitimi ve değerlendirme buluta devredilir.
* **Amaç:** GPU makineleri saatlik ücretlendirildiği için kod yazarken boş yere GPU çalıştırmamak ve maliyeti minimuma indirmek.

---

## 2. Bulut Platform Seçenekleri ve Karşılaştırma

| Platform | Kullanım Şekli | Tahmini Maliyet | Avantajları |
| :--- | :--- | :--- | :--- |
| **AWS EC2 (G5 Serisi - A10G 24GB)** | Sanal Sunucu (SSH ile bağlanma) | ~$0.30 - $1.00/saat (Spot/On-demand) | Tam kontrol, yüksek AWS bant genişliği, S3 ile sıfır veri transfer maliyeti. |
| **AWS SageMaker Training Jobs** | Sunucusuz Eğitim Görevi (Serverless) | Eğitim süresince dakika bazlı | Kod bittiğinde makine otomatik kapanır; boşta kalma (idle) maliyeti sıfırdır. |
| **Hugging Face AutoTrain / Spaces** | Yönetilen Arayüz / Spaces GPU | ~$0.60 - $1.05/saat (T4 / A10G) | Altyapı kurulumu gerektirmez, doğrudan Hub modelleriyle entegredir. |
| **RunPod / Vast.ai** | GPU Kiralama Pazaryeri | ~$0.25 - $0.45/saat (RTX 3090/4090) | Bütçe dostu GPU'lar, pratik şablonlar. |

---

## 3. Veri ve Depolama Stratejisi (AWS S3)

Veri seti ve eğitilen model ağırlıkları **mutlaka AWS S3** üzerinde tutulmalıdır:

* **Neden S3?**
  1. **Kalıcılık:** Geçici EC2 makineleri silindiğinde (terminate) disklerindeki veriler uçar; S3'teki veri asla kaybolmaz.
  2. **Düşük Maliyet:** S3 depolaması GB başına aylık ~$0.02'dir (EBS disklerine kıyasla çok daha ucuz).
  3. **Hızlı ve Ücretsiz Transfer:** S3 ile aynı bölgedeki (örn. `eu-central-1`) EC2 veya SageMaker arasındaki veri transferi saniyede yüzlerce megabayttır ve **tamamen ücretsizdir**.

### Önerilen S3 Dizin Yapısı
```text
s3://<bucket-adi>/
├── data/
│   ├── images/              # İndirilen görseller
│   └── metadata.csv         # image_path, caption kolonları
└── checkpoints/             # Eğitilen model ağırlıkları ve loglar
```

---

## 4. Kimlik Doğrulama ve Güvenlik (Credentials)

> [!CAUTION]
> Kod dosyalarına (`train.py`, `dataset.py`) kesinlikle `AKIA...` gibi Access Key veya Secret Key yazılmamalı ve GitHub'a atılmamalıdır.

* **Yerel Bilgisayarınızda (Local):**
  * AWS CLI kurulur: `aws configure`
  * IAM User bilgileri bilgisayarın `~/.aws/credentials` dosyasına kaydedilir.
* **AWS Üzerinde (EC2 / SageMaker):**
  * **Sıfır Key Kuralı:** Makineye hiçbir şifre girilmez.
  * S3 okuma/yazma yetkisine sahip bir **IAM Role** (Örn: `EC2-S3-FullAccess-Role`) oluşturulup EC2 veya SageMaker servisine atanır.
  * AWS SDK (`boto3`) ve AWS CLI, kimliği otomatik olarak bu rolden çeker.

---

## 5. Docker Gerekli mi?

* **Hayır, Docker zorunlu değildir.**
* **EC2 Çözümü:** AWS'nin resmi **Deep Learning AMI (DLAMI)** seçildiğinde NVIDIA sürücüleri, CUDA, PyTorch ve Python kurulu gelir. Sadece `pip install -r requirements.txt` çalıştırmak yeterlidir.
* **SageMaker Çözümü:** AWS'nin hazır PyTorch container'ları kullanılır; sadece kod ve `requirements.txt` iletilir.
* Docker yalnızca standart dışı C++ bağımlılıkları veya karmaşık kurumsal orkestrasyonlar için opsiyoneldir.

---

## 6. Önerilen İş Akışı (Roadmap)

1. **Konfigürasyon Hazırlığı (`config.yaml`):**
   * Ortam değişkenlerini (yerel/S3 bayrağı, batch size, model tipi, epoch) tek bir dosyadan yönetecek yapı kurulur.
2. **Local Dry-Run Testi:**
   * Mevcut `dataset.py` ve `eval.py` scriptleri birleştirilerek CPU üzerinde 5-10 örnekle mini bir eğitim döngüsü çalıştırılır.
3. **Verinin S3'e Yüklenmesi:**
   * Hazır olan veriler veya indirme scripti S3 ile senkronize edilir:
     `aws s3 sync ./data s3://<bucket-adi>/data/`
4. **AWS Üzerinde Eğitimi Başlatma:**
   * Bir EC2 `g5.xlarge` Spot Instance başlatılır (veya SageMaker Job tetiklenir).
   * Veri S3'ten çekilir, GPU üzerinde eğitim tamamlanır, çıktılar S3'e geri yüklenir.
   * Makine kapatılır/silinir.

---

## 7. AWS EC2 Üzerinde Adım Adım Çalıştırma Kılavuzu

Mevcut `eval.py` veya ilerideki `train.py` scriptlerini AWS EC2 üzerinde çalıştırmak için pratik adımlar:

### Adım 1: S3 Erişim İzni (IAM Role) Oluşturma *(Bir Seferlik)*
1. AWS Console -> **IAM** -> **Roles** -> **Create Role**.
2. **Trusted entity:** `AWS service`, kullanım alanı: **EC2**.
3. **Permissions:** `AmazonS3FullAccess` politikasını ekleyin.
4. Rol adına **`EC2-S3-Access-Role`** verip kaydedin.

### Adım 2: EC2 GPU Makinesini Başlatma (Launch)
1. AWS Console -> **EC2** -> **Launch Instance**.
2. **Name:** `clip-evaluation`
3. **AMI:** `Deep Learning OSS Nvidia Driver AMI GPU (Ubuntu 22.04)` (NVIDIA sürücüleri ve CUDA hazır gelir).
4. **Instance Type:**
   * `eval.py` için: `g4dn.xlarge` (NVIDIA T4 16GB VRAM, ~$0.52/saat).
   * Büyük fine-tuning için: `g5.xlarge` (NVIDIA A10G 24GB VRAM, ~$1.00/saat).
5. **Key Pair:** Yeni bir `.pem` anahtarı oluşturun ve bilgisayarınıza indirin (örn. `clip-key.pem`).
6. **Storage (Disk):** En az **50 GB - 80 GB** (gp3) olarak ayarlayın.
7. **Advanced Details:**
   * **IAM instance profile:** `EC2-S3-Access-Role` seçin.
   * **Spot Instance:** *Request Spot Instances* kutucuğunu işaretleyin (fiyat ~$0.15 - $0.25/saate düşer, %70+ tasarruf).
8. **Launch Instance** butonuna tıklayın.

### Adım 3: VS Code ile Makineye Bağlanma
1. VS Code'da **Remote Explorer (SSH)** sekmesine gelin.
2. Bağlantı komutunu girin:
   ```bash
   ssh -i "C:/path/to/clip-key.pem" ubuntu@<EC2-PUBLIC-IP>
   ```
3. VS Code penceresinde sunucunun dosya yöneticisi ve terminali açılır.

### Adım 4: Ortamı Kurma ve Veriyi Çekme
Sunucu terminalinde:
```bash
# 1. Kodu sunucuya alın
git clone <github-repo-url>
cd IMAGE_RETRIEVAL

# 2. Python paketlerini kurun
pip install -r requirements.txt

# 3. S3'ten verileri çekin (~10-15 saniye)
aws s3 sync s3://<YOUR_S3_BUCKET>/data/ ./data/
```

### Adım 5: Değerlendirmeyi (`eval.py`) Çalıştırma
```bash
python eval.py
```
* Kod otomatik olarak GPU'yu (`cuda`) tanır, test embedding'lerini çıkarır ve metrikleri ekrana basar.
* Sonuçlar `analysis/zero_shot_results.json` dosyasına yazılır.

### Adım 6: Sonuçları Alma ve Makineyi Kapatma (Maliyeti Durdurma)
1. Sonuç dosyasını S3'e yedekleyin veya yerel bilgisayarınıza indirin:
   ```bash
   aws s3 cp analysis/zero_shot_results.json s3://<YOUR_S3_BUCKET>/results/
   ```
2. **Kapatma:** AWS Console -> EC2 -> Instance seçin -> **Instance State -> Terminate** (veya Stop).
   * Böylece boşta kalma faturası ödemezsiniz; tüm testin toplam maliyeti birkaç sent ile sınırlı kalır.

