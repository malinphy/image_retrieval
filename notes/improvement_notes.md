# CLIP Fine-Tuning Sonuç Analizi, Dosya Konumları ve İyileştirme Notları

Bu doküman, `openai/clip-vit-large-patch14` modelinin Amazon ESCI + SQID veri seti üzerinde gerçekleştirilen **3 ana deneyin** sonuçlarını, tüm sonuç dosyalarının **yerel (Local)** ve **bulut (AWS S3)** konumlarını, hangi sonucun **Test** veya **Validation** veri setine ait olduğunu ve sonraki iterasyonlar için teknik iyileştirme önerilerini içerir:
1. **Zero-Shot Baseline** (`openai/clip-vit-large-patch14`)
2. **Deney 1 — LoRA Fine-Tuning (`lr = 1e-4`)**: `r=16, alpha=32, bs=64, lr=1e-4, 5 epoch`
3. **Deney 2 — LoRA Fine-Tuning (`lr = 1e-5`)**: `r=16, alpha=32, bs=64, lr=1e-5, 5 epoch` *(Overfitting'i önlemek için düşürülmüş öğrenme oranı — **En İyi Model**)*

---

## 1. Sonuç Dosyalarının Konumları ve Veri Seti Dağılımları (Test vs. Validation)

Deneylerde kullanılan veri setleri (`data/processed/splits/`) birbirinden tamamen bağımsız (sıfır sorgu kesişimi / *zero query overlap*) olacak şekilde ayrılmıştır:
- **Train Split (`train.parquet`):** `155,104` toplam satır (`67,799` geçerli görselli `'E'` Exact eşleşmesi), `8,046` tekil sorgu.
- **Validation Split (`val.parquet`):** `8,622` satır (`3,657` `'E'` eşleşmesi), `447` tekil sorgu, `8,554` tekil ürün görseli.
- **Test Split (`test.parquet`):** `8,816` satır (`3,925` `'E'` eşleşmesi), `448` tekil sorgu, `8,725` tekil ürün görseli.

### Dosya Konumları Özeti Tablosu

| Dosya / Klasör | Hangi Split? | İçerik Açıklaması | Yerel (Local) Konum | AWS S3 Konumu |
| :--- | :---: | :--- | :--- | :--- |
| **`zero_shot_results.json`** | **TEST SPLIT** (`test.parquet`) | Eğitimsiz (`openai/clip-vit-large-patch14`) modelin **Test** seti (448 sorgu, 8,725 görsel) sonuçları. | `E:\PERSONAL_PROJ\IMAGE_RETRIEVAL\analysis\zero_shot_results.json` | `s3://<YOUR_S3_BUCKET>/results/zero_shot_results.json` |
| **`finetuned_test_results.json`** | **TEST SPLIT** (`test.parquet`) | **Deney 1 (`lr=1e-4`)** en iyi modelinin **Test** seti sonuçları. | `E:\PERSONAL_PROJ\IMAGE_RETRIEVAL\analysis\finetuned_test_results.json` | `s3://<YOUR_S3_BUCKET>/checkpoints/vit-large-p14_lora-r16-a32_bs64_lr1e-04_ep5/finetuned_test_results.json` |
| **`finetuned_test_results_lr1e-05.json`** | **TEST SPLIT** (`test.parquet`) | **Deney 2 (`lr=1e-5`)** en iyi modelinin **Test** seti sonuçları (**En Yüksek Performans**). | `E:\PERSONAL_PROJ\IMAGE_RETRIEVAL\analysis\finetuned_test_results_lr1e-05.json` | `s3://<YOUR_S3_BUCKET>/checkpoints/vit-large-p14_lora-r16-a32_bs64_lr1e-05_ep5/finetuned_test_results.json` |
| **`training_history.json`** | **TRAIN + VALIDATION** (`val.parquet`) | **Deney 1 (`lr=1e-4`)** 5 epochluk Train loss ve **Validation** metrikleri. | `E:\PERSONAL_PROJ\IMAGE_RETRIEVAL\analysis\training_history.json` | `s3://<YOUR_S3_BUCKET>/checkpoints/vit-large-p14_lora-r16-a32_bs64_lr1e-04_ep5/training_history.json` |
| **`training_history_lr1e-05.json`** | **TRAIN + VALIDATION** (`val.parquet`) | **Deney 2 (`lr=1e-5`)** 5 epochluk Train loss ve **Validation** metrikleri. | `E:\PERSONAL_PROJ\IMAGE_RETRIEVAL\analysis\training_history_lr1e-05.json` | `s3://<YOUR_S3_BUCKET>/checkpoints/vit-large-p14_lora-r16-a32_bs64_lr1e-05_ep5/training_history.json` |

---

## 2. TEST SPLIT Sonuçları: Zero-Shot vs. `lr=1e-4` vs. `lr=1e-5` Karşılaştırması

Aşağıdaki tüm metrikler **TEST SPLIT (`test.parquet` — 448 sorgu, 8,725 tekil ürün görseli)** üzerinde ölçülmüştür.

### A. Global Vector Retrieval (8,725 Görselin Tamamı İçinden Arama)

| Metrik (Test Split - Global) | Zero-Shot CLIP | LoRA (`lr=1e-4`) | LoRA (`lr=1e-5`) | Zero-Shot'a Göre Fark (`1e-5`) | Yorum |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **`NDCG@10`** | `0.5430` | `0.5581` | **`0.5804`** | **+3.74 puan** (`+6.89%`) | İlk 10 sıralama kalitesinde muazzam sıçrama. |
| **`NDCG@20`** | `0.5914` | `0.6028` | **`0.6281`** | **+3.67 puan** (`+6.21%`) | İlk 20 sıralama kalitesinde büyük artış. |
| **`MRR`** | `0.8028` | `0.8048` | **`0.8325`** | **+2.97 puan** | İlk doğru ürün çok daha üst sırada geliyor. |
| **`HitRate@1`** | `48.77%` | `50.78%` | **`54.81%`** | **+6.04 puan** | Sorguların **~%55'inde 1. sıradaki görsel tam isabet (`E`)**! |
| **`HitRate@5`** | `81.88%` | `81.88%` | **`82.55%`** | **+0.67 puan** | İlk 5'te en az 1 doğru ürün bulma oranı arttı. |
| **`HitRate@10`** | `88.37%` | `87.70%` | **`88.81%`** | **+0.44 puan** | `1e-4`'teki hafif düşüş tamamen düzeldi ve geçti. |
| **`HitRate@20`** | `92.17%` | `91.28%` | **`93.51%`** | **+1.34 puan** | İlk 20'de doğru ürün yakalama oranı %93.5'e ulaştı. |
| **`HitRate@50`** | `93.96%` | `95.75%` | **`96.42%`** | **+2.46 puan** | İlk 50'de sorguların **%96.4'ünde** doğru ürün yakalanıyor. |
| **`Recall@1`** | `9.08%` | `9.39%` | **`10.22%`** | **+1.14 puan** (`+12.5%` bağıl) | 1. sırada yakalanan toplam doğru ürün oranı. |
| **`Recall@5`** | `32.98%` | `34.42%` | **`35.15%`** | **+2.17 puan** | İlk 5'te yakalanan doğru ürün oranı. |
| **`Recall@10`** | `50.73%` | `52.89%` | **`54.53%`** | **+3.80 puan** (`+7.49%` bağıl) | İlk 10'da tüm doğru ürünlerin yarıdan fazlası yakalanıyor. |
| **`Recall@20`** | `68.06%` | `69.18%` | **`71.44%`** | **+3.38 puan** | İlk 20'de yakalanan doğru ürün oranı %71.4'e çıktı. |
| **`Recall@50`** | `81.47%` | `84.54%` | **`85.66%`** | **+4.19 puan** | İlk 50'de yakalanan doğru ürün oranında +4.19 puan artış. |
| **`Recall@100`** | `87.35%` | `89.68%` | **`90.79%`** | **+3.44 puan** | İlk 100'de katalogdaki doğru ürünlerin **~%91'i** bulunuyor. |

### B. Candidate Re-ranking (Sorgu Başına ~20 Hazır Aday İçinden Sıralama)
`lr=1e-4` denemesinde Candidate Re-ranking metrikleri hafif gerilerken, **`lr=1e-5` ile Candidate Re-ranking metriklerinin tamamı da Zero-Shot baseline'ı geçmiştir:**

| Metrik (Test Split - Reranking) | Zero-Shot CLIP | LoRA (`lr=1e-4`) | LoRA (`lr=1e-5`) | Zero-Shot'a Göre Fark (`1e-5`) |
| :--- | :---: | :---: | :---: | :---: |
| **`NDCG@10`** | `0.7276` | `0.7237` | **`0.7339`** | **+0.63 puan** |
| **`NDCG@20`** | `0.8005` | `0.7969` | **`0.8054`** | **+0.49 puan** |
| **`HitRate@1`** | `60.63%` | `59.96%` | **`64.65%`** | **+4.02 puan** |
| **`Recall@1`** | `11.78%` | `12.09%` | **`12.96%`** | **+1.18 puan** |
| **`Recall@10`** | `70.07%` | `70.29%` | **`70.14%`** | **+0.07 puan** |
| **`MRR`** | `0.9405` | `0.9290` | **`0.9433`** | **+0.28 puan** |

---

## 3. VALIDATION SPLIT Sonuçları: `lr=1e-4` vs. `lr=1e-5` Eğitim Dinamikleri

Aşağıdaki tablo **VALIDATION SPLIT (`val.parquet` — 447 sorgu, 8,554 görsel)** üzerinde `lr=1e-5` modelinin 5 epoch boyunca nasıl çok daha dengeli ve ezberlemeden (overfit olmadan) ilerlediğini gösterir:

| Epoch (`lr=1e-5`) | Train Loss (`1e-5` vs `1e-4`) | Val Loss (`1e-5` vs `1e-4`) | Val Global `NDCG@10` | Val Global `Recall@10` | Val Global `HitRate@1` | Val Global `MRR` |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Epoch 1** *(Best Val Loss)* | `0.5170` *(vs 0.4031)* | **`3.2330`** *(vs 3.2428)* | `0.5445` | `50.85%` | **`54.52%`** | `0.8071` |
| **Epoch 2** | `0.3366` *(vs 0.1906)* | `3.2403` *(vs 3.4647)* | `0.5425` | `51.73%` | `51.13%` | `0.8065` |
| **Epoch 3** *(Best Val NDCG & Recall@10)* | `0.2652` *(vs 0.0984)* | `3.3097` *(vs 3.7185)* | **`0.5457`** | **`51.98%`** | `52.26%` | **`0.8195`** |
| **Epoch 4** | `0.2385` *(vs 0.0703)* | `3.3297` *(vs 3.7932)* | `0.5439` | `51.89%` | `51.58%` | `0.8137` |
| **Epoch 5** | `0.2299` *(vs 0.0579)* | `3.3322` *(vs 3.8277)* | `0.5439` | `51.77%` | `52.04%` | `0.8158` |

### Neden `lr=1e-5` Çok Daha Başarılı Oldu?
1. **Pretrained Uzayın Korunması (Catastrophic Forgetting Önleme):** `1e-4` öğrenme oranı ilk epoch içinde CLIP'in önceden eğitilmiş görsel-metin hizalamasını fazla sarsarken, `1e-5` mevcut bilgiyi bozmadan e-ticaret alanına (domain adaptation) ince ayar yaptı.
2. **Overfitting'in Frenlenmesi:** `1e-4` ile `train_loss` `0.0579`'a kadar düşüp `val_loss` `3.8277`'ye fırlamıştı. `1e-5` ile `train_loss` `0.23` seviyesinde dengelendi ve `val_loss` 5 epoch boyunca `3.23 – 3.33` bandında son derece kararlı kaldı.
3. **Hem Global Arama Hem Re-ranking Artışı:** Ağırlıklar aşırı özgüvenle şişmediği için yalnızca genel kategori ayrımı değil, aynı sorgunun ~20 yakın adayı arasındaki ince sıralama (`Candidate Re-ranking HitRate@1`: `%60.63` $\rightarrow$ **`%64.65`**) da belirgin şekilde iyileşti.

---

## 4. Sonraki İterasyonlar İçin Ek İyileştirme Önerileri (V2 Roadmap)

1. **Query-Balanced Sampling (Epoch Başına Tekil Sorgu Örnekleme):**
   - Her epoch'ta `8,046` tekil sorgunun her biri için rastgele 1 adet `'E'` ürün görseli seçerek hem sorgu tekrarını tamamen sıfırlamak hem de 1 epoch süresini 30 dakikadan **~3.5 dakikaya** düşürmek.
2. **ESCI `'I'` (Irrelevant) ve `'S'` (Substitute) Etiketleriyle Hard Negative Mining:**
   - Eğitim sırasında aynı sorguya ait `'I'` (Irrelevant) ürünleri zor negatif (hard negative) olarak batch içine dahil etmek.
3. **Epoch 3 Checkpoint'ini Test Setinde Değerlendirmek (`best_metric: "val_NDCG@10"`):**
   - `lr=1e-5` eğitiminde **Epoch 3**, Validation setinde en yüksek `NDCG@10` (`0.5457`), `Recall@10` (`51.98%`) ve `MRR` (`0.8195`) değerlerine ulaşmıştır. `best_metric` olarak `val_NDCG@10` kullanıldığında Epoch 3 otomatik olarak seçilecektir.
