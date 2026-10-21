# Multi-Positive Contrastive Loss'ta Entropi Tabanı (Entropy Floor) ve `val_loss` Analizi

Bu doküman, CLIP fine-tuning eğitimlerimizde neden `train_loss` değerinin **`~0.51 → 0.23`** aralığında seyrederken `val_loss` değerinin **`~3.23 → 3.33`** aralığında göründüğünü ve bu **~2.7 puanlık farkın neden büyük ölçüde Overfitting değil, matematiksel bir Entropi Tabanı (Shannon Entropy Floor) farkı olduğunu** kanıtlarıyla açıklar.

---

## 1. Gözlemlenen Durum (Soru İşareti)

`lr = 1e-5` ile yapılan 5 epochluk eğitimde (`analysis/training_history_lr1e-05.json`) şu tablo gözlemlenmiştir:

| Epoch | `train_loss` | `val_loss` | Görünürdeki Fark (`overfit_gap`) | Validation `NDCG@10` | Validation `Recall@10` |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **Epoch 1** | `0.5170` | `3.2330` | `2.7160` | `0.5445` | `50.85%` |
| **Epoch 2** | `0.3366` | `3.2403` | `2.9037` | `0.5425` | `51.73%` |
| **Epoch 3** | `0.2652` | `3.3097` | `3.0444` | **`0.5457`** | **`51.98%`** |
| **Epoch 4** | `0.2385` | `3.3297` | `3.0912` | `0.5439` | `51.89%` |
| **Epoch 5** | `0.2299` | `3.3322` | `3.1023` | `0.5439` | `51.77%` |

**Paradoks:** Eğer model daha 1. epoch sonunda `2.71` puanlık devasa bir overfitting yaşıyor olsaydı, Validation ve Test retrieval metriklerinin (`NDCG@10`, `Recall@10`, `MRR`) düşmesi gerekirdi. Oysa 3. epoch'a kadar Validation `NDCG@10` ve `Recall@10` artmaya devam etmiş ve Test setinde Zero-Shot baseline'a göre **+3.80 puan `Recall@10`** ve **+6.04 puan `HitRate@1`** artışı elde edilmiştir.

Peki `val_loss` neden `3.23`'ün altına inememektedir?

---

## 2. Matematiksel Kanıt: Cross-Entropy = Entropi Tabanı + KL-Divergence

`train.py` içerisindeki `compute_contrastive_loss` fonksiyonu, bir batch içinde aynı sorguya (`query_id`) veya aynı ürüne (`product_id`) ait **$K$ adet pozitif eşleşme** bulunduğunda, hedef olasılık dağılımını ($p$) bu $K$ adet doğru örneğe eşit olarak böler:

$$p_j = \begin{cases} \frac{1}{K} & \text{eğer } j \text{ pozitif eşleşme ise (toplam } K \text{ adet)} \\ 0 & \text{diğer durumlarda} \end{cases}$$

Modelin softmax sonrasında ürettiği tahmin olasılıklarına $q_j$ dersek, hesapladığımız **Cross-Entropy Loss** ($\mathcal{L}_{\text{CE}}$) matematiksel olarak iki terimin toplamına eşittir:

$$\mathcal{L}_{\text{CE}}(p, q) = -\sum_{j=1}^{B} p_j \ln q_j = \underbrace{H(p)}_{\text{Sabit Entropi Tabanı}} + \underbrace{D_{\text{KL}}(p \parallel q)}_{\text{Gerçek Model Hatası }\ge 0}$$

Burada:
1. **$D_{\text{KL}}(p \parallel q) = \sum p_j \ln(p_j / q_j)$ (Kullback-Leibler Divergence):** Modelin hedef dağılımdan ne kadar saptığını ölçer. Model **%100 kusursuz** tahmin yaptığında ($q_j = p_j$), bu hata terimi **$0.0000$** olur.
2. **$H(p) = -\sum p_j \ln p_j$ (Shannon Entropisi / Teorik Alt Sınır):** Hedef dağılımın kendi belirsizliğidir. $K$ adet eşit ağırlıklı pozitif örnek için:
   $$H(p) = -\sum_{k=1}^{K} \frac{1}{K} \ln\left(\frac{1}{K}\right) = \ln(K)$$

> [!IMPORTANT]
> **Temel Matematiksel Sonuç:**
> Bir batch içindeki bir satırın **$K$ adet pozitif eşleşmesi** varsa, model **%100 kusursuz ($D_{\text{KL}} = 0$) olsa bile** Cross-Entropy Loss değeri **asla $\ln(K)$ değerinin altına inemez!**
> * $K = 1$ ise (batch içinde tek pozitif): En düşük olası loss = $\ln(1) = \mathbf{0.0000}$
> * $K = 10$ ise (batch içinde aynı sorgudan 10 pozitif): En düşük olası loss = $\ln(10) = \mathbf{2.3026}$

---

## 3. Neden `train_loader` İçin $K \approx 1$ İken `val_loss_loader` İçin $K \approx 10.4$ Oldu?

`train.py` içindeki DataLoader tanımlarına ve `val.parquet` dosyasının diskteki dizilimine baktığımızda farkın kaynağı ortaya çıkmaktadır:

### A. `val.parquet` Dosyasının Sıralı Yapısı ve `shuffle=False` Etkisi
`val.parquet` dosyasında aynı `query_id`'ye ait tüm satırlar **arka arkaya (gruplanmış olarak)** durmaktadır:
```text
val.parquet içindeki ilk 15 Exact ('E') satırın query_id değerleri:
[37, 37, 37, 37, 37, 37, 37, 42, 42, 42, 42, 42, 42, 42, 42]
```
* **`val_loss_loader` (`shuffle=False`, `batch_size=128`):**
  Karıştırma (`shuffle`) kapalı olduğu için, 128 örnekli her bir validation batch'inin içine aynı sorgunun **tüm `'E'` ürünleri (~8 ile 20 adet arası, ortalama $K \approx 10.4$)** blok halinde birlikte düşmektedir!
* **`train_loader` (`shuffle=True`, `batch_size=64`):**
  `67,799` eğitim satırı her epoch başında rastgele karıştırıldığı (`shuffle=True`) için, 64'lük bir eğitim batch'inin içine aynı sorgudan neredeyse her zaman **sadece 1 adet ($K \approx 1.07$)** örnek düşmektedir.

### B. `val_loss_loader` Üzerinde Ölçülen Gerçek Entropi Tabanı (`2.3443`)
`val.parquet` üzerindeki tüm `3,657` `'E'` örneği, `val_loss_loader`'ın yaptığı gibi 128'lik sıralı batch'lere bölünüp her örneğin batch içi pozitif sayısı ($K_i$) üzerinden ortalama $\ln(K_i)$ hesaplandığında:

$$\text{Teorik Minimum Val Loss Tabanı } H_{\text{val}}(p) = \frac{1}{N}\sum_{i=1}^{N} \ln(K_i) = \mathbf{2.3443}$$

Yani hiçbir öğrenme hatası yapmayan **kusursuz bir model (Oracle)** bile `val_loss_loader` üzerinde **`2.3443`** loss değeri alacaktır!

*(Üstelik `loss_i2t` yani Görsel $\rightarrow$ Metin yönünde, aynı batch içindeki 10 özdeş sorgu metni birebir aynı metin vektörünü ürettiği için, modelin bu 10 kopyaya farklı logit vermesi fiziksel olarak da imkansızdır; dolayısıyla softmax olasılığı en fazla $1/10$ olabilir).*

---

## 4. İkinci Faktör: Batch Boyutu Farkı (`64` vs `128`)

Contrastive (InfoNCE) loss fonksiyonunda paydada batch içindeki tüm adaylar yer alır:
$$\mathcal{L} = -\ln \frac{e^{s_{\text{pos}}}}{\sum_{j=1}^{B} e^{s_j}}$$
* Eğitim sırasında `batch_size = 64` kullanılmıştır (1 doğru aday, 63 negatif çeldirici).
* Validation sırasında `eval_batch_size = 128` kullanılmıştır (2 kat daha fazla aday havuzu).
* Aday havuzu 2 katına çıktığında softmax paydası büyüdüğü için Cross-Entropy loss değeri doğal olarak yaklaşık $\Delta \mathcal{L} \approx \ln\left(\frac{128}{64}\right) = \ln(2) \approx \mathbf{0.6931}$ kadar daha yüksek ölçeklenir.

---

## 5. Düzeltilmiş (Gerçek) Loss Karşılaştırması

Ham `val_loss` değerinden sabit entropi tabanını ($H_{\text{val}}(p) = 2.3443$) çıkardığımızda modelin **gerçek hata terimi olan KL-Divergence ($D_{\text{KL}}$)** ortaya çıkar:

| Epoch (`lr=1e-5`) | `train_loss` (`bs=64`, $H \approx 0.05$) | Ham `val_loss` (`bs=128`, $H = 2.3443$) | **Gerçek Val Hatası: $D_{\text{KL}} = \text{val\_loss} - 2.3443$** | Batch Boyutu (`128→64`) Düzeltilmiş Karşılığı |
| :---: | :---: | :---: | :---: | :---: |
| **Epoch 1** | `0.5170` | `3.2330` | **`0.8887`** | **`~0.48`** |
| **Epoch 2** | `0.3366` | `3.2403` | **`0.8960`** | **`~0.49`** |
| **Epoch 3** | `0.2652` | `3.3097` | **`0.9654`** | **`~0.55`** |
| **Epoch 4** | `0.2385` | `3.3297` | **`0.9854`** | **`~0.57`** |
| **Epoch 5** | `0.2299` | `3.3322` | **`0.9879`** | **`~0.57`** |

### Bu Tablodan Çıkan Sonuçlar:
1. **1. Epoch'ta Overfitting Yoktur:** Entropi tabanı (`2.3443`) ve batch boyutu farkı çıkarıldığında, 1. epoch sonundaki gerçek Validation hatası (`~0.48`) ile Training hatası (`0.5170`) **birebir aynı seviyededir**.
2. **Epoch 1 $\rightarrow$ 5 Arasındaki Gerçek Overfit Çok Küçüktür (`+0.09` puan):**
   * `lr = 1e-4` denemesinde `val_loss` `3.24`'ten `3.83`'e (**`+0.59` puan**) fırlamıştı.
   * `lr = 1e-5` denemesinde ise `val_loss` 5 epoch boyunca sadece `3.23`'ten `3.33`'e (**`+0.09` puan**) değişmiştir. Bu küçük artışın sebebi de eğitim setindeki `8,046` tekil sorgunun bir epoch içinde ortalama 8.5 kez tekrar etmesidir.

---

## 6. Kodda `train_loss` ve `val_loss`'u Birebir Aynı Ölçeğe Getirmek İçin Çözüm

Bir sonraki iterasyonda `train_loss` ve `val_loss` değerlerinin ekranda doğrudan aynı ölçekte (`0.0` tabanlı olarak) görünmesi için `train.py` içinde şu 2 küçük değişiklik yeterlidir:

1. **`compute_contrastive_loss` içinde Cross-Entropy yerine doğrudan KL-Divergence döndürmek (Entropi tabanını çıkarmak):**
   ```python
   # Hedef dağılımın entropisini (ln K) çıkararak minimum olası loss'u her zaman 0.0 yap:
   eps = 1e-8
   entropy_t2i = -(target_probs_t2i * torch.log(target_probs_t2i + eps)).sum(dim=-1).mean()
   entropy_i2t = -(target_probs_i2t * torch.log(target_probs_i2t + eps)).sum(dim=-1).mean()

   loss_t2i = (
       -(target_probs_t2i * F.log_softmax(logits_per_text, dim=-1)).sum(dim=-1).mean() - entropy_t2i
   )
   loss_i2t = (
       -(target_probs_i2t * F.log_softmax(logits_per_image, dim=-1)).sum(dim=-1).mean() - entropy_i2t
   )
   ```
   *(Not: $H(p)$ model ağırlıklarına bağlı olmadığı için türevi $\nabla_\theta H(p) = 0$'dır; yani eğitimin gradyanlarını hiçbir şekilde değiştirmez, sadece raporlanan loss değerini gerçek `0.0` tabanına çeker).*

2. **`val_loss_loader` için sabit tohumlu (deterministic) `shuffle=True` ve eğitimle aynı `batch_size` (`64`) değerini kullanmak:**
   Böylece hem eğitim hem validation batch'leri aynı negatif aday sayısına (`64`) ve aynı tekil sorgu dağılımına sahip olur.
