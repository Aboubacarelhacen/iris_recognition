# 🎓 İris Tanıma Sistemi — 10 Dakikalık Sunum

Bu doküman tek başına yeterli olacak şekilde hazırlanmıştır. İçindekiler:

1. **Sunum Senaryosu** (10 dakika, bölüm bölüm, zamanlamalı)
2. **Dashboard Rehberi** — her ekranda ne göstereceğin ve ne anlama geldiği
3. **Soru & Cevap Hazırlığı** — beklenen sorulara hazır cevaplar
4. **Sunum Öncesi Kontrol Listesi**

---

## 🕐 BÖLÜM 1 — SUNUM SENARYOSU

### ⏱ Sunum Öncesi Hazırlık (kontrolden geçirilmesi gereken hususlar)

- Dashboard'u `http://localhost:5173` adresinde tarayıcıda açık tut.
- Masaüstünde `demo/` klasörü oluştur ve içine 4 iris görüntüsü koy:
  - **Aynı kişi**: Konu 000'dan iki görüntü (örn. `S5000L00.jpg`, `S5000L01.jpg`)
  - **Farklı kişiler**: Konu 001 ve Konu 002'den birer görüntü
- Yedek video veya ekran görüntüsü hazır olsun (sunumda bağlantı kesilirse).
- Ollama çalışıyor olsun (`ollama serve`).
- Su iç, derin nefes al, **yavaş konuş**.

---

### 🎬 Bölüm 1 — Giriş ve Problem (0:00 → 1:00)

> "Herkese merhaba. Bugün sizinle uçtan uca inşa ettiğim bir **İris Tanıma Sistemini** paylaşmak istiyorum."
>
> "Bilmeyenler için kısaca: irisiniz, parmak izinizden **altı kat daha fazla benzersiz özelliğe** sahiptir. Yaşla birlikte değişmez ve yüz tanımanın aksine, kontakt lens olmadan taklit etmek neredeyse imkansızdır. Bu yüzden iris tanıma, havalimanlarında ve ulusal kimlik sistemlerinde kullanılan en güvenilir biyometrik yöntemlerden biridir."

**Göster**: Dashboard ana sayfasını sessizce aç.

> "Çözmeye çalıştığım problem şu: iki iris görüntüsü verildiğinde, aynı kişiye mi yoksa farklı kişilere mi ait olduklarına karar vermek. Buna **doğrulama** denir. Kullandığım veri seti **CASIA-Iris-Thousand** — 1000 kişi, yaklaşık 20.000 görüntü, kontrollü kızılötesi aydınlatma altında çekilmiş."

---

### 🔬 Bölüm 2 — Klasik Algoritma: Daugman'ın İşlem Hattı (1:00 → 3:00)

**Tıkla**: Sol menüden `Sistem` (System) sayfasına git.

> "Sistemde iki farklı yaklaşım var. İlki, **1993 yılında John Daugman tarafından geliştirilen klasik IrisCode algoritması**. Bu algoritma, dünya çapında milyarlarca biyometrik kayıtta kullanılıyor. Dört adımı var:"

**Göster**: "Daugman's algorithm" kartını işaret et.

> "**Birinci adım — Bölütleme.** CLAHE kontrast iyileştirmesinden sonra **Hough Dönüşümü** ile iris ve gözbebeği sınırlarını tespit ediyoruz."
>
> "**İkinci adım — Normalizasyon.** Daugman'ın **kauçuk levha modelini** kullanarak dairesel iris halkasını 64×512 boyutunda düz bir dikdörtgene açıyoruz. Bu sayede göz dönmüş veya gözbebeği büyümüş olsa bile iki irisi piksel piksel karşılaştırabiliyoruz."
>
> "**Üçüncü adım — Kodlama.** Dört farklı yönelimde **Gabor filtreleri** uygulayıp her yanıtın işaretini alıyoruz. Sonuç: **2048 bitlik ikili bir IrisCode**."
>
> "**Dördüncü adım — Eşleştirme.** İki kodu **Hamming Uzaklığı** ile karşılaştırıyoruz — yani farklı olan bitlerin oranı. Aynı kişi için yaklaşık 0.20, farklı kişiler için yaklaşık 0.50 olması beklenir."

**Göster**: "End-to-end pipeline" kartındaki görseli işaret et.

> "Bu görseldeki akışı görüyorsunuz: iki giriş görüntüsü, bölütleme, ResNet-18 kodlayıcı, karşılaştırma ve sonuç olarak eşleşme kararı."

---

### 🤖 Bölüm 3 — Modern Yaklaşım: CNN ile Derin Öğrenme (3:00 → 5:00)

> "Ama Daugman'ın algoritması artık 32 yıllık. Modern bir **Evrişimli Sinir Ağı** bunu geçebilir mi diye merak ettim. **ResNet18 omurgalı ve ArcFace metrik öğrenme başlıklı** bir model eğittim. ArcFace, modern yüz tanıma sistemlerinde kullanılan en güçlü yöntemlerden biridir."

> "Modeli **Kaggle GPU üzerinde 35 epoch** boyunca eğittim. Önemli olan şu: **800 kişi eğitim**, **200 kişi doğrulama** için kullanıldı. Bu 200 kişi eğitim sırasında **hiç görülmedi**. Bu, gerçek dünyadaki kullanımı simüle eder — sisteme tanımadığı biri geldiğinde nasıl davranır?"

**Önemli teknik kararlar**:

> "Üç kritik teknik karar aldım:"
>
> "**Birincisi**: ImageNet üzerinde önceden eğitilmiş ağırlıkları kullandım. Modeli sıfırdan eğitmek 20.000 görüntü ile yeterli olmazdı."
>
> "**İkincisi**: ArcFace marjını sıfırdan 0.15'e yumuşak bir şekilde yükselttim. Bu, embedding uzayının önce yayılmasına izin verir, sonra ayrımı keskinleştirir."
>
> "**Üçüncüsü**: Diferansiyel öğrenme oranları — omurga yavaş öğrenir, yeni başlık hızlı öğrenir. Ayrıca rastgele kirpik benzeri silme ile veri çeşitlendirmesi yaptım."

---

### 📊 Bölüm 4 — Sonuçlar ve Dürüst Analiz (5:00 → 7:00)

**Tıkla**: Sol menüden ana sayfaya (`/`) dön.

> "Şimdi gerçek sayılara bakalım."

**İki büyük metrik kartını işaret et**.

> "**Daugman: %42 EER, AUC 0.61** — açıkça söyleyeyim, bu veri setinde Daugman algoritması iyi çalışmıyor. Sebebi bölütleme gürültüsü: Hough çemberleri iris sınırını birazcık kaçırınca, oluşan bit kodu neredeyse rastgele oluyor."

> "**CNN: %16 EER, AUC 0.92** — göreli olarak **%60'lık bir iyileşme** ve daha da önemlisi, **0.92'lik AUC değeri, modelin çiftleri %92 doğrulukla sıraladığı anlamına gelir**. Geriye kalan %16'lık hata, model kapasitesi tarafından değil, bölütleme kalitesi tarafından sınırlanıyor."

**Bu çok önemli — bunu vurgula**:

> "Pratikte en kritik nokta şu: CNN'in **aynı-kişi ile farklı-kişi çiftleri arasındaki ayrımı Daugman'ınkinden 30 kat daha büyük**. Yani teorik EER sayısı dışında, gerçek dünyada CNN çok daha güvenilir."

---

#### 🎯 CANLI DEMO (4:50 → 6:30)

**Tıkla**: Sol menüden `Verify` sayfasına git.
**Seç**: "Both (compare)" eşleştiriciyi.

> "Şimdi canlı bir gösterim yapayım."

**Adım 1 — Aynı kişi**:
- Image A → Konu 000'dan birinci görüntüyü yükle
- Image B → Konu 000'dan ikinci görüntüyü yükle
- "Verify Identity" butonuna tıkla

**Sonucu göster ve şöyle de**:

> "İki yöntem de yeşil verdi — **AYNI KİŞİ**. Daugman'ın Hamming uzaklığı 0.22, CNN'in kosinüs uzaklığı 0.4 civarında. Eşik değerlerinin altında, dolayısıyla eşleşme onaylandı."

**Adım 2 — Farklı kişiler**:
- Sayfayı yenile veya yeni dosya seç
- Image A → Konu 000'dan bir görüntü
- Image B → Konu 002'den bir görüntü
- "Verify Identity"

**Sonuçları göster — özellikle Daugman'ın hatasını vurgula**:

> "İşte ilginç olan kısım. Bakın, **Daugman bu iki farklı kişiyi neredeyse eşleştirmek üzere** — Hamming uzaklığı yine 0.22 civarında, eşiğin sınırında. Ama **CNN tamamen emin, kosinüs uzaklığı neredeyse 1.0, kesinlikle FARKLI KİŞİ diyor**."
>
> "Bu tek bir örnek, neden öğrenilmiş özelliklere ihtiyacımız olduğunun tüm hikayesini anlatıyor."

---

### 🏗 Bölüm 5 — Mimari (7:00 → 8:30)

> "Bu sadece bir Python defteri değil. Az önce gösterdiğim her şey gerçek bir **üretim seviyesinde web uygulaması yığını** üzerinde çalışıyor:"

**Bahsedilecek katmanlar**:

> "**Arka uçta** FastAPI çalışıyor — Python tabanlı, iris işlem hattını tipli bir HTTP API olarak sunuyor."
>
> "**Ön uçta** TanStack Start, React 19 ve shadcn/ui ile inşa edilmiş bir kontrol paneli var. Bu gördüğünüz her şey React bileşeni."
>
> "**PyTorch** modeli MacBook'umun Apple Silicon GPU'sunda Metal üzerinden çalıştırıyor — saniyenin yarısı kadar bir çıkarım süresi."

**Tıkla**: `System` sayfasına git, "LLM analysis — Türkçe" kartına kaydır.

> "Ve son rötuş — **Ollama üzerinden çalışan yerel bir LLM, llama3.2 modeli**, değerlendirme metriklerini okuyup otomatik olarak **Türkçe akademik analiz raporu** üretiyor. Burada Bulgular, Tartışma, Sonuç ve Gelecek Çalışmalar bölümlerini görüyorsunuz. Bu rapor uydurma değil — gerçek sayısal sonuçlardan üretildi."

---

### 🧱 Bölüm 6 — Sınırlamalar ve Sonraki Adımlar (8:30 → 9:30)

> "Bir mühendis olarak, sistemin sınırlarını da dürüstçe paylaşmam gerekiyor:"

> "**Birinci sınırlama**: Hough bölütlemem kapalı göz kapaklarında ve yoğun kirpiklerde başarısız oluyor. Görüntülerin yaklaşık **%10'u atlanıyor**. Bir **U-Net bölütleme modeli** EER'yi muhtemelen %5'in altına indirir."
>
> "**İkinci sınırlama**: Değerlendirme **veri seti içinde**. Gerçek biyometri sistemleri **çapraz sensör** testi gerektirir — örneğin telefon kamerası ile profesyonel iris tarayıcısının uyumu. Bunu yapmadım."
>
> "**Üçüncü sınırlama**: **Canlılık tespiti** yok. Basılı bir fotoğraf hala sistemi kandırabilir. Üretim seviyesindeki iris tarayıcılar, tanımayı kontrollü ışık altında gözbebeği genişleme testleriyle birleştirir."

---

### 🎤 Bölüm 7 — Kapanış (9:30 → 10:00)

> "Özetlemek gerekirse:"
>
> "1️⃣ Daugman'ın 1993 algoritmasını sıfırdan uyguladım."
>
> "2️⃣ Modern bir ResNet18 + ArcFace modeli eğittim ve **AUC 0.92'ye** ulaştım."
>
> "3️⃣ Tüm sistemi gerçek bir web kontrol paneli üzerinden sundum."
>
> "4️⃣ Otomatik analiz için yerel bir LLM ekledim."
>
> "En önemli ders şu: **klasik algoritmalar size anlayış ve referans noktası verir; öğrenilmiş özellikler size performans verir**. Bu projenin tek bir cümlede özetlemesi budur."
>
> "Beni dinlediğiniz için teşekkür ederim. Sorularınızı memnuniyetle alırım."

---

## 🖥 BÖLÜM 2 — DASHBOARD REHBERİ

### 🏠 Sayfa 1: Dashboard (Ana Sayfa) — `/`

| Eleman | Ne Söyleyeceksin |
|---|---|
| **Subjects** kartı (999) | "Veri tabanında neredeyse 1000 kişi var, hepsi başarıyla kodlandı." |
| **CNN EER** kartı (%16.22) | "Ana metrik — iki iris görüntüsünü doğrularken hata oranı." |
| **Model Health** kartı (koyu, AUC 0.919) | "**En kritik sayı**. AUC 0.92, modelin çiftleri %92 doğrulukla sıraladığını gösterir." |
| **End-to-end pipeline** kartı (görsel) | "Burada sistemin tam akışını görüyorsunuz — iki giriş, ResNet-18, karşılaştırma, sonuç." |
| **3 alt karşılaştırma kartı** | "Daugman %42, CNN %16, hata oranında **%60 göreceli iyileşme**." |
| Sağ taraftaki küçük kartlar | "Hızlı istatistikler ve sayfa kısayolları." |

---

### 🔍 Sayfa 2: Verify (Doğrulama) — `/verify`

| Eleman | Ne Söyleyeceksin |
|---|---|
| **Eşleştirici seçici** (Daugman / CNN / Both) | "Klasik algoritma ile, CNN ile veya her ikisi yan yana karşılaştırmalı olarak doğrulama yapabiliyorum." |
| **Yükleme alanları** | "İki iris görüntüsünü yükle. Tüm işlem hattı canlı olarak çalışır." |
| **Karar kutuları** (yeşil/kırmızı) | "Yeşil eşleşme, kırmızı eşleşmeme. Sayı uzaklık değeri; eşik karar sınırı." |
| **Pipeline artifacts** | "Bu küçük görseller rubber-sheet çıktısını ve ikili IrisCode'u gösteriyor — algoritmanın gerçekten ne gördüğünü." |

---

### 👥 Sayfa 3: Subjects (Kişiler) — `/subjects`

| Eleman | Ne Söyleyeceksin |
|---|---|
| **Arama kutusu** | "999 kişiyi ID'ye göre filtreleme." |
| **Sıralama** | "Tutarlılığa, örnek sayısına veya ID'ye göre sırala." |
| **Kalite rozeti** (high/medium/low) | "Aynı kişinin kendi iris kodları arasındaki Hamming uzaklığına göre renk kodlu." |
| **Open** butonu | "Herhangi bir satıra tıklayarak detayına in." |

---

### 👤 Sayfa 4: Subject Detail — `/subjects/$id`

| Eleman | Ne Söyleyeceksin |
|---|---|
| **Samples** | "Bu kişi için kaç iris görüntüsü kodlandığını gösterir." |
| **Intra-subject HD** | "Kişinin kendi kodları arasındaki ortalama Hamming uzaklığı — iris imzasının ne kadar tekrarlanabilir olduğunu ölçer." |
| **IrisCodes grid** | "Aynı kişinin benzer ikili desenler ürettiğinin görsel kanıtı — hepsi birbirine benziyor." |

---

### 🔬 Sayfa 5: System (Sistem) — `/system`

| Kart | Ne Söyleyeceksin |
|---|---|
| **Daugman's algorithm** | "1993 klasik işlem hattının adım adım açıklaması." |
| **Model architecture** (görsel) | "İşte yeni eklediğim mimari diyagramı — iki giriş, ResNet-18 omurgası, kosinüs karşılaştırma, eşleşme çıktısı." |
| **LLM analysis — Türkçe** | "Regenerate butonuna basarsam, Ollama llama3.2 modeli güncel metriklerden taze bir Türkçe rapor üretir." |
| **Best CNN checkpoint** kartı | "En iyi CNN modelinin tam sayıları: AUC, EER, eşik, FAR, FRR ve değerlendirme çift sayıları." |
| **Daugman baseline** kartı | "Klasik referans için aynı detaylar, böylece eşit karşılaştırma yapabilirsin." |

---

## 🆘 BÖLÜM 3 — SORU & CEVAP HAZIRLIĞI

### Soru: *"Neden önce derin öğrenme ile başlamadın?"*

> "Daugman'ın algoritması her iris tanıma makalesinin karşılaştırma referansıdır. Sıfırdan uygulamak problemi gerçekten anlamamı sağladı. CNN sonra geldi — aynı ön işleme, farklı kodlayıcı, kontrollü bir deney."

### Soru: *"Neden Hamming uzaklığı, neden Öklid değil?"*

> "Çünkü IrisCode ikili. Hamming uzaklığı = farklı bit sayısı bölü toplam bit sayısı. Bu Bernoulli denemesi olarak temiz istatistiksel garantiler sağlar — Daugman impostor dağılımının teorik özelliklerini bunun üzerinden ispat ediyor."

### Soru: *"Sistem nasıl kandırılabilir?"*

> "Yüksek çözünürlüklü basılı fotoğraf veya desenli kontakt lens. Modern üretim sistemleri tanımayı **canlılık tespiti** ile birleştirir — gözbebeği tepkileri, mikro hareketler, kızılötesi yansıma analizi gibi."

### Soru: *"Modelin boyutu ne kadar? Gerçek zamanlı çalışabilir mi?"*

> "ResNet-18, embedding katmanıyla birlikte yaklaşık 11 milyon parametre, diskte 45 MB. CPU üzerinde çıkarım 50 milisaniyenin altında. Bir akıllı telefon bile bunu rahat çalıştırır."

### Soru: *"Veri setinde olmayan birini tanıyabilir mi?"*

> "Evet — açık-küme bölünmesinin tüm amacı bu. Veri setindeki kişilerin %20'si eğitim sırasında hiç görülmedi ve raporladığım EER değeri bu yeni kişiler üzerinde hesaplandı. Sistem yeni bir kişiyi 'bilmediği' birisi olarak doğru şekilde işleyebilir."

### Soru: *"Daugman neden bu kadar kötü performans gösterdi?"*

> "Daugman algoritması sağlam ama bölütlemeye çok hassas. Hough çemberleri iris sınırını birkaç piksel kaçırırsa, kauçuk levha eşleşmiyor ve bit kodu rastgele oluyor. Eğer bölütlemeyi bir U-Net ile değiştirseydim, Daugman muhtemelen %5 EER altına inerdi. Bu, mimari değil veri-akış problemi."

### Soru: *"Neden ArcFace, neden Triplet Loss değil?"*

> "Triplet Loss dikkatli negatif örnek seçimi gerektirir ve batch kompozisyonuna çok hassastır. ArcFace, bunu softmax tabanlı margin yaklaşımıyla daha kararlı bir şekilde elde ediyor — eğitim çok daha kolay."

### Soru: *"Eğitim ne kadar sürdü?"*

> "Kaggle T4 GPU üzerinde 35 epoch, toplam yaklaşık 40 dakika. Veri ön işleme — yani 20.000 görüntüyü tek tek rubber-sheet'e çevirmek — 10 dakika sürdü ve sonuç önbelleğe alındı."

### Soru: *"LLM kullanmak neden mantıklı?"*

> "İki sebep: Birincisi, sonuçları otomatik olarak doğal dilde özetlemek araştırmacılar için zaman kazandırır. İkincisi, yerel bir LLM (llama3.2 Ollama üzerinden) kullandığım için **veri dışarı çıkmıyor** — biyometrik bir uygulamada bu kritik."

---

## ✅ BÖLÜM 4 — SUNUM ÖNCESİ KONTROL LİSTESİ

**Sunumdan 1 saat önce**:

- [ ] Backend ve dashboard çalışıyor mu? → `./scripts/dev.sh`
- [ ] Tarayıcıda `http://localhost:5173` açık ve hata yok mu?
- [ ] Demo klasöründe 4 iris görüntüsü hazır mı?
  - 2 tane Konu 000'dan (aynı kişi)
  - 1 tane Konu 001'den
  - 1 tane Konu 002'den
- [ ] Ollama çalışıyor mu? → `ollama serve` (LLM göstermek istiyorsan)
- [ ] Yedek video/ekran görüntüsü masaüstünde mi?
- [ ] Telefonu sessize al

**Sunum sırasında**:

- [ ] Yavaş konuş — 10 dakika çok zaman, acele etme
- [ ] Demo sırasında **sayıları yüksek sesle oku** — izleyici küçük yazıyı göremez
- [ ] Sayfa geçişlerinde 1-2 saniye bekle, izleyicinin görmesine izin ver
- [ ] Mouse hareketini yavaş yap — hızlı imleç gözleri yorar
- [ ] Hata olursa: "Bu canlı sistemler, bazen aksaklık olabilir" de ve yedek videoya geç

**Sunumdan sonra**:

- [ ] Sorulara cevap verirken **kısa ve net** ol
- [ ] Bilmediğin bir soru gelirse: *"Çok güzel soru — bunu test etmem gerekir. Tahminim X çünkü Y"* de
- [ ] Asla blöf yapma, asla "bilmiyorum" demekten korkma

---

## 📊 EZBERLE — Kritik Sayılar

| Metrik | Değer | Söyleyiş |
|---|---|---|
| Toplam kişi | 999 | "Neredeyse bin kişi" |
| Toplam kodlanmış görüntü | 18.148 | "On sekiz bin küsür görüntü" |
| Daugman EER | %42.06 | "Yüzde kırk iki" |
| Daugman AUC | 0.610 | "Yaklaşık 0.6" |
| CNN EER | %16.22 | "Yüzde on altı" |
| **CNN AUC** | **0.919** | **"Sıfır nokta doksan iki" — bu sayı en önemlisi!** |
| EER iyileştirme | %60 | "Göreli olarak %60 daha iyi" |
| Pratikteki ayrım kazancı | 30 kat | "Otuz kat daha iyi ayrım" |
| Eğitim süresi | 35 epoch / ~40 dk | "Kaggle GPU'da kırk dakika" |
| Model boyutu | 11 M parametre / 45 MB | "On bir milyon parametre" |

---

İyi şanslar! 🍀 Bu sunumun arkasında gerçek bir mühendislik çalışması var — kendine güven, yavaş konuş, sayıları net telaffuz et.
