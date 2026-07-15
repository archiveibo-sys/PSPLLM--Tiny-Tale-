# PSP TinyTale LLM — Yapım Rehberi

> ## ⚠️ Ön koşul — bu kısım size ait
> Bu proje, kullandığınız işletim sisteminde (**Ubuntu** veya **WSL Ubuntu**)
> PSP için uygulama geliştirme ortamının (`pspdev` / `pspsdk`) **hâlihazırda
> kurulu ve çalışır** olduğunu, yani **`EBOOT.PBP` üretebildiğinizi** varsayar.
> Bu ortamın kurulumu bu rehberin kapsamı **dışındadır** ve sizin
> sorumluluğunuzdadır. Eğer `make` ile PSP homebrew derleyemiyorsanız, önce o
> ortamı kurun — bkz. [pspdev](https://github.com/pspdev/pspdev).

Bu belge, Sony PSP-E1000 üzerinde tamamen cihaz içinde çalışan, kendi
eğittiğimiz Türkçe bir dil modelinin (transformer) nasıl yapıldığını,
**yalnızca gerekli adımlarla** anlatır. Deneme-yanılma sırasında yaşanan
çıkmazlar (ortam sorunları, erişilemeyen veri setleri vb.) bilinçli olarak
atlanmıştır; aşağıdaki sıra, en baştan yapsaydık izleyeceğimiz temiz yoldur.

---

## 1. Ne yaptık? (Özet)

- **Donanım:** PSP-E1000 — 333 MHz tek çekirdek MIPS (Allegrex) + VFPU, 64 MB RAM.
- **Model:** ~8M parametreli decoder-only transformer (Llama mimarisi: RMSNorm,
  RoPE, SwiGLU). Karpathy'nin `llama2.c` mimarisi temel alındı.
- **Çalışma biçimi:** Ağırlıklar **int8 (Q8_0)** olarak quantize edilir, modelin
  tamamı cihaz RAM'ine yüklenir, çıkarım (token üretimi) tamamen PSP'nin
  CPU'sunda olur. İnternet/sunucu yoktur.
- **Dil:** Türkçe. Hazır Türkçe veri seti bulunmadığı için İngilizce TinyStories
  yerel bir çeviri modeliyle Türkçeye çevrilerek korpus üretildi.

İş bölümü: **model PC'de eğitilir ve quantize edilir, PSP yalnızca bitmiş modeli
çalıştırır.** (Eğitim PSP'de yapılamaz; çok yavaş olurdu.)

```
[PC]  İngilizce TinyStories ──çeviri──> Türkçe korpus
                                            │
                                       tokenizer (8K) + eğitim
                                            │
                                   export + int8 quantize
                                            │
                              model_q80.bin + tokenizer.bin
                                            │
[PSP] ◄──────────── Memory Stick ──────────┘
        EBOOT.PBP (int8 runtime) bunları okuyup Türkçe üretir
```

---

## 2. Gereksinimler

**PSP tarafı**
- Özel yazılım (CFW) çalıştırabilen bir PSP (E1000 dahil).
- `pspdev` / `pspsdk` kurulu bir geliştirme ortamı (Windows + WSL veya Linux).
- `make` ile `EBOOT.PBP` üretebiliyor olmak.

**PC tarafı (eğitim/çeviri)**
- **Python 3.12 veya 3.13** (3.14 KULLANMA — PyTorch paketi henüz yok).
  İzole ortam için Miniconda en pürüzsüz yoldur:
  ```bash
  conda create -n llama python=3.12 -y
  conda activate llama
  ```
- PyTorch + araçlar:
  ```bash
  # GPU yoksa veya disk/kurulum sorunundan kaçınmak için CPU sürümü yeterli:
  pip install torch --index-url https://download.pytorch.org/whl/cpu
  # (NVIDIA GPU varsa ve hız isteniyorsa CPU yerine: .../whl/cu121)
  pip install transformers sacremoses sentencepiece datasets numpy
  ```
- HuggingFace'e giriş (veri/model indirmede sürtünmeyi azaltır):
  ```bash
  huggingface-cli login    # https://huggingface.co/settings/tokens 'tan Read token
  ```

**Kullanılan kod dosyaları**
| Dosya | Görev |
|------|------|
| `make_turkish_data.py` | İngilizce TinyStories'i Türkçeye çevirip korpus üretir |
| `train_turkish.py` | Tokenizer eğitimi + model eğitimi + export + quantize (hepsi bir arada) |
| `main_q.c` + `Makefile` | PSP int8 çıkarım motoru (EBOOT.PBP) |
| `quantize.py` | (Opsiyonel) bağımsız float32→int8 dönüştürücü |

---

## 3. Adım adım işlem sırası

### Adım 1 — Türkçe korpusu üret (çeviri)

```bash
python make_turkish_data.py --n 5000
```

- İngilizce `roneneldan/TinyStories`'ten ilk 5000 hikâyeyi çeker (streaming).
- `Helsinki-NLP/opus-mt-tc-big-en-tr` ile Türkçeye çevirir.
- `data/turkish_stories.txt` (her satır bir hikâye) üretir.
- Yarıda kesilirse tekrar çalıştırınca kaldığı yerden devam eder.

> İlk denemede 5000 hikâye yeterlidir (zincirin çalıştığını görmek için).
> Kaliteyi artırmak için sonradan `--n 50000` ile büyütülür (GPU önerilir).

### Adım 2 — Tokenizer + modeli eğit

```bash
python train_turkish.py --text data/turkish_stories.txt
```

Bu tek komut sırasıyla şunları yapar:
1. Türkçe metin üzerinde **8K'lık SentencePiece BPE** tokenizer eğitir
   (`character_coverage=1.0` + `byte_fallback` → ğ/ş/ı/İ/ç/ö/ü tam desteklenir).
2. Korpusu tokenize eder (`train.bin` / `val.bin`).
3. ~8M parametreli modeli eğitir; her 500 adımda **örnek Türkçe metin** basar,
   böylece ilerlemeyi canlı görürsün.
4. Bitince export + int8 quantize yapıp **`out/model_q80.bin`** ve
   **`out/tokenizer.bin`** üretir.

> Eğitimi istediğin an `Ctrl+C` ile durdurup `python train_turkish.py --stage export`
> diyerek o ana kadarki en iyi modelden dosyaları üretebilirsin. CPU'da eğitim
> yavaştır; örnek üretimleri tatmin edici Türkçeye dönünce durmak yeterlidir.

### Adım 3 — PSP çıkarım motorunu derle

`main_q.c` + `Makefile`'ı bir klasöre koyup:

```bash
make
```

`EBOOT.PBP` üretir. `main_q.c` içindeki PSP'ye özgü kritik noktalar (zaten
koddadır, bilgi amaçlı):
- Model `mmap` yerine `malloc` + `fread` ile tek seferde RAM'e yüklenir.
- Kelime gömme tablosu **tamamen** float'a açılmaz (32K vocab'da ~37 MB ederdi);
  her token için yalnızca ilgili satır anlık çözülür.
- CPU 333 MHz'e kilitlenir, ana iş parçacığı VFPU özniteliğiyle açılır.
- Ekran çıktısı **UTF-8 → ASCII** çevrilir (PSP debug ekranı UTF-8 bilmez):
  ğ→g, ş→s, ı→i, İ→I, ç→c, ö→o, ü→u. Çıktı şapkasız ama okunaklıdır.
- Varsayılan başlangıç metni `"Bir zamanlar"`.

### Adım 4 — PSP'ye kur

Memory Stick'te aşağıdaki klasöre **üç dosyayı** koy:

```
ms0:/PSP/GAME/tinytale/
        ├─ EBOOT.PBP        (Adım 3)
        ├─ model_q80.bin    (Adım 2, out/ içinden)
        └─ tokenizer.bin    (Adım 2, out/ içinden)
```

> Model boyutları (dim, katman, vocab, group size) dosya başlığından otomatik
> okunur; `main_q.c`'de elle ayar gerekmez.

### Adım 5 — Çalıştır

XMB → Oyun → Memory Stick → uygulamayı başlat. Ekranda "Bir zamanlar..." ile
başlayan Türkçe metin akar ve en altta saniyedeki token (tok/s) yazar.

---

## 4. Neden bu kararlar? (Tasarımın özü)

- **int8 (Q8_0) quantize:** float32 model RAM'e sığmaz (15M model ~60 MB).
  int8'e indirince ~dörtte birine düşer; hem sığar hem bellek bant genişliği
  azaldığı için PSP'de hızlanır.
- **8K kelime hazinesi (32K değil):** Modelin işinin büyük kısmı son katmandadır
  ve kelime sayısıyla orantılıdır. 8K seçmek bu katmanı ~4 kat ucuzlatır →
  hem hız hem bellek kazancı. Türkçe için 8K BPE yeterli temsil verir.
- **TinyStories yaklaşımı (basit/dar veri):** Küçük modeller ancak basit ve
  sınırlı kelimeli veride tutarlı metin üretir. Karmaşık genel metinle (ör.
  ansiklopedi) ~8M model tutarsız çıktı verirdi. Bu yüzden basit hikâye korpusu.
- **PC'de eğit, PSP'de çalıştır:** 333 MHz'de eğitim aylar sürerdi; PSP yalnızca
  çıkarım yapar.

---

## 5. Ölçülen performans ve sınırlar

- **Hız (gerçek cihaz):** İngilizce 15M int8 model PSP'de ~**1.26 tok/s** ölçüldü.
  8K vocab'lı ~8M Türkçe model bundan daha hızlıdır.
- **Eğitim (CPU):** 5000 hikâye ile en iyi nokta ~500 adımdı; sonrasında model
  veriyi ezberlemeye başladı (doğrulama kaybı yükseldi). Bu, "5000 hikâye az"
  demektir — ilk sürüm için yeterli, kalite için veri büyütülmeli.
- **Yetenek sınırı:** ~8M parametre "her konuyu konuşan" bir asistan değildir;
  dar alanda (basit hikâye) akıcıdır, açık uçlu sohbette zayıftır. Bu bir kusur
  değil, 64 MB'lık donanım için bilinçli ölçek seçimidir.

---

## 6. Daha ileri götürmek (opsiyonel)

1. **Kalite (en yüksek etki):** `make_turkish_data.py --n 50000` ile daha çok
   veri çevir, sonra yeniden eğit. Ezberlemeyi azaltır, Türkçeyi zenginleştirir.
   CPU'da uzun sürer → bu aşamada NVIDIA GPU'lu PyTorch (`cu121`) önerilir.
2. **Hız:** Çıkarımın sıcak döngüsündeki matris çarpımını PSP'nin VFPU vektör
   birimiyle elle yazmak token/sn'yi katlayabilir.
3. **Gerçek Türkçe harfler:** ASCII eşleme yerine, ekran tamponuna kendi bitmap
   fontumuzu çizen bir katman yazılarak gerçek ğ/ş/ı gösterilebilir.
4. **Sohbet botu:** Base modeli Türkçe diyalog verisiyle ince ayar (fine-tune)
   ederek hikâye anlatıcıdan basit sohbete dönüştürmek.

---

## 7. Tek bakışta komut özeti

```bash
# Ortam (bir kez)
conda create -n llama python=3.12 -y && conda activate llama
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install transformers sacremoses sentencepiece datasets numpy
huggingface-cli login

# Veri  ->  Eğitim+Export
python make_turkish_data.py --n 5000
python train_turkish.py --text data/turkish_stories.txt
#   -> out/model_q80.bin , out/tokenizer.bin

# PSP motoru
make            # main_q.c + Makefile  ->  EBOOT.PBP

# Kur:  EBOOT.PBP + model_q80.bin + tokenizer.bin
#       -> ms0:/PSP/GAME/tinytale/  ->  çalıştır
```
