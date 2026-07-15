#!/usr/bin/env python3
"""
make_turkish_data.py  -  İngilizce TinyStories -> Türkçe (yerel çeviri)

Hazır Türkçe TinyStories bulunmadığı için (SoAp9035 silinmiş) kendimiz üretiyoruz:
İngilizce TinyStories'i Helsinki-NLP/opus-mt-tc-big-en-tr ile Türkçeye çevirir,
satır satır data/turkish_stories.txt'e yazar (her satır bir hikâye).

  python make_turkish_data.py --n 5000      # önce küçük dene (hızlı kanıt)
  python make_turkish_data.py --n 50000     # beğenince büyüt

Özellikler:
  - cache/resume: yarıda kesilirse tekrar çalıştır, kaldığı yerden devam eder
  - GPU varsa GPU, yoksa CPU (kullanıcının kurulumuna göre otomatik)

Bağımlılıklar: torch, transformers, sentencepiece, sacremoses, datasets
  pip install transformers sacremoses
"""
import os, sys, time, argparse

OUT_DEFAULT = "data/turkish_stories.txt"
SRC_DATASET = "roneneldan/TinyStories"
MT_MODEL = "Helsinki-NLP/opus-mt-tc-big-en-tr"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5000, help="kaç hikâye çevrilsin")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--out", default=OUT_DEFAULT)
    ap.add_argument("--max_len", type=int, default=512)
    args = ap.parse_args()

    import torch
    from transformers import MarianMTModel, MarianTokenizer
    from datasets import load_dataset

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[mt] cihaz: {device}")

    # kaç tane zaten çevrilmiş? (resume)
    done = 0
    if os.path.exists(args.out):
        with open(args.out, encoding="utf-8") as f:
            done = sum(1 for _ in f)
        print(f"[mt] {done} hikâye zaten çevrilmiş, devam ediliyor")
    if done >= args.n:
        print(f"[mt] hedefe ({args.n}) ulaşılmış. Bitti.")
        return

    # İngilizce hikâyeleri streaming ile çek (tüm 2GB inmez)
    print(f"[mt] {SRC_DATASET} (streaming) ilk {args.n} hikâye...")
    ds = load_dataset(SRC_DATASET, split="train", streaming=True)
    english = []
    for i, ex in enumerate(ds):
        if i >= args.n:
            break
        t = (ex.get("text") or "").strip().replace("\n", " ")
        if t:
            english.append(t)
    print(f"[mt] {len(english)} İngilizce hikâye alındı")
    todo = english[done:args.n]
    if not todo:
        print("[mt] çevrilecek yeni hikâye yok.")
        return

    print(f"[mt] çeviri modeli yükleniyor: {MT_MODEL}")
    tok = MarianTokenizer.from_pretrained(MT_MODEL)
    model = MarianMTModel.from_pretrained(MT_MODEL).to(device).eval()

    @torch.no_grad()
    def translate(batch_texts):
        enc = tok(batch_texts, return_tensors="pt", padding=True,
                  truncation=True, max_length=args.max_len).to(device)
        gen = model.generate(**enc, max_length=args.max_len, num_beams=1)
        return tok.batch_decode(gen, skip_special_tokens=True)

    t0 = time.time()
    written = done
    with open(args.out, "a", encoding="utf-8") as out:
        for b in range(0, len(todo), args.batch):
            chunk = todo[b:b + args.batch]
            try:
                tr = translate(chunk)
            except Exception as e:
                print(f"\n[mt] batch hata ({e}); tek tek deneniyor...")
                tr = []
                for s in chunk:
                    try: tr.append(translate([s])[0])
                    except Exception: tr.append("")  # atla
            for line in tr:
                line = line.strip().replace("\n", " ")
                if line:
                    out.write(line + "\n")
                    written += 1
            out.flush()
            elapsed = time.time() - t0
            rate = (b + len(chunk)) / max(1e-9, elapsed)
            remaining = (len(todo) - (b + len(chunk))) / max(1e-9, rate)
            print(f"  {written}/{args.n}  ({rate:.1f} hikâye/sn, ~{remaining/60:.0f} dk kaldı)", end="\r")

    print(f"\n[mt] bitti: {written} hikâye -> {args.out}")
    print(f"\nSonraki adım:")
    print(f"  python train_turkish.py --text {args.out}")


if __name__ == "__main__":
    main()
