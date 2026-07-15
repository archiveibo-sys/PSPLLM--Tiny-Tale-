Ya aslında ekran görüntüsü koyacaktım ama cihaz arkadaşta kalmış. bir iki screenshot koyacam.
I had started the project with Fable 5. Then, during that time, it was banned, so I was forced to continue with OPUS 4.8; however, I had achieved the result I was aiming for.

# PSP TinyTale LLM 🎮📖

> ## ⚠️ Prerequisite — this part is on you
> This project assumes you **already have a working PSP homebrew development
> environment** (`pspdev` / `pspsdk`) installed on your system
> (**Ubuntu** or **WSL Ubuntu**) and that you can **already compile
> `EBOOT.PBP`** files. Setting up that toolchain is **NOT** covered here and is
> your responsibility. If `make` doesn't already build PSP homebrew on your
> machine, set that up first — see [pspdev](https://github.com/pspdev/pspdev).

**A tiny Turkish storyteller that runs entirely on the Sony PSP.**

PSP TinyTale LLM (Turkish: *Masalcı*, "storyteller") is a small decoder-only
transformer (~8M parameters, int8 quantized) that runs **fully on-device** on a
PSP-E1000 — a 333 MHz handheld from 2004 with 64 MB of RAM. No internet,
no server: the model is loaded into the console's RAM and generates Turkish
text on its MIPS CPU.

It uses the same architecture as modern LLMs (RMSNorm, RoPE, SwiGLU — based on
Andrej Karpathy's [`llama2.c`](https://github.com/karpathy/llama2.c)), just at
a scale that fits a 20-year-old game console.

> **Scale note:** This is a *small* language model (~8M params). It won't chat
> about anything like ChatGPT — but it tells coherent little Turkish stories,
> and it does so on a PSP. The architecture and training are real; only the
> size is tiny, by design.

---

## How it works

The model is **trained and quantized on a PC; the PSP only runs inference.**

```
[PC]  English TinyStories ──translate──> Turkish corpus
                                             │
                                   8K SentencePiece tokenizer + training
                                             │
                                    export + int8 (Q8_0) quantize
                                             │
                                model_q80.bin  +  tokenizer.bin
                                             │
[PSP] ◄──────────────── Memory Stick ────────┘
        EBOOT.PBP (int8 inference engine) reads them and generates Turkish
```

Since there is no ready-made Turkish TinyStories dataset, we **build one**:
English TinyStories is translated to Turkish locally with
[`Helsinki-NLP/opus-mt-tc-big-en-tr`](https://huggingface.co/Helsinki-NLP/opus-mt-tc-big-en-tr).

---

## Repository layout

```
psp-tinytale-llm/
├── psp/                    # On-device inference engine (the PSP app)
│   ├── main.c              #   int8 llama2 runtime, PSP-specific
│   └── Makefile            #   builds EBOOT.PBP with pspsdk
├── train/                  # PC-side pipeline
│   ├── make_turkish_data.py#   English TinyStories -> Turkish (translation)
│   ├── train_turkish.py    #   tokenizer + training + export + quantize
│   └── quantize.py          #   standalone float32 -> int8 converter (optional)
└── docs/
    └── REHBER.md           # Detailed build guide (Turkish)
```

---

## Requirements

**PSP side** — *(see the prerequisite warning at the top)*
- A PSP capable of running homebrew (custom firmware). Works on PSP-E1000.
- A working [`pspdev` / `pspsdk`](https://github.com/pspdev/pspdev) toolchain
  that you have **already installed and verified**.

**PC side (training / translation)**
- **Python 3.12 or 3.13** (not 3.14 — PyTorch wheels aren't available there yet).
- PyTorch + tools:
  ```bash
  # CPU is enough; use the cu121 index instead if you have an NVIDIA GPU
  pip install torch --index-url https://download.pytorch.org/whl/cpu
  pip install transformers sacremoses sentencepiece datasets numpy
  ```
- A (free) HuggingFace login helps with downloads: `huggingface-cli login`.

---

## Quick start

### 1. Build the Turkish corpus (translation)
```bash
cd train
python make_turkish_data.py --n 5000     # start small; --n 50000 for quality
```
Produces `data/turkish_stories.txt`. Resumable if interrupted.

### 2. Train + export
```bash
python train_turkish.py --text data/turkish_stories.txt
```
Trains an 8K SentencePiece tokenizer, trains the model (prints Turkish samples
every 500 steps), then exports **`out/model_q80.bin`** and **`out/tokenizer.bin`**.
Stop anytime with `Ctrl+C` and run `python train_turkish.py --stage export`.

### 3. Build the PSP app
```bash
cd ../psp
make                                      # produces EBOOT.PBP
```

### 4. Deploy & run
Copy the three files to your Memory Stick:
```
ms0:/PSP/GAME/tinytale/
        ├─ EBOOT.PBP
        ├─ model_q80.bin
        └─ tokenizer.bin
```
Launch from the PSP's Game menu. It generates Turkish text starting from
"Bir zamanlar" (Turkish for "Once upon a time") and prints tokens/sec at the end.

---

## Technical notes

A few PSP-specific decisions that make this work on 64 MB / 333 MHz:

- **int8 (Q8_0) quantization** — a float32 model doesn't fit in RAM
  (a 15M model is ~60 MB). int8 cuts that ~4× and also speeds up inference,
  since the PSP is memory-bandwidth bound.
- **8K vocabulary (not 32K)** — the final projection layer dominates compute and
  scales with vocab size; 8K makes it ~4× cheaper, saving both time and memory.
- **On-the-fly embedding dequant** — the token embedding table is kept int8 and
  only the needed row is dequantized per token, instead of expanding the whole
  table to float (~37 MB for a 32K vocab — it wouldn't fit).
- **No `mmap`** — the checkpoint is loaded into RAM once with `malloc` + `fread`.
- **333 MHz + VFPU** — the CPU is clocked up and the main thread is VFPU-enabled
  (ready for a future hand-written vector matmul).
- **UTF-8 → ASCII on screen** — the PSP debug screen has no UTF-8 font, so Turkish
  letters are mapped to ASCII (ğ→g, ş→s, ı→i, İ→I, ç→c, ö→o, ü→u). Output is
  readable (without diacritics). A real bitmap font is a possible improvement.

The training pipeline follows the TinyStories recipe: a **small, simple corpus**
lets a tiny model produce coherent text. Complex general text (e.g. encyclopedia
articles) would yield garbage at ~8M parameters.

---

## Performance & limitations

- **Measured speed:** an English 15M int8 model ran at ~**1.26 tok/s** on real
  hardware; the 8K-vocab ~8M Turkish model is faster.
- **Capability:** great at simple Turkish stories within its narrow domain; weak
  at open-ended conversation. That's the intended trade-off for the hardware.
- **Data scale:** 5,000 translated stories is enough to prove the pipeline but
  the model overfits (memorizes). Translate more (`--n 50000`) for real quality.

---

## Roadmap

- [ ] More translated data (`--n 50000+`) + GPU training to reduce overfitting.
- [ ] Hand-written VFPU matmul to multiply tokens/sec.
- [ ] Custom bitmap font for real ğ/ş/ı rendering on screen.
- [ ] Chat fine-tune on a Turkish dialog dataset.

---

## Credits

- [`llama2.c`](https://github.com/karpathy/llama2.c) by Andrej Karpathy — model
  architecture and the run/runq inference reference.
- [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) by
  Ronen Eldan & Yuanzhi Li — the source corpus.
- [OPUS-MT](https://huggingface.co/Helsinki-NLP/opus-mt-tc-big-en-tr)
  (Helsinki-NLP) — English→Turkish translation.
- [`pspdev`](https://github.com/pspdev/pspdev) — the PSP homebrew toolchain.

## License

MIT — see [LICENSE](LICENSE).

---

🇹🇷 **Türkçe ayrıntılı yapım rehberi için: [docs/REHBER.md](docs/REHBER.md)**
