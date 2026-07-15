#!/usr/bin/env python3
"""
train_turkish.py  -  TEK KOMUTLA Türkçe model -> PSP'ye hazır dosyalar

  python train_turkish.py

Yaptıkları (sırayla, otomatik):
  1. SoAp9035/Turkish_TinyStories indir
  2. 8K SentencePiece tokenizer eğit (ğ/ş/ı/İ destekli)
  3. Korpusu tokenize et (train.bin / val.bin)
  4. ~8M parametreli llama2 modelini GPU'da eğit (eğitim sırasında Türkçe örnek basar)
  5. PSP'ye hazır iki dosya üret:  out/model_q80.bin   ve   out/tokenizer.bin

Devam etmek için (eğitim yarıda kaldıysa):  python train_turkish.py --resume
Sadece export (eğitim bitti, dosyaları üret):  python train_turkish.py --stage export

Bağımlılıklar: torch, sentencepiece, datasets, numpy
"""
import os, math, glob, json, time, struct, argparse
import numpy as np

DATA_DIR = "data"
OUT_DIR  = "out"
DATASET  = "SoAp9035/Turkish_TinyStories"
TEXT_FILE = None  # set via --text to read stories from a local file instead of HF

# ----------------------------------------------------------------------------
# Model hyperparameters (tuned for a 2 GB GPU; ~8M params)
# ----------------------------------------------------------------------------
class Cfg:
    dim         = 288
    n_layers    = 6
    n_heads     = 6
    n_kv_heads  = 6          # = n_heads (no GQA), keeps it simple
    vocab_size  = 8192
    seq_len     = 256
    multiple_of = 32
    # training
    batch_size  = 8
    grad_accum  = 8          # effective batch 64
    max_iters   = 10000
    eval_interval = 500
    eval_iters  = 50
    lr          = 5e-4
    min_lr      = 5e-5
    warmup      = 200
    weight_decay = 0.1
    grad_clip   = 1.0

# ============================================================================
# STEP 1-3: data + tokenizer
# ============================================================================
def _load_rows():
    """Load dataset rows robustly: try `datasets`, else snapshot_download + read files."""
    try:
        from datasets import load_dataset
        print(f"[data] load_dataset('{DATASET}') deneniyor...")
        ds = load_dataset(DATASET)
        split = "train" if "train" in ds else list(ds.keys())[0]
        d = ds[split]
        return [d[i] for i in range(len(d))], list(d.column_names)
    except Exception as e:
        print(f"[data] load_dataset olmadi ({type(e).__name__}); snapshot_download deneniyor...")

    from huggingface_hub import snapshot_download
    local = snapshot_download(repo_id=DATASET, repo_type="dataset")
    print(f"[data] repo indirildi: {local}")
    rows, cols = [], None

    pqs = sorted(glob.glob(os.path.join(local, "**", "*.parquet"), recursive=True))
    if pqs:
        import pyarrow.parquet as pq
        for f in pqs:
            tbl = pq.read_table(f).to_pylist()
            if tbl:
                cols = cols or list(tbl[0].keys())
                rows.extend(tbl)
        return rows, cols

    for f in sorted(glob.glob(os.path.join(local, "**", "*.json*"), recursive=True)):
        with open(f, encoding="utf-8") as fh:
            txt = fh.read().strip()
        try:
            obj = json.loads(txt)
            rows.extend(obj if isinstance(obj, list) else [obj])
        except json.JSONDecodeError:
            for line in txt.splitlines():
                line = line.strip()
                if line:
                    try: rows.append(json.loads(line))
                    except json.JSONDecodeError: rows.append({"text": line})
    if not rows:
        for f in sorted(glob.glob(os.path.join(local, "**", "*.txt"), recursive=True)):
            with open(f, encoding="utf-8") as fh:
                for line in fh:
                    if line.strip():
                        rows.append({"text": line.strip()})
    if not rows:
        raise SystemExit(f"{DATASET} icinde okunabilir veri yok (parquet/json/txt bulunamadi).")
    cols = cols or (list(rows[0].keys()) if isinstance(rows[0], dict) else ["text"])
    return rows, cols


def prepare_data():
    os.makedirs(DATA_DIR, exist_ok=True)
    model_path = os.path.join(DATA_DIR, f"tok{Cfg.vocab_size}.model")
    train_bin  = os.path.join(DATA_DIR, "train.bin")
    val_bin    = os.path.join(DATA_DIR, "val.bin")
    if os.path.exists(train_bin) and os.path.exists(val_bin) and os.path.exists(model_path):
        print("[data] already prepared, skipping (delete data/ to redo)")
        return model_path

    import sentencepiece as spm

    if TEXT_FILE and os.path.exists(TEXT_FILE):
        print(f"[data] yerel metin dosyasi: {TEXT_FILE}")
        with open(TEXT_FILE, encoding="utf-8") as f:
            stories = [ln.strip() for ln in f if ln.strip()]
        print(f"[data] {len(stories)} hikaye (yerel dosyadan)")
    else:
        rows, cols = _load_rows()
        text_col = next((c for c in ["story","text","content","turkish","tr","translation","output","story_tr","Story"] if c in cols), None)
        if text_col is None:
            text_col = next(c for c in cols if isinstance((rows[0].get(c) if isinstance(rows[0], dict) else rows[0]), str))
        print(f"[data] rows={len(rows)} text_col='{text_col}'")
        stories = []
        for r in rows:
            s = r.get(text_col) if isinstance(r, dict) else r
            if isinstance(s, str) and s.strip():
                stories.append(s.strip())
        print(f"[data] {len(stories)} stories")

    # --- train tokenizer (no input() prompt; llama2.c settings) ---
    tiny = os.path.join(DATA_DIR, "tiny.txt")
    with open(tiny, "w", encoding="utf-8") as f:
        for s in stories:
            f.write(s.replace("\n", " ") + "\n")
    print(f"[tok] training SentencePiece BPE vocab={Cfg.vocab_size} ...")
    spm.SentencePieceTrainer.train(
        input=tiny, model_prefix=os.path.join(DATA_DIR, f"tok{Cfg.vocab_size}"),
        model_type="bpe", vocab_size=Cfg.vocab_size, self_test_sample_size=0,
        input_format="text", character_coverage=1.0, num_threads=os.cpu_count(),
        split_digits=True, allow_whitespace_only_pieces=True, byte_fallback=True,
        unk_surface=r" \342\201\207 ", normalization_rule_name="identity",
        bos_id=1, eos_id=2, unk_id=0, pad_id=-1,
    )
    sp = spm.SentencePieceProcessor(model_file=model_path)

    # --- tokenize: each story gets a BOS(1) prefix, then concatenated ---
    print("[tok] tokenizing corpus ...")
    all_ids = []
    for i, s in enumerate(stories):
        ids = sp.encode(s, out_type=int)
        all_ids.append(1)            # BOS delimiter
        all_ids.extend(ids)
        if (i + 1) % 20000 == 0:
            print(f"  {i+1}/{len(stories)}")
    arr = np.array(all_ids, dtype=np.uint16)
    n_val = max(1, int(len(arr) * 0.02))
    arr[:-n_val].tofile(train_bin)
    arr[-n_val:].tofile(val_bin)
    print(f"[data] total tokens={len(arr):,}  train={len(arr)-n_val:,}  val={n_val:,}")
    print(f"[data] chars/token ~= {sum(len(s) for s in stories)/max(1,len(arr)):.2f}")
    return model_path

# ============================================================================
# STEP 4: the model (llama2 architecture, RoPE matched to run.c/PSP)
# ============================================================================
def build_model():
    import torch, torch.nn as nn, torch.nn.functional as F

    def precompute_freqs_cis(dim, end, theta=10000.0):
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: dim // 2].float() / dim))
        t = torch.arange(end)
        freqs = torch.outer(t, freqs).float()
        return torch.cos(freqs), torch.sin(freqs)

    def reshape_for_broadcast(freqs, x):
        ndim = x.ndim
        shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
        return freqs.view(shape)

    def apply_rotary(xq, xk, cos, sin):
        xqr, xqi = xq.float().reshape(*xq.shape[:-1], -1, 2).unbind(-1)
        xkr, xki = xk.float().reshape(*xk.shape[:-1], -1, 2).unbind(-1)
        cos = reshape_for_broadcast(cos, xqr); sin = reshape_for_broadcast(sin, xqr)
        xqo = torch.stack([xqr*cos - xqi*sin, xqr*sin + xqi*cos], dim=-1).flatten(3)
        xko = torch.stack([xkr*cos - xki*sin, xkr*sin + xki*cos], dim=-1).flatten(3)
        return xqo.type_as(xq), xko.type_as(xk)

    class RMSNorm(nn.Module):
        def __init__(self, dim, eps=1e-5):
            super().__init__(); self.eps = eps; self.weight = nn.Parameter(torch.ones(dim))
        def forward(self, x):
            n = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
            return self.weight * n.type_as(x)

    class Attention(nn.Module):
        def __init__(s, c):
            super().__init__()
            s.nh = c.n_heads; s.hd = c.dim // c.n_heads
            s.wq = nn.Linear(c.dim, c.n_heads*s.hd, bias=False)
            s.wk = nn.Linear(c.dim, c.n_kv_heads*s.hd, bias=False)
            s.wv = nn.Linear(c.dim, c.n_kv_heads*s.hd, bias=False)
            s.wo = nn.Linear(c.n_heads*s.hd, c.dim, bias=False)
        def forward(s, x, cos, sin):
            B,T,_ = x.shape
            q = s.wq(x).view(B,T,s.nh,s.hd); k = s.wk(x).view(B,T,s.nh,s.hd); v = s.wv(x).view(B,T,s.nh,s.hd)
            q,k = apply_rotary(q,k,cos,sin)
            q,k,v = q.transpose(1,2),k.transpose(1,2),v.transpose(1,2)
            y = F.scaled_dot_product_attention(q,k,v,is_causal=True)
            y = y.transpose(1,2).contiguous().view(B,T,-1)
            return s.wo(y)

    class FFN(nn.Module):
        def __init__(s, c):
            super().__init__()
            hidden = int(2*4*c.dim/3)
            hidden = c.multiple_of*((hidden+c.multiple_of-1)//c.multiple_of)
            s.hidden = hidden
            s.w1 = nn.Linear(c.dim, hidden, bias=False)
            s.w2 = nn.Linear(hidden, c.dim, bias=False)
            s.w3 = nn.Linear(c.dim, hidden, bias=False)
        def forward(s, x): return s.w2(F.silu(s.w1(x)) * s.w3(x))

    class Block(nn.Module):
        def __init__(s, c):
            super().__init__()
            s.attention_norm = RMSNorm(c.dim); s.attention = Attention(c)
            s.ffn_norm = RMSNorm(c.dim); s.feed_forward = FFN(c)
        def forward(s, x, cos, sin):
            x = x + s.attention(s.attention_norm(x), cos, sin)
            x = x + s.feed_forward(s.ffn_norm(x))
            return x

    class Llama(nn.Module):
        def __init__(s, c):
            super().__init__(); s.c = c
            s.tok_embeddings = nn.Embedding(c.vocab_size, c.dim)
            s.layers = nn.ModuleList([Block(c) for _ in range(c.n_layers)])
            s.norm = RMSNorm(c.dim)
            s.output = nn.Linear(c.dim, c.vocab_size, bias=False)
            s.output.weight = s.tok_embeddings.weight  # shared (matches v0 shared classifier)
            cos, sin = precompute_freqs_cis(c.dim // c.n_heads, c.seq_len)
            s.register_buffer("freqs_cos", cos, persistent=True)
            s.register_buffer("freqs_sin", sin, persistent=True)
            s.apply(s._init)
        def _init(s, m):
            if isinstance(m, nn.Linear): torch.nn.init.normal_(m.weight, 0.0, 0.02)
            elif isinstance(m, nn.Embedding): torch.nn.init.normal_(m.weight, 0.0, 0.02)
        def forward(s, idx, targets=None):
            B,T = idx.shape
            x = s.tok_embeddings(idx)
            cos = s.freqs_cos[:T]; sin = s.freqs_sin[:T]
            for l in s.layers: x = l(x, cos, sin)
            x = s.norm(x)
            if targets is not None:
                logits = s.output(x)
                loss = torch.nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
                return logits, loss
            logits = s.output(x[:, [-1], :])
            return logits, None

    return Llama(Cfg)

# ============================================================================
# training utilities
# ============================================================================
def get_batch(split, device):
    import torch
    path = os.path.join(DATA_DIR, "train.bin" if split == "train" else "val.bin")
    data = np.memmap(path, dtype=np.uint16, mode="r")
    ix = np.random.randint(0, len(data) - Cfg.seq_len - 1, (Cfg.batch_size,))
    x = np.stack([data[i:i+Cfg.seq_len].astype(np.int64) for i in ix])
    y = np.stack([data[i+1:i+1+Cfg.seq_len].astype(np.int64) for i in ix])
    x = torch.from_numpy(x).to(device); y = torch.from_numpy(y).to(device)
    return x, y

def lr_at(it):
    if it < Cfg.warmup: return Cfg.lr * (it + 1) / Cfg.warmup
    if it > Cfg.max_iters: return Cfg.min_lr
    r = (it - Cfg.warmup) / (Cfg.max_iters - Cfg.warmup)
    return Cfg.min_lr + 0.5 * (1 + math.cos(math.pi * r)) * (Cfg.lr - Cfg.min_lr)

@__import__("contextlib").contextmanager
def nullguard():
    yield

def sample_text(model, sp, device, n=60):
    import torch
    model.eval()
    ids = [1]  # BOS
    with torch.no_grad():
        for _ in range(n):
            x = torch.tensor([ids[-Cfg.seq_len:]], dtype=torch.long, device=device)
            logits, _ = model(x)
            logits = logits[0, -1, :] / 0.8
            probs = torch.softmax(logits, dim=-1)
            nxt = int(torch.multinomial(probs, 1))
            if nxt == 1: break
            ids.append(nxt)
    model.train()
    return sp.decode(ids[1:])

def train(resume=False):
    import torch
    model_path = prepare_data()
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor(model_file=model_path)
    os.makedirs(OUT_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[train] device={device}")
    if device == "cpu":
        print("[train] UYARI: GPU yok, CPU'da eğitim çok yavaş olur.")

    model = build_model().to(device)
    nparams = sum(p.numel() for p in model.parameters())
    # subtract shared output (counted once via embedding tie)
    print(f"[train] params ~= {nparams/1e6:.2f}M  (dim={Cfg.dim} layers={Cfg.n_layers} vocab={Cfg.vocab_size})")

    opt = torch.optim.AdamW(model.parameters(), lr=Cfg.lr, weight_decay=Cfg.weight_decay, betas=(0.9, 0.95))
    start_iter = 0; best_val = 1e9
    ckpt_path = os.path.join(OUT_DIR, "ckpt.pt")
    if resume and os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        start_iter = ck["iter"] + 1; best_val = ck.get("best_val", 1e9)
        print(f"[train] resumed from iter {start_iter}")

    t0 = time.time()
    for it in range(start_iter, Cfg.max_iters + 1):
        for g in opt.param_groups: g["lr"] = lr_at(it)

        if it % Cfg.eval_interval == 0:
            model.eval()
            with torch.no_grad():
                losses = []
                for _ in range(Cfg.eval_iters):
                    x, y = get_batch("val", device); _, l = model(x, y); losses.append(l.item())
            vloss = sum(losses)/len(losses)
            model.train()
            dt = time.time() - t0
            print(f"\n[iter {it}] val_loss={vloss:.4f}  lr={lr_at(it):.2e}  elapsed={dt/60:.1f}min")
            try:
                print(f"  ÖRNEK: {sample_text(model, sp, device)}")
            except Exception as e:
                print(f"  (örnek üretilemedi: {e})")
            cfg_dict = {k: v for k, v in vars(Cfg).items() if isinstance(v, (int, float, str, bool))}
            ck = {"model": model.state_dict(), "opt": opt.state_dict(),
                  "iter": it, "best_val": min(best_val, vloss), "cfg": cfg_dict}
            torch.save(ck, ckpt_path)
            best_val = min(best_val, vloss)

        # gradient accumulation step
        opt.zero_grad(set_to_none=True)
        for micro in range(Cfg.grad_accum):
            x, y = get_batch("train", device)
            _, loss = model(x, y)
            (loss / Cfg.grad_accum).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), Cfg.grad_clip)
        opt.step()

        if it % 50 == 0:
            print(f"  iter {it}: loss={loss.item():.4f}", end="\r")

    print("\n[train] done.")
    return model_path

# ============================================================================
# STEP 5: export to PSP files (v0 -> q80, plus tokenizer.bin)
# ============================================================================
def export_v0_bytes(model):
    """Serialize weights in llama2.c v0 (legacy fp32) order. Returns bytes."""
    import io, torch
    buf = io.BytesIO()
    c = Cfg
    buf.write(struct.pack("iiiiiii", c.dim, c.feed_forward_hidden(), c.n_layers,
                          c.n_heads, c.n_kv_heads, c.vocab_size, c.seq_len))
    def w(t): buf.write(t.detach().cpu().float().numpy().astype(np.float32).tobytes())
    w(model.tok_embeddings.weight)
    for l in model.layers: w(l.attention_norm.weight)
    for l in model.layers: w(l.attention.wq.weight)
    for l in model.layers: w(l.attention.wk.weight)
    for l in model.layers: w(l.attention.wv.weight)
    for l in model.layers: w(l.attention.wo.weight)
    for l in model.layers: w(l.ffn_norm.weight)
    for l in model.layers: w(l.feed_forward.w1.weight)
    for l in model.layers: w(l.feed_forward.w2.weight)
    for l in model.layers: w(l.feed_forward.w3.weight)
    w(model.norm.weight)
    w(model.freqs_cos[:c.seq_len])
    w(model.freqs_sin[:c.seq_len])
    # shared classifier -> nothing extra
    return buf.getvalue()

# helper bound to Cfg (hidden dim used in header)
def _hidden():
    h = int(2*4*Cfg.dim/3)
    return Cfg.multiple_of*((h+Cfg.multiple_of-1)//Cfg.multiple_of)
Cfg.feed_forward_hidden = staticmethod(_hidden)

# ---- v0 (fp32) -> v2 (Q8_0) quantization, same layout as runq.c ----
def quantize_q80(w, gs):
    w = w.astype(np.float32).reshape(-1, gs)
    wmax = np.abs(w).max(axis=1); scale = wmax/127.0
    nz = scale > 0; q = np.zeros_like(w)
    q[nz] = np.round(w[nz]/scale[nz, None])
    return np.clip(q, -127, 127).astype(np.int8).reshape(-1), scale.astype(np.float32)

def v0_bytes_to_q80(v0: bytes, out_path: str):
    import io
    f = io.BytesIO(v0)
    dim, hidden, nl, nh, nkv, vocab, seq = struct.unpack("iiiiiii", f.read(28))
    shared = vocab > 0; vocab = abs(vocab); head = dim // nh
    def rd(count):
        a = np.frombuffer(f.read(count*4), dtype=np.float32); assert a.size == count; return a.copy()
    emb = rd(vocab*dim)
    att_norm = rd(nl*dim)
    wq = rd(nl*dim*(nh*head)); wk = rd(nl*dim*(nkv*head)); wv = rd(nl*dim*(nkv*head)); wo = rd(nl*(nh*head)*dim)
    ffn_norm = rd(nl*dim)
    w1 = rd(nl*dim*hidden); w2 = rd(nl*hidden*dim); w3 = rd(nl*dim*hidden)
    final_norm = rd(dim)
    rd(seq*(head//2)); rd(seq*(head//2))  # skip RoPE tables
    out = None if shared else rd(vocab*dim)
    sp = lambda a, per: [a[i*per:(i+1)*per] for i in range(nl)]

    gs = 64
    while dim % gs != 0: gs //= 2
    quant = [emb] + sp(wq, dim*dim) + sp(wk, dim*(nkv*head)) + sp(wv, dim*(nkv*head)) + sp(wo, dim*dim) \
            + sp(w1, dim*hidden) + sp(w2, hidden*dim) + sp(w3, dim*hidden)
    if not shared: quant.append(out)

    with open(out_path, "wb") as g:
        g.write(struct.pack("I", 0x616b3432)); g.write(struct.pack("i", 2))
        g.write(struct.pack("iiiiiii", dim, hidden, nl, nh, nkv, vocab, seq))
        g.write(struct.pack("B", 1 if shared else 0)); g.write(struct.pack("i", gs))
        g.write(b"\0" * (256 - g.tell()))
        for n in sp(att_norm, dim): g.write(n.tobytes())
        for n in sp(ffn_norm, dim): g.write(n.tobytes())
        g.write(final_norm.tobytes())
        for t in quant:
            q, s = quantize_q80(t, gs); g.write(q.tobytes()); g.write(s.tobytes())
    return gs

def export_tokenizer_bin(model_path, out_path):
    """Write the tokenizer in the format PSP build_tokenizer() reads."""
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor(model_file=model_path)
    toks, scores = [], []
    for i in range(sp.get_piece_size()):
        t = sp.id_to_piece(i).replace("\u2581", " ")  # SentencePiece space marker -> ' '
        toks.append(t.encode("utf-8")); scores.append(sp.get_score(i))
    max_len = max(len(b) for b in toks)
    with open(out_path, "wb") as f:
        f.write(struct.pack("i", max_len))
        for b, s in zip(toks, scores):
            f.write(struct.pack("fi", s, len(b))); f.write(b)

def export_all():
    import torch
    ckpt = os.path.join(OUT_DIR, "ckpt.pt")
    if not os.path.exists(ckpt):
        raise SystemExit("out/ckpt.pt yok — önce eğitim yap (python train_turkish.py)")
    model = build_model()
    ck = torch.load(ckpt, map_location="cpu"); model.load_state_dict(ck["model"]); model.eval()
    v0 = export_v0_bytes(model)
    q80_path = os.path.join(OUT_DIR, "model_q80.bin")
    gs = v0_bytes_to_q80(v0, q80_path)
    tok_path = os.path.join(OUT_DIR, "tokenizer.bin")
    export_tokenizer_bin(os.path.join(DATA_DIR, f"tok{Cfg.vocab_size}.model"), tok_path)
    sz = os.path.getsize(q80_path)/1e6
    print(f"\n=== PSP DOSYALARI HAZIR (group_size={gs}) ===")
    print(f"  {q80_path}   ({sz:.1f} MB)")
    print(f"  {tok_path}")
    print("\nBunları PSP'de ms0:/PSP/GAME/llama2q/ klasörüne kopyala:")
    print("  model_q80.bin   ve   tokenizer.bin (mevcut tokenizer.bin'in yerine)")

# ============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["all","data","train","export"], default="all")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--dataset", default=DATASET, help="HuggingFace dataset repo id")
    ap.add_argument("--text", default=None, help="yerel metin dosyasi (her satir bir hikaye)")
    a = ap.parse_args()
    globals()["DATASET"] = a.dataset
    globals()["TEXT_FILE"] = a.text
    if a.stage == "data":   prepare_data()
    elif a.stage == "train": train(resume=a.resume)
    elif a.stage == "export": export_all()
    else:
        train(resume=a.resume)
        export_all()

if __name__ == "__main__":
    main()
