#!/usr/bin/env python3
"""
quantize.py  -  llama2.c v0 (legacy float32) .bin  ->  v2 (Q8_0 int8) .bin

Replicates export.py's version2_export byte layout, but reads from an already
exported v0 float32 .bin (e.g. stories15M.bin) instead of a PyTorch checkpoint.
No torch dependency: numpy only.

Usage:
    python quantize.py stories15M.bin stories15M_q80.bin

The output is readable by the stock runq.c from karpathy/llama2.c.
"""
import sys
import struct
import numpy as np

MAGIC = 0x616b3432  # "ak42"


def read_v0(path):
    with open(path, "rb") as f:
        dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len = \
            struct.unpack("iiiiiii", f.read(28))
        shared = vocab_size > 0
        vocab_size = abs(vocab_size)
        head_size = dim // n_heads

        def rd(count):
            buf = f.read(count * 4)
            arr = np.frombuffer(buf, dtype=np.float32)
            if arr.size != count:
                raise ValueError(f"short read: got {arr.size}, want {count}")
            return arr.copy()

        emb      = rd(vocab_size * dim)
        att_norm = rd(n_layers * dim)
        wq = rd(n_layers * dim * (n_heads * head_size))
        wk = rd(n_layers * dim * (n_kv_heads * head_size))
        wv = rd(n_layers * dim * (n_kv_heads * head_size))
        wo = rd(n_layers * (n_heads * head_size) * dim)
        ffn_norm = rd(n_layers * dim)
        w1 = rd(n_layers * dim * hidden_dim)
        w2 = rd(n_layers * hidden_dim * dim)
        w3 = rd(n_layers * dim * hidden_dim)
        final_norm = rd(dim)
        # skip the precomputed RoPE tables (recomputed on the fly in C)
        rd(seq_len * (head_size // 2))  # freq_cos
        rd(seq_len * (head_size // 2))  # freq_sin
        out = None if shared else rd(vocab_size * dim)

        trailing = f.read()
        if trailing:
            raise ValueError(f"unexpected {len(trailing)} trailing bytes "
                             f"(header/shape mismatch?)")

    def split(arr, per):
        return [arr[i * per:(i + 1) * per] for i in range(n_layers)]

    cfg = dict(dim=dim, hidden_dim=hidden_dim, n_layers=n_layers, n_heads=n_heads,
               n_kv_heads=n_kv_heads, vocab_size=vocab_size, seq_len=seq_len,
               shared=shared)
    w = dict(
        emb=emb, final_norm=final_norm, out=out,
        att_norm=split(att_norm, dim),
        ffn_norm=split(ffn_norm, dim),
        wq=split(wq, dim * (n_heads * head_size)),
        wk=split(wk, dim * (n_kv_heads * head_size)),
        wv=split(wv, dim * (n_kv_heads * head_size)),
        wo=split(wo, (n_heads * head_size) * dim),
        w1=split(w1, dim * hidden_dim),
        w2=split(w2, hidden_dim * dim),
        w3=split(w3, dim * hidden_dim),
    )
    return cfg, w


def quantize_q80(w, gs):
    """Symmetric int8 quantization in groups of gs. Returns (int8 flat, fp32 scales)."""
    w = w.astype(np.float32).reshape(-1, gs)
    wmax = np.abs(w).max(axis=1)
    scale = wmax / 127.0
    nz = scale > 0
    q = np.zeros_like(w)
    q[nz] = np.round(w[nz] / scale[nz, None])
    q = np.clip(q, -127, 127).astype(np.int8).reshape(-1)
    return q, scale.astype(np.float32)


def write_v2(path, cfg, w):
    dim = cfg["dim"]
    gs = 64
    while dim % gs != 0:
        gs //= 2  # back off so group size divides dim (matches export.py)

    quant_order = ([w["emb"]] + w["wq"] + w["wk"] + w["wv"] + w["wo"]
                   + w["w1"] + w["w2"] + w["w3"])
    if not cfg["shared"]:
        quant_order.append(w["out"])
    for t in quant_order:
        if t.size % gs != 0:
            raise ValueError(f"tensor numel {t.size} not divisible by group size {gs}")

    max_err = 0.0
    with open(path, "wb") as f:
        f.write(struct.pack("I", MAGIC))
        f.write(struct.pack("i", 2))
        f.write(struct.pack("iiiiiii", dim, cfg["hidden_dim"], cfg["n_layers"],
                            cfg["n_heads"], cfg["n_kv_heads"], cfg["vocab_size"],
                            cfg["seq_len"]))
        f.write(struct.pack("B", 1 if cfg["shared"] else 0))
        f.write(struct.pack("i", gs))
        pad = 256 - f.tell()
        assert pad >= 0, "header overflow"
        f.write(b"\0" * pad)

        # fp32 norms: all attention norms, all ffn norms, final norm
        for n in w["att_norm"]:
            f.write(n.astype(np.float32).tobytes())
        for n in w["ffn_norm"]:
            f.write(n.astype(np.float32).tobytes())
        f.write(w["final_norm"].astype(np.float32).tobytes())

        # quantized weights
        for t in quant_order:
            q, s = quantize_q80(t, gs)
            # measure reconstruction error for sanity
            deq = (q.astype(np.float32).reshape(-1, gs) * s[:, None]).reshape(-1)
            err = float(np.abs(deq - t.astype(np.float32)).max())
            max_err = max(max_err, err)
            f.write(q.tobytes())
            f.write(s.tobytes())

    return gs, max_err


def main():
    if len(sys.argv) != 3:
        print("usage: python quantize.py <in_v0.bin> <out_q80.bin>")
        sys.exit(1)
    src, dst = sys.argv[1], sys.argv[2]
    cfg, w = read_v0(src)
    gs, max_err = write_v2(dst, cfg, w)
    import os
    in_sz = os.path.getsize(src) / 1e6
    out_sz = os.path.getsize(dst) / 1e6
    print(f"config: dim={cfg['dim']} layers={cfg['n_layers']} heads={cfg['n_heads']} "
          f"kv_heads={cfg['n_kv_heads']} hidden={cfg['hidden_dim']} "
          f"vocab={cfg['vocab_size']} seq_len={cfg['seq_len']} shared={cfg['shared']}")
    print(f"group_size: {gs}")
    print(f"max quantization error: {max_err:.6f}  (should be small, ~0.001-0.01)")
    print(f"size: {in_sz:.1f} MB -> {out_sz:.1f} MB")
    print(f"wrote {dst}")


if __name__ == "__main__":
    main()
