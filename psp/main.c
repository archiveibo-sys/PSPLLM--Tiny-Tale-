/* ============================================================================
 * llama2q.psp  -  Llama-2 INT8 (Q8_0) inference on the Sony PSP
 * ----------------------------------------------------------------------------
 * Port of Andrej Karpathy's llama2.c runq.c (int8 quantized forward) to PSPSDK.
 *
 * Key PSP change vs stock runq.c: we do NOT dequantize the whole token
 * embedding table to float (that would be ~37 MB for a 32k vocab and never
 * fit in PSP RAM). Instead we keep it quantized and dequantize a single row
 * per token, on the fly, in forward().
 *
 * Stage 2 goal: run the int8 15M model on hardware and measure real tok/s.
 *
 * Build:  make            (produces EBOOT.PBP)
 * Place:  EBOOT.PBP + model_q80.bin + tokenizer.bin together in
 *         ms0:/PSP/GAME/tinytale/
 * ==========================================================================*/

#ifdef HOST_CHECK
/* gcc -fsyntax-only -DHOST_CHECK -include psp_stubs.h */
#else
#include <pspkernel.h>
#include <pspdebug.h>
#include <pspdisplay.h>
#include <pspctrl.h>
#include <psppower.h>
#endif

#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <stdint.h>
#include <math.h>
#include <string.h>

PSP_MODULE_INFO("tinytale", 0, 1, 0);
PSP_MAIN_THREAD_ATTR(PSP_THREAD_ATTR_USER | PSP_THREAD_ATTR_VFPU);
PSP_HEAP_SIZE_KB(-1024); /* grab all RAM but 1 MB; 15M int8 needs ~21 MB */

#define ERREXIT(...) do { \
        pspDebugScreenPrintf(__VA_ARGS__); \
        sceKernelDelayThread(5 * 1000 * 1000); \
        sceKernelExitGame(); \
    } while (0)

/* ----------------------------------------------------------------------------
 * Globals
 */
int GS = 0; // group size for weight quantization (read from model header)

/* ----------------------------------------------------------------------------
 * Transformer model
 */
typedef struct {
    int dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len;
} Config;

typedef struct {
    int8_t* q; // quantized values
    float*  s; // scaling factors
} QuantizedTensor;

typedef struct {
    QuantizedTensor *q_tokens; // (vocab_size, dim) — kept quantized; row-dequantized on the fly
    float* rms_att_weight;     // (layer, dim)
    float* rms_ffn_weight;     // (layer, dim)
    QuantizedTensor *wq, *wk, *wv, *wo;
    QuantizedTensor *w1, *w2, *w3;
    float* rms_final_weight;   // (dim,)
    QuantizedTensor *wcls;     // classifier (shared with q_tokens unless separate)
} TransformerWeights;

typedef struct {
    float *x, *xb, *xb2, *hb, *hb2;
    QuantizedTensor xq; // quantized x (dim,)
    QuantizedTensor hq; // quantized hb (hidden_dim,)
    float *q, *k, *v;
    float *att;
    float *logits;
    float* key_cache;
    float* value_cache;
} RunState;

typedef struct {
    Config config;
    TransformerWeights weights;
    RunState state;
    void*  data;       // malloc'd buffer holding the whole checkpoint file
    size_t file_size;
} Transformer;

void malloc_run_state(RunState* s, Config* p) {
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    s->x   = calloc(p->dim, sizeof(float));
    s->xb  = calloc(p->dim, sizeof(float));
    s->xb2 = calloc(p->dim, sizeof(float));
    s->hb  = calloc(p->hidden_dim, sizeof(float));
    s->hb2 = calloc(p->hidden_dim, sizeof(float));
    s->xq.q = calloc(p->dim, sizeof(int8_t));
    s->xq.s = calloc(p->dim, sizeof(float));
    s->hq.q = calloc(p->hidden_dim, sizeof(int8_t));
    s->hq.s = calloc(p->hidden_dim, sizeof(float));
    s->q = calloc(p->dim, sizeof(float));
    s->k = calloc(kv_dim, sizeof(float));
    s->v = calloc(kv_dim, sizeof(float));
    s->att = calloc(p->n_heads * p->seq_len, sizeof(float));
    s->logits = calloc(p->vocab_size, sizeof(float));
    s->key_cache   = calloc(p->n_layers * p->seq_len * kv_dim, sizeof(float));
    s->value_cache = calloc(p->n_layers * p->seq_len * kv_dim, sizeof(float));
    if (!s->x || !s->xb || !s->xb2 || !s->hb || !s->hb2 || !s->q || !s->k || !s->v
     || !s->att || !s->logits || !s->key_cache || !s->value_cache
     || !s->xq.q || !s->xq.s || !s->hq.q || !s->hq.s) {
        ERREXIT("malloc_run_state failed (out of memory)\n");
    }
}

void free_run_state(RunState* s) {
    free(s->x); free(s->xb); free(s->xb2); free(s->hb); free(s->hb2);
    free(s->xq.q); free(s->xq.s); free(s->hq.q); free(s->hq.s);
    free(s->q); free(s->k); free(s->v);
    free(s->att); free(s->logits); free(s->key_cache); free(s->value_cache);
}

/* ----------------------------------------------------------------------------
 * Quantization
 */
void dequantize(QuantizedTensor *qx, float* x, int n) {
    for (int i = 0; i < n; i++) { x[i] = qx->q[i] * qx->s[i / GS]; }
}

void quantize(QuantizedTensor *qx, float* x, int n) {
    int num_groups = n / GS;
    float Q_MAX = 127.0f;
    for (int group = 0; group < num_groups; group++) {
        float wmax = 0.0f;
        for (int i = 0; i < GS; i++) {
            float val = fabsf(x[group * GS + i]);
            if (val > wmax) wmax = val;
        }
        float scale = wmax / Q_MAX;
        qx->s[group] = scale;
        for (int i = 0; i < GS; i++) {
            float quant_value = (scale > 0.0f) ? (x[group * GS + i] / scale) : 0.0f;
            qx->q[group * GS + i] = (int8_t)roundf(quant_value);
        }
    }
}

/* Map n quantized tensors (each size_each elements) out of the file buffer. */
QuantizedTensor *init_quantized_tensors(void **ptr, int n, int size_each) {
    void *p = *ptr;
    QuantizedTensor *res = malloc(n * sizeof(QuantizedTensor));
    for (int i = 0; i < n; i++) {
        res[i].q = (int8_t*)p;
        p = (int8_t*)p + size_each;
        res[i].s = (float*)p;
        p = (float*)p + size_each / GS;
    }
    *ptr = p;
    return res;
}

void memory_map_weights(TransformerWeights *w, Config* p, void* ptr, uint8_t shared_classifier) {
    int head_size = p->dim / p->n_heads;
    float* fptr = (float*) ptr;
    w->rms_att_weight = fptr;   fptr += p->n_layers * p->dim;
    w->rms_ffn_weight = fptr;   fptr += p->n_layers * p->dim;
    w->rms_final_weight = fptr; fptr += p->dim;
    ptr = (void*)fptr;
    w->q_tokens = init_quantized_tensors(&ptr, 1, p->vocab_size * p->dim);
    /* NB: no full-table dequantize here (would be ~37 MB). forward() does one row. */
    w->wq = init_quantized_tensors(&ptr, p->n_layers, p->dim * (p->n_heads * head_size));
    w->wk = init_quantized_tensors(&ptr, p->n_layers, p->dim * (p->n_kv_heads * head_size));
    w->wv = init_quantized_tensors(&ptr, p->n_layers, p->dim * (p->n_kv_heads * head_size));
    w->wo = init_quantized_tensors(&ptr, p->n_layers, (p->n_heads * head_size) * p->dim);
    w->w1 = init_quantized_tensors(&ptr, p->n_layers, p->dim * p->hidden_dim);
    w->w2 = init_quantized_tensors(&ptr, p->n_layers, p->hidden_dim * p->dim);
    w->w3 = init_quantized_tensors(&ptr, p->n_layers, p->dim * p->hidden_dim);
    w->wcls = shared_classifier ? w->q_tokens
                                : init_quantized_tensors(&ptr, 1, p->dim * p->vocab_size);
}

void read_checkpoint(char* checkpoint, Config* config, TransformerWeights* weights,
                     void** data, size_t* file_size) {
    FILE* file = fopen(checkpoint, "rb");
    if (!file) ERREXIT("Couldn't open model:\n%s\n", checkpoint);
    fseek(file, 0, SEEK_END);
    *file_size = (size_t)ftell(file);
    fseek(file, 0, SEEK_SET);
    *data = malloc(*file_size);
    if (!*data) ERREXIT("Model too big for RAM: %u bytes\n", (unsigned)*file_size);
    if (fread(*data, 1, *file_size, file) != *file_size) ERREXIT("Short read on model\n");
    fclose(file);

    /* parse the 256-byte v2 header (use memcpy: group_size sits at an odd offset) */
    char* p = (char*)*data;
    uint32_t magic; memcpy(&magic, p, 4);
    if (magic != 0x616b3432) ERREXIT("Bad magic number (not a v2 q80 file)\n");
    int version; memcpy(&version, p + 4, 4);
    if (version != 2) ERREXIT("Bad version %d, need 2\n", version);
    memcpy(config, p + 8, sizeof(Config));                 /* 7 ints = 28 bytes */
    uint8_t shared_classifier; memcpy(&shared_classifier, p + 36, 1);
    int group_size; memcpy(&group_size, p + 37, 4);
    GS = group_size;

    void* weights_ptr = p + 256; /* weights begin after the fixed 256-byte header */
    memory_map_weights(weights, config, weights_ptr, shared_classifier);
}

void build_transformer(Transformer *t, char* checkpoint_path) {
    read_checkpoint(checkpoint_path, &t->config, &t->weights, &t->data, &t->file_size);
    malloc_run_state(&t->state, &t->config);
}

void free_transformer(Transformer* t) {
    free(t->weights.q_tokens);
    free(t->weights.wq); free(t->weights.wk); free(t->weights.wv); free(t->weights.wo);
    free(t->weights.w1); free(t->weights.w2); free(t->weights.w3);
    if (t->weights.wcls != t->weights.q_tokens) free(t->weights.wcls);
    if (t->data) { free(t->data); t->data = NULL; }
    free_run_state(&t->state);
}

/* ----------------------------------------------------------------------------
 * Neural net blocks
 */
void rmsnorm(float* o, float* x, float* weight, int size) {
    float ss = 0.0f;
    for (int j = 0; j < size; j++) ss += x[j] * x[j];
    ss /= size; ss += 1e-5f; ss = 1.0f / sqrtf(ss);
    for (int j = 0; j < size; j++) o[j] = weight[j] * (ss * x[j]);
}

void softmax(float* x, int size) {
    float max_val = x[0];
    for (int i = 1; i < size; i++) if (x[i] > max_val) max_val = x[i];
    float sum = 0.0f;
    for (int i = 0; i < size; i++) { x[i] = expf(x[i] - max_val); sum += x[i]; }
    for (int i = 0; i < size; i++) x[i] /= sum;
}

/* Quantized matmul: W (d,n) @ x (n,) -> xout (d,). Both operands int8. Hot loop. */
void matmul(float* xout, QuantizedTensor *x, QuantizedTensor *w, int n, int d) {
    for (int i = 0; i < d; i++) {
        float val = 0.0f;
        int32_t ival = 0;
        int in = i * n;
        int j;
        for (j = 0; j <= n - GS; j += GS) {
            for (int k = 0; k < GS; k++) {
                ival += ((int32_t) x->q[j + k]) * ((int32_t) w->q[in + j + k]);
            }
            val += ((float) ival) * w->s[(in + j) / GS] * x->s[j / GS];
            ival = 0;
        }
        xout[i] = val;
    }
}

float* forward(Transformer* transformer, int token, int pos) {
    Config* p = &transformer->config;
    TransformerWeights* w = &transformer->weights;
    RunState* s = &transformer->state;
    float *x = s->x;
    int dim = p->dim;
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    int kv_mul = p->n_heads / p->n_kv_heads;
    int hidden_dim = p->hidden_dim;
    int head_size = dim / p->n_heads;

    /* token embedding: dequantize a single row on the fly (avoids the 37 MB table) */
    {
        int base = token * dim;
        int8_t* tq = w->q_tokens->q + base;
        float*  ts = w->q_tokens->s;
        for (int i = 0; i < dim; i++) {
            x[i] = tq[i] * ts[(base + i) / GS];
        }
    }

    for (int l = 0; l < p->n_layers; l++) {
        rmsnorm(s->xb, x, w->rms_att_weight + l*dim, dim);

        quantize(&s->xq, s->xb, dim);
        matmul(s->q, &s->xq, w->wq + l, dim, dim);
        matmul(s->k, &s->xq, w->wk + l, dim, kv_dim);
        matmul(s->v, &s->xq, w->wv + l, dim, kv_dim);

        for (int i = 0; i < dim; i += 2) {
            int head_dim = i % head_size;
            float freq = 1.0f / powf(10000.0f, head_dim / (float)head_size);
            float val = pos * freq;
            float fcr = cosf(val);
            float fci = sinf(val);
            int rotn = i < kv_dim ? 2 : 1;
            for (int v = 0; v < rotn; v++) {
                float* vec = v == 0 ? s->q : s->k;
                float v0 = vec[i];
                float v1 = vec[i+1];
                vec[i]   = v0 * fcr - v1 * fci;
                vec[i+1] = v0 * fci + v1 * fcr;
            }
        }

        int loff = l * p->seq_len * kv_dim;
        float* key_cache_row = s->key_cache + loff + pos * kv_dim;
        float* value_cache_row = s->value_cache + loff + pos * kv_dim;
        memcpy(key_cache_row, s->k, kv_dim * sizeof(float));
        memcpy(value_cache_row, s->v, kv_dim * sizeof(float));

        for (int h = 0; h < p->n_heads; h++) {
            float* q = s->q + h * head_size;
            float* att = s->att + h * p->seq_len;
            for (int t = 0; t <= pos; t++) {
                float* k = s->key_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
                float score = 0.0f;
                for (int i = 0; i < head_size; i++) score += q[i] * k[i];
                score /= sqrtf(head_size);
                att[t] = score;
            }
            softmax(att, pos + 1);
            float* xb = s->xb + h * head_size;
            memset(xb, 0, head_size * sizeof(float));
            for (int t = 0; t <= pos; t++) {
                float* v = s->value_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
                float a = att[t];
                for (int i = 0; i < head_size; i++) xb[i] += a * v[i];
            }
        }

        quantize(&s->xq, s->xb, dim);
        matmul(s->xb2, &s->xq, w->wo + l, dim, dim);
        for (int i = 0; i < dim; i++) x[i] += s->xb2[i];

        rmsnorm(s->xb, x, w->rms_ffn_weight + l*dim, dim);
        quantize(&s->xq, s->xb, dim);
        matmul(s->hb,  &s->xq, w->w1 + l, dim, hidden_dim);
        matmul(s->hb2, &s->xq, w->w3 + l, dim, hidden_dim);
        for (int i = 0; i < hidden_dim; i++) {
            float val = s->hb[i];
            val *= (1.0f / (1.0f + expf(-val)));
            val *= s->hb2[i];
            s->hb[i] = val;
        }
        quantize(&s->hq, s->hb, hidden_dim);
        matmul(s->xb, &s->hq, w->w2 + l, hidden_dim, dim);
        for (int i = 0; i < dim; i++) x[i] += s->xb[i];
    }

    rmsnorm(x, x, w->rms_final_weight, dim);
    quantize(&s->xq, x, dim);
    matmul(s->logits, &s->xq, w->wcls, dim, p->vocab_size);
    return s->logits;
}

/* ----------------------------------------------------------------------------
 * BPE Tokenizer
 */
typedef struct { char *str; int id; } TokenIndex;
typedef struct {
    char** vocab;
    float* vocab_scores;
    TokenIndex *sorted_vocab;
    int vocab_size;
    unsigned int max_token_length;
    unsigned char byte_pieces[512];
} Tokenizer;

int compare_tokens(const void *a, const void *b) {
    return strcmp(((TokenIndex*)a)->str, ((TokenIndex*)b)->str);
}

void build_tokenizer(Tokenizer* t, char* tokenizer_path, int vocab_size) {
    t->vocab_size = vocab_size;
    t->vocab = (char**)malloc(vocab_size * sizeof(char*));
    t->vocab_scores = (float*)malloc(vocab_size * sizeof(float));
    t->sorted_vocab = NULL;
    for (int i = 0; i < 256; i++) {
        t->byte_pieces[i * 2] = (unsigned char)i;
        t->byte_pieces[i * 2 + 1] = '\0';
    }
    FILE *file = fopen(tokenizer_path, "rb");
    if (!file) ERREXIT("Couldn't open tokenizer:\n%s\n", tokenizer_path);
    if (fread(&t->max_token_length, sizeof(int), 1, file) != 1) ERREXIT("Bad tokenizer\n");
    int len;
    for (int i = 0; i < vocab_size; i++) {
        if (fread(t->vocab_scores + i, sizeof(float), 1, file) != 1) ERREXIT("Bad tokenizer (scores)\n");
        if (fread(&len, sizeof(int), 1, file) != 1) ERREXIT("Bad tokenizer (len)\n");
        t->vocab[i] = (char *)malloc(len + 1);
        if (fread(t->vocab[i], len, 1, file) != 1) ERREXIT("Bad tokenizer (str)\n");
        t->vocab[i][len] = '\0';
    }
    fclose(file);
}

void free_tokenizer(Tokenizer* t) {
    for (int i = 0; i < t->vocab_size; i++) free(t->vocab[i]);
    free(t->vocab); free(t->vocab_scores); free(t->sorted_vocab);
}

char* decode(Tokenizer* t, int prev_token, int token) {
    char *piece = t->vocab[token];
    if (prev_token == 1 && piece[0] == ' ') piece++;
    unsigned char byte_val;
    if (sscanf(piece, "<0x%02hhX>", &byte_val) == 1) {
        piece = (char*)t->byte_pieces + byte_val * 2;
    }
    return piece;
}

/* PSP debug screen has no UTF-8 font; map multi-byte Turkish letters to the
 * nearest ASCII (ğ->g, ş->s, ı->i, İ->I, ç->c, ö->o, ü->u, and uppercase). */
void safe_print(char *piece) {
    if (piece == NULL || piece[0] == '\0') return;
    const unsigned char *p = (const unsigned char *)piece;
    while (*p) {
        unsigned char c = *p;
        if (c < 0x80) {
            if (isprint(c) || isspace(c)) pspDebugScreenPrintf("%c", c);
            p++;
        } else if (c == 0xC3 && p[1]) {        /* ç ö ü Ç Ö Ü */
            char o; unsigned char d = p[1];
            switch (d) {
                case 0xA7: o='c'; break; case 0x87: o='C'; break;
                case 0xB6: o='o'; break; case 0x96: o='O'; break;
                case 0xBC: o='u'; break; case 0x9C: o='U'; break;
                default:   o='?'; break;
            }
            pspDebugScreenPrintf("%c", o); p += 2;
        } else if (c == 0xC4 && p[1]) {        /* ğ Ğ ı İ */
            char o; unsigned char d = p[1];
            switch (d) {
                case 0x9F: o='g'; break; case 0x9E: o='G'; break;
                case 0xB1: o='i'; break; case 0xB0: o='I'; break;
                default:   o='?'; break;
            }
            pspDebugScreenPrintf("%c", o); p += 2;
        } else if (c == 0xC5 && p[1]) {        /* ş Ş */
            char o; unsigned char d = p[1];
            switch (d) {
                case 0x9F: o='s'; break; case 0x9E: o='S'; break;
                default:   o='?'; break;
            }
            pspDebugScreenPrintf("%c", o); p += 2;
        } else {
            p++;  /* skip unknown multi-byte sequence */
        }
    }
}

int str_lookup(char *str, TokenIndex *sorted_vocab, int vocab_size) {
    TokenIndex tok = { .str = str };
    TokenIndex *res = bsearch(&tok, sorted_vocab, vocab_size, sizeof(TokenIndex), compare_tokens);
    return res != NULL ? res->id : -1;
}

void encode(Tokenizer* t, char *text, int8_t bos, int8_t eos, int *tokens, int *n_tokens) {
    if (text == NULL) ERREXIT("cannot encode NULL text\n");
    if (t->sorted_vocab == NULL) {
        t->sorted_vocab = malloc(t->vocab_size * sizeof(TokenIndex));
        for (int i = 0; i < t->vocab_size; i++) {
            t->sorted_vocab[i].str = t->vocab[i];
            t->sorted_vocab[i].id = i;
        }
        qsort(t->sorted_vocab, t->vocab_size, sizeof(TokenIndex), compare_tokens);
    }
    char* str_buffer = malloc((t->max_token_length*2 + 1 + 2) * sizeof(char));
    size_t str_len = 0;
    *n_tokens = 0;
    if (bos) tokens[(*n_tokens)++] = 1;
    if (text[0] != '\0') {
        int dummy_prefix = str_lookup(" ", t->sorted_vocab, t->vocab_size);
        tokens[(*n_tokens)++] = dummy_prefix;
    }
    for (char *c = text; *c != '\0'; c++) {
        if ((*c & 0xC0) != 0x80) str_len = 0;
        str_buffer[str_len++] = *c;
        str_buffer[str_len] = '\0';
        if ((*(c+1) & 0xC0) == 0x80 && str_len < 4) continue;
        int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);
        if (id != -1) {
            tokens[(*n_tokens)++] = id;
        } else {
            for (size_t i = 0; i < str_len; i++)
                tokens[(*n_tokens)++] = (unsigned char)str_buffer[i] + 3;
        }
        str_len = 0;
    }
    while (1) {
        float best_score = -1e10;
        int best_id = -1, best_idx = -1;
        for (int i = 0; i < (*n_tokens - 1); i++) {
            sprintf(str_buffer, "%s%s", t->vocab[tokens[i]], t->vocab[tokens[i+1]]);
            int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);
            if (id != -1 && t->vocab_scores[id] > best_score) {
                best_score = t->vocab_scores[id]; best_id = id; best_idx = i;
            }
        }
        if (best_idx == -1) break;
        tokens[best_idx] = best_id;
        for (int i = best_idx + 1; i < (*n_tokens - 1); i++) tokens[i] = tokens[i+1];
        (*n_tokens)--;
    }
    if (eos) tokens[(*n_tokens)++] = 2;
    free(str_buffer);
}

/* ----------------------------------------------------------------------------
 * Sampler
 */
typedef struct { float prob; int index; } ProbIndex;
typedef struct {
    int vocab_size;
    ProbIndex* probindex;
    float temperature;
    float topp;
    unsigned long long rng_state;
} Sampler;

int sample_argmax(float* p, int n) {
    int max_i = 0; float max_p = p[0];
    for (int i = 1; i < n; i++) if (p[i] > max_p) { max_i = i; max_p = p[i]; }
    return max_i;
}
int sample_mult(float* p, int n, float coin) {
    float cdf = 0.0f;
    for (int i = 0; i < n; i++) { cdf += p[i]; if (coin < cdf) return i; }
    return n - 1;
}
int compare(const void* a, const void* b) {
    ProbIndex* a_ = (ProbIndex*)a; ProbIndex* b_ = (ProbIndex*)b;
    if (a_->prob > b_->prob) return -1;
    if (a_->prob < b_->prob) return 1;
    return 0;
}
int sample_topp(float* p, int n, float topp, ProbIndex* probindex, float coin) {
    int n0 = 0;
    const float cutoff = (1.0f - topp) / (n - 1);
    for (int i = 0; i < n; i++) {
        if (p[i] >= cutoff) { probindex[n0].index = i; probindex[n0].prob = p[i]; n0++; }
    }
    qsort(probindex, n0, sizeof(ProbIndex), compare);
    float cumulative_prob = 0.0f;
    int last_idx = n0 - 1;
    for (int i = 0; i < n0; i++) {
        cumulative_prob += probindex[i].prob;
        if (cumulative_prob > topp) { last_idx = i; break; }
    }
    float r = coin * cumulative_prob;
    float cdf = 0.0f;
    for (int i = 0; i <= last_idx; i++) {
        cdf += probindex[i].prob;
        if (r < cdf) return probindex[i].index;
    }
    return probindex[last_idx].index;
}
void build_sampler(Sampler* s, int vocab_size, float temperature, float topp, unsigned long long rng_seed) {
    s->vocab_size = vocab_size; s->temperature = temperature; s->topp = topp;
    s->rng_state = rng_seed;
    s->probindex = malloc(vocab_size * sizeof(ProbIndex));
}
void free_sampler(Sampler* s) { free(s->probindex); }
unsigned int random_u32(unsigned long long *state) {
    *state ^= *state >> 12; *state ^= *state << 25; *state ^= *state >> 27;
    return (*state * 0x2545F4914F6CDD1Dull) >> 32;
}
float random_f32(unsigned long long *state) { return (random_u32(state) >> 8) / 16777216.0f; }
int sample(Sampler* s, float* logits) {
    int next;
    if (s->temperature == 0.0f) {
        next = sample_argmax(logits, s->vocab_size);
    } else {
        for (int q = 0; q < s->vocab_size; q++) logits[q] /= s->temperature;
        softmax(logits, s->vocab_size);
        float coin = random_f32(&s->rng_state);
        if (s->topp <= 0 || s->topp >= 1) next = sample_mult(logits, s->vocab_size, coin);
        else next = sample_topp(logits, s->vocab_size, s->topp, s->probindex, coin);
    }
    return next;
}

/* ----------------------------------------------------------------------------
 * Timing
 */
long time_in_ms(void) { return (long)(sceKernelGetSystemTimeWide() / 1000); }

/* ----------------------------------------------------------------------------
 * Generation
 */
void generate(Transformer *transformer, Tokenizer *tokenizer, Sampler *sampler,
              char *prompt, int steps) {
    char *empty_prompt = "";
    if (prompt == NULL) prompt = empty_prompt;
    int num_prompt_tokens = 0;
    int* prompt_tokens = (int*)malloc((strlen(prompt) + 3) * sizeof(int));
    encode(tokenizer, prompt, 1, 0, prompt_tokens, &num_prompt_tokens);
    if (num_prompt_tokens < 1) ERREXIT("expected at least 1 prompt token\n");

    long start = 0;
    int next;
    int token = prompt_tokens[0];
    int pos = 0;
    while (pos < steps) {
        float* logits = forward(transformer, token, pos);
        if (pos < num_prompt_tokens - 1) next = prompt_tokens[pos + 1];
        else next = sample(sampler, logits);
        pos++;
        if (next == 1) break;
        char* piece = decode(tokenizer, token, next);
        safe_print(piece);
        token = next;
        if (start == 0) start = time_in_ms();
    }
    pspDebugScreenPrintf("\n");
    if (pos > 1) {
        long end = time_in_ms();
        double tps = (pos - 1) / (double)(end - start) * 1000.0;
        pspDebugScreenPrintf("\n--- achieved tok/s: %.3f ---\n", tps);
    }
    free(prompt_tokens);
}

/* ----------------------------------------------------------------------------
 * PSP plumbing
 */
#ifndef HOST_CHECK
static int exit_callback(int arg1, int arg2, void *common) {
    (void)arg1; (void)arg2; (void)common;
    sceKernelExitGame();
    return 0;
}
static int callback_thread(SceSize args, void *argp) {
    (void)args; (void)argp;
    int cbid = sceKernelCreateCallback("Exit Callback", exit_callback, NULL);
    sceKernelRegisterExitCallback(cbid);
    sceKernelSleepThreadCB();
    return 0;
}
static int setup_callbacks(void) {
    int thid = sceKernelCreateThread("update_thread", callback_thread, 0x11, 0xFA0, 0, 0);
    if (thid >= 0) sceKernelStartThread(thid, 0, 0);
    return thid;
}
#endif

static void wait_for_start(void) {
    SceCtrlData pad;
    sceCtrlSetSamplingCycle(0);
    sceCtrlSetSamplingMode(PSP_CTRL_MODE_DIGITAL);
    while (1) {
        sceCtrlReadBufferPositive(&pad, 1);
        if (pad.Buttons & PSP_CTRL_START) break;
        sceDisplayWaitVblankStart();
    }
}

static void make_path(char* out, size_t outsz, const char* argv0, const char* name) {
    char dir[256];
    strcpy(dir, "ms0:/PSP/GAME/tinytale/");
    if (argv0 && argv0[0]) {
        strncpy(dir, argv0, sizeof(dir) - 1);
        dir[sizeof(dir) - 1] = '\0';
        char* slash = strrchr(dir, '/');
        if (slash) slash[1] = '\0';
    }
    snprintf(out, outsz, "%s%s", dir, name);
}

int main(int argc, char *argv[]) {
    pspDebugScreenInit();
    setup_callbacks();
    scePowerSetClockFrequency(333, 333, 166);

    const char* argv0 = (argc > 0) ? argv[0] : NULL;
    char model_path[300], tok_path[300];
    make_path(model_path, sizeof(model_path), argv0, "model_q80.bin");
    make_path(tok_path,   sizeof(tok_path),   argv0, "tokenizer.bin");

    pspDebugScreenPrintf("llama2q.psp  (int8 / Q8_0)\n");
    pspDebugScreenPrintf("CPU 333 MHz. Loading model...\n");
    pspDebugScreenPrintf("%s\n\n", model_path);

    float temperature = 0.9f;
    float topp        = 0.9f;
    int   steps       = 256;
    char* prompt      = "Bir zamanlar";
    unsigned long long rng_seed = (unsigned long long)sceKernelGetSystemTimeWide();

    Transformer transformer;
    build_transformer(&transformer, model_path);
    if (steps <= 0 || steps > transformer.config.seq_len) steps = transformer.config.seq_len;

    pspDebugScreenPrintf("dim=%d layers=%d vocab=%d GS=%d\n",
                         transformer.config.dim, transformer.config.n_layers,
                         transformer.config.vocab_size, GS);

    Tokenizer tokenizer;
    build_tokenizer(&tokenizer, tok_path, transformer.config.vocab_size);

    Sampler sampler;
    build_sampler(&sampler, transformer.config.vocab_size, temperature, topp, rng_seed);

    pspDebugScreenPrintf("--- generating ---\n");
    generate(&transformer, &tokenizer, &sampler, prompt, steps);

    free_sampler(&sampler);
    free_tokenizer(&tokenizer);
    free_transformer(&transformer);

    pspDebugScreenPrintf("\nDone. Press START to exit.\n");
    wait_for_start();
    sceKernelExitGame();
    return 0;
}
