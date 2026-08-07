// ggml_delta_net's per-head forget gate, checked against the recurrence written
// out longhand.
//
// The gate and beta arrive HEAD-fastest: build_fused_delta_net permutes them
// without ggml_cont, so the pointer the kernel receives is the pre-permute
// buffer and successive tokens are n_heads apart. iqk_fused_delta_net reads it
// that way; the portable path in ggml.c read it token-fastest, which is the
// transpose, and produced silently wrong results.
//
// HEAD_DIM selects which implementation runs, and both matter:
// iqk_fused_delta_net only accepts 64 and 128, so HEAD_DIM=8 is the ONLY way to
// reach the portable path. A suite that ran at 128 alone would not have caught
// this.
//
//   ctest -R delta-net-gate
#include "ggml.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef HEAD_DIM
#define HEAD_DIM 128
#endif

#define S HEAD_DIM
#define H 3
#define T 5
#define NS 1

static float frand(unsigned * st) {
    *st = *st * 1103515245u + 12345u;
    return (float)((*st >> 9) & 0xFFFF) / 32768.0f - 1.0f;
}

// Contiguous layouts, read right to left from the ggml_new_tensor_4d shapes.
#define QKV_AT(h, t, s)   ((s) + (t)*S + (h)*S*T)   // [S, T, H, NS]
#define STATE_AT(h, c, r) ((r) + (c)*S + (h)*S*S)   // [S, S*H, 1, NS]
#define OUT_AT(h, t, s)   ((s) + (h)*S + (t)*S*H)   // NOT the input layout
#define GATE_AT(h, t)     ((h) + (t)*H)             // HEAD-fastest: the point of this test

static float * run(const float * g, const float * beta, const float * q, const float * k,
                   const float * v, const float * state) {
    struct ggml_init_params ip = { 512u*1024*1024, NULL, false };
    struct ggml_context * ctx = ggml_init(ip);

    struct ggml_tensor * tq = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, T, H, NS);
    struct ggml_tensor * tk = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, T, H, NS);
    struct ggml_tensor * tv = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, T, H, NS);
    struct ggml_tensor * tb = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, T, H, NS);
    struct ggml_tensor * ts = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, S*H, 1, NS);
    struct ggml_tensor * tg = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, T, 1, H, NS);

    memcpy(tq->data, q, ggml_nbytes(tq));
    memcpy(tk->data, k, ggml_nbytes(tk));
    memcpy(tv->data, v, ggml_nbytes(tv));
    memcpy(tb->data, beta, ggml_nbytes(tb));
    memcpy(ts->data, state, ggml_nbytes(ts));
    memcpy(tg->data, g, ggml_nbytes(tg));

    struct ggml_tensor * r = ggml_delta_net(ctx, tq, tk, tv, tg, tb, ts, NULL);
    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, r);
    ggml_graph_compute_with_ctx(ctx, gf, 1);

    const int n = S*H*T*NS;
    float * out = (float *) malloc(n * sizeof(float));
    memcpy(out, r->data, n * sizeof(float));
    ggml_free(ctx);
    return out;
}

// The gated delta rule, longhand.
static float * reference(const float * g, const float * beta, const float * q, const float * k,
                         const float * v, const float * state_in) {
    const float scale = 1.0f / sqrtf((float) S);
    float * out  = (float *) malloc(S*H*T*NS * sizeof(float));
    float * st   = (float *) malloc(S*S*H * sizeof(float));
    float * vnew = (float *) malloc(S * sizeof(float));
    memcpy(st, state_in, S*S*H * sizeof(float));

    for (int h = 0; h < H; ++h) {
        for (int t = 0; t < T; ++t) {
            const float * q_t = q + QKV_AT(h, t, 0);
            const float * k_t = k + QKV_AT(h, t, 0);
            const float * v_t = v + QKV_AT(h, t, 0);

            const float b_val = 1.0f / (1.0f + expf(-beta[GATE_AT(h, t)]));
            const float d     = expf(fminf(g[GATE_AT(h, t)], 50.0f));

            float kq = 0.0f;
            for (int i = 0; i < S; ++i) kq += k_t[i] * q_t[i];
            const float attn = kq * scale;

            for (int c = 0; c < S; ++c)
                for (int r = 0; r < S; ++r) st[STATE_AT(h, c, r)] *= d;

            for (int r = 0; r < S; ++r) {
                float vp = 0.0f, ov = 0.0f;
                for (int c = 0; c < S; ++c) {
                    const float s = st[STATE_AT(h, c, r)];
                    vp += s * k_t[c];
                    ov += s * q_t[c];
                }
                vnew[r] = v_t[r] * b_val - vp * b_val;
                out[OUT_AT(h, t, r)] = ov * scale + vnew[r] * attn;
            }

            for (int c = 0; c < S; ++c)
                for (int r = 0; r < S; ++r) {
                    float s = st[STATE_AT(h, c, r)] + vnew[r] * k_t[c];
                    st[STATE_AT(h, c, r)] = fminf(fmaxf(s, -1e6f), 1e6f);
                }
        }
    }
    free(st); free(vnew);
    return out;
}

int main(void) {
    unsigned st = 12345;
    static float q[S*T*H*NS], k[S*T*H*NS], v[S*T*H*NS], state[S*S*H*NS];
    static float g[T*H*NS], beta[T*H*NS];

    for (int i = 0; i < S*T*H*NS; ++i) { q[i] = frand(&st); k[i] = frand(&st); v[i] = frand(&st); }
    for (int i = 0; i < S*S*H*NS; ++i) state[i] = frand(&st) * 0.1f;

    // Vary the gate per (head, token). A constant gate would pass even with the
    // indexing transposed, which is exactly how this went unnoticed.
    for (int h = 0; h < H; ++h)
        for (int t = 0; t < T; ++t) {
            g[GATE_AT(h, t)]    = -0.05f * (float)((h*7 + t*11) % 40);
            beta[GATE_AT(h, t)] = frand(&st);
        }

    // l2-normalise q and k per (head, token): build_qkv does this in the graph,
    // and the two implementations differ on whether they redo it internally.
    for (int h = 0; h < H; ++h) for (int t = 0; t < T; ++t) {
        double nq = 0.0, nk = 0.0;
        for (int i = 0; i < S; ++i) { nq += q[QKV_AT(h,t,i)]*q[QKV_AT(h,t,i)];
                                      nk += k[QKV_AT(h,t,i)]*k[QKV_AT(h,t,i)]; }
        const float iq = 1.0f/sqrtf((float)nq + 1e-12f), ik = 1.0f/sqrtf((float)nk + 1e-12f);
        for (int i = 0; i < S; ++i) { q[QKV_AT(h,t,i)] *= iq; k[QKV_AT(h,t,i)] *= ik; }
    }

    const int fused = (S == 64 || S == 128);
    printf("head_dim %d -> %s path\n", S, fused ? "fused (iqk_fused_delta_net)" : "portable (ggml.c)");

    float * a = run(g, beta, q, k, v, state);
    float * b = reference(g, beta, q, k, v, state);

    double worst = 0.0;
    for (int i = 0; i < S*H*T*NS; ++i) {
        double d = fabs((double)a[i] - (double)b[i]);
        if (d > worst) worst = d;
    }
    const int ok = worst < 1e-4;
    printf("  max|kernel - reference| = %.3e   %s\n", worst, ok ? "ok" : "MISMATCH");
    free(a); free(b);

    printf("\n%s\n", ok ? "PASSED" : "FAILED: per-head gate indexing does not match the recurrence");
    return ok ? 0 : 1;
}
