// Self-validating test of ggml_delta_net's per-channel forget gate (Kimi-K3 KDA).
//
// Two independent checks, because either alone misses a whole class of bug:
//
//   A. REDUCTION. A per-channel gate whose channels all hold the same value is
//      mathematically identical to a per-head gate holding that value, so the
//      two must agree. Catches errors in the per-channel MATH - a decay applied
//      twice, applied to the wrong side of the state update, or left out of one
//      of the two sums.
//
//      It cannot catch an indexing error: with every channel equal, a
//      transposed or mis-strided read still lands on the same number. That is
//      what B is for.
//
//   B. LAYOUT. A gate that varies per (token, channel, head), checked against
//      the recurrence written out longhand below. The test builds g as a
//      [n_tokens, head_dim, n_heads, n_seqs] tensor and lets ggml lay it out,
//      so the reference's indexing is derived from the tensor shape rather than
//      from the kernel's pointer arithmetic - a transpose in the kernel shows up
//      as a mismatch.
//
// HEAD_DIM selects which implementation runs, and both are worth testing:
// iqk_fused_delta_net only accepts 64 and 128, so HEAD_DIM=8 exercises the
// portable scalar path in ggml.c and HEAD_DIM=128 the fused AVX-512 one.
//
// Runs in milliseconds; no model load.
//
//   cmake --build build --target test-delta-net-gate-8 test-delta-net-gate-64 \
//       test-delta-net-gate-128
//   ctest --test-dir build -R delta-net-gate --output-on-failure
#include "ggml.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef HEAD_DIM
#define HEAD_DIM 128
#endif

#define S HEAD_DIM   // head dim  (S_k == S_v)
#define H 3          // heads
#define T 5          // tokens
#define NS 1         // sequences

static float frand(unsigned * st) {
    *st = *st * 1103515245u + 12345u;
    return (float)((*st >> 9) & 0xFFFF) / 32768.0f - 1.0f;
}

// Index helpers, one per tensor, matching the ggml_new_tensor_4d shapes below.
// Everything is contiguous, so these are just the shapes read right to left.
#define QKV_AT(h, t, s)   ((s) + (t)*S + (h)*S*T)        // [S, T, H, NS]
#define BETA_AT(h, t)     ((t) + (h)*T)                  // [1, T, H, NS]
#define G_AT(h, c, t, w)  ((t) + (c)*T + (h)*T*(w))      // [T, w, H, NS]
#define STATE_AT(h, c, r) ((r) + (c)*S + (h)*S*S)        // [S, S*H, 1, NS]
#define OUT_AT(h, t, s)   ((s) + (h)*S + (t)*S*H)        // note: NOT the input layout

// Run delta_net once through ggml. g is supplied already laid out for `w`
// channels per (head, token): w == 1 is a per-head gate, w == S a per-channel
// one. Returns a malloc'd copy of the output region.
static float * run(int w, const float * g, const float * q, const float * k,
                   const float * v, const float * beta, const float * state) {
    struct ggml_init_params ip = { 512u*1024*1024, NULL, false };
    struct ggml_context * ctx = ggml_init(ip);

    struct ggml_tensor * tq = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, T, H, NS);
    struct ggml_tensor * tk = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, T, H, NS);
    struct ggml_tensor * tv = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, T, H, NS);
    struct ggml_tensor * tb = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, T, H, NS);
    struct ggml_tensor * ts = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, S*H, 1, NS);
    struct ggml_tensor * tg = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, T, w, H, NS);

    memcpy(tq->data, q,     ggml_nbytes(tq));
    memcpy(tk->data, k,     ggml_nbytes(tk));
    memcpy(tv->data, v,     ggml_nbytes(tv));
    memcpy(tb->data, beta,  ggml_nbytes(tb));
    memcpy(ts->data, state, ggml_nbytes(ts));
    memcpy(tg->data, g,     ggml_nbytes(tg));

    struct ggml_tensor * r = ggml_delta_net(ctx, tq, tk, tv, tg, tb, ts, NULL);
    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, r);
    ggml_graph_compute_with_ctx(ctx, gf, 1);

    const int n = S*H*T*NS;   // the output region, before the trailing state
    float * out = (float *) malloc(n * sizeof(float));
    memcpy(out, r->data, n * sizeof(float));
    ggml_free(ctx);
    return out;
}

// The gated delta rule, written out longhand, for a per-channel gate. Decay is
// indexed by COLUMN - the key dimension - and is applied to the state before it
// is read, which is the whole reason a per-channel gate needs its own code path:
// a scalar decay would factor out of both sums below and could be folded into
// the outputs instead.
static float * reference(const float * g, const float * q, const float * k,
                         const float * v, const float * beta, const float * state_in) {
    const float scale = 1.0f / sqrtf((float) S);
    float * out   = (float *) malloc(S*H*T*NS * sizeof(float));
    float * st    = (float *) malloc(S*S*H * sizeof(float));
    float * vnew  = (float *) malloc(S * sizeof(float));
    memcpy(st, state_in, S*S*H * sizeof(float));

    for (int h = 0; h < H; ++h) {
        for (int t = 0; t < T; ++t) {
            const float * q_t = q + QKV_AT(h, t, 0);
            const float * k_t = k + QKV_AT(h, t, 0);
            const float * v_t = v + QKV_AT(h, t, 0);

            const float b_val = 1.0f / (1.0f + expf(-beta[BETA_AT(h, t)]));

            float kq = 0.0f;
            for (int i = 0; i < S; ++i) kq += k_t[i] * q_t[i];
            const float attn = kq * scale;

            for (int c = 0; c < S; ++c) {
                const float d = expf(fminf(g[G_AT(h, c, t, S)], 50.0f));
                for (int r = 0; r < S; ++r) st[STATE_AT(h, c, r)] *= d;
            }

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

            for (int c = 0; c < S; ++c) {
                for (int r = 0; r < S; ++r) {
                    float s = st[STATE_AT(h, c, r)] + vnew[r] * k_t[c];
                    st[STATE_AT(h, c, r)] = fminf(fmaxf(s, -1e6f), 1e6f);
                }
            }
        }
    }
    free(st); free(vnew);
    return out;
}

static double max_abs_diff(const float * a, const float * b, int n) {
    double worst = 0.0;
    for (int i = 0; i < n; ++i) {
        double d = fabs((double)a[i] - (double)b[i]);
        if (d > worst) worst = d;
    }
    return worst;
}

int main(void) {
    unsigned st = 12345;
    static float q[S*T*H*NS], k[S*T*H*NS], v[S*T*H*NS], beta[T*H*NS], state[S*S*H*NS];
    static float g_head[T*1*H*NS], g_chan[T*S*H*NS];

    for (int i = 0; i < S*T*H*NS; ++i) { q[i] = frand(&st); k[i] = frand(&st); v[i] = frand(&st); }
    for (int i = 0; i < T*H*NS; ++i)   beta[i]  = frand(&st);
    for (int i = 0; i < S*S*H*NS; ++i) state[i] = frand(&st) * 0.1f;

    // L2-normalise q and k per (head, token). The graph does this via
    // build_qkv's ggml_l2_norm before delta_net ever sees them, and the two
    // implementations differ on whether they redo it internally - so feeding
    // unnormalised vectors would compare paths that are not supposed to agree.
    for (int h = 0; h < H; ++h) for (int t = 0; t < T; ++t) {
        double nq = 0.0, nk = 0.0;
        for (int i = 0; i < S; ++i) { nq += q[QKV_AT(h,t,i)]*q[QKV_AT(h,t,i)];
                                      nk += k[QKV_AT(h,t,i)]*k[QKV_AT(h,t,i)]; }
        const float iq = 1.0f/sqrtf((float)nq + 1e-12f), ik = 1.0f/sqrtf((float)nk + 1e-12f);
        for (int i = 0; i < S; ++i) { q[QKV_AT(h,t,i)] *= iq; k[QKV_AT(h,t,i)] *= ik; }
    }

    const int n = S*H*T*NS;
    int rc = 0;
    const int fused = (S == 64 || S == 128);   // what iqk_fused_delta_net accepts
    printf("head_dim %d -> %s path\n\n", S,
           fused ? "fused (iqk_fused_delta_net)" : "portable scalar (ggml.c)");

    // A. per-channel with all channels equal must reduce to per-head.
    //
    // A PER-HEAD gate and beta are head-fastest, on BOTH paths. That is not
    // arbitrary: build_fused_delta_net permutes them without ggml_cont, so the
    // pointer both kernels receive is the pre-permute buffer, and head-fastest
    // is what is actually in it. ggml.c's portable path used to read it
    // token-fastest - a silent transpose that only a non-x86 build or an
    // unusual head_dim could reach, and this test is what found it.
    //
    // A per-channel gate IS cont'd, and is token-fastest on both paths, which
    // is why only the per-head case needs laying out here.
    static float beta_ph[T*H*NS];
    for (int h = 0; h < H; ++h) for (int t = 0; t < T; ++t) {
        beta_ph[t*H + h] = beta[BETA_AT(h, t)];
    }

    printf("A. reduction to the per-head case\n");
    const float gvals[] = { -0.25f, -1.0f, -3.0f };
    for (int i = 0; i < 3; ++i) {
        const float gv = gvals[i];
        for (int j = 0; j < T*1*H*NS; ++j) g_head[j] = gv;
        for (int j = 0; j < T*S*H*NS; ++j) g_chan[j] = gv;

        float * a = run(1, g_head, q, k, v, beta_ph, state);
        float * b = run(S, g_chan, q, k, v, beta,    state);
        const double worst = max_abs_diff(a, b, n);
        const int ok = worst < 1e-4;
        printf("   g = %-6.2f  max|per_head - per_channel| = %.3e   %s\n",
               gv, worst, ok ? "ok" : "MISMATCH");
        if (!ok) rc = 1;
        free(a); free(b);
    }

    // B. a gate that actually varies, against the longhand recurrence.
    printf("\nB. varying gate vs reference recurrence\n");
    for (int h = 0; h < H; ++h) for (int c = 0; c < S; ++c) for (int t = 0; t < T; ++t) {
        // Spread over roughly [-4, 0): decays from ~0.02 to ~1, and distinct
        // enough per (h, c, t) that any two indices swapped is visible.
        g_chan[G_AT(h, c, t, S)] = -0.05f * (float)((h*7 + c*3 + t*11) % 80);
    }
    {
        float * a = run(S, g_chan, q, k, v, beta, state);
        float * b = reference(g_chan, q, k, v, beta, state);
        const double worst = max_abs_diff(a, b, n);
        const int ok = worst < 1e-4;
        printf("   max|kernel - reference| = %.3e   %s\n", worst, ok ? "ok" : "MISMATCH");
        if (!ok) rc = 1;
        free(a); free(b);
    }

    printf("\n%s\n", rc ? "FAILED: the per-channel gate path is wrong"
                        : "PASSED: per-channel gate reduces correctly AND matches the recurrence");
    return rc;
}
