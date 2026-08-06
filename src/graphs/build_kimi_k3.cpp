#include "../llama-build-context.h"
#include "../llama-model.h"
#include "../llama-context.h"
#include "../llama-delta-net.h"

//
// Kimi-K3.
//
// Hybrid KDA (linear) + MLA (full) attention, plus four things no other
// architecture here has:
//
//   1. cross-layer residual attention (AttnRes) - replaces the residual add
//   2. latent MoE                     - routed experts run at a narrower width
//   3. situ activation                - replaces SwiGLU (see LLM_FFN_SITU)
//   4. MLA output gate                - sigmoid gate before o_proj
//
// See porting/k3/GRAPH-BUILDER-SPEC.md in the glm-cpu-kit repo for the derivation
// and for the specific ways each of these fails SILENTLY when written wrong.
//

//
// ---- cross-layer residual attention -----------------------------------------
//
// Rather than h += f(h) at every layer, each site takes a softmax-weighted
// average over previously banked residual checkpoints plus the live stream.
//
// NOTE: this deliberately does NOT use ggml_hc_pre. ik's hc_pre is DeepSeek-V4's
// Sinkhorn-normalised hyper-connection op (scale[3], bias[S*S+2S], n_iters) and
// is unrelated to mainline's ggml_dsv4_hc_pre, which is a plain weighted sum
// over ne1. They share a name fragment and nothing else. With attn_res_block_size
// = 12 over 93 layers there are at most 8 checkpoints, so accumulating them
// explicitly costs n_embd*8*n_tokens FLOPs - nothing beside one MoE layer - and
// is obviously correct.
//
struct kimi_k3_res {
    std::vector<ggml_tensor *> bank;   // each [n_embd, 1, n_tokens]

    void push(ggml_context * ctx, ggml_tensor * cur, int64_t n_embd, int64_t n_tokens) {
        bank.push_back(ggml_reshape_3d(ctx, cur, n_embd, 1, n_tokens));
    }

    // score(x) = sum_rows(rms_norm(x) * w)   -> [1, 1, n_tokens]
    //
    // The RMSNorm carries NO weight of its own: score_w multiplies inside the
    // scoring path only, and the weighted sum below consumes the RAW tensors.
    // Normalising the output instead is the obvious wrong turn.
    static ggml_tensor * score(ggml_context * ctx, ggml_tensor * x, ggml_tensor * w, float eps) {
        ggml_tensor * s = ggml_rms_norm(ctx, x, eps);
        s = ggml_mul(ctx, s, w);
        return ggml_sum_rows(ctx, s);
    }

    ggml_tensor * mix(ggml_context * ctx, ggml_tensor * cur, ggml_tensor * score_w,
                      int64_t n_embd, int64_t n_tokens, float eps,
                      const llm_build_cb & cb, int il) {
        const int n_ckpt = (int) bank.size();
        if (n_ckpt == 0) {
            return cur;   // layer 0: nothing banked yet
        }

        // Scores for the banked checkpoints, then the live stream. The live one
        // is scored but never banked, which keeps the bank append-only.
        ggml_tensor * scores = nullptr;
        for (int c = 0; c < n_ckpt; ++c) {
            ggml_tensor * s = score(ctx, bank[c], score_w, eps);           // [1,1,n_tokens]
            s = ggml_reshape_2d(ctx, s, 1, n_tokens);
            scores = scores ? ggml_concat(ctx, scores, s, 0) : s;
        }
        ggml_tensor * s_cur = score(ctx, ggml_reshape_3d(ctx, cur, n_embd, 1, n_tokens), score_w, eps);
        s_cur  = ggml_reshape_2d(ctx, s_cur, 1, n_tokens);
        scores = ggml_concat(ctx, scores, s_cur, 0);                        // [n_ckpt+1, n_tokens]

        ggml_tensor * probs = ggml_soft_max(ctx, scores);                   // over ne0
        cb(probs, "res_probs", il);

        // Weighted sum over the RAW (un-normalised) tensors.
        ggml_tensor * out = nullptr;
        for (int c = 0; c < n_ckpt; ++c) {
            ggml_tensor * p = ggml_cont(ctx, ggml_view_2d(ctx, probs, 1, n_tokens,
                                                          probs->nb[1], (size_t) c * probs->nb[0]));
            p = ggml_reshape_3d(ctx, p, 1, 1, n_tokens);                    // broadcasts over n_embd
            ggml_tensor * term = ggml_mul(ctx, bank[c], p);
            out = out ? ggml_add(ctx, out, term) : term;
        }
        ggml_tensor * p_cur = ggml_cont(ctx, ggml_view_2d(ctx, probs, 1, n_tokens,
                                                          probs->nb[1], (size_t) n_ckpt * probs->nb[0]));
        p_cur = ggml_reshape_3d(ctx, p_cur, 1, 1, n_tokens);
        out = ggml_add(ctx, out, ggml_mul(ctx, ggml_reshape_3d(ctx, cur, n_embd, 1, n_tokens), p_cur));

        return ggml_reshape_2d(ctx, out, n_embd, n_tokens);
    }
};

ggml_cgraph * llm_build_context::build_kimi_k3() {

    ggml_cgraph * gf = new_graph_custom();

    delta_net delta(lctx, batch);

    ggml_tensor * inpL = llm_build_inp_embd(ctx0, lctx, hparams, batch, model.tok_embd, cb);
    ggml_tensor * inp_out_ids = n_tokens > 1 ? build_inp_out_ids() : nullptr;
    ggml_tensor * KQ_mask = build_inp_KQ_mask();

    lctx.inp_s_seq_qnext = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, 1, n_tokens);
    cb(lctx.inp_s_seq_qnext, "inp_s_seq_qnext", -1);
    ggml_set_input(lctx.inp_s_seq_qnext);

    const int64_t n_embd_full = hparams.n_embd;
    const float   eps         = hparams.f_norm_rms_eps;

    const uint32_t res_bs       = hparams.attn_res_block_size;
    const bool     use_attn_res = res_bs > 0;

    const int64_t n_latent = hparams.n_expert_latent > 0 ? hparams.n_expert_latent : n_embd_full;

    // MLA widths come from the dedicated *_mla hparams, NOT n_embd_head_k(0):
    // for K3 that is 576, the compressed MQA width (kv_lora 512 + rope 64),
    // rather than the 192/128 the q_b and v_b projections actually use.
    const int64_t n_embd_head_k_mla = hparams.n_embd_head_k_mla ? hparams.n_embd_head_k_mla : n_embd_head_k;
    const float   kq_scale_mla      = 1.0f / sqrtf((float) n_embd_head_k_mla);

    kimi_k3_res res;

    ggml_tensor * cur = nullptr;

    for (int il = 0; il < n_layer; ++il) {
        const auto & layer = model.layers[il];

        // `prefix_sum` is the running residual stream.
        ggml_tensor * prefix_sum = inpL;

        cur = use_attn_res
            ? res.mix(ctx0, prefix_sum, layer.attn_res_score, n_embd_full, n_tokens, eps, cb, il)
            : prefix_sum;

        // Bank the RAW layer input, not the mixed `cur`.
        bool banked = false;
        if (use_attn_res && (uint32_t) il % res_bs == 0) {
            res.push(ctx0, prefix_sum, n_embd_full, n_tokens);
            banked = true;
        }

        cur = llm_build_norm(ctx0, cur, hparams, layer.attn_norm, nullptr, LLM_NORM_RMS, cb, il);
        cb(cur, "attn_norm", il);

        if (hparams.is_recurrent(il)) {
            cur = build_kimi_k3_kda(gf, cur, il == n_layer - 1 ? inp_out_ids : nullptr, il);
        } else {
            cur = build_kimi_k3_mla(gf, cur, KQ_mask, il == n_layer - 1 ? inp_out_ids : nullptr,
                                    kq_scale_mla, il);
        }

        // THE subtle line. On a banking layer the running residual RESTARTS from
        // the attention output alone, because the stream it would have added to
        // has just been checkpointed. Writing the usual unconditional add here
        // yields a model that runs and generates plausible, slightly-worse text.
        prefix_sum = banked ? cur : ggml_add(ctx0, prefix_sum, cur);
        cb(prefix_sum, "prefix_sum_attn", il);

        cur = use_attn_res
            ? res.mix(ctx0, prefix_sum, layer.ffn_res_score, n_embd_full, n_tokens, eps, cb, il)
            : prefix_sum;

        if (il < (int) hparams.n_layer_dense_lead) {
            cur = llm_build_ffn(ctx0, lctx, layer.ffn_norm, cur,
                    layer.ffn_up,   nullptr, nullptr,
                    layer.ffn_gate, nullptr, nullptr,
                    layer.ffn_down, nullptr, nullptr,
                    nullptr,
                    LLM_FFN_SITU, LLM_FFN_PAR, cb, il, gf, true);
            cb(cur, "ffn_out", il);
        } else {
            cur = build_kimi_k3_latent_moe(gf, cur, layer, n_latent, eps, il);
        }

        prefix_sum = ggml_add(ctx0, prefix_sum, cur);
        prefix_sum = lctx.cvec.apply_to(ctx0, prefix_sum, il);
        cb(prefix_sum, "l_out", il);

        inpL = prefix_sum;
    }

    cur = inpL;

    // One more mix at the output site before the final norm.
    if (use_attn_res) {
        cur = res.mix(ctx0, cur, model.output_res_score, n_embd_full, n_tokens, eps, cb, -1);
    }

    cur = build_output(lctx, ctx0, cur, model.output, model.output_norm, cb);
    cb(cur, "result_output", -1);

    ggml_build_forward_expand(gf, cur);

    return gf;
}

//
// ---- latent MoE --------------------------------------------------------------
//
// Three widths in one block. The router and the shared experts see the FULL
// residual width; the routed experts live at n_expert_latent, with a
// down-projection into that space and an up-projection back out. The shared
// experts consume the block INPUT, not the down-projected tensor.
//
ggml_tensor * llm_build_context::build_kimi_k3_latent_moe(ggml_cgraph * gf, ggml_tensor * cur,
        const llama_layer & layer, int64_t n_latent, float eps, int il) {

    ggml_tensor * inp = llm_build_norm(ctx0, cur, hparams, layer.ffn_norm, nullptr, LLM_NORM_RMS, cb, il);
    cb(inp, "ffn_norm", il);

    // routed path: 7168 -> 3584, norm, experts, 3584 -> 7168
    ggml_tensor * h = ggml_mul_mat(ctx0, layer.ffn_routed_down, inp);
    cb(h, "ffn_routed_down", il);
    h = llm_build_norm(ctx0, h, hparams, layer.ffn_routed_norm, nullptr, LLM_NORM_RMS, cb, il);
    cb(h, "ffn_routed_norm", il);

    // The router scores the FULL residual width, not the down-projected tensor -
    // ffn_gate_inp is [n_embd, n_expert]. So the logits are computed here from
    // `inp` and handed in, with `h` supplying only the expert input. Passing `h`
    // as both is a shape error at 7168 vs 3584, which is the loud version of this
    // mistake; a model where the two happened to match would just route badly.
    ggml_tensor * logits = ggml_mul_mat(ctx0, layer.ffn_gate_inp, inp);
    cb(logits, "ffn_moe_logits", il);

    ggml_tensor * moe = llm_build_moe_ffn(ctx0, lctx, h,
            nullptr,
            layer.ffn_up_exps,
            layer.ffn_gate_exps,
            layer.ffn_down_exps,
            layer.ffn_exp_probs_b,
            n_expert, n_expert_used,
            LLM_FFN_SITU,
            hparams.expert_weights_norm,
            true, hparams.expert_weights_scale,
            (llm_expert_gating_func_type) hparams.expert_gating_func,
            cb, il, gf, false, nullptr, nullptr, logits);
    cb(moe, "ffn_moe_out", il);

    moe = ggml_mul_mat(ctx0, layer.ffn_routed_up, moe);
    cb(moe, "ffn_routed_up", il);

    // shared experts, at the FULL width, on the block input
    ggml_tensor * shexp = llm_build_ffn(ctx0, lctx, nullptr, inp,
            layer.ffn_up_shexp,   nullptr, nullptr,
            layer.ffn_gate_shexp, nullptr, nullptr,
            layer.ffn_down_shexp, nullptr, nullptr,
            nullptr,
            LLM_FFN_SITU, LLM_FFN_PAR, cb, il, gf, false);
    cb(shexp, "ffn_shexp", il);

    ggml_tensor * out = ggml_add(ctx0, moe, shexp);
    cb(out, "ffn_out", il);
    GGML_UNUSED(n_latent);
    return out;
}

//
// ---- attention layers --------------------------------------------------------
//
// NOT YET IMPLEMENTED. Everything above - AttnRes, the latent MoE, situ, the
// residual/banking structure - is complete and compiles; these two are the
// remaining work. They abort loudly rather than returning something plausible,
// because the failure mode this architecture punishes is code that runs and is
// quietly wrong.
//
// See porting/k3/GRAPH-BUILDER-SPEC.md for what each has to do, in particular:
//
//  KDA: ik's delta_net::build_layer_attn_linear_core is Qwen3-Next-shaped (one
//       fused ssm_in projection, one conv1d). K3 has separate attn_q/k/v and
//       THREE conv1ds sharing one conv-state buffer by thirds. The reusable
//       piece is the static delta_net::build_fused_delta_net. The gate needs
//       ssm_a read as -exp(A_log) (pre-folded by the converter), the lower-bound
//       form of the activation, and a PERMUTE into ik's [n_tokens, head_dim,
//       n_head, n_seqs] gate layout - mainline emits channel-major.
//
//  MLA: DeepSeek-style, nope-only (no RoPE despite the GGUF's rope keys), plus
//       a sigmoid output gate from attn_gate computed on the LAYER INPUT and
//       applied before wo.
//
ggml_tensor * llm_build_context::build_kimi_k3_kda(ggml_cgraph * gf, ggml_tensor * cur,
        ggml_tensor * inp_out_ids, int il) {

    const auto & layer = model.layers[il];

    const int64_t head_dim   = hparams.kda_head_dim;              // 128
    const int64_t n_head_kda = hparams.n_head();                  // 96
    const int64_t d_inner    = head_dim * n_head_kda;             // 12288
    const int64_t n_seqs       = 1;
    const int64_t n_seq_tokens = n_tokens;
    const float   eps          = hparams.f_norm_rms_eps;

    ggml_tensor * input = cur;

    // ---- q/k/v, presented as the fused layout build_qkv expects ----
    // Qwen3-Next projects q/k/v/z with one fused matrix; K3 keeps them separate.
    // Concatenating here costs one copy and lets the whole conv + state +
    // recurrence path below be reused rather than reimplemented.
    ggml_tensor * q = ggml_mul_mat(ctx0, layer.wq, cur);
    ggml_tensor * k = ggml_mul_mat(ctx0, layer.wk, cur);
    ggml_tensor * v = ggml_mul_mat(ctx0, layer.wv, cur);
    ggml_tensor * qkv_mixed = ggml_concat(ctx0, ggml_concat(ctx0, q, k, 0), v, 0);
    qkv_mixed = ggml_reshape_3d(ctx0, qkv_mixed, 3*d_inner, n_seq_tokens, n_seqs);
    cb(qkv_mixed, "kda_qkv", il);

    // Same story for the three short convolutions: one fused [d_conv, 1, 3*d_inner]
    // weight, matching the single conv-state buffer build_qkv reads and writes.
    // ggml_ssm_conv wants the weight as a 2-D [d_conv, d_inner] matrix, while the
    // GGUF ships each conv as [d_conv, 1, d_inner]. The middle axis is 1, so the
    // reshape is a relabel rather than a copy.
    ggml_tensor * conv_w = ggml_concat(ctx0,
            ggml_concat(ctx0, layer.ssm_conv1d_q, layer.ssm_conv1d_k, 2),
            layer.ssm_conv1d_v, 2);
    conv_w = ggml_reshape_2d(ctx0, conv_w, hparams.ssm_d_conv, 3*d_inner);

    // ---- forget gate ----
    // gate_lower_bound is NOT a clamp; it selects the activation:
    //   unset:            g = -exp(A_log) * softplus(f_b(f_a(x)) + dt_bias)
    //   set (K3, -5.0):   g = lower_bound * sigmoid(exp(A_log) * (f_b(f_a(x)) + dt_bias))
    // ssm_a stores -exp(A_log) folded at conversion time, so the scale(-1) below
    // is what turns A back into +exp(A_log) inside the sigmoid.
    ggml_tensor * g = ggml_mul_mat(ctx0, layer.ssm_f_b, ggml_mul_mat(ctx0, layer.ssm_f_a, cur));
    g = ggml_add(ctx0, g, layer.ssm_dt_b);
    g = ggml_reshape_3d(ctx0, g, head_dim, n_head_kda, n_tokens);

    ggml_tensor * A = ggml_reshape_3d(ctx0, layer.ssm_a, 1, n_head_kda, 1);
    g = ggml_mul(ctx0, g, A);                        // per-head scalar over 128 channels
    if (hparams.kda_gate_lower_bound > -INFINITY) {
        g = ggml_sigmoid(ctx0, ggml_scale(ctx0, g, -1.0f));
        g = ggml_scale(ctx0, g, hparams.kda_gate_lower_bound);
    } else {
        g = ggml_mul(ctx0, ggml_softplus(ctx0, g), A);
    }
    cb(g, "kda_gate", il);

    // Leave it channel-major as [S_v, H_v, n_tokens, n_seqs]; build_fused_delta_net
    // recognises that shape as the per-channel gate and permutes it into the
    // token-first layout the kernel wants.
    g = ggml_reshape_4d(ctx0, g, head_dim, n_head_kda, n_seq_tokens, n_seqs);

    // ---- beta: one per head, shaped [H_v, 1, n_tokens, n_seqs] ----
    // Passed RAW. ik's delta-net kernel applies the sigmoid itself
    // (beta_val = 1/(1+exp(-beta_raw))), so sigmoiding here would apply it twice
    // - which stays in range and merely flattens the gate, i.e. fails quietly.
    // Mainline sigmoids at this point because its kernel does not.
    ggml_tensor * beta = ggml_mul_mat(ctx0, layer.ssm_beta, cur);     // [n_head, n_tokens]
    beta = ggml_reshape_4d(ctx0, beta, n_head_kda, 1, n_seq_tokens, n_seqs);
    cb(beta, "kda_beta", il);

    // ---- conv + recurrence, reusing the delta-net plumbing ----
    const uint32_t slots = llama_kv_qnext_state_slots(lctx.kv_self);
    GGML_ASSERT(slots > 0);

    ggml_tensor * out = delta_net::build_qkv(ctx0, lctx.kv_self.s_l[il], conv_w,
            qkv_mixed, lctx.inp_s_seq_qnext, beta, g,
            head_dim, n_head_kda, head_dim, n_head_kda, hparams.ssm_d_conv,
            0, slots, false, eps, 1, il, cb, gf);
    cb(out, "kda_out", il);

    // ---- output gate, norm, projection ----
    // Identical to Qwen3-Next's tail, so reuse it: per-head RMSNorm with
    // ssm_norm, a SiLU gate from ssm_g, then the output projection.
    ggml_tensor * z = ggml_mul_mat(ctx0, layer.ssm_g, cur);
    out = delta_net::build_gated_output(lctx, ctx0, layer.ssm_norm, layer.wo,
            out, z, head_dim, n_head_kda, n_tokens, il, cb);
    cb(out, "kda_gated", il);

    if (inp_out_ids) {
        out = ggml_get_rows(ctx0, out, inp_out_ids);
        GGML_UNUSED(input);
    }
    return out;
}

ggml_tensor * llm_build_context::build_kimi_k3_mla(ggml_cgraph * gf, ggml_tensor * cur,
        ggml_tensor * KQ_mask, ggml_tensor * inp_out_ids, float kq_scale, int il) {
    GGML_UNUSED(gf); GGML_UNUSED(cur); GGML_UNUSED(KQ_mask);
    GGML_UNUSED(inp_out_ids); GGML_UNUSED(kq_scale); GGML_UNUSED(il);
    GGML_ABORT("kimi-k3: MLA layer not implemented yet (see porting/k3/GRAPH-BUILDER-SPEC.md)");
}
