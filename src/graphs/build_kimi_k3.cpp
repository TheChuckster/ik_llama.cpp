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

    ggml_tensor * moe = llm_build_moe_ffn(ctx0, lctx, h,
            layer.ffn_gate_inp,
            layer.ffn_up_exps,
            layer.ffn_gate_exps,
            layer.ffn_down_exps,
            layer.ffn_exp_probs_b,
            n_expert, n_expert_used,
            LLM_FFN_SITU,
            hparams.expert_weights_norm,
            true, hparams.expert_weights_scale,
            (llm_expert_gating_func_type) hparams.expert_gating_func,
            cb, il, gf);
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
    GGML_UNUSED(gf); GGML_UNUSED(cur); GGML_UNUSED(inp_out_ids); GGML_UNUSED(il);
    GGML_ABORT("kimi-k3: KDA layer not implemented yet (see porting/k3/GRAPH-BUILDER-SPEC.md)");
}

ggml_tensor * llm_build_context::build_kimi_k3_mla(ggml_cgraph * gf, ggml_tensor * cur,
        ggml_tensor * KQ_mask, ggml_tensor * inp_out_ids, float kq_scale, int il) {
    GGML_UNUSED(gf); GGML_UNUSED(cur); GGML_UNUSED(KQ_mask);
    GGML_UNUSED(inp_out_ids); GGML_UNUSED(kq_scale); GGML_UNUSED(il);
    GGML_ABORT("kimi-k3: MLA layer not implemented yet (see porting/k3/GRAPH-BUILDER-SPEC.md)");
}
