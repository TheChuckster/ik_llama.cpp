#include "common.h"
#include "llama.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#define CHECK(condition) do { \
    if (!(condition)) { \
        std::fprintf(stderr, "CHECK failed at %s:%d: %s\n", __FILE__, __LINE__, #condition); \
        std::abort(); \
    } \
} while (0)

static std::vector<float> evaluate(
        llama_context * ctx,
        const std::vector<llama_token> & tokens,
        int32_t n_vocab) {
    llama_kv_cache_clear(ctx);
    llama_batch batch = llama_batch_get_one(
        const_cast<llama_token *>(tokens.data()), tokens.size(), 0, 0);
    CHECK(llama_decode(ctx, batch) == 0);
    const float * logits = llama_get_logits_ith(ctx, -1);
    CHECK(logits != nullptr);
    return { logits, logits + n_vocab };
}

static float maximum_difference(
        const std::vector<float> & left,
        const std::vector<float> & right) {
    CHECK(left.size() == right.size());
    float result = 0.0f;
    for (size_t index = 0; index < left.size(); ++index) {
        CHECK(std::isfinite(left[index]));
        CHECK(std::isfinite(right[index]));
        result = std::max(result, std::abs(left[index] - right[index]));
    }
    return result;
}

int main(int argc, char ** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: %s MODEL.gguf\n", argv[0]);
        return 2;
    }

    llama_backend_init();
    llama_model_params model_params = llama_model_default_params();
    llama_model * model = llama_model_load_from_file(argv[1], model_params);
    CHECK(model != nullptr);

    llama_context_params context_params = llama_context_default_params();
    context_params.n_ctx = 256;
    context_params.n_batch = 256;
    context_params.n_ubatch = 256;
    context_params.n_threads = 1;
    context_params.n_threads_batch = 1;
    context_params.graph_reuse = true;
    llama_context * ctx = llama_init_from_model(model, context_params);
    CHECK(ctx != nullptr);

    const int32_t n_embd = llama_model_n_embd(model);
    const int32_t n_layer = llama_n_layer(model);
    const int32_t n_vocab = llama_n_vocab(model);
    CHECK(n_embd > 1);
    CHECK(n_layer > 1);
    CHECK(n_vocab > 0);

    const std::vector<llama_token> tokens = common_tokenize(
        model, "Once upon a time", true, false);
    CHECK(!tokens.empty());

    std::vector<float> projection((size_t) n_embd * (n_layer - 1), 0.0f);
    std::vector<float> offset(projection.size(), 0.0f);
    for (int32_t layer = 1; layer < n_layer; ++layer) {
        projection[(size_t) n_embd * (layer - 1)] = 1.0f;
        // A parallel offset proves it is added after projection; the reverse
        // order would erase it.
        offset[(size_t) n_embd * (layer - 1)] = 50.0f;
    }

    const auto baseline = evaluate(ctx, tokens, n_vocab);
    CHECK(llama_control_vector_projection_apply(
        ctx, projection.data(), projection.size(), n_embd, 1, n_layer - 1) == 0);
    const auto projected = evaluate(ctx, tokens, n_vocab);
    CHECK(maximum_difference(baseline, projected) > 1e-5f);

    CHECK(llama_control_vector_apply(
        ctx, offset.data(), offset.size(), n_embd, 1, n_layer - 1) == 0);
    const auto affine = evaluate(ctx, tokens, n_vocab);
    CHECK(maximum_difference(projected, affine) > 1e-5f);

    CHECK(llama_control_vector_apply(ctx, nullptr, 0, 0, 0, 0) == 0);
    const auto projected_restored = evaluate(ctx, tokens, n_vocab);
    CHECK(maximum_difference(projected, projected_restored) <= 1e-6f);

    CHECK(llama_control_vector_projection_apply(ctx, nullptr, 0, 0, 0, 0) == 0);
    const auto baseline_restored = evaluate(ctx, tokens, n_vocab);
    CHECK(maximum_difference(baseline, baseline_restored) <= 1e-6f);

    std::vector<float> basis((size_t) n_embd * 2, 0.0f);
    std::vector<float> affine_offset(n_embd, 0.0f);
    basis[0] = 1.0f;
    basis[n_embd + 1] = 1.0f;
    affine_offset[0] = 50.0f;
    CHECK(llama_control_vector_affine_subspace_apply(
        ctx,
        basis.data(), basis.size(),
        affine_offset.data(), affine_offset.size(),
        n_embd, 2, 1) == 0);
    const auto subspace = evaluate(ctx, tokens, n_vocab);
    const auto subspace_reused = evaluate(ctx, tokens, n_vocab);
    CHECK(maximum_difference(baseline, subspace) > 1e-5f);
    CHECK(maximum_difference(subspace, subspace_reused) <= 1e-6f);

    // The startup affine state is exclusive even at the low-level API.
    CHECK(llama_control_vector_apply(
        ctx, offset.data(), offset.size(), n_embd, 1, n_layer - 1) != 0);
    CHECK(llama_control_vector_projection_apply(
        ctx, projection.data(), projection.size(), n_embd, 1, n_layer - 1) != 0);

    CHECK(llama_control_vector_affine_subspace_apply(
        ctx, nullptr, 0, nullptr, 0, 0, 0, 0) == 0);
    const auto affine_restored = evaluate(ctx, tokens, n_vocab);
    CHECK(maximum_difference(baseline, affine_restored) <= 1e-6f);

    // Clearing the new state leaves the established rank-one path byte-stable.
    CHECK(llama_control_vector_projection_apply(
        ctx, projection.data(), projection.size(), n_embd, 1, n_layer - 1) == 0);
    const auto projected_reapplied = evaluate(ctx, tokens, n_vocab);
    CHECK(maximum_difference(projected, projected_reapplied) <= 1e-6f);
    CHECK(llama_control_vector_projection_apply(ctx, nullptr, 0, 0, 0, 0) == 0);

    llama_free(ctx);
    llama_free_model(model);
    llama_backend_free();
    std::puts("control-vector state tests passed");
    return 0;
}
