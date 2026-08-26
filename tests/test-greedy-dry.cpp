#include "get-model.h"
#include "common.h"
#include "sampling.h"

#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr llama_token TOKEN_A      = 101;
constexpr llama_token TOKEN_B      = 102;
constexpr llama_token TOKEN_REPEAT = 103;
constexpr llama_token TOKEN_OTHER  = 104;

void require(bool condition, const char * message) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", message);
        std::exit(EXIT_FAILURE);
    }
}

common_sampler * make_sampler(
        const llama_model * model,
        float temp,
        float dry_multiplier,
        bool include_dry,
        std::vector<std::string> sequence_breakers = {}) {
    common_params_sampling params;
    params.temp = temp;
    params.total_context_size = 64;
    params.dry_multiplier = dry_multiplier;
    params.dry_base = 1.75f;
    params.dry_allowed_length = 2;
    params.dry_penalty_last_n = -1;
    params.dry_sequence_breakers = std::move(sequence_breakers);
    params.samplers_sequence = include_dry
        ? std::vector<llama_sampler_type>{llama_sampler_type::DRY}
        : std::vector<llama_sampler_type>{llama_sampler_type::TOP_K, llama_sampler_type::TEMPERATURE};

    common_sampler * sampler = common_sampler_init(model, params);
    require(sampler != nullptr, "common_sampler_init returned null");

    // TOKEN_REPEAT would extend the repeated TOKEN_A, TOKEN_B suffix.
    for (const llama_token token : {TOKEN_A, TOKEN_B, TOKEN_REPEAT, TOKEN_A, TOKEN_B}) {
        common_sampler_accept(sampler, nullptr, token, true);
    }

    return sampler;
}

common_sampler * make_breaker_sampler(
        const llama_model * model,
        llama_token breaker,
        std::vector<std::string> sequence_breakers) {
    common_sampler * sampler = make_sampler(
        model, 0.0f, 2.0f, true, std::move(sequence_breakers));
    for (const llama_token token : {TOKEN_A, TOKEN_B, breaker, TOKEN_A, TOKEN_B}) {
        common_sampler_accept(sampler, nullptr, token, true);
    }
    return sampler;
}

std::array<llama_token_data, 3> make_candidates() {
    return {{
        {TOKEN_REPEAT, 10.0f, 0.0f},
        {TOKEN_OTHER,   9.0f, 0.0f},
        {TOKEN_A,      -1.0f, 0.0f},
    }};
}

template <size_t N>
llama_token_data_array as_array(std::array<llama_token_data, N> & candidates) {
    return {candidates.data(), candidates.size(), -1, false};
}

void test_zero_temperature_dry_changes_argmax(const llama_model * model) {
    common_sampler * sampler = make_sampler(model, 0.0f, 2.0f, true);
    auto candidates = make_candidates();
    auto cur_p = as_array(candidates);

    const llama_token selected = common_sampler_sample_greedy(sampler, nullptr, cur_p);

    require(selected == TOKEN_OTHER, "temp=0 DRY did not change the repeated greedy choice");
    require(candidates[0].logit < candidates[1].logit, "temp=0 DRY did not penalize the repeated token logit");
    require(candidates[1].logit == 9.0f, "temp=0 DRY changed an unrelated token logit");
    common_sampler_free(sampler);
}

void test_negative_temperature_dry_changes_argmax_and_computes_probabilities(const llama_model * model) {
    common_sampler * sampler = make_sampler(model, -1.0f, 2.0f, true);
    auto candidates = make_candidates();
    auto cur_p = as_array(candidates);

    const llama_token selected = common_sampler_sample_greedy(sampler, nullptr, cur_p);

    require(selected == TOKEN_OTHER, "temp<0 DRY did not change the repeated greedy choice");
    require(cur_p.sorted, "temp<0 greedy sampling did not sort probabilities");
    require(cur_p.data[0].id == TOKEN_OTHER, "temp<0 top probability does not match the selected token");
    require(cur_p.data[0].p > cur_p.data[1].p, "temp<0 probabilities do not preserve the DRY-adjusted ordering");
    const float probability_sum = cur_p.data[0].p + cur_p.data[1].p + cur_p.data[2].p;
    require(std::fabs(probability_sum - 1.0f) < 1e-6f, "temp<0 probabilities do not sum to one");
    common_sampler_free(sampler);
}

void test_disabled_dry_preserves_zero_temperature_greedy(const llama_model * model) {
    common_sampler * sampler = make_sampler(model, 0.0f, 0.0f, true);
    auto candidates = make_candidates();
    const auto original = candidates;
    auto cur_p = as_array(candidates);

    const llama_token selected = common_sampler_sample_greedy(sampler, nullptr, cur_p);

    require(selected == TOKEN_REPEAT, "disabled DRY changed the greedy token");
    require(candidates[0].logit == original[0].logit &&
            candidates[1].logit == original[1].logit &&
            candidates[2].logit == original[2].logit,
            "disabled DRY changed candidate logits");
    common_sampler_free(sampler);
}

void test_absent_dry_preserves_zero_temperature_greedy(const llama_model * model) {
    common_sampler * sampler = make_sampler(model, 0.0f, 2.0f, false);
    auto candidates = make_candidates();
    const auto original = candidates;
    auto cur_p = as_array(candidates);

    const llama_token selected = common_sampler_sample_greedy(sampler, nullptr, cur_p);

    require(selected == TOKEN_REPEAT, "a sampler chain without DRY changed the greedy token");
    require(candidates[0].logit == original[0].logit &&
            candidates[1].logit == original[1].logit &&
            candidates[2].logit == original[2].logit,
            "a sampler chain without DRY changed candidate logits");
    common_sampler_free(sampler);
}

void test_colon_breaker_exempts_repeated_colon_at_zero_temperature(
        const llama_model * model) {
    const auto colon_tokens = common_tokenize(
        llama_model_get_vocab(model), ":", false, false);
    require(colon_tokens.size() == 1, "test vocabulary does not have one colon token");
    const llama_token colon = colon_tokens.front();

    common_sampler * without_breaker = make_breaker_sampler(model, colon, {});
    std::array<llama_token_data, 2> penalized_candidates {{
        {colon,       10.0f, 0.0f},
        {TOKEN_OTHER,  9.0f, 0.0f},
    }};
    auto penalized = as_array(penalized_candidates);
    require(
        common_sampler_sample_greedy(without_breaker, nullptr, penalized) == TOKEN_OTHER,
        "colon was not penalized when absent from the DRY breakers");
    require(
        penalized_candidates[0].logit < penalized_candidates[1].logit,
        "colon logit was not reduced when absent from the DRY breakers");
    common_sampler_free(without_breaker);

    common_sampler * with_breaker = make_breaker_sampler(model, colon, {":"});
    std::array<llama_token_data, 2> exempt_candidates {{
        {colon,       10.0f, 0.0f},
        {TOKEN_OTHER,  9.0f, 0.0f},
    }};
    auto exempt = as_array(exempt_candidates);
    require(
        common_sampler_sample_greedy(with_breaker, nullptr, exempt) == colon,
        "configured colon breaker did not preserve the repeated colon");
    require(
        exempt_candidates[0].logit == 10.0f,
        "configured colon breaker changed the repeated colon logit");
    common_sampler_free(with_breaker);
}

void test_newline_breaker_exempts_repeated_newline_at_zero_temperature(
        const llama_model * model) {
    const auto newline_tokens = common_tokenize(
        llama_model_get_vocab(model), "\n", false, false);
    require(!newline_tokens.empty(), "test vocabulary does not tokenize newline");
    const llama_token newline = newline_tokens.back();
    require(
        common_token_to_piece(llama_model_get_vocab(model), newline, true) == "\n",
        "test vocabulary newline token does not decode to newline");

    common_sampler * without_breaker = make_breaker_sampler(model, newline, {});
    std::array<llama_token_data, 2> penalized_candidates {{
        {newline,     10.0f, 0.0f},
        {TOKEN_OTHER,  9.0f, 0.0f},
    }};
    auto penalized = as_array(penalized_candidates);
    require(
        common_sampler_sample_greedy(without_breaker, nullptr, penalized) == TOKEN_OTHER,
        "newline was not penalized when absent from the DRY breakers");
    require(
        penalized_candidates[0].logit < penalized_candidates[1].logit,
        "newline logit was not reduced when absent from the DRY breakers");
    common_sampler_free(without_breaker);

    common_sampler * with_breaker = make_breaker_sampler(model, newline, {"\n"});
    std::array<llama_token_data, 2> exempt_candidates {{
        {newline,     10.0f, 0.0f},
        {TOKEN_OTHER,  9.0f, 0.0f},
    }};
    auto exempt = as_array(exempt_candidates);
    require(
        common_sampler_sample_greedy(with_breaker, nullptr, exempt) == newline,
        "configured newline breaker did not preserve the repeated newline");
    require(
        exempt_candidates[0].logit == 10.0f,
        "configured newline breaker changed the repeated newline logit");
    common_sampler_free(with_breaker);
}

void test_quote_star_breakers_penalize_colon_and_newline_at_zero_temperature(
        const llama_model * model) {
    const auto colon_tokens = common_tokenize(
        llama_model_get_vocab(model), ":", false, false);
    require(colon_tokens.size() == 1, "test vocabulary does not have one colon token");
    const llama_token colon = colon_tokens.front();

    const auto newline_tokens = common_tokenize(
        llama_model_get_vocab(model), "\n", false, false);
    require(!newline_tokens.empty(), "test vocabulary does not tokenize newline");
    const llama_token newline = newline_tokens.back();
    require(
        common_token_to_piece(llama_model_get_vocab(model), newline, true) == "\n",
        "test vocabulary newline token does not decode to newline");

    for (const auto delimiter : {colon, newline}) {
        common_sampler * sampler = make_breaker_sampler(model, delimiter, {"\"", "*"});
        std::array<llama_token_data, 2> candidates {{
            {delimiter,   10.0f, 0.0f},
            {TOKEN_OTHER,  9.0f, 0.0f},
        }};
        auto cur_p = as_array(candidates);
        require(
            common_sampler_sample_greedy(sampler, nullptr, cur_p) == TOKEN_OTHER,
            "quote/star-only breakers did not penalize a removed delimiter");
        require(
            candidates[0].logit < candidates[1].logit,
            "quote/star-only breakers did not reduce a removed delimiter logit");
        common_sampler_free(sampler);
    }
}

} // namespace

int main(int argc, char ** argv) {
    const char * model_path = get_model_or_exit(argc, argv);

    llama_backend_init();
    llama_model_params model_params = llama_model_default_params();
    model_params.vocab_only = true;
    llama_model * model = llama_model_load_from_file(model_path, model_params);
    require(model != nullptr, "failed to load the vocabulary-only test model");

    test_zero_temperature_dry_changes_argmax(model);
    test_negative_temperature_dry_changes_argmax_and_computes_probabilities(model);
    test_disabled_dry_preserves_zero_temperature_greedy(model);
    test_absent_dry_preserves_zero_temperature_greedy(model);
    test_colon_breaker_exempts_repeated_colon_at_zero_temperature(model);
    test_newline_breaker_exempts_repeated_newline_at_zero_temperature(model);
    test_quote_star_breakers_penalize_colon_and_newline_at_zero_temperature(model);

    llama_free_model(model);
    llama_backend_free();
    std::puts("PASS: greedy DRY sampling");
    return 0;
}
