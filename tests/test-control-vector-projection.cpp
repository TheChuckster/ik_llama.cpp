#include "llama-control-vector.h"

#include "ggml.h"

#include <cmath>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <limits>

#define CHECK(condition) do { \
    if (!(condition)) { \
        std::fprintf(stderr, "CHECK failed at %s:%d: %s\n", __FILE__, __LINE__, #condition); \
        std::abort(); \
    } \
} while (0)

static bool close(float actual, float expected, float tolerance = 1e-6f) {
    return std::abs(actual - expected) <= tolerance;
}

static void test_unit_validation() {
    const float unit[] = { 0.6f, 0.8f };
    const float too_short[] = { 0.6f, 0.79f };
    const float not_finite[] = { 0.0f, std::numeric_limits<float>::infinity() };

    CHECK(llama_control_vector_is_unit_f32(unit, 2));
    CHECK(!llama_control_vector_is_unit_f32(too_short, 2));
    CHECK(!llama_control_vector_is_unit_f32(not_finite, 2));
    CHECK(!llama_control_vector_is_unit_f32(nullptr, 2));
    CHECK(!llama_control_vector_is_unit_f32(unit, 0));
    CHECK(!llama_control_vector_is_unit_f32(unit, 2, -1.0));
}

static void test_tokenwise_projection_graph() {
    ggml_init_params params = {
        /* .mem_size = */ 1024 * 1024,
        /* .mem_base = */ nullptr,
        /* .no_alloc = */ false,
    };
    ggml_context * ctx = ggml_init(params);
    CHECK(ctx != nullptr);

    ggml_tensor * input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 3, 2);
    ggml_tensor * direction = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 3);

    const float input_values[] = {
        3.0f,  4.0f, 0.0f,
        0.8f, -0.6f, 2.0f,
    };
    const float direction_values[] = { 0.6f, 0.8f, 0.0f };
    std::memcpy(input->data, input_values, sizeof(input_values));
    std::memcpy(direction->data, direction_values, sizeof(direction_values));

    ggml_tensor * output = llama_control_vector_project(ctx, input, direction);
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, output);
    CHECK(ggml_graph_compute_with_ctx(ctx, graph, 1) == GGML_STATUS_SUCCESS);

    const float expected[] = {
        0.0f,  0.0f, 0.0f,
        0.8f, -0.6f, 2.0f,
    };
    const float * actual = static_cast<const float *>(output->data);
    for (size_t index = 0; index < sizeof(expected) / sizeof(expected[0]); ++index) {
        CHECK(close(actual[index], expected[index]));
    }

    ggml_free(ctx);
}

int main() {
    test_unit_validation();
    test_tokenwise_projection_graph();
    std::puts("control-vector projection tests passed");
    return 0;
}
