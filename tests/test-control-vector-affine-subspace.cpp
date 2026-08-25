#include "llama-control-vector.h"

#include "ggml.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#define CHECK(condition) do { \
    if (!(condition)) { \
        std::fprintf(stderr, "CHECK failed at %s:%d: %s\n", __FILE__, __LINE__, #condition); \
        std::abort(); \
    } \
} while (0)

static bool close(float actual, float expected, float tolerance = 1e-6f) {
    return std::abs(actual - expected) <= tolerance;
}

static void test_geometry_validation() {
    const float basis[] = {
        1.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f,
    };
    const float swapped[] = {
        0.0f, 1.0f, 0.0f,
        1.0f, 0.0f, 0.0f,
    };
    const float nonunit[] = {
        0.5f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f,
    };
    const float nonorthogonal[] = {
        1.0f, 0.0f, 0.0f,
        1.0f, 0.0f, 0.0f,
    };
    const float in_span[] = { 2.0f, -3.0f, 0.0f };
    const float zero[] = { 0.0f, 0.0f, 0.0f };
    const float outside_span[] = { 0.0f, 0.0f, 1.0f };
    const float nonfinite_basis[] = {
        std::numeric_limits<float>::quiet_NaN(), 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f,
    };
    const float nonfinite_offset[] = {
        std::numeric_limits<float>::infinity(), 0.0f, 0.0f,
    };

    CHECK(llama_control_vector_is_orthonormal_f32(basis, 3, 2));
    CHECK(llama_control_vector_is_orthonormal_f32(swapped, 3, 2));
    CHECK(!llama_control_vector_is_orthonormal_f32(nullptr, 3, 2));
    CHECK(!llama_control_vector_is_orthonormal_f32(basis, 0, 2));
    CHECK(!llama_control_vector_is_orthonormal_f32(basis, 3, 0));
    CHECK(!llama_control_vector_is_orthonormal_f32(nonunit, 3, 2));
    CHECK(!llama_control_vector_is_orthonormal_f32(nonorthogonal, 3, 2));
    CHECK(!llama_control_vector_is_orthonormal_f32(nonfinite_basis, 3, 2));

    CHECK(llama_control_vector_offset_in_span_f32(basis, in_span, 3, 2));
    CHECK(llama_control_vector_offset_in_span_f32(basis, zero, 3, 2));
    CHECK(!llama_control_vector_offset_in_span_f32(
        basis, outside_span, 3, 2));
    CHECK(!llama_control_vector_offset_in_span_f32(
        basis, nonfinite_offset, 3, 2));
    CHECK(!llama_control_vector_offset_in_span_f32(
        basis, in_span, 3, 2, -1.0));
}

static std::vector<float> evaluate(
        const std::vector<float> & basis_values,
        int64_t width,
        int64_t rank,
        const std::vector<float> & input_values,
        int64_t tokens,
        const std::vector<float> & offset_values) {
    CHECK((int64_t) basis_values.size() == width * rank);
    CHECK((int64_t) input_values.size() == width * tokens);
    CHECK((int64_t) offset_values.size() == width);
    ggml_init_params params = {
        /* .mem_size = */ 1024 * 1024,
        /* .mem_base = */ nullptr,
        /* .no_alloc = */ false,
    };
    ggml_context * context = ggml_init(params);
    CHECK(context != nullptr);

    ggml_tensor * input = ggml_new_tensor_2d(
        context, GGML_TYPE_F32, width, tokens);
    ggml_tensor * basis = ggml_new_tensor_2d(
        context, GGML_TYPE_F32, width, rank);
    ggml_tensor * offset = ggml_new_tensor_1d(
        context, GGML_TYPE_F32, width);
    std::memcpy(input->data, input_values.data(), input_values.size() * sizeof(float));
    std::memcpy(basis->data, basis_values.data(), basis_values.size() * sizeof(float));
    std::memcpy(offset->data, offset_values.data(), offset_values.size() * sizeof(float));

    ggml_tensor * projected = llama_control_vector_project_subspace(
        context, input, basis, rank);
    ggml_tensor * output = ggml_add(context, projected, offset);
    ggml_cgraph * graph = ggml_new_graph(context);
    ggml_build_forward_expand(graph, output);
    CHECK(ggml_graph_compute_with_ctx(context, graph, 1) == GGML_STATUS_SUCCESS);

    const float * data = static_cast<const float *>(output->data);
    std::vector<float> result(data, data + input_values.size());
    ggml_free(context);
    return result;
}

static void test_project_then_offset_and_basis_order() {
    const std::vector<float> basis = {
        1.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f,
    };
    const std::vector<float> swapped = {
        0.0f, 1.0f, 0.0f,
        1.0f, 0.0f, 0.0f,
    };
    const std::vector<float> rotated = {
        0.6f, 0.8f, 0.0f,
        -0.8f, 0.6f, 0.0f,
    };
    const std::vector<float> rotated_swapped = {
        -0.8f, 0.6f, 0.0f,
        0.6f, 0.8f, 0.0f,
    };
    const std::vector<float> input = {
        3.0f, 4.0f, 5.0f,
        -2.0f, 8.0f, -1.0f,
    };
    const std::vector<float> offset = { 7.0f, -9.0f, 0.0f };
    const std::vector<float> actual = evaluate(basis, 3, 2, input, 2, offset);
    const std::vector<float> reordered = evaluate(swapped, 3, 2, input, 2, offset);
    const std::vector<float> rotated_actual = evaluate(rotated, 3, 2, input, 2, offset);
    const std::vector<float> rotated_reordered = evaluate(
        rotated_swapped, 3, 2, input, 2, offset);
    const float expected[] = {
        7.0f, -9.0f, 5.0f,
        7.0f, -9.0f, -1.0f,
    };
    CHECK(actual.size() == 6);
    CHECK(reordered.size() == actual.size());
    for (size_t index = 0; index < actual.size(); ++index) {
        if (!close(actual[index], expected[index])
                || !close(reordered[index], actual[index])) {
            std::fprintf(stderr,
                "index %zu: actual=%g reordered=%g expected=%g\n",
                index, actual[index], reordered[index], expected[index]);
        }
        CHECK(close(actual[index], expected[index]));
        CHECK(close(reordered[index], actual[index]));
        CHECK(close(rotated_actual[index], expected[index], 2e-6f));
        CHECK(close(rotated_reordered[index], rotated_actual[index], 2e-6f));
    }
}

static void test_rank_seven_single_token() {
    constexpr int64_t width = 8;
    constexpr int64_t rank = 7;
    std::vector<float> basis(width * rank, 0.0f);
    std::vector<float> reversed(width * rank, 0.0f);
    std::vector<float> input(width);
    std::vector<float> offset(width, 0.0f);
    for (int64_t row = 0; row < rank; ++row) {
        basis[row * width + row] = 1.0f;
        reversed[row * width + (rank - row - 1)] = 1.0f;
        input[row] = (float) row + 1.0f;
        offset[row] = (float) row + 10.0f;
    }
    input[width - 1] = 8.0f;

    const std::vector<float> actual = evaluate(
        basis, width, rank, input, 1, offset);
    const std::vector<float> reordered = evaluate(
        reversed, width, rank, input, 1, offset);
    for (int64_t column = 0; column < rank; ++column) {
        if (!close(actual[column], offset[column])
                || !close(reordered[column], actual[column])) {
            std::fprintf(stderr,
                "rank-seven column %ld: actual=%g reordered=%g expected=%g\n",
                (long) column, actual[column], reordered[column], offset[column]);
        }
        CHECK(close(actual[column], offset[column]));
        CHECK(close(reordered[column], actual[column]));
    }
    CHECK(close(actual[width - 1], input[width - 1]));
    CHECK(close(reordered[width - 1], actual[width - 1]));
}

int main() {
    test_geometry_validation();
    test_project_then_offset_and_basis_order();
    test_rank_seven_single_token();
    std::puts("control-vector affine-subspace tests passed");
    return 0;
}
