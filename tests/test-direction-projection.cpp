#include "llama-direction-projection.h"
#include "llama-direction-subspace.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
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

static ggml_tensor tensor_2d(int64_t ne0, int64_t ne1, int64_t ne2 = 1) {
    ggml_tensor tensor = {};
    tensor.ne[0] = ne0;
    tensor.ne[1] = ne1;
    tensor.ne[2] = ne2;
    tensor.ne[3] = 1;
    return tensor;
}

static void test_axis_0() {
    const std::vector<float> direction = { 0.6f, 0.8f };
    std::vector<float> values = {
        3.0f,  4.0f,
        0.8f, -0.6f,
        6.0f,  8.0f,
    };
    auto tensor = tensor_2d(2, 3);
    const auto stats = llama_direction_orthogonalize_f32(
            values.data(), &tensor, direction, 1.0f, 0, 4);

    const std::vector<float> expected = { 0.0f, 0.0f, 0.8f, -0.6f, 0.0f, 0.0f };
    for (size_t i = 0; i < values.size(); ++i) {
        CHECK(close(values[i], expected[i]));
    }
    CHECK(std::abs(stats.tensor_norm_sq - 126.0) < 1e-6);
    CHECK(std::abs(stats.component_norm_sq - 125.0) < 1e-5);
}

static void test_axis_1_and_slices() {
    const std::vector<float> direction = { 0.6f, 0.8f };
    const std::vector<float> one_slice = {
        3.0f,  0.8f, 6.0f,
        4.0f, -0.6f, 8.0f,
    };
    std::vector<float> values = one_slice;
    values.insert(values.end(), one_slice.begin(), one_slice.end());
    auto tensor = tensor_2d(3, 2, 2);
    const auto stats = llama_direction_orthogonalize_f32(
            values.data(), &tensor, direction, 1.0f, 1, 4);

    const std::vector<float> expected_slice = { 0.0f, 0.8f, 0.0f, 0.0f, -0.6f, 0.0f };
    for (int slice = 0; slice < 2; ++slice) {
        for (size_t i = 0; i < expected_slice.size(); ++i) {
            CHECK(close(values[slice * expected_slice.size() + i], expected_slice[i]));
        }
    }
    CHECK(std::abs(stats.tensor_norm_sq - 252.0) < 1e-6);
    CHECK(std::abs(stats.component_norm_sq - 250.0) < 2e-5);
}

static void test_scale_and_thread_equivalence() {
    const std::vector<float> direction = { 0.6f, 0.8f };
    std::vector<float> scaled = { 3.0f, 4.0f };
    auto row_tensor = tensor_2d(2, 1);
    llama_direction_orthogonalize_f32(scaled.data(), &row_tensor, direction, 0.25f, 0, 2);
    CHECK(close(scaled[0], 2.25f));
    CHECK(close(scaled[1], 3.00f));

    constexpr int64_t ne0 = 17;
    constexpr int64_t ne1 = 11;
    std::vector<float> wide_direction(ne1);
    double norm_sq = 0.0;
    for (int64_t row = 0; row < ne1; ++row) {
        wide_direction[row] = 0.1f + 0.03f * row;
        norm_sq += double(wide_direction[row]) * wide_direction[row];
    }
    const float inv_norm = 1.0f / std::sqrt(norm_sq);
    for (float & value : wide_direction) {
        value *= inv_norm;
    }
    std::vector<float> serial(ne0 * ne1);
    for (size_t i = 0; i < serial.size(); ++i) {
        serial[i] = std::sin(0.17f * i) + std::cos(0.031f * i);
    }
    auto parallel = serial;
    auto tensor = tensor_2d(ne0, ne1);
    llama_direction_orthogonalize_f32(serial.data(), &tensor, wide_direction, 1.0f, 1, 1);
    llama_direction_orthogonalize_f32(parallel.data(), &tensor, wide_direction, 1.0f, 1, 4);
    for (size_t i = 0; i < serial.size(); ++i) {
        CHECK(close(serial[i], parallel[i], 2e-6f));
    }
    for (int64_t col = 0; col < ne0; ++col) {
        double residual = 0.0;
        for (int64_t row = 0; row < ne1; ++row) {
            residual += parallel[row * ne0 + col] * wide_direction[row];
        }
        CHECK(std::abs(residual) < 2e-6);
    }
}

static void test_measurement_only_and_zero_tensor() {
    const std::vector<float> direction = { 0.6f, 0.8f };
    auto tensor = tensor_2d(3, 2, 2);
    std::vector<float> values = {
         3.0f,  0.8f,  6.0f,
         4.0f, -0.6f,  8.0f,
        -3.0f, -0.8f, -6.0f,
        -4.0f,  0.6f, -8.0f,
    };
    const auto original = values;
    const auto stats = llama_direction_orthogonalize_f32(
            values.data(), &tensor, direction, 0.0f, 1, 64);
    CHECK(values == original);
    CHECK(std::abs(stats.tensor_norm_sq - 252.0) < 1e-6);
    CHECK(std::abs(stats.component_norm_sq - 250.0) < 2e-5);

    std::fill(values.begin(), values.end(), 0.0f);
    const auto zero_stats = llama_direction_orthogonalize_f32(
            values.data(), &tensor, direction, 1.0f, 1, 0);
    CHECK(zero_stats.tensor_norm_sq == 0.0);
    CHECK(zero_stats.component_norm_sq == 0.0);
    CHECK(std::all_of(values.begin(), values.end(), [](float value) { return value == 0.0f; }));

    llama_direction_projection_stats source_component;
    source_component.component_norm_sq = 25.0;
    llama_direction_projection_stats post_quant_component;
    post_quant_component.component_norm_sq = 0.01;
    CHECK(std::abs(llama_direction_retained_component_ratio(
            source_component, post_quant_component) - 0.02) < 1e-12);
}

static std::vector<float> normalized_direction(int64_t width) {
    std::vector<float> result(width);
    double norm_sq = 0.0;
    for (int64_t index = 0; index < width; ++index) {
        result[index] = std::sin(0.37f * (index + 1)) + 0.2f * std::cos(0.11f * index);
        norm_sq += double(result[index]) * result[index];
    }
    const float inv_norm = 1.0f / std::sqrt(norm_sq);
    for (float & value : result) {
        value *= inv_norm;
    }
    return result;
}

static void test_projection_invariants() {
    constexpr int64_t width = 7;
    const auto direction = normalized_direction(width);

    // Embedding layout: the residual dimension is contiguous (axis 0).
    auto embedding = tensor_2d(width, 31, 3);
    std::vector<float> axis0(ggml_nelements(&embedding));
    for (size_t index = 0; index < axis0.size(); ++index) {
        axis0[index] = std::sin(0.013f * index) - 0.4f * std::cos(0.071f * index);
    }
    const auto axis0_before = axis0;
    const auto axis0_stats = llama_direction_orthogonalize_f32(
            axis0.data(), &embedding, direction, 1.0f, 0, 8);
    double expected_axis0_norm = 0.0;
    double expected_axis0_component = 0.0;
    for (int64_t row = 0; row < embedding.ne[1] * embedding.ne[2]; ++row) {
        double original_dot = 0.0;
        double residual_dot = 0.0;
        for (int64_t col = 0; col < width; ++col) {
            const size_t index = row * width + col;
            expected_axis0_norm += double(axis0_before[index]) * axis0_before[index];
            original_dot += double(axis0_before[index]) * direction[col];
            residual_dot += double(axis0[index]) * direction[col];
        }
        expected_axis0_component += original_dot * original_dot;
        CHECK(std::abs(residual_dot) < 2e-6);
    }
    CHECK(std::abs(axis0_stats.tensor_norm_sq - expected_axis0_norm) < 1e-9);
    CHECK(std::abs(axis0_stats.component_norm_sq - expected_axis0_component) < 1e-9);

    // Linear-map layout: residual outputs are rows (axis 1). A partial
    // projection must leave exactly (1-scale) of each original component.
    auto linear = tensor_2d(29, width, 3);
    std::vector<float> axis1(ggml_nelements(&linear));
    for (size_t index = 0; index < axis1.size(); ++index) {
        axis1[index] = std::cos(0.019f * index) + 0.3f * std::sin(0.053f * index);
    }
    const auto axis1_before = axis1;
    constexpr float scale = 0.25f;
    llama_direction_orthogonalize_f32(axis1.data(), &linear, direction, scale, 1, 8);
    for (int64_t slice = 0; slice < linear.ne[2]; ++slice) {
        const int64_t base = slice * linear.ne[0] * linear.ne[1];
        for (int64_t col = 0; col < linear.ne[0]; ++col) {
            double original_dot = 0.0;
            double projected_dot = 0.0;
            for (int64_t row = 0; row < width; ++row) {
                const size_t index = base + row * linear.ne[0] + col;
                original_dot += double(axis1_before[index]) * direction[row];
                projected_dot += double(axis1[index]) * direction[row];
            }
            CHECK(std::abs(projected_dot - (1.0 - scale) * original_dot) < 3e-6);
        }
    }
}

static void test_counterfactual_reflection() {
    const std::vector<float> direction = { 0.6f, 0.8f };

    // Rank-one embedding orientation: the selected component flips while an
    // orthogonal row remains unchanged. A reflection preserves total norm.
    auto embedding = tensor_2d(2, 2);
    std::vector<float> axis0 = {
        3.0f,  4.0f,
        0.8f, -0.6f,
    };
    const auto axis0_before = axis0;
    const auto axis0_stats = llama_direction_orthogonalize_f32(
            axis0.data(), &embedding, direction, 2.0f, 0, 4);
    const std::vector<float> expected_axis0 = {
       -3.0f, -4.0f,
        0.8f, -0.6f,
    };
    for (size_t index = 0; index < axis0.size(); ++index) {
        CHECK(close(axis0[index], expected_axis0[index]));
    }
    double before_norm_sq = 0.0;
    double after_norm_sq = 0.0;
    for (size_t index = 0; index < axis0.size(); ++index) {
        before_norm_sq += double(axis0_before[index]) * axis0_before[index];
        after_norm_sq += double(axis0[index]) * axis0[index];
    }
    CHECK(std::abs(before_norm_sq - after_norm_sq) < 1e-6);
    CHECK(std::abs(axis0_stats.tensor_norm_sq - 26.0) < 1e-6);
    CHECK(std::abs(axis0_stats.component_norm_sq - 25.0) < 1e-5);

    // Rank-two linear-map orientation: reflect the e0/e1 rowspace and retain
    // the e2 orthogonal row exactly.
    const std::vector<std::vector<float>> basis = {
        { 1.0f, 0.0f, 0.0f },
        { 0.0f, 1.0f, 0.0f },
    };
    auto linear = tensor_2d(2, 3);
    std::vector<float> axis1 = {
        1.0f, 4.0f,
        2.0f, 5.0f,
        3.0f, 6.0f,
    };
    const auto axis1_before = axis1;
    const auto axis1_stats = llama_direction_orthogonalize_subspace_f32(
            axis1.data(), &linear, basis, 2.0f, 1, 4);
    const std::vector<float> expected_axis1 = {
       -1.0f, -4.0f,
       -2.0f, -5.0f,
        3.0f,  6.0f,
    };
    CHECK(axis1 == expected_axis1);
    before_norm_sq = 0.0;
    after_norm_sq = 0.0;
    for (size_t index = 0; index < axis1.size(); ++index) {
        before_norm_sq += double(axis1_before[index]) * axis1_before[index];
        after_norm_sq += double(axis1[index]) * axis1[index];
    }
    CHECK(before_norm_sq == after_norm_sq);
    CHECK(std::abs(axis1_stats.tensor_norm_sq - 91.0) < 1e-9);
    CHECK(std::abs(axis1_stats.component_norm_sq - 46.0) < 1e-9);
}

static void test_quantization_residual_compensation() {
    const std::vector<float> direction = { 0.6f, 0.8f };

    // Axis 0: measure the residual in a decoded quantized buffer while
    // correcting a separate copy of the original projected F32 values.
    auto embedding = tensor_2d(2, 2);
    std::vector<float> embedding_target = {
         0.8f, -0.6f,
        -0.8f,  0.6f,
    };
    const std::vector<float> embedding_decoded = {
         1.1f, -0.2f, // target + 0.5 * direction
        -0.5f,  1.0f, // target + 0.5 * direction
    };
    std::vector<double> embedding_coefficients;
    const auto embedding_stats = llama_direction_remove_measured_component_f32(
            embedding_target.data(), embedding_decoded.data(), &embedding,
            direction, 1.0f, 0, 8, &embedding_coefficients);
    const std::vector<float> expected_embedding = {
         0.5f, -1.0f,
        -1.1f,  0.2f,
    };
    for (size_t index = 0; index < embedding_target.size(); ++index) {
        CHECK(close(embedding_target[index], expected_embedding[index]));
    }
    CHECK(std::abs(embedding_stats.component_norm_sq - 0.5) < 1e-6);
    CHECK(embedding_coefficients.size() == 2);
    CHECK(std::abs(embedding_coefficients[0] - 0.5) < 1e-6);
    CHECK(std::abs(embedding_coefficients[1] - 0.5) < 1e-6);

    // Axis 1: each input column has an independently measured coefficient.
    auto linear = tensor_2d(3, 2);
    std::vector<float> linear_target(6, 0.0f);
    const std::vector<float> linear_decoded = {
        3.0f,  0.8f, 6.0f,
        4.0f, -0.6f, 8.0f,
    };
    std::vector<double> linear_coefficients;
    const auto linear_stats = llama_direction_remove_measured_component_f32(
            linear_target.data(), linear_decoded.data(), &linear,
            direction, 1.0f, 1, 8, &linear_coefficients);
    const std::vector<float> expected_linear = {
        -3.0f, 0.0f, -6.0f,
        -4.0f, 0.0f, -8.0f,
    };
    for (size_t index = 0; index < linear_target.size(); ++index) {
        CHECK(close(linear_target[index], expected_linear[index]));
    }
    CHECK(std::abs(linear_stats.tensor_norm_sq - 126.0) < 1e-6);
    CHECK(std::abs(linear_stats.component_norm_sq - 125.0) < 1e-5);
    CHECK(linear_coefficients.size() == 3);
    CHECK(std::abs(linear_coefficients[0] - 5.0) < 1e-6);
    CHECK(std::abs(linear_coefficients[1]) < 1e-6);
    CHECK(std::abs(linear_coefficients[2] - 10.0) < 1e-6);
}

static void test_target_relative_residual_compensation() {
    const std::vector<float> direction = { 0.6f, 0.8f };
    const std::vector<std::vector<float>> basis = { direction };
    auto embedding = tensor_2d(2, 1);

    // At scale one the intended target has no selected component, so the new
    // target-relative measurement is exactly the legacy absolute measurement.
    const std::vector<float> scale1_target = { 0.0f, 0.0f };
    const std::vector<float> scale1_decoded = { 0.06f, 0.08f };
    auto legacy_correction = scale1_target;
    auto relative_correction = scale1_target;
    const auto legacy_stats = llama_direction_remove_measured_subspace_f32(
            legacy_correction.data(), scale1_decoded.data(), &embedding,
            basis, 0.5f, 0, 2);
    const auto relative_stats = llama_direction_remove_measured_subspace_difference_f32(
            relative_correction.data(), scale1_decoded.data(), scale1_target.data(),
            &embedding, basis, 0.5f, 0, 2);
    CHECK(legacy_correction == relative_correction);
    CHECK(legacy_stats.tensor_norm_sq == relative_stats.tensor_norm_sq);
    CHECK(legacy_stats.component_norm_sq == relative_stats.component_norm_sq);

    // At scale two the desired selected component is intentionally nonzero and
    // reversed. Measure only the encode/decode error around that target, then
    // apply a damped correction to a separate mutable F32 input.
    const std::vector<float> source = { 3.0f, 4.0f };
    auto intended = source;
    const auto source_stats = llama_direction_orthogonalize_subspace_f32(
            intended.data(), &embedding, basis, 2.0f, 0, 2);
    CHECK(close(intended[0], -3.0f));
    CHECK(close(intended[1], -4.0f));
    const std::vector<float> decoded = {
        intended[0] + 0.06f,
        intended[1] + 0.08f,
    };
    auto corrected = intended;
    std::vector<double> error_magnitudes;
    const auto error_stats = llama_direction_remove_measured_subspace_difference_f32(
            corrected.data(), decoded.data(), intended.data(), &embedding,
            basis, 0.5f, 0, 2, &error_magnitudes);
    CHECK(close(corrected[0], -3.03f));
    CHECK(close(corrected[1], -4.04f));
    CHECK(std::abs(error_stats.component_norm_sq - 0.01) < 1e-7);
    CHECK(error_magnitudes.size() == 1);
    CHECK(std::abs(error_magnitudes[0] - 0.1) < 1e-6);
    CHECK(std::abs(llama_direction_retained_component_ratio(
            source_stats, error_stats) - 0.02) < 1e-7);
}

static void test_subspace_projection() {
    const std::vector<std::vector<float>> basis = {
        { 1.0f, 0.0f, 0.0f },
        { 0.0f, 1.0f, 0.0f },
    };

    auto embedding = tensor_2d(3, 2);
    std::vector<float> axis0 = {
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f,
    };
    std::vector<double> magnitudes;
    const auto axis0_stats = llama_direction_remove_measured_subspace_f32(
            axis0.data(), axis0.data(), &embedding, basis, 1.0f, 0, 4, &magnitudes);
    const std::vector<float> expected_axis0 = {
        0.0f, 0.0f, 3.0f,
        0.0f, 0.0f, 6.0f,
    };
    CHECK(axis0 == expected_axis0);
    CHECK(std::abs(axis0_stats.tensor_norm_sq - 91.0) < 1e-9);
    CHECK(std::abs(axis0_stats.component_norm_sq - 46.0) < 1e-9);
    CHECK(magnitudes.size() == 2);
    CHECK(std::abs(magnitudes[0] - std::sqrt(5.0)) < 1e-9);
    CHECK(std::abs(magnitudes[1] - std::sqrt(41.0)) < 1e-9);

    auto linear = tensor_2d(2, 3);
    std::vector<float> axis1 = {
        1.0f, 4.0f,
        2.0f, 5.0f,
        3.0f, 6.0f,
    };
    const auto axis1_stats = llama_direction_orthogonalize_subspace_f32(
            axis1.data(), &linear, basis, 1.0f, 1, 4);
    const std::vector<float> expected_axis1 = {
        0.0f, 0.0f,
        0.0f, 0.0f,
        3.0f, 6.0f,
    };
    CHECK(axis1 == expected_axis1);
    CHECK(std::abs(axis1_stats.tensor_norm_sq - 91.0) < 1e-9);
    CHECK(std::abs(axis1_stats.component_norm_sq - 46.0) < 1e-9);
}

static void test_principal_subspace() {
    const std::vector<std::vector<float>> layers = {
        { 1.0f, 0.0f, 0.0f },
        { 2.0f, 0.0f, 0.0f }, // normalization makes this a second e0 sample
        { 0.0f, 3.0f, 0.0f },
    };
    const auto subspace = llama_direction_principal_subspace(layers, 2);
    CHECK(subspace.basis.size() == 2);
    CHECK(subspace.eigenvalues.size() == 2);
    CHECK(std::abs(subspace.eigenvalues[0] - 2.0) < 1e-9);
    CHECK(std::abs(subspace.eigenvalues[1] - 1.0) < 1e-9);
    CHECK(std::abs(subspace.captured_energy_fraction - 1.0) < 1e-12);
    CHECK(std::abs(std::abs(subspace.basis[0][0]) - 1.0f) < 1e-6f);
    CHECK(std::abs(std::abs(subspace.basis[1][1]) - 1.0f) < 1e-6f);
    double dot = 0.0;
    for (size_t index = 0; index < subspace.basis[0].size(); ++index) {
        dot += double(subspace.basis[0][index]) * subspace.basis[1][index];
    }
    CHECK(std::abs(dot) < 1e-7);

    bool rejected_excess_rank = false;
    try {
        (void) llama_direction_principal_subspace(layers, 3);
    } catch (const std::runtime_error &) {
        rejected_excess_rank = true;
    }
    CHECK(rejected_excess_rank);

    // Exercise the Jacobi rotations with a non-diagonal covariance matrix.
    // The normalized rows below produce X^T X = [[1.64, 0.48],
    // [0.48, 1.36]], whose eigenpairs are (2, [0.8, 0.6]) and
    // (1, [-0.6, 0.8]).
    const std::vector<std::vector<float>> rotated_layers = {
        { 1.0f, 0.0f },
        { 0.8f, 0.6f },
        { 0.0f, 1.0f },
    };
    const auto rotated = llama_direction_principal_subspace(rotated_layers, 2);
    CHECK(std::abs(rotated.eigenvalues[0] - 2.0) < 1e-9);
    CHECK(std::abs(rotated.eigenvalues[1] - 1.0) < 1e-9);
    CHECK(std::abs(rotated.captured_energy_fraction - 1.0) < 1e-12);
    CHECK(close(rotated.basis[0][0],  0.8f));
    CHECK(close(rotated.basis[0][1],  0.6f));
    CHECK(close(rotated.basis[1][0], -0.6f));
    CHECK(close(rotated.basis[1][1],  0.8f));
}

int main() {
    test_axis_0();
    test_axis_1_and_slices();
    test_scale_and_thread_equivalence();
    test_measurement_only_and_zero_tensor();
    test_projection_invariants();
    test_counterfactual_reflection();
    test_quantization_residual_compensation();
    test_target_relative_residual_compensation();
    test_subspace_projection();
    test_principal_subspace();
    std::puts("direction projection tests passed");
    return 0;
}
