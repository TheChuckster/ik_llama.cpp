#pragma once

#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

inline bool llama_control_vector_is_unit_f32(
        const float * data, size_t count, double tolerance = 1e-6) {
    if (data == nullptr || count == 0 || !std::isfinite(tolerance) || tolerance < 0.0) {
        return false;
    }
    double squared_norm = 0.0;
    for (size_t index = 0; index < count; ++index) {
        if (!std::isfinite(data[index])) {
            return false;
        }
        squared_norm += (double) data[index] * data[index];
    }
    const double norm = std::sqrt(squared_norm);
    return std::isfinite(norm) && std::abs(norm - 1.0) <= tolerance;
}

inline bool llama_control_vector_is_orthonormal_f32(
        const float * data,
        size_t n_embd,
        size_t rank,
        double unit_tolerance = 1e-6,
        double gram_tolerance = 2e-6) {
    if (data == nullptr || n_embd == 0 || rank == 0
            || !std::isfinite(unit_tolerance) || unit_tolerance < 0.0
            || !std::isfinite(gram_tolerance) || gram_tolerance < 0.0) {
        return false;
    }
    for (size_t row = 0; row < rank; ++row) {
        if (!llama_control_vector_is_unit_f32(
                data + row * n_embd, n_embd, unit_tolerance)) {
            return false;
        }
        for (size_t previous = 0; previous < row; ++previous) {
            double dot = 0.0;
            for (size_t column = 0; column < n_embd; ++column) {
                dot += (double) data[row * n_embd + column]
                    * data[previous * n_embd + column];
            }
            if (!std::isfinite(dot) || std::abs(dot) > gram_tolerance) {
                return false;
            }
        }
    }
    return true;
}

inline bool llama_control_vector_offset_in_span_f32(
        const float * basis,
        const float * offset,
        size_t n_embd,
        size_t rank,
        double tolerance = 1e-5) {
    if (basis == nullptr || offset == nullptr || n_embd == 0 || rank == 0
            || !std::isfinite(tolerance) || tolerance < 0.0) {
        return false;
    }
    std::vector<double> coefficients(rank, 0.0);
    double offset_squared_norm = 0.0;
    for (size_t column = 0; column < n_embd; ++column) {
        if (!std::isfinite(offset[column])) {
            return false;
        }
        offset_squared_norm += (double) offset[column] * offset[column];
        for (size_t row = 0; row < rank; ++row) {
            coefficients[row] += (double) basis[row * n_embd + column]
                * offset[column];
        }
    }
    double residual_squared_norm = 0.0;
    for (size_t column = 0; column < n_embd; ++column) {
        double reconstructed = 0.0;
        for (size_t row = 0; row < rank; ++row) {
            reconstructed += coefficients[row] * basis[row * n_embd + column];
        }
        const double residual = offset[column] - reconstructed;
        residual_squared_norm += residual * residual;
    }
    const double denominator = std::max(
        std::sqrt(offset_squared_norm), std::numeric_limits<double>::min());
    const double relative_residual = std::sqrt(residual_squared_norm) / denominator;
    return std::isfinite(relative_residual) && relative_residual <= tolerance;
}

inline ggml_tensor * llama_control_vector_parallel_component(
        ggml_context * ctx, ggml_tensor * cur, ggml_tensor * unit_direction) {
    ggml_tensor * repeated_direction = ggml_repeat(ctx, unit_direction, cur);
    ggml_tensor * coefficient = ggml_sum_rows(
        ctx, ggml_mul(ctx, cur, repeated_direction));
    ggml_tensor * repeated_coefficient = ggml_repeat(ctx, coefficient, cur);
    return ggml_mul(ctx, repeated_direction, repeated_coefficient);
}

inline ggml_tensor * llama_control_vector_project(
        ggml_context * ctx, ggml_tensor * cur, ggml_tensor * unit_direction) {
    return ggml_sub(
        ctx, cur,
        llama_control_vector_parallel_component(ctx, cur, unit_direction));
}

inline ggml_tensor * llama_control_vector_project_subspace(
        ggml_context * ctx,
        ggml_tensor * cur,
        ggml_tensor * basis,
        int32_t rank) {
    GGML_ASSERT(ctx != nullptr);
    GGML_ASSERT(cur != nullptr);
    GGML_ASSERT(basis != nullptr);
    GGML_ASSERT(rank > 0);
    GGML_ASSERT(basis->ne[0] == cur->ne[0]);
    GGML_ASSERT(basis->ne[1] == rank);
    ggml_tensor * parallel = nullptr;
    for (int32_t row = 0; row < rank; ++row) {
        ggml_tensor * direction = ggml_view_1d(
            ctx, basis, basis->ne[0], row * basis->nb[1]);
        ggml_tensor * component = llama_control_vector_parallel_component(
            ctx, cur, direction);
        parallel = parallel == nullptr ? component : ggml_add(ctx, parallel, component);
    }
    GGML_ASSERT(parallel != nullptr);
    return ggml_sub(ctx, cur, parallel);
}
