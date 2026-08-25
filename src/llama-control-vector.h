#pragma once

#include "ggml.h"

#include <cmath>
#include <cstddef>

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

inline ggml_tensor * llama_control_vector_project(
        ggml_context * ctx, ggml_tensor * cur, ggml_tensor * unit_direction) {
    ggml_tensor * repeated_direction = ggml_repeat(ctx, unit_direction, cur);
    ggml_tensor * coefficient = ggml_sum_rows(
        ctx, ggml_mul(ctx, cur, repeated_direction));
    ggml_tensor * repeated_coefficient = ggml_repeat(ctx, coefficient, cur);
    ggml_tensor * parallel = ggml_mul(
        ctx, repeated_direction, repeated_coefficient);
    return ggml_sub(ctx, cur, parallel);
}
