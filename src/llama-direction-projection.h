#ifndef LLAMA_DIRECTION_PROJECTION_H
#define LLAMA_DIRECTION_PROJECTION_H

#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <thread>
#include <vector>

struct llama_direction_projection_stats {
    double tensor_norm_sq = 0.0;
    double component_norm_sq = 0.0;
};

inline double llama_direction_retained_component_ratio(
        const llama_direction_projection_stats & source,
        const llama_direction_projection_stats & post_quant) {
    GGML_ASSERT(source.component_norm_sq > 0.0);
    GGML_ASSERT(post_quant.component_norm_sq >= 0.0);
    return std::sqrt(post_quant.component_norm_sq / source.component_norm_sq);
}

// Measure the direction component in `measured` and remove it from `target`.
// The buffers may alias. Keeping them separate lets the quantizer measure the
// component reintroduced by an encode/decode pass and pre-compensate the
// original projected F32 values without requantizing a lossy decode.
//
// Dequantization gives us a dense row-major F32 buffer regardless of the source
// GGML type. axis 0 is used for token embeddings ([n_embd, n_token]); axis 1 is
// used for linear maps whose output is the residual stream ([n_input, n_embd]).
inline llama_direction_projection_stats llama_direction_remove_measured_component_f32(
        float * target,
        const float * measured,
        const ggml_tensor * tensor,
        const std::vector<float> & direction,
        float scale,
        int axis,
        int nthread,
        std::vector<double> * component_coefficients = nullptr) {
    const int64_t ne0 = tensor->ne[0];
    const int64_t ne1 = tensor->ne[1];
    const int64_t n_slices = tensor->ne[2] * tensor->ne[3];
    llama_direction_projection_stats result;

    auto run_parallel = [](int workers, const auto & fn) {
        if (workers <= 1) {
            fn(0, 1);
            return;
        }
        std::vector<std::thread> threads;
        threads.reserve(workers);
        for (int worker = 0; worker < workers; ++worker) {
            threads.emplace_back(fn, worker, workers);
        }
        for (auto & thread : threads) {
            thread.join();
        }
    };

    if (axis == 0) {
        GGML_ASSERT(ne0 == (int64_t) direction.size());
        const int64_t n_rows = ne1 * n_slices;
        if (component_coefficients) {
            component_coefficients->assign(n_rows, 0.0);
        }
        const int workers = (int) std::max<int64_t>(1, std::min<int64_t>(nthread, n_rows));
        std::vector<llama_direction_projection_stats> partial(workers);
        run_parallel(workers, [&](int worker, int n_workers) {
            const int64_t row_begin = n_rows * worker / n_workers;
            const int64_t row_end   = n_rows * (worker + 1) / n_workers;
            auto & stats = partial[worker];
            for (int64_t row = row_begin; row < row_end; ++row) {
                float * target_values = target + row * ne0;
                const float * measured_values = measured + row * ne0;
                double dot = 0.0;
                for (int64_t col = 0; col < ne0; ++col) {
                    dot += double(measured_values[col]) * direction[col];
                    stats.tensor_norm_sq += double(measured_values[col]) * measured_values[col];
                }
                stats.component_norm_sq += dot * dot;
                if (component_coefficients) {
                    (*component_coefficients)[row] = dot;
                }
                const float coeff = scale * dot;
                for (int64_t col = 0; col < ne0; ++col) {
                    target_values[col] -= coeff * direction[col];
                }
            }
        });
        for (const auto & stats : partial) {
            result.tensor_norm_sq += stats.tensor_norm_sq;
            result.component_norm_sq += stats.component_norm_sq;
        }
        return result;
    }

    GGML_ASSERT(axis == 1);
    GGML_ASSERT(ne1 == (int64_t) direction.size());
    if (component_coefficients) {
        component_coefficients->assign(n_slices * ne0, 0.0);
    }
    const int workers = (int) std::max<int64_t>(1, std::min<int64_t>(nthread, ne1));
    for (int64_t slice = 0; slice < n_slices; ++slice) {
        float * target_matrix = target + slice * ne0 * ne1;
        const float * measured_matrix = measured + slice * ne0 * ne1;
        std::vector<std::vector<double>> partial_coeffs(workers, std::vector<double>(ne0, 0.0));
        std::vector<double> partial_norms(workers, 0.0);

        // Each worker reads whole contiguous output rows and contributes a
        // private coefficient vector. This avoids a cache-hostile column walk.
        run_parallel(workers, [&](int worker, int n_workers) {
            const int64_t row_begin = ne1 * worker / n_workers;
            const int64_t row_end   = ne1 * (worker + 1) / n_workers;
            auto & coeffs = partial_coeffs[worker];
            double norm_sq = 0.0;
            for (int64_t row = row_begin; row < row_end; ++row) {
                const float * values = measured_matrix + row * ne0;
                const double dir = direction[row];
                for (int64_t col = 0; col < ne0; ++col) {
                    coeffs[col] += dir * values[col];
                    norm_sq += double(values[col]) * values[col];
                }
            }
            partial_norms[worker] = norm_sq;
        });

        std::vector<double> coeffs(ne0, 0.0);
        for (int worker = 0; worker < workers; ++worker) {
            result.tensor_norm_sq += partial_norms[worker];
            for (int64_t col = 0; col < ne0; ++col) {
                coeffs[col] += partial_coeffs[worker][col];
            }
        }
        for (double coeff : coeffs) {
            result.component_norm_sq += coeff * coeff;
        }
        if (component_coefficients) {
            std::copy(coeffs.begin(), coeffs.end(),
                    component_coefficients->begin() + slice * ne0);
        }

        run_parallel(workers, [&](int worker, int n_workers) {
            const int64_t row_begin = ne1 * worker / n_workers;
            const int64_t row_end   = ne1 * (worker + 1) / n_workers;
            for (int64_t row = row_begin; row < row_end; ++row) {
                float * values = target_matrix + row * ne0;
                const double row_scale = scale * direction[row];
                for (int64_t col = 0; col < ne0; ++col) {
                    values[col] -= row_scale * coeffs[col];
                }
            }
        });
    }
    return result;
}

// Remove scale * r r^T from a tensor's own residual-stream axis.
inline llama_direction_projection_stats llama_direction_orthogonalize_f32(
        float * data,
        const ggml_tensor * tensor,
        const std::vector<float> & direction,
        float scale,
        int axis,
        int nthread) {
    return llama_direction_remove_measured_component_f32(
            data, data, tensor, direction, scale, axis, nthread);
}

// Measure and remove an orthonormal direction basis. Component energy is the
// sum of squared coefficients across the basis. When requested, per-vector
// coefficient arrays are reduced to one Euclidean magnitude per tensor row or
// input column; the quantizer uses those magnitudes to retain the best encoded
// embedding row across correction passes.
inline llama_direction_projection_stats llama_direction_remove_measured_subspace_f32(
        float * target,
        const float * measured,
        const ggml_tensor * tensor,
        const std::vector<std::vector<float>> & basis,
        float scale,
        int axis,
        int nthread,
        std::vector<double> * component_magnitudes = nullptr) {
    GGML_ASSERT(!basis.empty());
    llama_direction_projection_stats result;
    std::vector<double> magnitude_sq;
    if (component_magnitudes) {
        const int64_t count = axis == 0
                ? tensor->ne[1] * tensor->ne[2] * tensor->ne[3]
                : tensor->ne[0] * tensor->ne[2] * tensor->ne[3];
        magnitude_sq.assign(count, 0.0);
    }

    bool first = true;
    for (const auto & direction : basis) {
        std::vector<double> coefficients;
        const auto stats = llama_direction_remove_measured_component_f32(
                target, measured, tensor, direction, scale, axis, nthread,
                component_magnitudes ? &coefficients : nullptr);
        if (first) {
            // If target and measured alias, later basis projections reduce the
            // target norm. The first pass is the norm of the original tensor.
            result.tensor_norm_sq = stats.tensor_norm_sq;
            first = false;
        }
        result.component_norm_sq += stats.component_norm_sq;
        if (component_magnitudes) {
            GGML_ASSERT(coefficients.size() == magnitude_sq.size());
            for (size_t index = 0; index < coefficients.size(); ++index) {
                magnitude_sq[index] += coefficients[index] * coefficients[index];
            }
        }
    }

    if (component_magnitudes) {
        component_magnitudes->resize(magnitude_sq.size());
        for (size_t index = 0; index < magnitude_sq.size(); ++index) {
            (*component_magnitudes)[index] = std::sqrt(magnitude_sq[index]);
        }
    }
    return result;
}

inline llama_direction_projection_stats llama_direction_orthogonalize_subspace_f32(
        float * data,
        const ggml_tensor * tensor,
        const std::vector<std::vector<float>> & basis,
        float scale,
        int axis,
        int nthread) {
    return llama_direction_remove_measured_subspace_f32(
            data, data, tensor, basis, scale, axis, nthread);
}

#endif // LLAMA_DIRECTION_PROJECTION_H
