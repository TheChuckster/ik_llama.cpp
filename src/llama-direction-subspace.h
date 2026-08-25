#ifndef LLAMA_DIRECTION_SUBSPACE_H
#define LLAMA_DIRECTION_SUBSPACE_H

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numeric>
#include <stdexcept>
#include <vector>

struct llama_direction_subspace {
    std::vector<std::vector<float>> basis;
    std::vector<double> eigenvalues;
    double captured_energy_fraction = 0.0;
};

// Return the leading right-singular vectors of a small collection of layer
// directions. Each input is normalized before the sample-space Gram matrix is
// formed. A cyclic Jacobi eigensolver keeps this dependency-free and makes the
// quantizer usable on CPU-only hosts without a BLAS/LAPACK installation.
inline llama_direction_subspace llama_direction_principal_subspace(
        const std::vector<std::vector<float>> & input,
        size_t rank) {
    if (input.empty()) {
        throw std::invalid_argument("direction subspace input is empty");
    }
    if (rank == 0 || rank > input.size()) {
        throw std::invalid_argument("direction subspace rank is outside the input range");
    }

    const size_t n = input.size();
    const size_t width = input.front().size();
    if (width == 0) {
        throw std::invalid_argument("direction subspace vectors are empty");
    }

    std::vector<std::vector<double>> vectors(n, std::vector<double>(width));
    for (size_t row = 0; row < n; ++row) {
        if (input[row].size() != width) {
            throw std::invalid_argument("direction subspace vectors have inconsistent widths");
        }
        double norm_sq = 0.0;
        for (size_t col = 0; col < width; ++col) {
            const double value = input[row][col];
            if (!std::isfinite(value)) {
                throw std::invalid_argument("direction subspace contains a non-finite value");
            }
            vectors[row][col] = value;
            norm_sq += value * value;
        }
        if (!(norm_sq > 0.0) || !std::isfinite(norm_sq)) {
            throw std::invalid_argument("direction subspace contains a zero-norm vector");
        }
        const double inv_norm = 1.0 / std::sqrt(norm_sq);
        for (double & value : vectors[row]) {
            value *= inv_norm;
        }
    }

    std::vector<std::vector<double>> gram(n, std::vector<double>(n, 0.0));
    for (size_t left = 0; left < n; ++left) {
        for (size_t right = left; right < n; ++right) {
            double value = 0.0;
            for (size_t col = 0; col < width; ++col) {
                value += vectors[left][col] * vectors[right][col];
            }
            gram[left][right] = value;
            gram[right][left] = value;
        }
    }

    std::vector<std::vector<double>> eigenvectors(n, std::vector<double>(n, 0.0));
    for (size_t index = 0; index < n; ++index) {
        eigenvectors[index][index] = 1.0;
    }

    constexpr int max_sweeps = 100;
    constexpr double relative_tolerance = 1e-13;
    bool converged = false;
    for (int sweep = 0; sweep < max_sweeps; ++sweep) {
        double max_diagonal = 0.0;
        double max_off_diagonal = 0.0;
        for (size_t row = 0; row < n; ++row) {
            max_diagonal = std::max(max_diagonal, std::abs(gram[row][row]));
            for (size_t col = row + 1; col < n; ++col) {
                max_off_diagonal = std::max(max_off_diagonal, std::abs(gram[row][col]));
            }
        }
        if (max_off_diagonal <= relative_tolerance * std::max(1.0, max_diagonal)) {
            converged = true;
            break;
        }

        for (size_t p = 0; p < n; ++p) {
            for (size_t q = p + 1; q < n; ++q) {
                const double apq = gram[p][q];
                const double local_scale = std::sqrt(
                        std::max(0.0, std::abs(gram[p][p] * gram[q][q])));
                if (std::abs(apq) <= relative_tolerance * std::max(1.0, local_scale)) {
                    continue;
                }

                const double app = gram[p][p];
                const double aqq = gram[q][q];
                const double angle = 0.5 * std::atan2(2.0 * apq, aqq - app);
                const double cosine = std::cos(angle);
                const double sine = std::sin(angle);

                for (size_t k = 0; k < n; ++k) {
                    if (k == p || k == q) {
                        continue;
                    }
                    const double akp = gram[k][p];
                    const double akq = gram[k][q];
                    gram[k][p] = gram[p][k] = cosine * akp - sine * akq;
                    gram[k][q] = gram[q][k] = sine * akp + cosine * akq;
                }
                gram[p][p] = cosine * cosine * app
                        - 2.0 * sine * cosine * apq + sine * sine * aqq;
                gram[q][q] = sine * sine * app
                        + 2.0 * sine * cosine * apq + cosine * cosine * aqq;
                gram[p][q] = gram[q][p] = 0.0;

                for (size_t k = 0; k < n; ++k) {
                    const double vkp = eigenvectors[k][p];
                    const double vkq = eigenvectors[k][q];
                    eigenvectors[k][p] = cosine * vkp - sine * vkq;
                    eigenvectors[k][q] = sine * vkp + cosine * vkq;
                }
            }
        }
    }
    if (!converged) {
        throw std::runtime_error("direction subspace eigensolver did not converge");
    }

    std::vector<size_t> order(n);
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](size_t left, size_t right) {
        return gram[left][left] > gram[right][right];
    });

    double total_energy = 0.0;
    for (size_t index = 0; index < n; ++index) {
        total_energy += std::max(0.0, gram[index][index]);
    }
    const double maximum_eigenvalue = std::max(0.0, gram[order.front()][order.front()]);
    if (!(total_energy > 0.0) || !(maximum_eigenvalue > 0.0)) {
        throw std::runtime_error("direction subspace has no positive energy");
    }

    llama_direction_subspace result;
    result.basis.reserve(rank);
    result.eigenvalues.reserve(rank);
    double captured_energy = 0.0;
    for (size_t component = 0; component < rank; ++component) {
        const size_t index = order[component];
        const double eigenvalue = std::max(0.0, gram[index][index]);
        if (!(eigenvalue > maximum_eigenvalue * 1e-10)) {
            throw std::runtime_error("requested direction subspace rank exceeds numerical rank");
        }

        std::vector<double> principal(width, 0.0);
        for (size_t row = 0; row < n; ++row) {
            const double coefficient = eigenvectors[row][index] / std::sqrt(eigenvalue);
            for (size_t col = 0; col < width; ++col) {
                principal[col] += coefficient * vectors[row][col];
            }
        }

        // Reorthogonalize twice to bound accumulated Jacobi/F32 conversion
        // error. Projection statistics assume an orthonormal basis.
        for (int pass = 0; pass < 2; ++pass) {
            for (const auto & previous : result.basis) {
                double dot = 0.0;
                for (size_t col = 0; col < width; ++col) {
                    dot += principal[col] * previous[col];
                }
                for (size_t col = 0; col < width; ++col) {
                    principal[col] -= dot * previous[col];
                }
            }
        }
        double norm_sq = 0.0;
        for (double value : principal) {
            norm_sq += value * value;
        }
        if (!(norm_sq > 0.0) || !std::isfinite(norm_sq)) {
            throw std::runtime_error("direction subspace produced a degenerate basis vector");
        }
        const double inv_norm = 1.0 / std::sqrt(norm_sq);
        std::vector<float> basis_vector(width);
        size_t sign_index = 0;
        for (size_t col = 0; col < width; ++col) {
            basis_vector[col] = static_cast<float>(principal[col] * inv_norm);
            if (std::abs(basis_vector[col]) > std::abs(basis_vector[sign_index])) {
                sign_index = col;
            }
        }
        if (basis_vector[sign_index] < 0.0f) {
            for (float & value : basis_vector) {
                value = -value;
            }
        }
        result.basis.push_back(std::move(basis_vector));
        result.eigenvalues.push_back(eigenvalue);
        captured_energy += eigenvalue;
    }
    result.captured_energy_fraction = captured_energy / total_energy;
    return result;
}

#endif // LLAMA_DIRECTION_SUBSPACE_H
