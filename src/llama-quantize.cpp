#include "llama-impl.h"
#include "llama-model.h"
#include "llama-model-loader.h"
#include "llama-quantize.h"
#include "llama-direction-projection.h"

#include "ggml.h"
#include "ggml-common.h"

#include "iqk/iqk_quantize.h"

#include <thread>
#include <regex>
#include <mutex>
#include <fstream>
#include <filesystem>
#include <algorithm>
#include <cmath>
#include <limits>
#include <unordered_map>

//
// quantization
//

// TODO: replace with ggml API call
#define QK_K 256
#define QK_IQ1BN 64

#if defined(_WIN32)
    #define WIN32_LEAN_AND_MEAN
    #ifndef NOMINMAX
        #define NOMINMAX
    #endif
    #include <windows.h>
    #ifndef PATH_MAX
        #define PATH_MAX MAX_PATH
    #endif
    #include <io.h>
#endif

static void zeros(std::ofstream & file, size_t n) {
    char zero = 0;
    for (size_t i = 0; i < n; ++i) {
        file.write(&zero, 1);
    }
}

static void ensure_output_directory(const std::string & filepath) {
    std::filesystem::path p(filepath);
    if (p.has_parent_path()) {
        std::error_code ec;
        std::filesystem::create_directories(p.parent_path(), ec);
        if (ec) {
            fprintf(stderr, "Failed to create directory '%s': %s\n", p.parent_path().string().c_str(), ec.message().c_str());
            exit(EXIT_FAILURE);
        }
    }
}

struct quantize_state_internal {
    const llama_model                 & model;
    const llama_model_quantize_params * params;

    int n_attention_wv    = 0;
    int n_ffn_down        = 0;
    int n_ffn_gate        = 0;
    int n_ffn_up          = 0;
    int i_attention_wv    = 0;
    int i_ffn_down        = 0;
    int i_ffn_gate        = 0;
    int i_ffn_up          = 0;

    int n_k_quantized     = 0;
    int n_fallback        = 0;

    bool has_imatrix      = false;

    // used to figure out if a model shares tok_embd with the output weight
    bool has_output       = false;

    quantize_state_internal(const llama_model & model, const llama_model_quantize_params * params)
        : model(model)
        , params(params)
        {}
};

std::pair<ggml_type, int> interleaved_properties(ggml_type type) {
    static std::unordered_map<ggml_type, std::pair<ggml_type, int>> k_map = {
        { GGML_TYPE_Q4_0_4_4,    { GGML_TYPE_Q4_0, 4} },
        { GGML_TYPE_Q4_0_4_8,    { GGML_TYPE_Q4_0, 4} },
        { GGML_TYPE_Q4_0_8_8,    { GGML_TYPE_Q4_0, 8} },
        { GGML_TYPE_Q4_0_R8,     { GGML_TYPE_Q4_0, 8} },
        { GGML_TYPE_Q5_0_R4,     { GGML_TYPE_Q5_0, 4} },
        { GGML_TYPE_Q6_0_R4,     { GGML_TYPE_Q6_0, 4} },
        { GGML_TYPE_Q8_0_R8,     { GGML_TYPE_Q8_0, 8} },
        { GGML_TYPE_Q2_K_R4,     { GGML_TYPE_Q2_K, 4} },
        { GGML_TYPE_Q3_K_R4,     { GGML_TYPE_Q3_K, 4} },
        { GGML_TYPE_Q4_K_R4,     { GGML_TYPE_Q4_K, 4} },
        { GGML_TYPE_Q5_K_R4,     { GGML_TYPE_Q5_K, 4} },
        { GGML_TYPE_Q6_K_R4,     { GGML_TYPE_Q6_K, 4} },
        { GGML_TYPE_IQ2_XXS_R4,  { GGML_TYPE_IQ2_XXS, 4} },
        { GGML_TYPE_IQ2_XS_R4,   { GGML_TYPE_IQ2_XS, 4} },
        { GGML_TYPE_IQ2_S_R4,    { GGML_TYPE_IQ2_S, 4} },
        { GGML_TYPE_IQ3_XXS_R4,  { GGML_TYPE_IQ3_XXS, 4} },
        { GGML_TYPE_IQ3_S_R4,    { GGML_TYPE_IQ3_S, 4} },
        { GGML_TYPE_IQ4_XS_R8,   { GGML_TYPE_IQ4_XS, 8} },
        { GGML_TYPE_IQ4_NL_R4,   { GGML_TYPE_IQ4_NL, 4} },
        { GGML_TYPE_IQ1_S_R4,    { GGML_TYPE_IQ1_S, 4} },
        { GGML_TYPE_IQ1_M_R4,    { GGML_TYPE_IQ1_M, 4} },
        { GGML_TYPE_IQ2_BN_R4,   { GGML_TYPE_IQ2_BN, 4} },
        { GGML_TYPE_IQ2_K_R4,    { GGML_TYPE_IQ2_K, 4} },
        { GGML_TYPE_IQ3_K_R4,    { GGML_TYPE_IQ3_K, 4} },
        { GGML_TYPE_IQ4_K_R4,    { GGML_TYPE_IQ4_K, 4} },
        { GGML_TYPE_IQ4_KS_R4,   { GGML_TYPE_IQ4_KS, 4} },
        { GGML_TYPE_IQ5_KS_R4,   { GGML_TYPE_IQ5_KS, 4} },
        { GGML_TYPE_IQ5_K_R4,    { GGML_TYPE_IQ5_K, 4} },
        { GGML_TYPE_MXFP4_R8,    { GGML_TYPE_MXFP4, 8} },
        { GGML_TYPE_Q8_KV_R8,    { GGML_TYPE_Q8_KV, 8} },
        { GGML_TYPE_Q8_K_R8,     { GGML_TYPE_Q8_0, 8} },
        { GGML_TYPE_BF16_R16,    { GGML_TYPE_BF16, 16} },
    };
    if (auto it = k_map.find(type); it != k_map.end()) return it->second;
    return {type, 1};
}

static void llama_tensor_dequantize_internal(
    struct ggml_tensor * tensor, std::vector<no_init<float>> & output, std::vector<std::thread> & workers,
    const size_t nelements, const int nthread
) {
    if (output.size() < nelements) {
        output.resize(nelements);
    }
    float * f32_output = (float *) output.data();

    ggml_type_traits_t qtype;
    if (ggml_is_quantized(tensor->type)) {
        qtype = ggml_internal_get_type_traits(tensor->type);
        if (qtype.to_float == NULL) {
            throw std::runtime_error(format("type %s unsupported for integer quantization: no dequantization available", ggml_type_name(tensor->type)));
        }
    } else if (tensor->type != GGML_TYPE_F16 &&
               tensor->type != GGML_TYPE_BF16) {
        throw std::runtime_error(format("cannot dequantize/convert tensor type %s", ggml_type_name(tensor->type)));
    }

    if (tensor->type == GGML_TYPE_I2_S) {
        // we need to dequantize the entire tensor for I2_S
        qtype.to_float(tensor->data, f32_output, nelements);
        return;
    }

    if (nthread < 2 || (ggml_is_quantized(tensor->type) && qtype.row_meta_size > 0)) {
        if (tensor->type == GGML_TYPE_F16) {
            ggml_fp16_to_fp32_row((ggml_fp16_t *)tensor->data, f32_output, nelements);
        } else if (tensor->type == GGML_TYPE_BF16) {
            ggml_bf16_to_fp32_row((ggml_bf16_t *)tensor->data, f32_output, nelements);
        } else if (ggml_is_quantized(tensor->type)) {
            auto row_size = ggml_row_size(tensor->type, tensor->ne[0]);
            int nrows = ggml_nrows(tensor);
            auto qsrc = (const char *)tensor->data;
            auto num_rows = interleaved_properties(tensor->type).second;
            for (int row = 0; row < nrows; row += num_rows) {
                qtype.to_float(qsrc, f32_output, num_rows*tensor->ne[0]);
                qsrc += num_rows*row_size;
                f32_output += num_rows*tensor->ne[0];
            }
        } else {
            GGML_ABORT("fatal error"); // unreachable
        }
        return;
    }

    auto num_rows = interleaved_properties(tensor->type).second;
    if (num_rows > 1) {
        int nrows = ggml_nrows(tensor);
        auto row_size = ggml_row_size(tensor->type, tensor->ne[0]);
        auto qsrc = (const char *)tensor->data;
        for (int row = 0; row < nrows; row += num_rows) {
            qtype.to_float(qsrc, f32_output, num_rows*tensor->ne[0]);
            qsrc += num_rows*row_size;
            f32_output += num_rows*tensor->ne[0];
        }
        return;
    }

    size_t block_size;
    if (tensor->type == GGML_TYPE_F16 ||
        tensor->type == GGML_TYPE_BF16) {
        block_size = 1;
    } else {
        block_size = (size_t)ggml_blck_size(tensor->type);
    }

    size_t block_size_bytes = ggml_type_size(tensor->type);

    GGML_ASSERT(nelements % block_size == 0);
    size_t nblocks = nelements / block_size;
    size_t blocks_per_thread = nblocks / nthread;
    size_t spare_blocks = nblocks - (blocks_per_thread * nthread); // if blocks aren't divisible by thread count

    size_t in_buff_offs = 0;
    size_t out_buff_offs = 0;

    for (int tnum = 0; tnum < nthread; tnum++) {
        size_t thr_blocks = blocks_per_thread + (tnum == nthread - 1 ? spare_blocks : 0); // num blocks for this thread
        size_t thr_elems = thr_blocks * block_size; // number of elements for this thread
        size_t thr_block_bytes = thr_blocks * block_size_bytes; // number of input bytes for this thread

        auto compute = [qtype] (ggml_type typ, uint8_t * inbuf, float * outbuf, int nels) {
            if (typ == GGML_TYPE_F16) {
                ggml_fp16_to_fp32_row((ggml_fp16_t *)inbuf, outbuf, nels);
            } else if (typ == GGML_TYPE_BF16) {
                ggml_bf16_to_fp32_row((ggml_bf16_t *)inbuf, outbuf, nels);
            } else {
                qtype.to_float(inbuf, outbuf, nels);
            }
        };
        workers.emplace_back(compute, tensor->type, (uint8_t *) tensor->data + in_buff_offs, f32_output + out_buff_offs, thr_elems);
        in_buff_offs += thr_block_bytes;
        out_buff_offs += thr_elems;
    }
    for (auto & w : workers) { w.join(); }
    workers.clear();
}

static ggml_type change_type_if_necessary(ggml_type new_type, int nx, int ny) {
    bool convert_incompatible_tensor = false;
    if (new_type == GGML_TYPE_Q2_K    || new_type == GGML_TYPE_Q3_K    || new_type == GGML_TYPE_Q4_K   ||
        new_type == GGML_TYPE_Q5_K    || new_type == GGML_TYPE_Q6_K    || new_type == GGML_TYPE_IQ4_XS ||
        new_type == GGML_TYPE_IQ2_XS  || new_type == GGML_TYPE_IQ2_XXS || new_type == GGML_TYPE_IQ2_S  ||
        new_type == GGML_TYPE_IQ3_XXS || new_type == GGML_TYPE_IQ1_S   || new_type == GGML_TYPE_IQ3_S  ||
        new_type == GGML_TYPE_IQ1_M   || new_type == GGML_TYPE_IQ4_K   || new_type == GGML_TYPE_IQ2_K  ||
        new_type == GGML_TYPE_IQ5_K   || new_type == GGML_TYPE_IQ3_K   || new_type == GGML_TYPE_Q4_K_R4 ||
        new_type == GGML_TYPE_IQ6_K   || new_type == GGML_TYPE_IQ4_KS  || new_type == GGML_TYPE_IQ4_XS_R8 ||
        new_type == GGML_TYPE_IQ2_KS  || new_type == GGML_TYPE_IQ4_KSS || new_type == GGML_TYPE_Q6_K_R4 ||
        new_type == GGML_TYPE_Q5_K_R4 || new_type == GGML_TYPE_Q3_K_R4 || new_type == GGML_TYPE_Q2_K_R4 ||
        new_type == GGML_TYPE_IQ4_K_R4|| new_type == GGML_TYPE_Q8_K_R8 || new_type == GGML_TYPE_IQ3_K_R4||
        new_type == GGML_TYPE_IQ2_K_R4|| new_type == GGML_TYPE_IQ5_K_R4|| new_type == GGML_TYPE_IQ4_KS_R4 ||
        new_type == GGML_TYPE_IQ3_XXS_R4 || new_type == GGML_TYPE_IQ2_XXS_R4 || new_type == GGML_TYPE_IQ2_XS_R4 ||
        new_type == GGML_TYPE_IQ2_S_R4|| new_type == GGML_TYPE_IQ3_S_R4|| new_type == GGML_TYPE_IQ3_KS ||
        new_type == GGML_TYPE_IQ2_KT  || new_type == GGML_TYPE_IQ3_KT  || new_type == GGML_TYPE_IQ4_KT ||
        new_type == GGML_TYPE_IQ5_KS || new_type == GGML_TYPE_IQ5_KS_R4|| new_type == GGML_TYPE_IQ2_KL ||
        new_type == GGML_TYPE_IQ1_KT) {
        if (nx % QK_K != 0) {
            LLAMA_LOG_WARN("\n\n%s : tensor cols %d x %d are not divisible by %d, required for %s", __func__, nx, ny, QK_K, ggml_type_name(new_type));
            convert_incompatible_tensor = true;
        }
    }
    if (new_type == GGML_TYPE_IQ1_BN || new_type == GGML_TYPE_IQ2_BN || new_type == GGML_TYPE_IQ2_BN_R4) {
        if (nx % QK_IQ1BN != 0) {
            convert_incompatible_tensor = true;
        }
    }
    if (convert_incompatible_tensor) {
        switch (new_type) {
            case GGML_TYPE_IQ2_XXS:
            case GGML_TYPE_IQ2_XXS_R4:
            case GGML_TYPE_IQ2_XS:
            case GGML_TYPE_IQ2_XS_R4:
            case GGML_TYPE_IQ2_KS:
            case GGML_TYPE_IQ2_S:
            case GGML_TYPE_IQ2_S_R4:
            case GGML_TYPE_IQ3_XXS:
            case GGML_TYPE_IQ3_XXS_R4:
            case GGML_TYPE_IQ3_S:
            case GGML_TYPE_IQ3_S_R4:
            case GGML_TYPE_IQ1_S:
            case GGML_TYPE_IQ1_M:
            case GGML_TYPE_Q2_K:
            case GGML_TYPE_Q2_K_R4:
            case GGML_TYPE_Q3_K:
            case GGML_TYPE_Q3_K_R4:
            case GGML_TYPE_IQ2_K:
            case GGML_TYPE_IQ2_K_R4:
            case GGML_TYPE_IQ2_KL:
            case GGML_TYPE_IQ3_KS:
            case GGML_TYPE_IQ3_K:
            case GGML_TYPE_IQ3_K_R4:
            case GGML_TYPE_IQ4_KSS:
            case GGML_TYPE_IQ4_KS:
            case GGML_TYPE_IQ4_KS_R4:
            case GGML_TYPE_IQ4_XS_R8:
            case GGML_TYPE_IQ1_KT:
            case GGML_TYPE_IQ2_KT:
            case GGML_TYPE_IQ3_KT:
            case GGML_TYPE_IQ4_KT:
            case GGML_TYPE_IQ4_XS: new_type = GGML_TYPE_IQ4_NL; break;
            case GGML_TYPE_IQ4_K:
            case GGML_TYPE_IQ4_K_R4:
            case GGML_TYPE_Q4_K_R4:
            case GGML_TYPE_IQ5_KS:
            case GGML_TYPE_IQ5_KS_R4:
            case GGML_TYPE_Q4_K:   new_type = GGML_TYPE_Q5_0;   break;
            case GGML_TYPE_IQ5_K:
            case GGML_TYPE_IQ5_K_R4:
            case GGML_TYPE_Q5_K_R4:
            case GGML_TYPE_Q5_K:   new_type = GGML_TYPE_Q6_0;   break;
            case GGML_TYPE_IQ6_K:
            case GGML_TYPE_Q6_K_R4:
            case GGML_TYPE_Q8_K_R8:
            case GGML_TYPE_Q6_K:   new_type = GGML_TYPE_Q8_0;   break;
            default: throw std::runtime_error("\nUnsupported tensor size encountered\n");
        }
        LLAMA_LOG_WARN(" - using fallback quantization %s\n", ggml_type_name(new_type));
    }
    return new_type;
}

static ggml_type llama_tensor_get_type(quantize_state_internal & qs, ggml_type new_type, const ggml_tensor * tensor, llama_ftype ftype) {
    const std::string name = ggml_get_name(tensor);

    // TODO: avoid hardcoded tensor names - use the TN_* constants
    const llm_arch arch = qs.model.arch;
    const auto       tn = LLM_TN(arch);

    auto use_more_bits = [](int i_layer, int n_layers) -> bool {
        return i_layer < n_layers/8 || i_layer >= 7*n_layers/8 || (i_layer - n_layers/8)%3 == 2;
    };

    auto custom_type = GGML_TYPE_COUNT;
    if (qs.params->custom_quants) {
        using CustomQ = std::pair<std::string, ggml_type>;
        auto& q_rules = *static_cast<const std::vector<CustomQ>*>(qs.params->custom_quants);
        for (auto& rule : q_rules) {
            std::regex pattern(rule.first);
            if (std::regex_search(name, pattern)) {
                custom_type = rule.second;
                break;
            }
        }
    }

    //auto get_layer = [] (const char * name) {
    //    int il;
    //    if (sscanf(name, "blk.%d.", &il) == 1) return il;
    //    return -1;
    //};
    //int il = get_layer(tensor->name);
    //int nl = qs.model.hparams.n_layer;
    //if (ftype == LLAMA_FTYPE_MOSTLY_IQ2_K && (il == 0 || il == nl-1)) {
    //    return GGML_TYPE_IQ3_K;
    //}

    const int n_expert = std::max(1, (int)qs.model.hparams.n_expert);
    auto layer_info = [n_expert] (int i_layer, int n_layer, const char * name) {
        if (n_expert > 1) {
            // Believe it or not, "experts" in the FFN of Mixtral-8x7B are not consecutive, but occasionally randomly
            // sprinkled in the model. Hence, simply dividing i_ffn_down by n_expert does not work
            // for getting the current layer as I initially thought, and we need to resort to parsing the
            // tensor name.
            if (sscanf(name, "blk.%d.", &i_layer) != 1) {
                throw std::runtime_error(format("Failed to determine layer for tensor %s", name));
            }
            if (i_layer < 0 || i_layer >= n_layer) {
                throw std::runtime_error(format("Bad layer %d for tensor %s. Must be in [0, %d)", i_layer, name, n_layer));
            }
        }
        return std::make_pair(i_layer, n_layer);
    };

    // for arches that share the same tensor between the token embeddings and the output, we quantize the token embeddings
    // with the quantization of the output tensor
    if (name == tn(LLM_TENSOR_OUTPUT, "weight") || (!qs.has_output && name == tn(LLM_TENSOR_TOKEN_EMBD, "weight"))) {
        if (qs.params->output_tensor_type < GGML_TYPE_COUNT) {
            new_type = qs.params->output_tensor_type;
        } else {
            int nx = tensor->ne[0];
            if (arch == LLM_ARCH_FALCON || nx % QK_K != 0) {
                new_type = GGML_TYPE_Q8_0;
            }
            else if (ftype == LLAMA_FTYPE_MOSTLY_IQ2_XXS || ftype == LLAMA_FTYPE_MOSTLY_IQ2_XS || ftype == LLAMA_FTYPE_MOSTLY_IQ3_XXS ||
                     ftype == LLAMA_FTYPE_MOSTLY_IQ1_S   || ftype == LLAMA_FTYPE_MOSTLY_IQ2_S  || ftype == LLAMA_FTYPE_MOSTLY_IQ2_M   ||
                     ftype == LLAMA_FTYPE_MOSTLY_IQ1_M   || ftype == LLAMA_FTYPE_MOSTLY_IQ2_K  || ftype == LLAMA_FTYPE_MOSTLY_IQ3_K   ||
                     ftype == LLAMA_FTYPE_MOSTLY_IQ2_KS     || ftype == LLAMA_FTYPE_MOSTLY_IQ3_K_R4   || ftype == LLAMA_FTYPE_MOSTLY_IQ3_KS ||
                     ftype == LLAMA_FTYPE_MOSTLY_IQ2_K_R4   || ftype == LLAMA_FTYPE_MOSTLY_IQ3_XXS_R4 || ftype == LLAMA_FTYPE_MOSTLY_IQ2_KL ||
                     ftype == LLAMA_FTYPE_MOSTLY_IQ3_XXS_R4 || ftype == LLAMA_FTYPE_MOSTLY_IQ2_M_R4   ||
                     ftype == LLAMA_FTYPE_MOSTLY_IQ1_S_R4   || ftype == LLAMA_FTYPE_MOSTLY_IQ1_M_R4   ||
                     ftype == LLAMA_FTYPE_MOSTLY_IQ2_KT || ftype == LLAMA_FTYPE_MOSTLY_IQ3_KT || ftype == LLAMA_FTYPE_MOSTLY_IQ1_KT) {
                new_type = !qs.has_output ? GGML_TYPE_IQ4_K : GGML_TYPE_Q5_K;
            }
            else if (ftype == LLAMA_FTYPE_MOSTLY_IQ2_XXS_R4 || ftype == LLAMA_FTYPE_MOSTLY_IQ2_XS_R4) {
                new_type = !qs.has_output ? GGML_TYPE_IQ4_K_R4 : GGML_TYPE_Q5_K_R4;
            }
            else if ((ftype == LLAMA_FTYPE_MOSTLY_IQ3_S || ftype == LLAMA_FTYPE_MOSTLY_IQ3_M || ftype == LLAMA_FTYPE_MOSTLY_IQ3_KL ||
                      ftype == LLAMA_FTYPE_MOSTLY_IQ4_XS || ftype == LLAMA_FTYPE_MOSTLY_IQ3_S_R4 ||
                      ftype == LLAMA_FTYPE_MOSTLY_IQ4_KS || ftype == LLAMA_FTYPE_MOSTLY_IQ4_KSS || ftype == LLAMA_FTYPE_MOSTLY_IQ4_KS_R4) && !qs.has_output) {
                new_type = GGML_TYPE_IQ5_K;
            }
            else if (new_type != GGML_TYPE_Q8_0 && new_type != GGML_TYPE_Q8_0_R8 && new_type != GGML_TYPE_IQ6_K && new_type != GGML_TYPE_Q6_K_R4 &&
                     new_type != GGML_TYPE_Q8_K_R8 && new_type != GGML_TYPE_Q8_KV && new_type != GGML_TYPE_Q8_KV_R8) {
                new_type = GGML_TYPE_Q6_K;
            }
        }
    } else if (name == "token_embd.weight") {
        if (qs.params->token_embedding_type < GGML_TYPE_COUNT) {
            new_type = qs.params->token_embedding_type;
        } else {
            if (ftype == LLAMA_FTYPE_MOSTLY_IQ2_XXS || ftype == LLAMA_FTYPE_MOSTLY_IQ2_XS ||
                ftype == LLAMA_FTYPE_MOSTLY_IQ1_S   || ftype == LLAMA_FTYPE_MOSTLY_IQ1_M  ||
                ftype == LLAMA_FTYPE_MOSTLY_IQ2_XXS_R4 || ftype == LLAMA_FTYPE_MOSTLY_IQ2_XS_R4 ||
                ftype == LLAMA_FTYPE_MOSTLY_IQ1_S_R4 || ftype == LLAMA_FTYPE_MOSTLY_IQ1_M_R4) {
                new_type = GGML_TYPE_Q2_K;
            }
            else if (ftype == LLAMA_FTYPE_MOSTLY_IQ2_S || ftype == LLAMA_FTYPE_MOSTLY_IQ2_M || ftype == LLAMA_FTYPE_MOSTLY_IQ2_M_R4) {
                new_type = GGML_TYPE_IQ3_S;
            }
            else if (ftype == LLAMA_FTYPE_MOSTLY_IQ3_XXS || ftype == LLAMA_FTYPE_MOSTLY_IQ3_KT) {
                new_type = GGML_TYPE_IQ3_S;
            }
            else if (ftype == LLAMA_FTYPE_MOSTLY_IQ3_XXS_R4) {
                new_type = GGML_TYPE_IQ3_K;
            }
            else if (ftype == LLAMA_FTYPE_MOSTLY_IQ1_BN || ftype == LLAMA_FTYPE_MOSTLY_IQ2_BN || ftype == LLAMA_FTYPE_MOSTLY_IQ2_BN_R4) {
                new_type = GGML_TYPE_IQ4_NL;
            }
        }
    } else if (name == "per_layer_token_embd.weight") {
        if (qs.params->per_layer_token_embedding_type < GGML_TYPE_COUNT) {
            new_type = qs.params->per_layer_token_embedding_type;
        } else {
            if (ftype == LLAMA_FTYPE_MOSTLY_IQ2_XXS || ftype == LLAMA_FTYPE_MOSTLY_IQ2_XS ||
                ftype == LLAMA_FTYPE_MOSTLY_IQ1_S   || ftype == LLAMA_FTYPE_MOSTLY_IQ1_M  ||
                ftype == LLAMA_FTYPE_MOSTLY_IQ2_XXS_R4 || ftype == LLAMA_FTYPE_MOSTLY_IQ2_XS_R4 ||
                ftype == LLAMA_FTYPE_MOSTLY_IQ1_S_R4 || ftype == LLAMA_FTYPE_MOSTLY_IQ1_M_R4) {
                new_type = GGML_TYPE_Q2_K;
            }
            else if (ftype == LLAMA_FTYPE_MOSTLY_IQ2_S || ftype == LLAMA_FTYPE_MOSTLY_IQ2_M || ftype == LLAMA_FTYPE_MOSTLY_IQ2_M_R4) {
                new_type = GGML_TYPE_IQ3_S;
            }
            else if (ftype == LLAMA_FTYPE_MOSTLY_IQ3_XXS || ftype == LLAMA_FTYPE_MOSTLY_IQ3_KT) {
                new_type = GGML_TYPE_IQ3_S;
            }
            else if (ftype == LLAMA_FTYPE_MOSTLY_IQ3_XXS_R4) {
                new_type = GGML_TYPE_IQ3_K;
            }
            else if (ftype == LLAMA_FTYPE_MOSTLY_IQ1_BN || ftype == LLAMA_FTYPE_MOSTLY_IQ2_BN || ftype == LLAMA_FTYPE_MOSTLY_IQ2_BN_R4) {
                new_type = GGML_TYPE_IQ4_NL;
            }
        }
    } else if (ftype == LLAMA_FTYPE_MOSTLY_IQ1_S_R4 || ftype == LLAMA_FTYPE_MOSTLY_IQ1_M_R4) {
        if (name.find("attn_v.weight") != std::string::npos) {
            if (qs.model.hparams.n_expert >= 4 || qs.model.hparams.n_gqa() >= 4) new_type = GGML_TYPE_IQ4_K_R4;
            else if (qs.model.hparams.n_gqa() >= 2) new_type = GGML_TYPE_IQ3_K_R4;
            else new_type = GGML_TYPE_Q2_K_R4;
            ++qs.i_attention_wv;
        }
        else if (qs.model.hparams.n_expert >= 8 && name.find("attn_k") != std::string::npos) {
            new_type = GGML_TYPE_Q4_K_R4;
        }
        else if (qs.model.hparams.n_expert >= 8 && (name.find("blk.0.ffn_down") != std::string::npos ||
                                                    name.find("blk.0.ffn_gate") != std::string::npos ||
                                                    name.find("blk.0.ffn_up") != std::string::npos)) {
            new_type = GGML_TYPE_IQ3_K_R4;
        }
        else if (qs.model.hparams.n_expert >= 8 && name.find("attn_q") != std::string::npos) {
            new_type = GGML_TYPE_Q4_K_R4;
        }
        else if (name.find("attn_qkv.weight") != std::string::npos) {
            new_type = GGML_TYPE_IQ2_K_R4;
        }
        else if (name.find("_shexp.weight") != std::string::npos) {
            new_type = GGML_TYPE_IQ4_K_R4;
        }
        else if (name.find("ffn_down") != std::string::npos) {
            auto [i_layer, n_layer] = layer_info(qs.i_ffn_down, qs.n_ffn_down, name.c_str());
            if (qs.params->ffn_down_type < GGML_TYPE_COUNT) new_type = qs.params->ffn_down_type;
            else if (i_layer < n_layer/8) {
                new_type = GGML_TYPE_Q2_K_R4;
            }
            ++qs.i_ffn_down;
        }
        else if (name.find("attn_output.weight") != std::string::npos) {
            new_type = qs.model.hparams.n_expert >= 4 ? GGML_TYPE_Q5_K_R4 : GGML_TYPE_IQ2_K_R4;
        }
    }
    else if (ftype == LLAMA_FTYPE_MOSTLY_IQ2_KT) {
        if (name.find("attn_v.weight") != std::string::npos) {
            if (qs.model.hparams.n_expert >= 4 || qs.model.hparams.n_gqa() >= 4) new_type = GGML_TYPE_IQ4_K;
            else if (qs.model.hparams.n_gqa() >= 2) new_type = GGML_TYPE_IQ3_K;
            else new_type = GGML_TYPE_Q2_K;
            ++qs.i_attention_wv;
        }
        else if (qs.model.hparams.n_expert >= 8 && name.find("attn_k") != std::string::npos) {
            new_type = GGML_TYPE_Q4_K;
        }
        else if (qs.model.hparams.n_expert >= 8 && (name.find("blk.0.ffn_down") != std::string::npos ||
                                                    name.find("blk.0.ffn_gate") != std::string::npos ||
                                                    name.find("blk.0.ffn_up") != std::string::npos)) {
            new_type = GGML_TYPE_IQ3_K;
        }
        else if (qs.model.hparams.n_expert >= 8 && name.find("attn_q") != std::string::npos) {
            new_type = GGML_TYPE_Q4_K;
        }
        else if (name.find("attn_qkv.weight") != std::string::npos) {
            new_type = GGML_TYPE_IQ3_K;
        }
        else if (name.find("_shexp.weight") != std::string::npos) {
            new_type = GGML_TYPE_IQ4_K;
        }
        else if (name.find("ffn_down") != std::string::npos) {
            auto [i_layer, n_layer] = layer_info(qs.i_ffn_down, qs.n_ffn_down, name.c_str());
            if (qs.params->ffn_down_type < GGML_TYPE_COUNT) new_type = qs.params->ffn_down_type;
            else if (i_layer < n_layer/8) {
                new_type = GGML_TYPE_IQ3_K;
            }
            ++qs.i_ffn_down;
        }
        else if (name.find("attn_output.weight") != std::string::npos) {
            new_type = qs.model.hparams.n_expert >= 4 ? GGML_TYPE_Q5_K : GGML_TYPE_IQ3_K;
        }
    } else if (ftype == LLAMA_FTYPE_MOSTLY_IQ2_XXS || ftype == LLAMA_FTYPE_MOSTLY_IQ2_XS || ftype == LLAMA_FTYPE_MOSTLY_IQ1_S ||
               ftype == LLAMA_FTYPE_MOSTLY_IQ2_S   || ftype == LLAMA_FTYPE_MOSTLY_IQ2_M  || ftype == LLAMA_FTYPE_MOSTLY_IQ1_M ||
               ftype == LLAMA_FTYPE_MOSTLY_IQ2_KS  || ftype == LLAMA_FTYPE_MOSTLY_IQ2_XXS_R4 || ftype == LLAMA_FTYPE_MOSTLY_IQ2_XS_R4 ||
               ftype == LLAMA_FTYPE_MOSTLY_IQ2_M_R4) {
        bool is_iq2_m = ftype == LLAMA_FTYPE_MOSTLY_IQ2_M || ftype == LLAMA_FTYPE_MOSTLY_IQ2_M_R4;
        if (name.find("attn_v.weight") != std::string::npos) {
            if      (qs.model.hparams.n_gqa() >= 4 || qs.model.hparams.n_expert >= 4) new_type = GGML_TYPE_IQ4_K;
            else if (qs.model.hparams.n_gqa() >= 2 || qs.model.hparams.n_expert >= 2) new_type = GGML_TYPE_IQ3_K;
            else new_type = ftype == LLAMA_FTYPE_MOSTLY_IQ2_S || is_iq2_m ? GGML_TYPE_IQ3_S : GGML_TYPE_Q2_K;
            ++qs.i_attention_wv;
        }
        else if (qs.model.hparams.n_expert >= 8 && name.find("attn_k") != std::string::npos) {
            new_type = GGML_TYPE_Q4_K;
        }
        else if (qs.model.hparams.n_expert >= 8 && name.find("attn_q") != std::string::npos) {
            new_type = GGML_TYPE_Q4_K;
        }
        else if (name.find("attn_qkv.weight") != std::string::npos) {
            new_type = ftype == LLAMA_FTYPE_MOSTLY_IQ2_S || is_iq2_m ? GGML_TYPE_IQ3_XXS : GGML_TYPE_IQ2_K;
        }
        else if (name.find("ffn_down") != std::string::npos) {
            if (qs.i_ffn_down < qs.n_ffn_down/8) {
                new_type = ftype == LLAMA_FTYPE_MOSTLY_IQ2_S || is_iq2_m ? GGML_TYPE_IQ3_S : GGML_TYPE_Q2_K;
            }
            ++qs.i_ffn_down;
        }
        else if (name.find("attn_output.weight") != std::string::npos) {
            if (qs.params->attn_output_type < GGML_TYPE_COUNT) new_type = qs.params->attn_output_type;
            else if (qs.model.hparams.n_expert >= 4) {
                new_type = GGML_TYPE_Q5_K;
            } else {
                if (ftype == LLAMA_FTYPE_MOSTLY_IQ1_S || ftype == LLAMA_FTYPE_MOSTLY_IQ1_M) new_type = GGML_TYPE_IQ2_K;
                else if (ftype == LLAMA_FTYPE_MOSTLY_IQ2_S || is_iq2_m) new_type = GGML_TYPE_IQ3_S;
            }
        }
    } else if (name.find("attn_v.weight") != std::string::npos) {
        if      (qs.params->attn_v_type < GGML_TYPE_COUNT) new_type = qs.params->attn_v_type;
        else if (qs.model.hparams.n_expert >= 4) {
            // for the 4-8-expert model, bumping this to Q8_0 trades just ~128MB
            // TODO: explore better strategies
            new_type = GGML_TYPE_Q8_0;
        }
        else if (qs.model.type == MODEL_70B) {
            // In the 70B model we have 8 heads sharing the same attn_v weights. As a result, the attn_v.weight tensor is
            // 8x smaller compared to attn_q.weight. Hence, we can get a nice boost in quantization accuracy with
            // nearly negligible increase in model size by quantizing this tensor with more bits:
            if (new_type == GGML_TYPE_Q3_K || new_type == GGML_TYPE_Q4_K) new_type = GGML_TYPE_Q5_K;
            if (new_type == GGML_TYPE_IQ3_K) new_type = GGML_TYPE_IQ5_K;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_Q2_K) {
            new_type = qs.model.hparams.n_gqa() >= 4 ? GGML_TYPE_Q4_K : GGML_TYPE_Q3_K;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_IQ2_K) {
            new_type = qs.model.hparams.n_gqa() >= 2 ? GGML_TYPE_IQ4_K : GGML_TYPE_IQ3_K;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_IQ2_K_R4) {
            new_type = qs.model.hparams.n_gqa() >= 2 ? GGML_TYPE_IQ4_K_R4 : GGML_TYPE_IQ3_K_R4;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_Q2_K_S && qs.model.hparams.n_gqa() >= 4) {
            new_type = GGML_TYPE_Q4_K;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_Q2_K_R4 && qs.model.hparams.n_gqa() >= 4) {
            new_type = GGML_TYPE_Q4_K_R4;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_IQ3_XXS) {
            new_type = qs.model.hparams.n_gqa() >= 4 ? GGML_TYPE_Q4_K : qs.model.hparams.n_gqa() >= 2 ? GGML_TYPE_IQ3_K
                     : !qs.has_imatrix ? GGML_TYPE_IQ3_S : GGML_TYPE_IQ3_XXS;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_IQ3_KT) {
            //new_type = qs.model.hparams.n_gqa() >= 4 ? GGML_TYPE_IQ4_K : qs.model.hparams.n_gqa() >= 2 ? GGML_TYPE_IQ3_K
            //         : !qs.has_imatrix ? GGML_TYPE_IQ3_K : GGML_TYPE_IQ3_KT;
            new_type = qs.model.hparams.n_gqa() >= 4 ? GGML_TYPE_IQ4_K : GGML_TYPE_IQ3_K;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_IQ4_KT) {
            //new_type = qs.model.hparams.n_gqa() >= 4 ? GGML_TYPE_IQ5_K : qs.model.hparams.n_gqa() >= 2 ? GGML_TYPE_IQ4_K
            //         : !qs.has_imatrix ? GGML_TYPE_IQ4_KS : GGML_TYPE_IQ4_KT;
            new_type = qs.model.hparams.n_gqa() >= 4 ? GGML_TYPE_IQ5_K : GGML_TYPE_IQ4_K;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_IQ3_XXS_R4) {
            new_type = qs.model.hparams.n_gqa() >= 4 ? GGML_TYPE_Q4_K_R4 : qs.model.hparams.n_gqa() >= 2 ? GGML_TYPE_IQ3_K_R4
                     : !qs.has_imatrix ? GGML_TYPE_IQ3_K_R4 : GGML_TYPE_IQ3_XXS_R4;
        }
        else if ((ftype == LLAMA_FTYPE_MOSTLY_IQ3_XS || ftype == LLAMA_FTYPE_MOSTLY_IQ3_S) && qs.model.hparams.n_gqa() >= 2) {
            new_type = GGML_TYPE_IQ4_K;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_IQ3_S_R4 && qs.model.hparams.n_gqa() >= 2) {
            new_type = GGML_TYPE_IQ4_K_R4;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_IQ3_K && qs.model.hparams.n_gqa() >= 2) {
            new_type = GGML_TYPE_IQ4_K;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_IQ3_KS && qs.model.hparams.n_gqa() >= 2) {
            new_type = GGML_TYPE_IQ4_KS;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_IQ2_KL && qs.model.hparams.n_gqa() >= 2) {
            new_type = GGML_TYPE_IQ4_KS;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_IQ3_K_R4 && qs.model.hparams.n_gqa() >= 2) {
            new_type = GGML_TYPE_IQ4_K_R4;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_IQ3_KL) {
            new_type = qs.model.hparams.n_gqa() >= 2 ? GGML_TYPE_IQ5_K : GGML_TYPE_IQ4_K;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_IQ3_M) {
            new_type = qs.model.hparams.n_gqa() >= 2 ? GGML_TYPE_IQ5_K : GGML_TYPE_IQ4_K;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_Q3_K_M) {
            new_type = qs.i_attention_wv < 2 ? GGML_TYPE_Q5_K : GGML_TYPE_Q4_K;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_Q3_K_L) new_type = GGML_TYPE_Q5_K;
        else if ((ftype == LLAMA_FTYPE_MOSTLY_IQ4_NL || ftype == LLAMA_FTYPE_MOSTLY_IQ4_XS ||
                  ftype == LLAMA_FTYPE_MOSTLY_IQ4_NL_R4 || ftype == LLAMA_FTYPE_MOSTLY_IQ4_XS_R8 ||
                  ftype == LLAMA_FTYPE_MOSTLY_IQ4_KS || ftype == LLAMA_FTYPE_MOSTLY_IQ4_KSS) && qs.model.hparams.n_gqa() >= 2) {
            new_type = GGML_TYPE_IQ5_K;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_IQ4_KS_R4 && qs.model.hparams.n_gqa() >= 2) {
            new_type = GGML_TYPE_IQ5_K_R4;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_IQ4_K && qs.model.hparams.n_gqa() >= 2) {
            new_type = GGML_TYPE_IQ5_K;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_IQ4_K_R4 && qs.model.hparams.n_gqa() >= 2) {
            new_type = GGML_TYPE_IQ5_K;
        }
        else if ((ftype == LLAMA_FTYPE_MOSTLY_Q4_K_M || ftype == LLAMA_FTYPE_MOSTLY_Q5_K_M) &&
                use_more_bits(qs.i_attention_wv, qs.n_attention_wv)) new_type = GGML_TYPE_Q6_K;
        else if (ftype == LLAMA_FTYPE_MOSTLY_Q4_K_S && qs.i_attention_wv < 4) new_type = GGML_TYPE_Q5_K;
        else if (ftype == LLAMA_FTYPE_MOSTLY_Q4_K_R4 && qs.i_attention_wv < 4) new_type = GGML_TYPE_Q5_K;
        else if (ftype == LLAMA_FTYPE_MOSTLY_Q5_K_S) {
            if (qs.model.hparams.n_vocab >= 127999 && (qs.model.type == MODEL_8B || qs.model.type == MODEL_70B))
                new_type = GGML_TYPE_Q6_K;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_IQ5_K || ftype == LLAMA_FTYPE_MOSTLY_IQ5_KS) {
            if (qs.model.hparams.n_vocab >= 127999 && (qs.model.type == MODEL_8B || qs.model.type == MODEL_70B))
                new_type = GGML_TYPE_IQ6_K;
        }
        else if (qs.model.hparams.n_gqa() >= 4 &&
                 !(arch == LLM_ARCH_DFLASH_DRAFT &&
                   (ftype == LLAMA_FTYPE_MOSTLY_Q4_K_M || ftype == LLAMA_FTYPE_MOSTLY_Q5_K_M))) {
            if      (new_type == GGML_TYPE_Q2_K || new_type == GGML_TYPE_IQ3_XXS) new_type = GGML_TYPE_IQ3_S;
            else if (new_type == GGML_TYPE_Q2_K_R4 || new_type == GGML_TYPE_IQ3_XXS_R4) new_type = GGML_TYPE_IQ3_K_R4;
            else if (new_type == GGML_TYPE_Q3_K || new_type == GGML_TYPE_IQ3_S) new_type = GGML_TYPE_Q4_K;
            else if (new_type == GGML_TYPE_IQ3_K) new_type = GGML_TYPE_IQ4_K;
            else if (new_type == GGML_TYPE_IQ3_KS) new_type = GGML_TYPE_IQ4_KS;
            else if (new_type == GGML_TYPE_IQ2_KL) new_type = GGML_TYPE_IQ4_KS;
            else if (new_type == GGML_TYPE_IQ3_S_R4) new_type = GGML_TYPE_Q4_K_R4;
            else if (new_type == GGML_TYPE_Q3_K_R4) new_type = GGML_TYPE_Q4_K_R4;
            else if (new_type == GGML_TYPE_Q4_K || new_type == GGML_TYPE_IQ4_XS) new_type = GGML_TYPE_Q5_K;
            else if (new_type == GGML_TYPE_IQ4_NL) new_type = GGML_TYPE_Q5_K;
            else if (new_type == GGML_TYPE_IQ4_K || new_type == GGML_TYPE_IQ4_KS) new_type = GGML_TYPE_IQ5_K;
            else if (new_type == GGML_TYPE_IQ4_NL_R4) new_type = GGML_TYPE_Q5_K;
            else if (new_type == GGML_TYPE_IQ4_XS_R8) new_type = GGML_TYPE_Q5_K;
            else if (new_type == GGML_TYPE_Q5_K) new_type = GGML_TYPE_Q6_K;
            else if (new_type == GGML_TYPE_IQ5_K || new_type == GGML_TYPE_IQ5_KS) new_type = GGML_TYPE_IQ6_K;
        }
        ++qs.i_attention_wv;
    } else if (name.find("attn_k") != std::string::npos) {
        if (qs.params->attn_k_type < GGML_TYPE_COUNT) new_type = qs.params->attn_k_type;
        else if (qs.model.hparams.n_expert >= 4) {
            // for the 4-8-expert model, bumping this to Q8_0 trades just ~128MB
            // TODO: explore better strategies
            new_type = GGML_TYPE_Q8_0;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_IQ3_XS) {
            new_type = GGML_TYPE_IQ3_XXS; // TODO: explore better strategies?
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_IQ3_XXS || ftype == LLAMA_FTYPE_MOSTLY_IQ3_XXS_R4) {
            new_type = GGML_TYPE_IQ2_S; // TODO: explore better strategies?
        }
    } else if (name.find("attn_q") != std::string::npos) {
        if (qs.params->attn_q_type < GGML_TYPE_COUNT) new_type = qs.params->attn_q_type;
        else if (ftype == LLAMA_FTYPE_MOSTLY_IQ3_XS) {
            new_type = GGML_TYPE_IQ3_XXS;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_IQ3_XXS || ftype == LLAMA_FTYPE_MOSTLY_IQ3_XXS_R4) {
            new_type = GGML_TYPE_IQ2_S;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_Q5_K_S) {
            if (qs.model.hparams.n_vocab >= 127999 && (qs.model.type == MODEL_8B || qs.model.type == MODEL_70B))
                new_type = GGML_TYPE_Q4_K;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_IQ5_K) {
            if (qs.model.hparams.n_vocab >= 127999 && (qs.model.type == MODEL_8B || qs.model.type == MODEL_70B))
                new_type = GGML_TYPE_IQ4_K;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_IQ5_KS) {
            if (qs.model.hparams.n_vocab >= 127999 && (qs.model.type == MODEL_8B || qs.model.type == MODEL_70B))
                new_type = GGML_TYPE_IQ4_KS;
        }
    } else if (name.find("ffn_down") != std::string::npos) {
        auto info = layer_info(qs.i_ffn_down, qs.n_ffn_down, name.c_str());
        int i_layer = info.first, n_layer = info.second;
        if (qs.params->ffn_down_type < GGML_TYPE_COUNT) new_type = qs.params->ffn_down_type;
        else if      (ftype == LLAMA_FTYPE_MOSTLY_Q2_K) new_type = GGML_TYPE_Q3_K;
        else if (ftype == LLAMA_FTYPE_MOSTLY_Q2_K_S) {
            if (i_layer < n_layer/8) new_type = GGML_TYPE_Q4_K;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_Q2_K_R4) {
            if (i_layer < n_layer/8) new_type = GGML_TYPE_Q4_K_R4;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_IQ3_XXS && !qs.has_imatrix) {
            new_type = i_layer < n_layer/8 ? GGML_TYPE_Q4_K : GGML_TYPE_Q3_K;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_IQ3_KT && !qs.has_imatrix) {
            new_type = i_layer < n_layer/8 ? GGML_TYPE_IQ4_K : GGML_TYPE_IQ3_K;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_IQ3_XXS_R4 && !qs.has_imatrix) {
            new_type = i_layer < n_layer/8 ? GGML_TYPE_Q4_K_R4 : GGML_TYPE_IQ3_K_R4;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_Q3_K_M) {
            new_type = i_layer < n_layer/16 ? GGML_TYPE_Q5_K
                     : arch != LLM_ARCH_FALCON || use_more_bits(i_layer, n_layer) ? GGML_TYPE_Q4_K
                     : GGML_TYPE_Q3_K;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_IQ3_M && (i_layer < n_layer/8 ||
                    (qs.model.hparams.n_expert >= 4 && use_more_bits(i_layer, n_layer)))) {
            new_type = GGML_TYPE_IQ4_K;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_Q3_K_L) {
            new_type = arch == LLM_ARCH_FALCON ? GGML_TYPE_Q4_K : GGML_TYPE_Q5_K;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_IQ3_KL) {
            new_type = use_more_bits(i_layer, n_layer) ? GGML_TYPE_IQ4_KS : GGML_TYPE_IQ3_K;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_Q4_K_M) {
            if (arch == LLM_ARCH_FALCON) {
                new_type = i_layer < n_layer/16 ? GGML_TYPE_Q6_K :
                           use_more_bits(i_layer, n_layer) ? GGML_TYPE_Q5_K : GGML_TYPE_Q4_K;
            } else {
                if (use_more_bits(i_layer, n_layer)) new_type = GGML_TYPE_Q6_K;
            }
        }
        else if (i_layer < n_layer/8 && !qs.has_imatrix &&
                (ftype == LLAMA_FTYPE_MOSTLY_IQ4_NL || ftype == LLAMA_FTYPE_MOSTLY_IQ4_XS ||
                 ftype == LLAMA_FTYPE_MOSTLY_IQ4_KS || ftype == LLAMA_FTYPE_MOSTLY_IQ4_KSS ||
                 ftype == LLAMA_FTYPE_MOSTLY_IQ4_NL_R4 || ftype == LLAMA_FTYPE_MOSTLY_IQ4_XS_R8)) {
            new_type = GGML_TYPE_Q5_K;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_IQ4_KS_R4 && i_layer < n_layer/8 && !qs.has_imatrix) {
            new_type = GGML_TYPE_Q5_K_R4;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_Q5_K_M && use_more_bits(i_layer, n_layer)) new_type = GGML_TYPE_Q6_K;
        else if (ftype == LLAMA_FTYPE_MOSTLY_Q4_K_S && arch != LLM_ARCH_FALCON && i_layer < n_layer/8) {
            new_type = GGML_TYPE_Q5_K;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_Q4_K_R4 && arch != LLM_ARCH_FALCON && i_layer < n_layer/8) {
            new_type = GGML_TYPE_Q5_K;
        }
        else if ((ftype == LLAMA_FTYPE_MOSTLY_Q4_0 || ftype == LLAMA_FTYPE_MOSTLY_Q5_0)
                && qs.has_imatrix && i_layer < n_layer/8) {
            // Guard against craziness in the first few ffn_down layers that can happen even with imatrix for Q4_0/Q5_0.
            // We only do it when an imatrix is provided because a) we want to make sure that one can always get the
            // same quantization as before imatrix stuff, and b) Q4_1/Q5_1 do go crazy on ffn_down without an imatrix.
            new_type = ftype == LLAMA_FTYPE_MOSTLY_Q4_0 ? GGML_TYPE_Q4_1 : GGML_TYPE_Q5_1;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_Q4_0_R8 && qs.has_imatrix && i_layer < n_layer/8) {
            new_type = GGML_TYPE_IQ4_NL_R4;
        }
        ++qs.i_ffn_down;
    } else if (name.find("attn_output.weight") != std::string::npos) {
        if (qs.params->attn_output_type < GGML_TYPE_COUNT) new_type = qs.params->attn_output_type;
        else if (arch != LLM_ARCH_FALCON) {
            if (qs.model.hparams.n_expert >= 4) {
                if (ftype == LLAMA_FTYPE_MOSTLY_Q2_K   || ftype == LLAMA_FTYPE_MOSTLY_IQ3_XS || ftype == LLAMA_FTYPE_MOSTLY_IQ3_XXS ||
                    ftype == LLAMA_FTYPE_MOSTLY_Q3_K_S || ftype == LLAMA_FTYPE_MOSTLY_Q3_K_M || ftype == LLAMA_FTYPE_MOSTLY_IQ4_NL  ||
                    ftype == LLAMA_FTYPE_MOSTLY_Q4_K_S || ftype == LLAMA_FTYPE_MOSTLY_Q4_K_M || ftype == LLAMA_FTYPE_MOSTLY_IQ3_S   ||
                    ftype == LLAMA_FTYPE_MOSTLY_IQ3_M  || ftype == LLAMA_FTYPE_MOSTLY_IQ4_XS || ftype == LLAMA_FTYPE_MOSTLY_IQ4_K   ||
                    ftype == LLAMA_FTYPE_MOSTLY_IQ4_KSS || ftype == LLAMA_FTYPE_MOSTLY_IQ4_KS ||  ftype == LLAMA_FTYPE_MOSTLY_IQ4_KS_R4 ||
                    ftype == LLAMA_FTYPE_MOSTLY_IQ5_KS || ftype == LLAMA_FTYPE_MOSTLY_IQ5_KS_R4 ||
                    ftype == LLAMA_FTYPE_MOSTLY_IQ2_K  || ftype == LLAMA_FTYPE_MOSTLY_IQ3_K  || ftype == LLAMA_FTYPE_MOSTLY_IQ3_KL  ||
                    ftype == LLAMA_FTYPE_MOSTLY_Q4_K_R4 || ftype == LLAMA_FTYPE_MOSTLY_IQ4_NL_R4 || ftype == LLAMA_FTYPE_MOSTLY_IQ4_XS_R8 ||
                    ftype == LLAMA_FTYPE_MOSTLY_Q3_K_R4 || ftype == LLAMA_FTYPE_MOSTLY_IQ3_KT || ftype == LLAMA_FTYPE_MOSTLY_IQ3_KS ||
                    ftype == LLAMA_FTYPE_MOSTLY_Q2_K_R4|| ftype == LLAMA_FTYPE_MOSTLY_IQ4_K_R4 || ftype == LLAMA_FTYPE_MOSTLY_IQ3_K_R4 ||
                    ftype == LLAMA_FTYPE_MOSTLY_IQ2_K_R4 || ftype == LLAMA_FTYPE_MOSTLY_IQ3_XXS_R4 || ftype == LLAMA_FTYPE_MOSTLY_IQ3_S_R4) {
                    new_type = GGML_TYPE_Q5_K; // should the IQ_K quants be applied here as the new type for the IQ_K ftypes ?
                    // also, this condition could be reproduced on attn_q, eventually with Q4_K instead of Q5_K.
                }
                else if (ftype == LLAMA_FTYPE_MOSTLY_IQ2_KL) {
                    new_type = GGML_TYPE_IQ4_KS;
                }
            } else {
                if      (ftype == LLAMA_FTYPE_MOSTLY_Q2_K   ) new_type = GGML_TYPE_Q3_K; // This list could be generalized and streamlined
                else if (ftype == LLAMA_FTYPE_MOSTLY_IQ3_XXS) new_type = GGML_TYPE_IQ3_S;
                else if (ftype == LLAMA_FTYPE_MOSTLY_IQ3_KT && qs.model.hparams.n_gqa() >= 4) new_type = GGML_TYPE_IQ3_K;
                else if (ftype == LLAMA_FTYPE_MOSTLY_IQ3_XXS_R4) new_type = GGML_TYPE_IQ3_K_R4;
                else if (ftype == LLAMA_FTYPE_MOSTLY_Q3_K_M ) new_type = GGML_TYPE_Q4_K;
                else if (ftype == LLAMA_FTYPE_MOSTLY_Q3_K_L ) new_type = GGML_TYPE_Q5_K;
                else if (ftype == LLAMA_FTYPE_MOSTLY_IQ3_M  ) new_type = GGML_TYPE_IQ4_K;
                else if (ftype == LLAMA_FTYPE_MOSTLY_IQ2_K  ) new_type = GGML_TYPE_IQ3_K;
                else if (ftype == LLAMA_FTYPE_MOSTLY_IQ2_K_R4) new_type = GGML_TYPE_IQ3_K_R4;
                else if (ftype == LLAMA_FTYPE_MOSTLY_IQ3_KL ) new_type = GGML_TYPE_IQ4_KS;
            }
        } else {
            if (ftype == LLAMA_FTYPE_MOSTLY_Q3_K_L) new_type = GGML_TYPE_Q4_K;
        }
    }
    else if (name.find("attn_qkv.weight") != std::string::npos) {
        if (qs.params->attn_qkv_type < GGML_TYPE_COUNT) new_type = qs.params->attn_qkv_type;
        else if (ftype == LLAMA_FTYPE_MOSTLY_Q3_K_M || ftype == LLAMA_FTYPE_MOSTLY_Q3_K_L) {
            new_type = GGML_TYPE_Q4_K; // That logic could either be generalized, either be ditched?
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_IQ3_M ) new_type = GGML_TYPE_IQ4_K;
        else if (ftype == LLAMA_FTYPE_MOSTLY_Q4_K_M) new_type = GGML_TYPE_Q5_K;
        else if (ftype == LLAMA_FTYPE_MOSTLY_Q5_K_M) new_type = GGML_TYPE_Q6_K;
    }
    else if (name.find("ffn_gate") != std::string::npos) {
        auto info = layer_info(qs.i_ffn_gate, qs.n_ffn_gate, name.c_str());
        int i_layer = info.first, n_layer = info.second;
        if (qs.params->ffn_gate_type < GGML_TYPE_COUNT) new_type = qs.params->ffn_gate_type;
        else if (ftype == LLAMA_FTYPE_MOSTLY_IQ3_XS && (i_layer >= n_layer/8 && i_layer < 7*n_layer/8)) {
            new_type = GGML_TYPE_IQ3_XXS;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_IQ3_KL && use_more_bits(i_layer, n_layer)) {
            new_type = GGML_TYPE_IQ4_KS;
        }
        ++qs.i_ffn_gate;
    }
    else if (name.find("ffn_up") != std::string::npos) {
        auto info = layer_info(qs.i_ffn_up, qs.n_ffn_up, name.c_str());
        int i_layer = info.first, n_layer = info.second;
        if (qs.params->ffn_up_type < GGML_TYPE_COUNT) new_type = qs.params->ffn_up_type;
        else if (ftype == LLAMA_FTYPE_MOSTLY_IQ3_XS && (i_layer >= n_layer/8 && i_layer < 7*n_layer/8)) {
            new_type = GGML_TYPE_IQ3_XXS;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_IQ3_KL && use_more_bits(i_layer, n_layer)) {
            new_type = GGML_TYPE_IQ4_KS;
        }
        ++qs.i_ffn_up;
    }

    if (custom_type < GGML_TYPE_COUNT) {
        new_type = custom_type;
        LLAMA_LOG_INFO("Using custom type %s for tensor %s\n", ggml_type_name(new_type), name.c_str());
    }

    auto working_type = change_type_if_necessary(new_type, tensor->ne[0], tensor->ne[1]);
    if (working_type != new_type) {
        ++qs.n_fallback;
        new_type = working_type;
    }

    if (name == "token_embd.weight") {
        auto working_type = interleaved_properties(new_type).first;
        if (working_type != new_type) {
            printf("\n============ Token embeddings cannot be quantized with row-interleaved quants\n");
            printf("---> Changed %s to %s\n", ggml_type_name(new_type), ggml_type_name(working_type));
            new_type = working_type;
        }
    }

    if (name == "per_layer_token_embd.weight") {
        auto working_type = interleaved_properties(new_type).first;
        if (working_type != new_type) {
            printf("\n============ Per-layer Token embeddings cannot be quantized with row-interleaved quants\n");
            printf("---> Changed %s to %s\n", ggml_type_name(new_type), ggml_type_name(working_type));
            new_type = working_type;
        }
    }

    return new_type;
}

static size_t llama_tensor_quantize_internal(enum ggml_type new_type, const float * f32_data, void * new_data, const int64_t chunk_size, int64_t nrows, int64_t n_per_row,
        const float * imatrix, const quantize_user_data * user_data, std::vector<std::thread> & workers, const int nthread) {
    if (nthread < 2) {
        // single-thread
        size_t new_size = ggml_quantize_chunk(new_type, f32_data, new_data, 0, nrows, n_per_row, imatrix, user_data);
        if (!ggml_validate_row_data(new_type, new_data, new_size)) {
            throw std::runtime_error("quantized data validation failed");
        }
        return new_size;
    }

    std::mutex mutex;
    int64_t counter = 0;
    size_t new_size = 0;
    bool valid = true;
    auto compute = [&mutex, &counter, &new_size, &valid, new_type, f32_data, new_data, chunk_size,
            nrows, n_per_row, imatrix, user_data]() {
        const int64_t nrows_per_chunk = chunk_size / n_per_row;
        size_t local_size = 0;
        while (true) {
            std::unique_lock<std::mutex> lock(mutex);
            int64_t first_row = counter; counter += nrows_per_chunk;
            if (first_row >= nrows) {
                if (local_size > 0) {
                    new_size += local_size;
                }
                break;
            }
            lock.unlock();
            const int64_t this_nrow = std::min(nrows - first_row, nrows_per_chunk);
            size_t this_size = ggml_quantize_chunk(new_type, f32_data, new_data, first_row * n_per_row, this_nrow, n_per_row, imatrix, user_data);
            local_size += this_size;

            // validate the quantized data
            const size_t row_size  = ggml_row_size(new_type, n_per_row);
            void * this_data = (char *) new_data + first_row * row_size;
            if (!ggml_validate_row_data(new_type, this_data, this_size)) {
                std::unique_lock<std::mutex> lock(mutex);
                valid = false;
                break;
            }
        }
    };
    for (int it = 0; it < nthread - 1; ++it) {
        workers.emplace_back(compute);
    }
    compute();
    for (auto & w : workers) { w.join(); }
    workers.clear();
    if (!valid) {
        throw std::runtime_error("quantized data validation failed");
    }
    return new_size;
}

static llama_ftype repacked_ftype(llama_ftype ftype) {
    static std::unordered_map<llama_ftype, llama_ftype> k_map = {
        { LLAMA_FTYPE_MOSTLY_Q4_0,    LLAMA_FTYPE_MOSTLY_Q4_0_R8    },
        { LLAMA_FTYPE_MOSTLY_Q8_0,    LLAMA_FTYPE_MOSTLY_Q8_0_R8    },
        { LLAMA_FTYPE_MOSTLY_Q5_0,    LLAMA_FTYPE_MOSTLY_Q5_0_R4    },
        { LLAMA_FTYPE_MOSTLY_Q2_K,    LLAMA_FTYPE_MOSTLY_Q2_K_R4    },
        { LLAMA_FTYPE_MOSTLY_Q3_K_S,  LLAMA_FTYPE_MOSTLY_Q3_K_R4    },
        { LLAMA_FTYPE_MOSTLY_Q3_K_M,  LLAMA_FTYPE_MOSTLY_Q3_K_R4    },
        { LLAMA_FTYPE_MOSTLY_Q3_K_L,  LLAMA_FTYPE_MOSTLY_Q3_K_R4    },
        { LLAMA_FTYPE_MOSTLY_Q4_K_S,  LLAMA_FTYPE_MOSTLY_Q4_K_R4    },
        { LLAMA_FTYPE_MOSTLY_Q4_K_M,  LLAMA_FTYPE_MOSTLY_Q4_K_R4    },
        { LLAMA_FTYPE_MOSTLY_Q5_K_S,  LLAMA_FTYPE_MOSTLY_Q5_K_R4    },
        { LLAMA_FTYPE_MOSTLY_Q5_K_M,  LLAMA_FTYPE_MOSTLY_Q5_K_R4    },
        { LLAMA_FTYPE_MOSTLY_Q6_K,    LLAMA_FTYPE_MOSTLY_Q6_K_R4    },
        { LLAMA_FTYPE_MOSTLY_IQ2_XXS, LLAMA_FTYPE_MOSTLY_IQ2_XXS_R4 },
        { LLAMA_FTYPE_MOSTLY_IQ2_XS,  LLAMA_FTYPE_MOSTLY_IQ2_XS_R4  },
        { LLAMA_FTYPE_MOSTLY_IQ3_XXS, LLAMA_FTYPE_MOSTLY_IQ3_XXS_R4 },
        { LLAMA_FTYPE_MOSTLY_IQ1_S,   LLAMA_FTYPE_MOSTLY_IQ1_S_R4   },
        { LLAMA_FTYPE_MOSTLY_IQ4_NL,  LLAMA_FTYPE_MOSTLY_IQ4_NL_R4  },
        { LLAMA_FTYPE_MOSTLY_IQ3_S,   LLAMA_FTYPE_MOSTLY_IQ3_S_R4   },
        { LLAMA_FTYPE_MOSTLY_IQ2_M,   LLAMA_FTYPE_MOSTLY_IQ2_M_R4   },
        { LLAMA_FTYPE_MOSTLY_IQ4_XS,  LLAMA_FTYPE_MOSTLY_IQ4_XS_R8  },
        { LLAMA_FTYPE_MOSTLY_IQ1_M,   LLAMA_FTYPE_MOSTLY_IQ1_M_R4   },
        { LLAMA_FTYPE_MOSTLY_Q6_0,    LLAMA_FTYPE_MOSTLY_Q6_0_R4    },
        { LLAMA_FTYPE_MOSTLY_BF16,    LLAMA_FTYPE_MOSTLY_BF16_R16   },
        { LLAMA_FTYPE_MOSTLY_IQ2_BN,  LLAMA_FTYPE_MOSTLY_IQ2_BN_R4  },
        { LLAMA_FTYPE_MOSTLY_IQ2_K,   LLAMA_FTYPE_MOSTLY_IQ2_K_R4   },
        { LLAMA_FTYPE_MOSTLY_IQ3_K,   LLAMA_FTYPE_MOSTLY_IQ3_K_R4   },
        { LLAMA_FTYPE_MOSTLY_IQ4_K,   LLAMA_FTYPE_MOSTLY_IQ4_K_R4   },
        { LLAMA_FTYPE_MOSTLY_IQ5_K,   LLAMA_FTYPE_MOSTLY_IQ5_K_R4   },
        { LLAMA_FTYPE_MOSTLY_IQ4_KS,  LLAMA_FTYPE_MOSTLY_IQ4_KS_R4  },
        { LLAMA_FTYPE_MOSTLY_IQ5_KS,  LLAMA_FTYPE_MOSTLY_IQ5_KS_R4  },
        { LLAMA_FTYPE_MOSTLY_Q8_KV,   LLAMA_FTYPE_MOSTLY_Q8_KV_R8   },
    };
    if (auto it = k_map.find(ftype); it != k_map.end()) return it->second;
    return ftype;
}

static void do_quantize(int nthread, const ggml_tensor * tensor, ggml_type new_type, const float * f32_data, char * new_data,
        const float * imatrix, std::vector<std::thread> & workers, size_t & new_size, int chunk_size_multiplier,
        const llama_model_quantize_params * params) {
    if (nthread > 1 && (tensor->ne[2] % nthread == 0 || tensor->ne[2] >= 2*nthread)) {
        std::mutex mutex;
        int counter = 0;
        bool valid = true;
        auto compute = [&mutex, &counter, &new_size, &valid, new_type, f32_data, new_data, tensor, imatrix, user_data = params->user_data] () {
            int ne2 = tensor->ne[2];
            auto row_size = ggml_row_size(new_type, tensor->ne[0]);
            auto matrix_size = row_size * tensor->ne[1];
            size_t local_size = 0;
            while (true) {
                std::unique_lock<std::mutex> lock(mutex);
                int i02 = counter++;
                if (i02 >= ne2) {
                    if (local_size > 0) {
                        new_size += local_size;
                    }
                    break;
                }
                lock.unlock();
                auto this_imatrix = imatrix ? imatrix + i02 * tensor->ne[0] : nullptr;
                auto this_data = (char *)new_data + i02*matrix_size;
                auto this_size = ggml_quantize_chunk(new_type, f32_data + i02*tensor->ne[0]*tensor->ne[1], this_data,
                        0, tensor->ne[1], tensor->ne[0], this_imatrix, user_data);
                local_size += this_size;

                // validate the quantized data
                if (!ggml_validate_row_data(new_type, this_data, matrix_size)) {
                    lock.lock();
                    valid = false;
                    break;
                }
            }
        };
        for (int it = 0; it < nthread; ++it) workers.emplace_back(std::thread(compute));
        for (auto & w : workers) w.join();
        workers.clear();
        if (!valid) {
            throw std::runtime_error("quantized data validation failed");
        }
    } else {
        static const int64_t min_chunk_size = 32 * 512;
        const int64_t n_per_row = tensor->ne[0];
        const int64_t nrows     = tensor->ne[1];
        const int64_t chunk_size = (n_per_row >= min_chunk_size
                                 ? n_per_row : n_per_row * ((min_chunk_size + n_per_row - 1)/n_per_row)) * chunk_size_multiplier;

        const int64_t nelements_matrix = tensor->ne[0] * tensor->ne[1];
        const int64_t nchunk = (nelements_matrix + chunk_size - 1)/chunk_size;
        const int64_t nthread_use = nthread > 1 ? std::max((int64_t)1, std::min((int64_t)nthread, nchunk)) : 1;

        // quantize each expert separately since they have different importance matrices
        new_size = 0;
        for (int64_t i03 = 0; i03 < tensor->ne[2]; ++i03) {
            const float * f32_data_03 = f32_data + i03 * nelements_matrix;
            void * new_data_03 = (char *)new_data + ggml_row_size(new_type, n_per_row) * i03 * nrows;
            const float * imatrix_03 = imatrix ? imatrix + i03 * n_per_row : nullptr;

            new_size += llama_tensor_quantize_internal(new_type, f32_data_03, new_data_03, chunk_size,
                    nrows, n_per_row, imatrix_03, params->user_data, workers, nthread_use);
        }
    }
}

static void llama_model_quantize_internal(const std::string & fname_inp, const std::string & fname_out, const llama_model_quantize_params * params) {
    ggml_type default_type;
    llama_ftype ftype = params->ftype;

    switch (ftype) {
        case LLAMA_FTYPE_MOSTLY_Q4_0: default_type = GGML_TYPE_Q4_0; break;
        case LLAMA_FTYPE_MOSTLY_Q4_1: default_type = GGML_TYPE_Q4_1; break;
        case LLAMA_FTYPE_MOSTLY_Q5_0: default_type = GGML_TYPE_Q5_0; break;
        case LLAMA_FTYPE_MOSTLY_Q5_1: default_type = GGML_TYPE_Q5_1; break;
        case LLAMA_FTYPE_MOSTLY_Q6_0: default_type = GGML_TYPE_Q6_0; break;
        case LLAMA_FTYPE_MOSTLY_Q8_0: default_type = GGML_TYPE_Q8_0; break;
        case LLAMA_FTYPE_MOSTLY_Q8_KV:default_type = GGML_TYPE_Q8_KV;break;
        case LLAMA_FTYPE_MOSTLY_F16:  default_type = GGML_TYPE_F16;  break;
        case LLAMA_FTYPE_MOSTLY_BF16: default_type = GGML_TYPE_BF16; break;
        case LLAMA_FTYPE_MOSTLY_BF16_R16: default_type = GGML_TYPE_BF16_R16; break;
        case LLAMA_FTYPE_ALL_F32:     default_type = GGML_TYPE_F32;  break;

        // K-quants
        case LLAMA_FTYPE_MOSTLY_Q2_K_S:
        case LLAMA_FTYPE_MOSTLY_Q2_K:    default_type = GGML_TYPE_Q2_K;    break;
        case LLAMA_FTYPE_MOSTLY_Q2_K_R4: default_type = GGML_TYPE_Q2_K_R4; break;
        case LLAMA_FTYPE_MOSTLY_IQ3_XS:  default_type = GGML_TYPE_IQ3_S;   break;
        case LLAMA_FTYPE_MOSTLY_Q3_K_S:
        case LLAMA_FTYPE_MOSTLY_Q3_K_M:
        case LLAMA_FTYPE_MOSTLY_Q3_K_L:  default_type = GGML_TYPE_Q3_K;    break;
        case LLAMA_FTYPE_MOSTLY_Q3_K_R4: default_type = GGML_TYPE_Q3_K_R4; break;
        case LLAMA_FTYPE_MOSTLY_Q4_K_S:
        case LLAMA_FTYPE_MOSTLY_Q4_K_M:  default_type = GGML_TYPE_Q4_K;    break;
        case LLAMA_FTYPE_MOSTLY_Q4_K_R4: default_type = GGML_TYPE_Q4_K_R4; break;
        case LLAMA_FTYPE_MOSTLY_Q5_K_S:
        case LLAMA_FTYPE_MOSTLY_Q5_K_M:  default_type = GGML_TYPE_Q5_K;    break;
        case LLAMA_FTYPE_MOSTLY_Q5_K_R4: default_type = GGML_TYPE_Q5_K_R4; break;
        case LLAMA_FTYPE_MOSTLY_Q6_K:    default_type = GGML_TYPE_Q6_K;    break;
        case LLAMA_FTYPE_MOSTLY_Q6_K_R4: default_type = GGML_TYPE_Q6_K_R4; break;
        case LLAMA_FTYPE_MOSTLY_Q8_K_R8: default_type = GGML_TYPE_Q8_K_R8; break;
        case LLAMA_FTYPE_MOSTLY_Q8_KV_R8: default_type = GGML_TYPE_Q8_KV_R8; break;
        case LLAMA_FTYPE_MOSTLY_IQ2_XXS: default_type = GGML_TYPE_IQ2_XXS; break;
        case LLAMA_FTYPE_MOSTLY_IQ2_XXS_R4:default_type = GGML_TYPE_IQ2_XXS_R4; break;
        case LLAMA_FTYPE_MOSTLY_IQ2_XS:  default_type = GGML_TYPE_IQ2_XS;  break;
        case LLAMA_FTYPE_MOSTLY_IQ2_XS_R4:default_type = GGML_TYPE_IQ2_XS_R4;  break;
        case LLAMA_FTYPE_MOSTLY_IQ2_KS:  default_type = GGML_TYPE_IQ2_KS;  break;
        case LLAMA_FTYPE_MOSTLY_IQ1_KT:  default_type = GGML_TYPE_IQ1_KT;  break;
        case LLAMA_FTYPE_MOSTLY_IQ2_KT:  default_type = GGML_TYPE_IQ2_KT;  break;
        case LLAMA_FTYPE_MOSTLY_IQ2_S:   default_type = GGML_TYPE_IQ2_XS;  break;
        case LLAMA_FTYPE_MOSTLY_IQ2_M:   default_type = GGML_TYPE_IQ2_S;   break;
        case LLAMA_FTYPE_MOSTLY_IQ2_M_R4:default_type = GGML_TYPE_IQ2_S_R4;break;
        case LLAMA_FTYPE_MOSTLY_IQ3_XXS: default_type = GGML_TYPE_IQ3_XXS; break;
        case LLAMA_FTYPE_MOSTLY_IQ3_KT:  default_type = GGML_TYPE_IQ3_KT;  break;
        case LLAMA_FTYPE_MOSTLY_IQ4_KT:  default_type = GGML_TYPE_IQ4_KT;  break;
        case LLAMA_FTYPE_MOSTLY_IQ3_XXS_R4: default_type = GGML_TYPE_IQ3_XXS_R4; break;
        case LLAMA_FTYPE_MOSTLY_IQ1_S:   default_type = GGML_TYPE_IQ1_S;   break;
        case LLAMA_FTYPE_MOSTLY_IQ1_S_R4:default_type = GGML_TYPE_IQ1_S_R4;break;
        case LLAMA_FTYPE_MOSTLY_IQ1_M_R4:default_type = GGML_TYPE_IQ1_M_R4;break;
        case LLAMA_FTYPE_MOSTLY_IQ1_M:   default_type = GGML_TYPE_IQ1_M;   break;
        case LLAMA_FTYPE_MOSTLY_IQ1_BN:  default_type = GGML_TYPE_IQ1_BN;  break;
        case LLAMA_FTYPE_MOSTLY_IQ2_BN:  default_type = GGML_TYPE_IQ2_BN;  break;
        case LLAMA_FTYPE_MOSTLY_IQ2_BN_R4:default_type = GGML_TYPE_IQ2_BN_R4;break;
        case LLAMA_FTYPE_MOSTLY_IQ4_NL:  default_type = GGML_TYPE_IQ4_NL;  break;
        case LLAMA_FTYPE_MOSTLY_IQ4_NL_R4:default_type = GGML_TYPE_IQ4_NL_R4;break;
        case LLAMA_FTYPE_MOSTLY_IQ4_XS_R8:default_type = GGML_TYPE_IQ4_XS_R8;break;
        case LLAMA_FTYPE_MOSTLY_Q4_0_R8: default_type = GGML_TYPE_Q4_0_R8; break;
        case LLAMA_FTYPE_MOSTLY_Q5_0_R4: default_type = GGML_TYPE_Q5_0_R4; break;
        case LLAMA_FTYPE_MOSTLY_Q6_0_R4: default_type = GGML_TYPE_Q6_0_R4; break;
        case LLAMA_FTYPE_MOSTLY_Q8_0_R8: default_type = GGML_TYPE_Q8_0_R8; break;
        case LLAMA_FTYPE_MOSTLY_MXFP4:   default_type = GGML_TYPE_MXFP4;   break;
        case LLAMA_FTYPE_MOSTLY_MXFP4_R8:default_type = GGML_TYPE_MXFP4_R8;break;
        case LLAMA_FTYPE_MOSTLY_Q1_0_G128: default_type = GGML_TYPE_Q1_0_G128; break;
        case LLAMA_FTYPE_MOSTLY_IQ4_XS:  default_type = GGML_TYPE_IQ4_XS;  break;
        case LLAMA_FTYPE_MOSTLY_IQ4_KS:  default_type = GGML_TYPE_IQ4_KS;  break;
        case LLAMA_FTYPE_MOSTLY_IQ4_KS_R4:default_type = GGML_TYPE_IQ4_KS_R4;break;
        case LLAMA_FTYPE_MOSTLY_IQ5_KS_R4:default_type = GGML_TYPE_IQ5_KS_R4;break;
        case LLAMA_FTYPE_MOSTLY_IQ4_KSS: default_type = GGML_TYPE_IQ4_KSS; break;
        case LLAMA_FTYPE_MOSTLY_IQ5_KS:  default_type = GGML_TYPE_IQ5_KS;  break;
        case LLAMA_FTYPE_MOSTLY_IQ2_K:   default_type = GGML_TYPE_IQ2_K;   break;
        case LLAMA_FTYPE_MOSTLY_IQ2_K_R4:default_type = GGML_TYPE_IQ2_K_R4;break;
        case LLAMA_FTYPE_MOSTLY_IQ3_KS:  default_type = GGML_TYPE_IQ3_KS;  break;
        case LLAMA_FTYPE_MOSTLY_IQ2_KL:  default_type = GGML_TYPE_IQ2_KL;  break;
        case LLAMA_FTYPE_MOSTLY_IQ3_K:   default_type = GGML_TYPE_IQ3_K;   break;
        case LLAMA_FTYPE_MOSTLY_IQ3_K_R4:default_type = GGML_TYPE_IQ3_K_R4;break;
        case LLAMA_FTYPE_MOSTLY_IQ3_KL:  default_type = GGML_TYPE_IQ3_K;   break;
        case LLAMA_FTYPE_MOSTLY_IQ4_K:   default_type = GGML_TYPE_IQ4_K;   break;
        case LLAMA_FTYPE_MOSTLY_IQ4_K_R4:default_type = GGML_TYPE_IQ4_K_R4;break;
        case LLAMA_FTYPE_MOSTLY_IQ5_K:   default_type = GGML_TYPE_IQ5_K;   break;
        case LLAMA_FTYPE_MOSTLY_IQ5_K_R4:default_type = GGML_TYPE_IQ5_K_R4;break;
        case LLAMA_FTYPE_MOSTLY_IQ6_K:   default_type = GGML_TYPE_IQ6_K;   break;
        case LLAMA_FTYPE_MOSTLY_IQ3_S:   default_type = GGML_TYPE_IQ3_S;   break;
        case LLAMA_FTYPE_MOSTLY_IQ3_S_R4:default_type = GGML_TYPE_IQ3_S_R4;break;
        case LLAMA_FTYPE_MOSTLY_IQ3_M:   default_type = GGML_TYPE_IQ3_S;   break;
        case LLAMA_FTYPE_MOSTLY_Q4_0_4_4: default_type = GGML_TYPE_Q4_0_4_4; break;
        case LLAMA_FTYPE_MOSTLY_Q4_0_4_8: default_type = GGML_TYPE_Q4_0_4_8; break;
        case LLAMA_FTYPE_MOSTLY_Q4_0_8_8: default_type = GGML_TYPE_Q4_0_8_8; break;

        default: throw std::runtime_error(format("invalid output file type %d\n", ftype));
    }

    int nthread = params->nthread;

    if (nthread <= 0) {
        nthread = std::thread::hardware_concurrency();
    }

    // mmap consistently increases speed Linux, and also increases speed on Windows with
    // hot cache. It may cause a slowdown on macOS, possibly related to free memory.
#if defined(__linux__) || defined(_WIN32)
    constexpr bool use_mmap = true;
#else
    constexpr bool use_mmap = false;
#endif

    llama_model_kv_override * kv_overrides = nullptr;
    if (params->kv_overrides) {
        auto v = (std::vector<llama_model_kv_override>*)params->kv_overrides;
        kv_overrides = v->data();
    }
    llama_model_loader ml(fname_inp, 0, use_mmap, /*check_tensors*/ true, /* repack_tensors */ false,
            /* use_thp */ false, /* merge_qkv */ false, /* merge_up_gate_exps */ false,
            /* defer_experts */ false, kv_overrides, nullptr);
    ml.init_mappings(false); // no prefetching

    llama_model model;
    try {
        llm_load_arch(ml, model);
    } catch(const std::exception & e) {
        LLAMA_LOG_WARN("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX %s\n", e.what());
    }
    try {
        llm_load_hparams(ml, model, true);
    } catch(const std::exception & e) {
        LLAMA_LOG_WARN("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX %s\n", e.what());
    }

    struct quantize_state_internal qs(model, params);

    if (params->only_copy) {
        ftype = model.ftype;
    }
    const std::unordered_map<std::string, std::vector<float>> * imatrix_data = nullptr;
    if (!params->only_repack && params->imatrix) {
        imatrix_data = static_cast<const std::unordered_map<std::string, std::vector<float>>*>(params->imatrix);
        if (imatrix_data) {
            LLAMA_LOG_INFO("================================ Have weights data with %d entries\n",int(imatrix_data->size()));
            qs.has_imatrix = true;
            // check imatrix for nans or infs
            for (const auto & kv : *imatrix_data) {
                for (float f : kv.second) {
                    if (!std::isfinite(f)) {
                        throw std::runtime_error(format("imatrix contains non-finite value %f\n", f));
                    }
                }
            }
        }
    }

    const size_t align = GGUF_DEFAULT_ALIGNMENT;

    ensure_output_directory(fname_out);

    struct gguf_context * ctx_out = gguf_init_empty();

    // Early exit if partial_requant is enabled and output file already exists
    if (params->partial_requant && !params->keep_split) {
        std::ifstream test_file(fname_out);
        if (test_file) {
            LLAMA_LOG_INFO("%s: output file %s exists, skipping\n", __func__, fname_out.c_str());
            gguf_free(ctx_out);
            return;
        }
    }

    // copy the KV pairs from the input file
    gguf_set_kv     (ctx_out, ml.meta);
    gguf_set_val_u32(ctx_out, "general.quantization_version", GGML_QNT_VERSION); // TODO: use LLM_KV

    // Remove split metadata
    gguf_remove_key(ctx_out, ml.llm_kv(LLM_KV_SPLIT_NO).c_str());
    gguf_remove_key(ctx_out, ml.llm_kv(LLM_KV_SPLIT_COUNT).c_str());
    gguf_remove_key(ctx_out, ml.llm_kv(LLM_KV_SPLIT_TENSORS_COUNT).c_str());

    if (params->kv_overrides) {
        const std::vector<llama_model_kv_override> & overrides = *(const std::vector<llama_model_kv_override> *)params->kv_overrides;
        for (auto & o : overrides) {
            if (o.key[0] == 0) break;
            if (o.tag == LLAMA_KV_OVERRIDE_TYPE_FLOAT) {
                gguf_set_val_f32(ctx_out, o.key, o.val_f64);
            } else if (o.tag == LLAMA_KV_OVERRIDE_TYPE_INT) {
                gguf_set_val_i32(ctx_out, o.key, o.val_i64);
            } else if (o.tag == LLAMA_KV_OVERRIDE_TYPE_BOOL) {
                gguf_set_val_bool(ctx_out, o.key, o.val_bool);
            } else if (o.tag == LLAMA_KV_OVERRIDE_TYPE_STR) {
                gguf_set_val_str(ctx_out, o.key, o.val_str);
            } else {
                LLAMA_LOG_WARN("%s: unknown KV override type for key %s\n", __func__, o.key);
            }
        }
    }

    bool is_repacked = ml.ftype >= LLAMA_FTYPE_MOSTLY_Q4_0_R8 && ml.ftype <= LLAMA_FTYPE_MOSTLY_Q8_K_R8;
    int n_to_repack = 0, n_to_modify = 0;
    const std::vector<std::string> * repack_pattern = nullptr;
    if (params->repack_pattern) repack_pattern = (const std::vector<std::string> *)params->repack_pattern;
    const std::vector<std::string> * keep_pattern = nullptr;
    if (params->keep_pattern) keep_pattern = (const std::vector<std::string> *)params->keep_pattern;

    std::vector<std::vector<float>> single_direction_basis;
    const std::vector<std::vector<float>> * orthogonalize_directions = nullptr;
    const std::vector<std::string> * orthogonalize_pattern = nullptr;
    std::unordered_map<std::string, int> orthogonalize_axes;
    size_t orthogonalize_f32_count = 0;
    const bool has_single_direction = params->orthogonalize_direction != nullptr;
    const bool has_direction_basis = params->orthogonalize_directions != nullptr;
    if (has_single_direction || has_direction_basis || params->orthogonalize_pattern) {
        if ((!has_single_direction && !has_direction_basis) || !params->orthogonalize_pattern) {
            throw std::runtime_error("an orthogonalization direction basis and pattern must be supplied together");
        }
        if (has_single_direction && has_direction_basis) {
            throw std::runtime_error("supply either one orthogonalization direction or a direction basis, not both");
        }
        if (params->only_copy || params->only_repack) {
            throw std::runtime_error("direction orthogonalization cannot be combined with copy-only or repack-only mode");
        }
        if (params->orthogonalize_patch_existing && params->partial_requant) {
            throw std::runtime_error("patch-existing orthogonalization cannot be combined with partial-requant mode");
        }
        if (params->orthogonalize_patch_existing && !params->keep_split) {
            throw std::runtime_error("patch-existing orthogonalization requires keep-split mode");
        }
        if (params->orthogonalize_patch_existing &&
                (params->kv_overrides || params->extra_output_type != GGML_TYPE_COUNT)) {
            throw std::runtime_error("patch-existing orthogonalization cannot change metadata or add an output tensor");
        }
        if (!std::isfinite(params->orthogonalize_scale) ||
                params->orthogonalize_scale <= 0.0f || params->orthogonalize_scale > 1.0f) {
            throw std::runtime_error("orthogonalize_scale must be finite and in (0, 1]");
        }
        if (!std::isfinite(params->orthogonalize_max_residual) ||
                params->orthogonalize_max_residual > 1.0f) {
            throw std::runtime_error("orthogonalize_max_residual must be finite and <= 1");
        }
        if (params->orthogonalize_quant_passes < 1 || params->orthogonalize_quant_passes > 16) {
            throw std::runtime_error("orthogonalize_quant_passes must be in [1, 16]");
        }
        if (params->orthogonalize_quant_passes > 1 && params->orthogonalize_max_residual < 0.0f) {
            throw std::runtime_error("orthogonalize_quant_passes above 1 requires a residual limit");
        }
        if (params->orthogonalize_quant_passes > 1 && params->orthogonalize_scale != 1.0f) {
            throw std::runtime_error("orthogonalize_quant_passes above 1 requires full projection (scale 1)");
        }
        if (has_single_direction) {
            single_direction_basis.push_back(
                    *(const std::vector<float> *) params->orthogonalize_direction);
            orthogonalize_directions = &single_direction_basis;
        } else {
            orthogonalize_directions =
                    (const std::vector<std::vector<float>> *) params->orthogonalize_directions;
        }
        orthogonalize_pattern = (const std::vector<std::string> *) params->orthogonalize_pattern;
        if (orthogonalize_directions->empty()) {
            throw std::runtime_error("orthogonalization direction basis is empty");
        }
        for (size_t index = 0; index < orthogonalize_directions->size(); ++index) {
            const auto & direction = (*orthogonalize_directions)[index];
            if (direction.size() != model.hparams.n_embd) {
                throw std::runtime_error(format(
                        "orthogonalization direction %zu has %zu elements, model residual width is %u",
                        index, direction.size(), model.hparams.n_embd));
            }
            double direction_norm_sq = 0.0;
            for (float value : direction) {
                if (!std::isfinite(value)) {
                    throw std::runtime_error("orthogonalization direction contains a non-finite value");
                }
                direction_norm_sq += double(value) * value;
            }
            if (std::abs(direction_norm_sq - 1.0) > 1e-3) {
                throw std::runtime_error(format(
                        "orthogonalization direction %zu must be normalized (squared norm is %.9f)",
                        index, direction_norm_sq));
            }
            for (size_t previous = 0; previous < index; ++previous) {
                double dot = 0.0;
                for (size_t col = 0; col < direction.size(); ++col) {
                    dot += double(direction[col]) * (*orthogonalize_directions)[previous][col];
                }
                if (std::abs(dot) > 1e-3) {
                    throw std::runtime_error(format(
                            "orthogonalization directions %zu and %zu are not orthogonal (dot %.9f)",
                            previous, index, dot));
                }
            }
        }

        std::vector<std::regex> patterns;
        patterns.reserve(orthogonalize_pattern->size());
        for (const auto & value : *orthogonalize_pattern) {
            patterns.emplace_back(value);
        }
        for (int i = 0; i < ml.n_tensors; ++i) {
            const ggml_tensor * meta = ml.get_tensor_meta(i);
            const std::string name = ggml_get_name(meta);
            bool selected = false;
            for (const auto & pattern : patterns) {
                if (std::regex_search(name, pattern)) {
                    selected = true;
                    break;
                }
            }
            if (!selected) {
                continue;
            }
            if (keep_pattern) {
                for (const auto & value : *keep_pattern) {
                    if (std::regex_search(name, std::regex(value))) {
                        throw std::runtime_error("tensor " + name + " matches both --keep-pattern and --orthogonalize-pattern");
                    }
                }
            }
            if (ggml_n_dims(meta) < 2 || ggml_n_dims(meta) > 4) {
                throw std::runtime_error("selected tensor " + name + " is not a matrix");
            }
            const bool is_token_embedding = name == "token_embd.weight";
            const int axis = is_token_embedding ? 0 : 1;
            const int64_t axis_size = meta->ne[axis];
            if (axis_size != (int64_t) orthogonalize_directions->front().size()) {
                throw std::runtime_error(format(
                        "selected tensor %s has size %lld on residual axis %d, expected %zu",
                        name.c_str(), (long long) axis_size, axis,
                        orthogonalize_directions->front().size()));
            }
            orthogonalize_axes.emplace(name, axis);
            orthogonalize_f32_count += meta->type == GGML_TYPE_F32;
        }
        if (orthogonalize_axes.empty()) {
            throw std::runtime_error("orthogonalization patterns did not match any tensors");
        }
        if (params->orthogonalize_expected_count > 0 &&
                (int) orthogonalize_axes.size() != params->orthogonalize_expected_count) {
            throw std::runtime_error(format(
                    "orthogonalization matched %zu tensors, expected exactly %d",
                    orthogonalize_axes.size(), params->orthogonalize_expected_count));
        }
        LLAMA_LOG_INFO("%s: orthogonalization preflight matched %zu tensors; selected-F32=%zu; basis-rank=%zu; scale %.4f; quant-passes %d; patch-existing=%s; input files remain read-only\n",
                __func__, orthogonalize_axes.size(), orthogonalize_f32_count,
                orthogonalize_directions->size(),
                params->orthogonalize_scale, params->orthogonalize_quant_passes,
                params->orthogonalize_patch_existing ? "yes" : "no");
    } else if (params->orthogonalize_expected_count > 0 ||
            params->orthogonalize_quant_passes != 1 ||
            params->orthogonalize_max_residual >= 0.0f ||
            params->orthogonalize_patch_existing) {
        throw std::runtime_error("orthogonalization controls require a direction and pattern");
    }

    for (int i = 0; i < ml.n_tensors; ++i) {
        const struct ggml_tensor * meta = ml.get_tensor_meta(i);

        const std::string name = ggml_get_name(meta);

        if (params->only_repack) {
            auto repacked_type = (ggml_type)iqk_repacked_type(meta);
            bool repack = false, modify = false;
            if (repacked_type != meta->type) {
                repack = true;
            } else if (!is_repacked) {
                if (iqk_should_modify_tensor(meta)) {
                    modify = true;
                }
            }
            if ((repack || modify) && repack_pattern) {
                bool found = false;
                for (auto& r : *repack_pattern) {
                    std::regex pattern(r);
                    if (std::regex_search(name, pattern)) {
                        found = true;
                        break;
                    }
                }
                if (!found) repack = modify = false;
            }
            if (repack) ++n_to_repack;
            else if (modify) ++n_to_modify;
        }

        // TODO: avoid hardcoded tensor names - use the TN_* constants
        if (name.find("attn_v.weight")   != std::string::npos ||
            name.find("attn_qkv.weight") != std::string::npos) {
            ++qs.n_attention_wv;
        } else if (name == LLM_TN(model.arch)(LLM_TENSOR_OUTPUT, "weight")) {
            qs.has_output = true;
        }
    }

    if (params->only_repack) {
        if (n_to_repack == 0 && n_to_modify == 0) {
            printf("=========================== %s: nothing to do for only_repack option\n", __func__);
            return;
        }
        ftype = repacked_ftype(model.ftype);
        printf("===================== Model ftype: %s: Repacked ftype: %s\n", llama_model_ftype_name(model.ftype).c_str(),
                llama_model_ftype_name(ftype).c_str());
    }

    gguf_set_val_u32(ctx_out, "general.file_type", ftype); // TODO: use LLM_KV

    qs.n_ffn_down = qs.n_ffn_gate = qs.n_ffn_up = (int)model.hparams.n_layer;

    // sanity checks
    //
    //  - qs.n_attention_wv == 0                         for Mamba           models
    //  - qs.n_attention_wv == model.hparams.n_layer     for Transformer     models
    //  - qs.n_attention_wv == 3 * model.hparams.n_layer for Encoder-Decoder models
    //  - model.arch == LLM_ARCH_DECI                    for Deci-Nemotron   models
    //  - model.arch == LLM_ARCH_KIMI_K3                 for hybrid KDA/MLA  models
    //
    // K3 interleaves 24 MLA layers among 69 KDA ones, so its attn_v count is
    // neither 0 nor any multiple of n_layer. The count only feeds the heuristic
    // that spends extra bits on the first and last few attention layers; being
    // wrong about it costs a slightly different bit allocation, not correctness.
    //
    GGML_ASSERT((qs.n_attention_wv == 0 ||
                 qs.n_attention_wv == (int)model.hparams.n_layer ||
                 qs.n_attention_wv == 3 * (int)model.hparams.n_layer ||
                 model.arch == LLM_ARCH_DECI ||
                 model.arch == LLM_ARCH_KIMI_K3 ||
                 model.arch == LLM_ARCH_GEMMA4 ||
                 model.arch == LLM_ARCH_UNKNOWN) && "n_attention_wv is unexpected");

    size_t total_size_org = 0;
    size_t total_size_new = 0;

    std::vector<std::thread> workers;
    workers.reserve(nthread);

    int idx = 0;

    std::vector<no_init<uint8_t>> read_data;
    std::vector<no_init<uint8_t>> work;
    std::vector<no_init<uint8_t>> best_quant_buf;
    std::vector<no_init<float>> f32_conv_buf;
    std::vector<no_init<float>> post_quant_buf;
    std::vector<double> orthogonalization_component_ratios;
    std::vector<double> orthogonalization_post_quant_ratios;

    uint16_t n_split = 1;
    // Assume split index is continuous
    if (params->keep_split) {
        for (int i = 0; i < ml.n_tensors; ++i) {
            n_split = std::max(uint16_t(ml.get_weight(i)->idx+1), n_split);
        }
    }

    // A layout-compatible existing model can act as a copy-on-write template.
    // This mode never regenerates headers or streams non-selected payloads: it
    // validates the complete tensor name/shape layout, then overwrites only the
    // selected, projected payload ranges. The caller is responsible for making
    // the output a distinct inode (the K3 kit uses cp --reflink=always and
    // verifies device/inode pairs before invoking us).
    std::unique_ptr<llama_model_loader> patch_ml;
    std::vector<std::unique_ptr<llama_file>> patch_files;
    if (params->orthogonalize_patch_existing) {
        char first_split_path[PATH_MAX] = {0};
        llama_split_path(first_split_path, sizeof(first_split_path), fname_out.c_str(), 0, n_split);
        patch_ml = std::make_unique<llama_model_loader>(
                first_split_path, 0, /* use_mmap */ false, /* check_tensors */ true,
                /* repack_tensors */ false, /* use_thp */ false,
                /* merge_qkv */ false, /* merge_up_gate_exps */ false,
                /* defer_experts */ false, nullptr, nullptr);
        if (patch_ml->n_tensors != ml.n_tensors || patch_ml->files.size() != n_split) {
            throw std::runtime_error(format(
                    "patch-existing output layout differs: tensors %d/%d, splits %zu/%u",
                    patch_ml->n_tensors, ml.n_tensors, patch_ml->files.size(), n_split));
        }
        for (int i = 0; i < ml.n_tensors; ++i) {
            const ggml_tensor * source = ml.get_tensor_meta(i);
            const auto * patch_weight = patch_ml->get_weight(ggml_get_name(source));
            if (!patch_weight) {
                throw std::runtime_error(
                        "patch-existing output is missing tensor " + std::string(ggml_get_name(source)));
            }
            const ggml_tensor * patch = patch_weight->tensor;
            if (ggml_n_dims(source) != ggml_n_dims(patch)) {
                throw std::runtime_error(
                        "patch-existing output rank differs for tensor " + std::string(ggml_get_name(source)));
            }
            for (int dim = 0; dim < ggml_n_dims(source); ++dim) {
                if (source->ne[dim] != patch->ne[dim]) {
                    throw std::runtime_error(
                            "patch-existing output shape differs for tensor " + std::string(ggml_get_name(source)));
                }
            }
        }
        if (!params->dry_run) {
            patch_files.reserve(patch_ml->files.size());
            for (const auto & file : patch_ml->files) {
                patch_files.emplace_back(
                        std::make_unique<llama_file>(file->get_path().c_str(), "r+b"));
            }
        }
        LLAMA_LOG_INFO("%s: patch-existing validated %d tensors across %zu existing output shards; only %zu selected payloads may be written\n",
                __func__, patch_ml->n_tensors, patch_ml->files.size(), orthogonalize_axes.size());
    }
    std::vector<gguf_context*> ctx_outs(n_split, NULL);
    ctx_outs[0] = ctx_out;

    ggml_tensor extra;
    ggml_tensor * output_meta = ml.get_tensor_meta("output.weight");
    if (!output_meta) {
        output_meta = ml.get_tensor_meta("token_embd.weight");
    }
    ggml_tensor * output_tensor = nullptr;
    if (params->extra_output_type != GGML_TYPE_COUNT) {
        auto meta = ml.get_tensor_meta("output.weight");
        if (!meta) {
            meta = ml.get_tensor_meta("token_embd.weight");
        }
        if (!meta) {
            LLAMA_LOG_WARN("Extra output tensor requested, but 'output.weight' or 'token_embd.weight' not found\n");
        } else {
            LLAMA_LOG_INFO("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX Will duplicate %s as %s\n", meta->name,
                    ggml_type_name(params->extra_output_type));
            auto weights = ml.get_weight(meta->name);
            output_tensor = weights->tensor;
            extra = *output_tensor;
            auto new_type = params->extra_output_type;
            extra.type = new_type;
            auto tt = ggml_internal_get_type_traits(extra.type);
            extra.nb[0] = tt.type_size;
            extra.nb[1] = ggml_row_size(extra.type, extra.ne[0]);
            extra.nb[2] = extra.nb[3] = extra.nb[1]*extra.ne[1];
            extra.data  = nullptr;
            strcpy(extra.name, "output_extra.weight");
            auto orig_size = ggml_nbytes(output_tensor);
            auto new_size  = ggml_nbytes(&extra);
            if (new_size >= orig_size) {
                LLAMA_LOG_INFO("No, duplicating it makes no sense as the new size (%zu) is greater than the original size (%zu)\n",
                        new_size, orig_size);
                output_tensor = nullptr;
            }
        }
    }

    // populate the original tensors so we get an initial meta data
    for (int i = 0; i < ml.n_tensors; ++i) {
        auto weight = ml.get_weight(i);
        uint16_t i_split = params->keep_split ? weight->idx : 0;
        struct ggml_tensor * tensor = weight->tensor;
        if (ctx_outs[i_split] == NULL) {
            ctx_outs[i_split] = gguf_init_empty();
        }
        gguf_add_tensor(ctx_outs[i_split], tensor);
        if (tensor == output_tensor) {
            gguf_add_tensor(ctx_outs[i_split], &extra);
        }
    }

    // Set split info if needed
    if (n_split > 1) {
        for (size_t i = 0; i < ctx_outs.size(); ++i) {
            gguf_set_val_u16(ctx_outs[i], ml.llm_kv(LLM_KV_SPLIT_NO).c_str(), i);
            gguf_set_val_u16(ctx_outs[i], ml.llm_kv(LLM_KV_SPLIT_COUNT).c_str(), n_split);
            gguf_set_val_i32(ctx_outs[i], ml.llm_kv(LLM_KV_SPLIT_TENSORS_COUNT).c_str(), ml.n_tensors);
        }
    }

    int cur_split = -1;
    std::ofstream fout;
    std::vector<bool> split_skipped(n_split, false);
    auto close_ofstream = [&]() {
        // Write metadata and close file handler
        if (fout.is_open()) {
            fout.seekp(0);
            std::vector<uint8_t> data(gguf_get_meta_size(ctx_outs[cur_split]));
            gguf_get_meta_data(ctx_outs[cur_split], data.data());
            fout.write((const char *) data.data(), data.size());
            fout.close();
        }
    };
    auto new_ofstream = [&](int index) {
        cur_split = index;
        if (params->dry_run || params->orthogonalize_patch_existing) {
            return;
        }
        GGML_ASSERT(ctx_outs[cur_split] && "Find uninitialized gguf_context");
        std::string fname = fname_out;
        if (params->keep_split) {
            char split_path[PATH_MAX] = {0};
            llama_split_path(split_path, sizeof(split_path), fname_out.c_str(), cur_split, n_split);
            fname = std::string(split_path);
        }

        if (params->partial_requant) {
            std::ifstream test_file(fname);
            if (test_file) {
                LLAMA_LOG_INFO("%s: split file %s exists, skipping\n", __func__, fname.c_str());
                split_skipped[cur_split] = true;
                fout = std::ofstream();
                return;
            }
        }

        ensure_output_directory(fname);
        fout = std::ofstream(fname, std::ios::binary);
        fout.exceptions(std::ofstream::failbit); // fail fast on write errors
        const size_t meta_size = gguf_get_meta_size(ctx_outs[cur_split]);
        // placeholder for the meta data
        ::zeros(fout, meta_size);
    };

    const auto tn = LLM_TN(model.arch);
    new_ofstream(0);
    for (int i = 0; i < ml.n_tensors; ++i) {
        auto weight = ml.get_weight(i);
        struct ggml_tensor * tensor = weight->tensor;
        if (weight->idx != cur_split && params->keep_split) {
            close_ofstream();
            new_ofstream(weight->idx);
        }

        if (params->partial_requant && split_skipped[cur_split]) {
            const std::string name = ggml_get_name(tensor);
            gguf_set_tensor_type(ctx_outs[cur_split], name.c_str(), tensor->type);
            gguf_set_tensor_data(ctx_outs[cur_split], name.c_str(), tensor->data, ggml_nbytes(tensor));
            continue;
        }

        std::string name = ggml_get_name(tensor);
        const auto orthogonalize_it = orthogonalize_axes.find(name);
        const bool needs_orthogonalization = orthogonalize_it != orthogonalize_axes.end();
        llama_direction_projection_stats orthogonalization_source_stats;

        if (params->orthogonalize_patch_existing && !needs_orthogonalization) {
            const auto * patch_weight = patch_ml->get_weight(name.c_str());
            GGML_ASSERT(patch_weight != nullptr);
            ++idx;
            total_size_org += ggml_nbytes(tensor);
            total_size_new += ggml_nbytes(patch_weight->tensor);
            continue;
        }

        if (!ml.use_mmap) {
            if (read_data.size() < ggml_nbytes(tensor)) {
                read_data.resize(ggml_nbytes(tensor));
            }
            tensor->data = read_data.data();
        }
        ml.load_data_for(tensor);

        LLAMA_LOG_INFO("[%4d/%4d] %36s - [%s], type = %6s, ",
               ++idx, ml.n_tensors,
               ggml_get_name(tensor),
               llama_format_tensor_shape(tensor).c_str(),
               ggml_type_name(tensor->type));

        bool quantize = tensor->type != GGML_TYPE_I32 &&
                        tensor->type != GGML_TYPE_I64 &&
                        tensor->type != GGML_TYPE_I16 &&
                        tensor->type != GGML_TYPE_I8; // i.e., do not quantize tensors holding int values

        // This used to be a regex, but <regex> has an extreme cost to compile times.
        quantize &= name.rfind("weight") == name.size() - 6; // ends with 'weight'?

        // quantize only 2D and 3D tensors (experts)
        quantize &= (ggml_n_dims(tensor) >= 2);

        // do not quantize norm tensors
        quantize &= name.find("_norm.weight") == std::string::npos;

        quantize &= params->quantize_output_tensor || name != "output.weight";
        quantize &= !params->only_copy;

        // --keep-f32: a tensor that arrives as F32 in an already-quantized model
        // is F32 because the publisher chose that, and the choice is usually
        // load-bearing - GLM-DSA's indexer.proj scores which tokens attention
        // sees, K3's ssm_conv1d is asserted F32 by ggml_ssm_conv. They are also
        // uniformly tiny, so quantizing them buys nothing and can break the
        // model or quietly degrade it. Only meaningful when requantizing; on a
        // real F32 source model this would quantize nothing, hence opt-in.
        if (params->keep_f32 && tensor->type == GGML_TYPE_F32 && !needs_orthogonalization) {
            quantize = false;
        }

        // --keep-pattern: copy these tensors through byte for byte.
        //
        // There is no "same type" shortcut in this function - asking for a type
        // a tensor already has still dequantizes and requantizes it, which for
        // an IQ2_XS expert is hours of work and a lossy round trip to produce
        // (nearly) what was already there. That makes "requantize only the
        // non-expert tensors" impossible to express, which matters because on a
        // model like Kimi K3 the experts are 93% of the file and 19% of what a
        // token reads: the tensors worth requantizing are exactly the ones a
        // whole-file pass spends all its time NOT changing.
        if (keep_pattern) {
            for (auto & r : *keep_pattern) {
                if (std::regex_search(tensor->name, std::regex(r))) {
                    quantize = false;
                    break;
                }
            }
        }

        // do not quantize expert gating tensors
        // NOTE: can't use LLM_TN here because the layer number is not known
        if (name.find("ffn_gate_inp.weight") != std::string::npos) {
            if (params->ffn_gate_inp_type == GGML_TYPE_COUNT || params->ffn_gate_inp_type == tensor->type) {
                quantize = false;
            }
        }
        //quantize &= name.find("ffn_gate_inp.weight") == std::string::npos;

        // do not quantize positional embeddings and token types (BERT)
        quantize &= name != LLM_TN(model.arch)(LLM_TENSOR_POS_EMBD,    "weight");
        quantize &= name != LLM_TN(model.arch)(LLM_TENSOR_TOKEN_TYPES, "weight");

        // do not quantize Mamba's small yet 2D weights
        // NOTE: can't use LLM_TN here because the layer number is not known
        // Match the PREFIX, not "ssm_conv1d.weight": Kimi K3 has a separate
        // conv per projection (ssm_conv1d_q/_k/_v.weight) and the literal name
        // never matched, so its conv weights were eligible for quantization.
        // They are 4 columns wide, so they cannot take a k-quant and
        // change_type_if_necessary silently fell them back from F32 to Q8_0 -
        // whereupon ggml_ssm_conv aborts on GGML_ASSERT(src2->nb[0] ==
        // sizeof(float)) at the first token. Quantizing a 4-wide kernel saves
        // nothing anyway.
        quantize &= name.find("ssm_conv1d")        == std::string::npos;
        quantize &= name.find("ssm_x.weight")      == std::string::npos;
        quantize &= name.find("ssm_dt.weight")     == std::string::npos;

        // do not quantize relative position bias (T5)
        quantize &= name.find("attn_rel_b.weight") == std::string::npos;

        // quantize the extra output tensor
        quantize = tensor == output_tensor || quantize;

        enum ggml_type new_type;
        void * new_data = nullptr;
        size_t new_size = 0;

        if (params->only_repack) {
            ggml_type repacked_type = (ggml_type)iqk_repacked_type(tensor);
            bool modify = !is_repacked && iqk_should_modify_tensor(tensor);
            if ((modify || repacked_type != tensor->type) && repack_pattern) {
                bool found = false;
                for (auto& r : *repack_pattern) {
                    std::regex pattern(r);
                    if (std::regex_search(tensor->name, pattern)) {
                        found = true; break;
                    }
                }
                if (!found) {
                    modify = false;
                    repacked_type = tensor->type;
                }
            }
            if (modify || repacked_type != tensor->type) {
                new_type = repacked_type;
                new_size = ggml_nbytes(tensor);
                if (!params->dry_run) {
                    if ((int)work.size() < new_size) work.resize(new_size);
                    new_data = work.data();

                    auto aux_tensor = *tensor;
                    aux_tensor.data = work.data();
                    std::memcpy(aux_tensor.data, tensor->data, new_size);

                    if (repacked_type != tensor->type) {
                        iqk_repack_tensor(&aux_tensor);
                        GGML_ASSERT(aux_tensor.type == repacked_type);
                    } else {
                        bool did_modify = iqk_modify_tensor(&aux_tensor);
                        GGML_ASSERT(did_modify);
                    }
                }
            }
            else {
                new_type = tensor->type;
                new_size = ggml_nbytes(tensor);
                new_data = tensor->data;
            }
            LLAMA_LOG_INFO("size = %8.3f MB, type = %s\n", new_size/1024.0/1024.0, ggml_type_name(new_type));
            goto QuantizationDone;
        }

        if (quantize) {

            new_type = default_type;

            // get more optimal quantization type based on the tensor shape, layer, etc.
            if (params->pure) {
                auto working_type = change_type_if_necessary(new_type, tensor->ne[0], tensor->ne[1]);
                if (working_type != new_type) {
                    ++qs.n_fallback;
                    new_type = working_type;
                }
            }
            else if (ggml_is_quantized(default_type)) {
                new_type = llama_tensor_get_type(qs, new_type, tensor, ftype);
            }
            if (params->token_embedding_type < GGML_TYPE_COUNT && strcmp(tensor->name, "token_embd.weight") == 0) {
                new_type = params->token_embedding_type;
            }
            if (params->per_layer_token_embedding_type < GGML_TYPE_COUNT && strcmp(tensor->name, "per_layer_token_embd.weight") == 0) {
                new_type = params->per_layer_token_embedding_type;
            }
            if (params->output_tensor_type < GGML_TYPE_COUNT && strcmp(tensor->name, "output.weight") == 0) {
                new_type = params->output_tensor_type;
            }
            else if (params->only_copy && tensor == output_tensor) {
                new_type = tensor->type;
            }
            if (params->ffn_gate_inp_type < GGML_TYPE_COUNT && name.find("ffn_gate_inp.weight") != std::string::npos) {
                new_type = params->ffn_gate_inp_type;
            }
            if (params->attn_q_type < GGML_TYPE_COUNT && strcmp(tensor->name, "attn_q.weight") == 0) {
                new_type = params->attn_q_type;
            }
            if (params->attn_k_type < GGML_TYPE_COUNT && strcmp(tensor->name, "attn_k.weight") == 0) {
                new_type = params->attn_k_type;
            }
            if (params->attn_v_type < GGML_TYPE_COUNT && strcmp(tensor->name, "attn_v.weight") == 0) {
                new_type = params->attn_v_type;
            }
            if (params->attn_qkv_type < GGML_TYPE_COUNT && strcmp(tensor->name, "attn_qkv.weight") == 0) {
                new_type = params->attn_qkv_type;
            }
            if (params->attn_output_type < GGML_TYPE_COUNT && strcmp(tensor->name, "attn_output.weight") == 0) {
                new_type = params->attn_output_type;
            }
            if (params->ffn_gate_type < GGML_TYPE_COUNT && strcmp(tensor->name, "ffn_gate") == 0) {
                new_type = params->ffn_gate_type;
            }
            if (params->ffn_down_type < GGML_TYPE_COUNT && strcmp(tensor->name, "ffn_down") == 0) {
                new_type = params->ffn_down_type;
            }
            if (params->ffn_up_type < GGML_TYPE_COUNT && strcmp(tensor->name, "ffn_up") == 0) {
                new_type = params->ffn_up_type;
            }

            if (strcmp(tensor->name, "token_embd.weight") == 0) {
                // token embeddings cannot be quantized with row-interleaved quants
                auto working_type = interleaved_properties(new_type).first;
                if (working_type != new_type) {
                    printf("\n============ Token embeddings cannot be quantized with row-interleaved quants\n");
                    printf("---> Changed %s to %s\n", ggml_type_name(new_type), ggml_type_name(working_type));
                    new_type = working_type;
                }
            }

            // If we've decided to quantize to the same type the tensor is already
            // in then there's nothing to do.
            if (tensor != output_tensor) {
                quantize &= tensor->type != new_type;
            }
            // A same-type conversion is normally skipped, but a selected
            // tensor must still be decoded, projected, and encoded again.
            if (needs_orthogonalization) {
                quantize = true;
            }
        }

        if (needs_orthogonalization && !quantize) {
            throw std::runtime_error("selected tensor " + name + " is not eligible for quantization/projection");
        }

        if (!quantize) {
            new_type = tensor->type;
            new_data = tensor->data;
            new_size = ggml_nbytes(tensor);
            LLAMA_LOG_INFO("size = %8.3f MB\n", ggml_nbytes(tensor)/1024.0/1024.0);
        } else {
            const int64_t nelements = ggml_nelements(tensor);

            const float * imatrix = nullptr;
            if (imatrix_data) {
                auto it = imatrix_data->find(tensor->name);
                if (it == imatrix_data->end()) {
                    if (auto pos1 = name.find("ffn_up_exps.weight"), pos2 = name.find("ffn_gate_exps.weight"); pos1 != std::string::npos || pos2 != std::string::npos) {
                        // Merged ffn_up/gate_exps hack
                        auto pos = pos1 != std::string::npos ? pos1 : pos2;
                        auto merged_name = name.substr(0, pos) + "ffn_gate_up_exps.weight";
                        it = imatrix_data->find(merged_name);
                        if (it == imatrix_data->end()) {
                            auto up_name = name.substr(0, pos) + "ffn_up_exps.weight";
                            it = imatrix_data->find(up_name);
                        }
                    } else if (auto pos = name.find("ffn_gate_up_exps.weight"); pos != std::string::npos) {
                        auto not_merged_name = name.substr(0, pos) + "ffn_up_exps.weight";
                        it = imatrix_data->find(not_merged_name);
                    } else if (auto pos2 = name.find("ffn_gate.weight"); pos2 != std::string::npos) {
                        auto up_name = name.substr(0, pos2) + "ffn_up.weight";
                        it = imatrix_data->find(up_name);
                    } else {
                        // MLA hack: most imatrix files floating around the Internet have been computed with standard attention.
                        //           This means that the imatrix file does not contain data for the *.attn_k_b.weight and *.attn_v_b.weight
                        //           required by MLA. But the *.attn_v_b.weight tensors "see" the exact same activations as the
                        //           *.attn_kv_b.weight tensors used in standard attention. Hence, if we find imatrix data for
                        //           *.attn_kv_b.weight we can use it for *.attn_v_b.weight and vice versa.
                        std::string name{tensor->name};
                        static std::array<std::string, 2> alternatives{".attn_v_b.weight", ".attn_kv_b.weight"};
                        for (int j = 0; j < int(alternatives.size()); ++j) {
                            if (auto pos = name.find(alternatives[j]); pos != std::string::npos) {
                                int j1 = (j + 1) % alternatives.size();
                                auto alternative_name = name.substr(0, pos) + alternatives[j1];
                                it = imatrix_data->find(alternative_name);
                                break;
                            }
                        }
                    }
                }
                if (it == imatrix_data->end()) {
                    LLAMA_LOG_INFO("\n====== %s: did not find weights for %s\n", __func__, tensor->name);
                } else {
                    if (it->second.size() == (size_t)tensor->ne[0]*tensor->ne[2]) {
                        imatrix = it->second.data();
                    } else {
                        LLAMA_LOG_INFO("\n====== %s: imatrix size %d is different from tensor size %d for %s\n", __func__,
                                int(it->second.size()), int(tensor->ne[0]*tensor->ne[2]), tensor->name);

                        // this can happen when quantizing an old mixtral model with split tensors with a new incompatible imatrix
                        // this is a significant error and it may be good idea to abort the process if this happens,
                        // since many people will miss the error and not realize that most of the model is being quantized without an imatrix
                        // tok_embd should be ignored in this case, since it always causes this warning
                        if (name != tn(LLM_TENSOR_TOKEN_EMBD, "weight")) {
                            throw std::runtime_error(format("imatrix size %d is different from tensor size %d for %s",
                                    int(it->second.size()), int(tensor->ne[0]*tensor->ne[2]), tensor->name));
                        }
                    }
                }
            }
            if (!params->ignore_imatrix_rules && !imatrix) {
                bool is_very_low_bpw_quant = new_type == GGML_TYPE_IQ2_XXS    ||
                                             new_type == GGML_TYPE_IQ2_XXS_R4 ||
                                             new_type == GGML_TYPE_IQ2_XS     ||
                                             new_type == GGML_TYPE_IQ2_XS_R4  ||
                                             new_type == GGML_TYPE_IQ2_S      ||
                                             new_type == GGML_TYPE_IQ2_S_R4   ||
                                             new_type == GGML_TYPE_IQ1_S      ||
                                             new_type == GGML_TYPE_IQ1_S_R4   ||
                                             new_type == GGML_TYPE_IQ1_M      ||
                                             new_type == GGML_TYPE_IQ1_M_R4   ||
                                             new_type == GGML_TYPE_IQ1_KT     ||
                                             new_type == GGML_TYPE_IQ2_KT     ||
                                            (new_type == GGML_TYPE_Q2_K && ftype == LLAMA_FTYPE_MOSTLY_Q2_K_S);
                if (is_very_low_bpw_quant && strcmp(tensor->name, "token_embd.weight") && strcmp(tensor->name, "output.weight")) {
                    LLAMA_LOG_ERROR("\n\n============================================================\n");
                    LLAMA_LOG_ERROR("Missing importance matrix for tensor %s in a very low-bit quantization\n", tensor->name);
                    LLAMA_LOG_ERROR("The result will be garbage, so bailing out\n");
                    LLAMA_LOG_ERROR("============================================================\n\n");
                    throw std::runtime_error(format("Missing importance matrix for tensor %s in a very low-bit quantization", tensor->name));
                }
            }

            int chunk_size_multiplier = 1;
            auto [working_type, num_rows] = interleaved_properties(new_type);
            if (tensor->ne[1] % num_rows != 0) {
                new_type = working_type;
            } else {
                chunk_size_multiplier = num_rows;
            }

            LLAMA_LOG_INFO("converting to %s .. ", ggml_type_name(new_type));
            fflush(stdout);

            if (params->dry_run) {
                new_size = tensor->ne[2] * tensor->ne[1] * ggml_row_size(new_type, tensor->ne[0]);
            } else {
                float * f32_data;

                if (tensor->type == GGML_TYPE_F32) {
                    if (needs_orthogonalization) {
                        if (f32_conv_buf.size() < (size_t) nelements) {
                            f32_conv_buf.resize(nelements);
                        }
                        std::memcpy(f32_conv_buf.data(), tensor->data, nelements * sizeof(float));
                        f32_data = (float *) f32_conv_buf.data();
                    } else {
                        f32_data = (float *) tensor->data;
                    }
                } else if (ggml_is_quantized(tensor->type) && !params->allow_requantize) {
                    throw std::runtime_error(format("requantizing from type %s is disabled", ggml_type_name(tensor->type)));
                } else {
                    llama_tensor_dequantize_internal(tensor, f32_conv_buf, workers, nelements, nthread);
                    f32_data = (float *) f32_conv_buf.data();
                }

                if (needs_orthogonalization) {
                    const auto stats = llama_direction_orthogonalize_subspace_f32(
                            f32_data, tensor, *orthogonalize_directions,
                            params->orthogonalize_scale, orthogonalize_it->second, nthread);
                    if (!(stats.component_norm_sq > 0.0) || !std::isfinite(stats.component_norm_sq) ||
                            !(stats.tensor_norm_sq > 0.0) || !std::isfinite(stats.tensor_norm_sq)) {
                        throw std::runtime_error(
                                "selected tensor " + name + " has invalid source projection statistics");
                    }
                    orthogonalization_source_stats = stats;
                    const double ratio = stats.tensor_norm_sq > 0.0
                            ? std::sqrt(stats.component_norm_sq / stats.tensor_norm_sq)
                            : 0.0;
                    orthogonalization_component_ratios.push_back(ratio);
                    LLAMA_LOG_INFO("orthogonalize: %s residual-axis=%d source-component=%.6f%% float-residual=%.6f%%\n",
                            name.c_str(), orthogonalize_it->second, 100.0 * ratio,
                            100.0 * ratio * std::abs(1.0f - params->orthogonalize_scale));
                }

                auto expected_size = ggml_row_size(new_type, tensor->ne[0])*tensor->ne[1]*tensor->ne[2]*tensor->ne[3];

                if (work.size() < expected_size) { //(size_t)nelements * 4) {
                    //work.resize(nelements * 4); // upper bound on size
                    work.resize(expected_size); // upper bound on size
                }
                new_data = work.data();

                if (params->extra_output_type != GGML_TYPE_COUNT && tensor == output_tensor) {
                    if (needs_orthogonalization) {
                        throw std::runtime_error(
                                "direction orthogonalization cannot target a duplicated extra output tensor");
                    }
                    auto cur_size = ggml_nbytes(tensor);
                    if (new_type != tensor->type) {
                        do_quantize(nthread, tensor, new_type, f32_data, (char *)new_data, imatrix, workers,
                                new_size, chunk_size_multiplier, params);
                        gguf_set_tensor_type(ctx_outs[cur_split], name.c_str(), new_type);
                        gguf_set_tensor_data(ctx_outs[cur_split], name.c_str(), new_data, new_size);
                        fout.write((const char *) new_data, new_size);
                        zeros(fout, GGML_PAD(new_size, align) - new_size);
                        total_size_new += new_size;
                        LLAMA_LOG_INFO("size = %8.2f MiB -> %8.2f MiB\n", cur_size/1024.0/1024.0, new_size/1024.0/1024.0);
                    } else {
                        gguf_set_tensor_type(ctx_outs[cur_split], name.c_str(), tensor->type);
                        gguf_set_tensor_data(ctx_outs[cur_split], name.c_str(), tensor->data, cur_size);
                        fout.write((const char *) tensor->data, cur_size);
                        zeros(fout, GGML_PAD(cur_size, align) - cur_size);
                        total_size_new += cur_size;
                        LLAMA_LOG_INFO("size = %8.2f MiB -> %8.2f MiB\n", cur_size/1024.0/1024.0, cur_size/1024.0/1024.0);
                    }

                    LLAMA_LOG_INFO("[%4d/%4d] %36s - [%s], type = %6s, ",
                           ++idx, ml.n_tensors,
                           ggml_get_name(tensor),
                           llama_format_tensor_shape(tensor).c_str(),
                           ggml_type_name(tensor->type));

                    new_type = params->extra_output_type;
                    chunk_size_multiplier = 1;
                    auto [working_type, num_rows] = interleaved_properties(new_type);
                    if (tensor->ne[1] % num_rows != 0) {
                        new_type = working_type;
                    } else {
                        chunk_size_multiplier = num_rows;
                    }
                    LLAMA_LOG_INFO("converting to %s .. ", ggml_type_name(new_type));
                    fflush(stdout);

                    do_quantize(nthread, tensor, new_type, f32_data, (char *)new_data, imatrix, workers,
                        new_size, 1, params);

                    name = extra.name;
                } else if (!needs_orthogonalization) {
                    do_quantize(nthread, tensor, new_type, f32_data, (char *)new_data, imatrix, workers,
                            new_size, chunk_size_multiplier, params);
                } else {
                    // Quantization can reintroduce a direction component. Keep
                    // the original projected F32 data in f32_conv_buf, decode
                    // each candidate encoding into a separate buffer, and
                    // subtract only that measured residue before retrying. This
                    // avoids accumulating loss by requantizing a decoded quant.
                    double best_retained_ratio = std::numeric_limits<double>::infinity();
                    std::vector<double> best_component_magnitudes;
                    for (int pass = 1; pass <= params->orthogonalize_quant_passes; ++pass) {
                        new_size = 0;
                        do_quantize(nthread, tensor, new_type, f32_data, (char *)new_data, imatrix, workers,
                                new_size, chunk_size_multiplier, params);

                        ggml_tensor quantized = *tensor;
                        quantized.type = new_type;
                        quantized.data = new_data;
                        llama_tensor_dequantize_internal(
                                &quantized, post_quant_buf, workers, nelements, nthread);
                        std::vector<double> component_magnitudes;
                        // A damped step explores the discontinuous quantized
                        // code space without the two-point oscillation caused
                        // by subtracting the full decoded residue.
                        const float correction_scale = pass < params->orthogonalize_quant_passes ? 0.25f : 0.0f;
                        const auto post_stats = llama_direction_remove_measured_subspace_f32(
                                f32_data, (const float *) post_quant_buf.data(), tensor,
                                *orthogonalize_directions, correction_scale,
                                orthogonalize_it->second, nthread,
                                orthogonalize_it->second == 0 ? &component_magnitudes : nullptr);
                        if (!(post_stats.component_norm_sq >= 0.0) ||
                                !std::isfinite(post_stats.component_norm_sq) ||
                                !(post_stats.tensor_norm_sq >= 0.0) ||
                                !std::isfinite(post_stats.tensor_norm_sq)) {
                            throw std::runtime_error(
                                    "selected tensor " + name + " has invalid post-quant projection statistics");
                        }
                        const double post_tensor_ratio = post_stats.tensor_norm_sq > 0.0
                                ? std::sqrt(post_stats.component_norm_sq / post_stats.tensor_norm_sq)
                                : 0.0;
                        const double retained_ratio = llama_direction_retained_component_ratio(
                                orthogonalization_source_stats, post_stats);
                        if (!std::isfinite(retained_ratio) || !std::isfinite(post_tensor_ratio)) {
                            throw std::runtime_error(
                                    "selected tensor " + name + " has invalid post-quant projection ratios");
                        }
                        if (best_quant_buf.size() < new_size) {
                            best_quant_buf.resize(new_size);
                        }
                        if (orthogonalize_it->second == 0) {
                            // Embedding vectors are quantization rows. Keep the
                            // best independently encoded row seen across passes;
                            // combining those rows remains a valid tensor and is
                            // strictly no worse than any whole-pass candidate.
                            const size_t row_size = ggml_row_size(new_type, tensor->ne[0]);
                            const int64_t n_rows = nelements / tensor->ne[0];
                            GGML_ASSERT(component_magnitudes.size() == (size_t) n_rows);
                            GGML_ASSERT(row_size * n_rows == new_size);
                            if (best_component_magnitudes.empty()) {
                                best_component_magnitudes.resize(n_rows);
                                std::memcpy(best_quant_buf.data(), new_data, new_size);
                                for (int64_t row = 0; row < n_rows; ++row) {
                                    best_component_magnitudes[row] = component_magnitudes[row];
                                }
                            } else {
                                for (int64_t row = 0; row < n_rows; ++row) {
                                    const double magnitude = component_magnitudes[row];
                                    if (magnitude < best_component_magnitudes[row]) {
                                        best_component_magnitudes[row] = magnitude;
                                        std::memcpy(best_quant_buf.data() + row * row_size,
                                                (const uint8_t *) new_data + row * row_size, row_size);
                                    }
                                }
                            }
                            double best_component_norm_sq = 0.0;
                            for (double magnitude : best_component_magnitudes) {
                                best_component_norm_sq += magnitude * magnitude;
                            }
                            best_retained_ratio = std::sqrt(
                                    best_component_norm_sq / orthogonalization_source_stats.component_norm_sq);
                        } else if (retained_ratio < best_retained_ratio) {
                            // For axis 1, quantization rows and independently
                            // measured residual columns are transverse, so only
                            // whole-tensor candidates can be safely selected.
                            best_retained_ratio = retained_ratio;
                            std::memcpy(best_quant_buf.data(), new_data, new_size);
                        }

                        LLAMA_LOG_INFO("orthogonalize: %s quant-pass=%d/%d retained-source-component=%.6f%% best=%.6f%% absolute-component=%.6f%%\n",
                                name.c_str(), pass, params->orthogonalize_quant_passes,
                                100.0 * retained_ratio, 100.0 * best_retained_ratio,
                                100.0 * post_tensor_ratio);

                        if (params->orthogonalize_max_residual < 0.0f ||
                                best_retained_ratio <= params->orthogonalize_max_residual) {
                            break;
                        }
                        if (pass < params->orthogonalize_quant_passes) {
                            LLAMA_LOG_INFO("orthogonalize: %s correcting measured quantization residue before pass %d\n",
                                    name.c_str(), pass + 1);
                        }
                    }

                    std::memcpy(new_data, best_quant_buf.data(), new_size);
                    ggml_tensor best_quantized = *tensor;
                    best_quantized.type = new_type;
                    best_quantized.data = new_data;
                    llama_tensor_dequantize_internal(
                            &best_quantized, post_quant_buf, workers, nelements, nthread);
                    const auto best_post_stats = llama_direction_orthogonalize_subspace_f32(
                            (float *) post_quant_buf.data(), tensor, *orthogonalize_directions,
                            0.0f, orthogonalize_it->second, nthread);
                    const double post_tensor_ratio = best_post_stats.tensor_norm_sq > 0.0
                            ? std::sqrt(best_post_stats.component_norm_sq / best_post_stats.tensor_norm_sq)
                            : 0.0;
                    const double retained_ratio = llama_direction_retained_component_ratio(
                            orthogonalization_source_stats, best_post_stats);
                    if (!std::isfinite(retained_ratio) || !std::isfinite(post_tensor_ratio)) {
                        throw std::runtime_error(
                                "selected tensor " + name + " has invalid best post-quant projection ratios");
                    }
                    if (params->orthogonalize_max_residual >= 0.0f &&
                            retained_ratio > params->orthogonalize_max_residual) {
                        throw std::runtime_error(format(
                                "selected tensor %s retained %.6f%% of its source direction component after %d quantization pass(es), above limit %.6f%%",
                                name.c_str(), 100.0 * retained_ratio,
                                params->orthogonalize_quant_passes,
                                100.0 * params->orthogonalize_max_residual));
                    }
                    orthogonalization_post_quant_ratios.push_back(retained_ratio);
                    LLAMA_LOG_INFO("orthogonalize: %s post-quant-residual=%.6f%% absolute-component=%.6f%%\n",
                            name.c_str(), 100.0 * retained_ratio, 100.0 * post_tensor_ratio);
                }

            }
            LLAMA_LOG_INFO("size = %8.2f MiB -> %8.2f MiB\n", ggml_nbytes(tensor)/1024.0/1024.0, new_size/1024.0/1024.0);
        }

QuantizationDone:;
        if (params->orthogonalize_patch_existing) {
            if (!needs_orthogonalization) {
                throw std::runtime_error("patch-existing reached an unselected tensor unexpectedly: " + name);
            }
            const auto * patch_weight = patch_ml->get_weight(name.c_str());
            if (!patch_weight) {
                throw std::runtime_error("patch-existing output lost tensor " + name);
            }
            if (patch_weight->tensor->type != new_type || ggml_nbytes(patch_weight->tensor) != new_size) {
                throw std::runtime_error(format(
                        "patch-existing layout mismatch for %s: output %s/%zu bytes, generated %s/%zu bytes",
                        name.c_str(), ggml_type_name(patch_weight->tensor->type),
                        ggml_nbytes(patch_weight->tensor), ggml_type_name(new_type), new_size));
            }
            if (!params->dry_run) {
                if (!new_data || patch_weight->idx >= patch_files.size()) {
                    throw std::runtime_error("patch-existing has no writable payload for " + name);
                }
                patch_files[patch_weight->idx]->seek(patch_weight->offs, SEEK_SET);
                patch_files[patch_weight->idx]->write_raw(new_data, new_size);
                LLAMA_LOG_INFO("orthogonalize: %s patched-existing shard=%u offset=%zu bytes=%zu\n",
                        name.c_str(), patch_weight->idx, patch_weight->offs, new_size);
            }
        }
        total_size_org += ggml_nbytes(tensor);
        total_size_new += new_size;

        if (!params->dry_run && !params->orthogonalize_patch_existing && !split_skipped[cur_split]) {
            // update the gguf meta data as we go
            gguf_set_tensor_type(ctx_outs[cur_split], name.c_str(), new_type);
            gguf_set_tensor_data(ctx_outs[cur_split], name.c_str(), new_data, new_size);

            // write tensor data + padding
            fout.write((const char *) new_data, new_size);
            zeros(fout, GGML_PAD(new_size, align) - new_size);
        }
    }
    close_ofstream();
    for (auto & c:ctx_outs) {
        gguf_free(c);
    }

    LLAMA_LOG_INFO("%s: model size  = %8.2f MB\n", __func__, total_size_org/1024.0/1024.0);
    LLAMA_LOG_INFO("%s: quant size  = %8.2f MB\n", __func__, total_size_new/1024.0/1024.0);

    if (!orthogonalization_component_ratios.empty()) {
        std::sort(orthogonalization_component_ratios.begin(), orthogonalization_component_ratios.end());
        LLAMA_LOG_INFO("%s: orthogonalized %zu tensors; source component ratio min/median/max %.6f%%/%.6f%%/%.6f%%\n",
                __func__, orthogonalization_component_ratios.size(),
                100.0 * orthogonalization_component_ratios.front(),
                100.0 * orthogonalization_component_ratios[orthogonalization_component_ratios.size()/2],
                100.0 * orthogonalization_component_ratios.back());
    }
    if (!orthogonalization_post_quant_ratios.empty()) {
        std::sort(orthogonalization_post_quant_ratios.begin(), orthogonalization_post_quant_ratios.end());
        LLAMA_LOG_INFO("%s: post-quant retained source component min/median/max %.6f%%/%.6f%%/%.6f%%\n",
                __func__,
                100.0 * orthogonalization_post_quant_ratios.front(),
                100.0 * orthogonalization_post_quant_ratios[orthogonalization_post_quant_ratios.size()/2],
                100.0 * orthogonalization_post_quant_ratios.back());
    }

    if (qs.n_fallback > 0) {
        LLAMA_LOG_WARN("%s: WARNING: %d of %d tensor(s) required fallback quantization\n",
                __func__, qs.n_fallback, qs.n_k_quantized + qs.n_fallback);
    }
}

uint32_t llama_model_quantize(
        const char * fname_inp,
        const char * fname_out,
        const llama_model_quantize_params * params) {
    try {
        llama_model_quantize_internal(fname_inp, fname_out, params);
        return 0;
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: failed to quantize: %s\n", __func__, err.what());
        return 1;
    }
}
