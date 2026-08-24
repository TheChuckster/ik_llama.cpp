# cvector-generator

This example demonstrates how to generate a control vector using gguf models.

Related PRs:
- [Add support for control vectors](https://github.com/ggerganov/llama.cpp/pull/5970)
- (Issue) [Generate control vector using llama.cpp](https://github.com/ggerganov/llama.cpp/issues/6880)
- [Add cvector-generator example](https://github.com/ggerganov/llama.cpp/pull/7514)

## Examples

```sh
# CPU only
./cvector-generator -m ./llama-3.Q4_K_M.gguf

# With GPU
./cvector-generator -m ./llama-3.Q4_K_M.gguf -ngl 99

# With advanced options
./cvector-generator -m ./llama-3.Q4_K_M.gguf -ngl 99 --pca-iter 2000 --pca-batch 100

# Using mean value instead of PCA
./cvector-generator -m ./llama-3.Q4_K_M.gguf --method mean

# Difference in means at the final, model-templated prompt position
./cvector-generator -m ./model.gguf \
    --positive-file harmful.txt --negative-file harmless.txt \
    --method mean-last --apply-chat-template --jinja

# To see help message
./cvector-generator -h
# Then, have a look at "cvector" section
```

`mean-last` captures one activation per prompt and layer instead of averaging
over every token. The positive and negative files must contain the same number
of non-empty lines. With `--apply-chat-template`, each line is rendered as one
user message with the GGUF's embedded template and an assistant generation
prompt; `--jinja` is required so extraction cannot silently fall back to a
different built-in template. As in the original control-vector generator, the
final transformer layer is omitted. Capture uses exact `l_out-N` node names so
architectures that delay output-row narrowing, including Kimi K3, cannot add an
extra final-layer vector or accidentally match similarly prefixed graph nodes.

## Tips and tricks

If you have multiple lines per prompt, you can escape the newline character (change it to `\n`). For example:

```
<|im_start|>system\nAct like a person who is extremely happy.<|im_end|>
<|im_start|>system\nYou are in a very good mood today<|im_end|>
```

Example to use output file with `llama-cli`:

(Tips: The control vector works better when apply to layers higher than 10)

```sh
./llama-cli -m ./llama-3.Q4_K_M.gguf -p "<|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nSing a song<|im_end|><|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n" --special --control-vector-scaled ./control_vector.gguf 0.8 --control-vector-layer-range 10 31
```
