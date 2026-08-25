#include "chat.h"

#undef NDEBUG
#include <cassert>
#include <stdexcept>
#include <string>

static common_chat_params valid_params() {
    common_chat_params params;
    params.prompt = "rendered<think>";
    params.generation_prompt = "<think>";
    params.supports_thinking = true;
    params.thinking_start_tag = "<think>";
    params.thinking_end_tag = "</think>";
    return params;
}

static void expect_invalid(common_chat_params params,
                           const std::string & prefill, bool enable_thinking) {
    const auto prompt = params.prompt;
    const auto generation_prompt = params.generation_prompt;
    bool threw = false;
    try {
        common_chat_apply_reasoning_prefill(params, prefill, enable_thinking);
    } catch (const std::invalid_argument &) {
        threw = true;
    }
    assert(threw);
    assert(params.prompt == prompt);
    assert(params.generation_prompt == generation_prompt);
}

static void test_parser_reconstruction() {
    const std::string template_str = R"(
{%- set ns_token = ']<]minimax[>[' -%}
{%- set toolcall_begin_token = ns_token ~ '<tool_call>' -%}
{%- set toolcall_end_token = ns_token ~ '</tool_call>' -%}
{%- for message in messages -%}
{{- message.role ~ ': ' ~ message.content ~ '\n' -}}
{%- endfor -%}
{%- if tools -%}
{{- toolcall_begin_token ~ ns_token ~ '<invoke name="example">' ~ ns_token ~ '</invoke>' ~ toolcall_end_token -}}
{%- endif -%}
{%- if add_generation_prompt -%}
{{- '<mm:think>' -}}
{%- endif -%}
)";

    common_chat_templates_ptr tmpls(
        common_chat_templates_init(nullptr, template_str));
    common_chat_templates_inputs inputs;
    common_chat_msg user;
    user.role = "user";
    user.content = "question";
    inputs.messages = {user};
    inputs.add_generation_prompt = true;
    inputs.reasoning_format = COMMON_REASONING_FORMAT_DEEPSEEK;
    inputs.enable_thinking = true;

    auto params = common_chat_templates_apply(tmpls.get(), inputs);
    common_chat_apply_reasoning_prefill(params, "I know that.", true);
    const std::string expected_suffix = "<mm:think>I know that.";
    assert(params.prompt.size() >= expected_suffix.size());
    assert(params.prompt.compare(params.prompt.size() - expected_suffix.size(),
                                 expected_suffix.size(), expected_suffix) == 0);
    assert(params.generation_prompt == "<mm:think>I know that.");

    common_chat_parser_params parser_params(params);
    parser_params.reasoning_format = COMMON_REASONING_FORMAT_DEEPSEEK;
    parser_params.parser.load(params.parser);
    const auto parsed =
        common_chat_parse("continued</mm:think>\nanswer", false, parser_params);
    assert(parsed.reasoning_content == "I know that.continued");
    assert(parsed.content == "\nanswer");

    const auto partial = common_chat_parse("continued", true, parser_params);
    assert(partial.reasoning_content == "I know that.continued");
    assert(partial.content.empty());
}

int main() {
    {
        auto params = valid_params();
        common_chat_apply_reasoning_prefill(params, "I know that.", true);
        assert(params.prompt == "rendered<think>I know that.");
        assert(params.generation_prompt == "<think>I know that.");
    }
    {
        auto params = valid_params();
        common_chat_apply_reasoning_prefill(params, "", false);
        assert(params.prompt == "rendered<think>");
        assert(params.generation_prompt == "<think>");
    }
    {
        auto params = valid_params();
        expect_invalid(params, "I know that.", false);
        params.supports_thinking = false;
        expect_invalid(params, "I know that.", true);
    }
    {
        auto params = valid_params();
        params.thinking_start_tag.clear();
        expect_invalid(params, "I know that.", true);
        params = valid_params();
        params.thinking_end_tag.clear();
        expect_invalid(params, "I know that.", true);
        params = valid_params();
        params.thinking_end_tag = params.thinking_start_tag;
        expect_invalid(params, "I know that.", true);
    }
    {
        auto params = valid_params();
        params.prompt += "not-at-start";
        expect_invalid(params, "I know that.", true);
        params = valid_params();
        params.generation_prompt += "not-at-start";
        expect_invalid(params, "I know that.", true);
    }
    {
        auto params = valid_params();
        expect_invalid(params, "seed<think>", true);
        expect_invalid(params, "seed</think>", true);
        expect_invalid(params, std::string(16 * 1024 + 1, 'x'), true);
    }
    test_parser_reconstruction();
    return 0;
}
