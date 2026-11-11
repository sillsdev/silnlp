from typing import Any, Dict

from silnlp.nmt.reasoning_estimate import (
    MeasuredReasoning,
    NoReasoning,
    ReasoningBudget,
    ReasoningRates,
    ReasoningToTheLimit,
    UnknownReasoning,
)

# --- no reasoning, or reasoning of unknown extent ---------------------------------------------


def test_a_model_that_does_not_reason_spends_no_tokens_on_it():
    assert NoReasoning().tokens_for(100) == 0


def test_a_model_that_does_not_reason_is_not_mentioned(caplog):
    with caplog.at_level("INFO"):
        NoReasoning().tokens_for(100)

    assert caplog.text == ""


def test_reasoning_of_unknown_extent_is_left_out_of_the_estimate():
    assert UnknownReasoning("a-model").tokens_for(100) == 0


def test_reasoning_of_unknown_extent_is_announced_with_a_warning_when_first_estimated(caplog):
    with caplog.at_level("WARNING"):
        UnknownReasoning("gemini/gemini-3-pro-preview").tokens_for(100)

    assert "gemini/gemini-3-pro-preview can spend output tokens on reasoning" in caplog.text


def test_reasoning_of_unknown_extent_is_announced_only_once(caplog):
    estimate = UnknownReasoning("a-model")

    with caplog.at_level("WARNING"):
        estimate.tokens_for(100)
        estimate.tokens_for(100)

    assert caplog.text.count("can spend output tokens on reasoning") == 1


def test_reasoning_of_unknown_extent_adds_nothing_to_the_total():
    assert UnknownReasoning("a-model").describe_total(0) == ""


# --- a thinking budget ------------------------------------------------------------------------


def test_a_thinking_budget_is_spent_in_full_on_every_request():
    assert ReasoningBudget("a-model", 4096, 1024).tokens_for(100) == 1024


def test_a_thinking_budget_gets_only_what_the_reply_leaves_of_max_new_tokens():
    assert ReasoningBudget("a-model", 1000, 1024).tokens_for(100) == 900


def test_a_reply_longer_than_max_new_tokens_leaves_no_room_for_reasoning():
    assert ReasoningBudget("a-model", 1000, 1024).tokens_for(1200) == 0


def test_a_thinking_budget_that_fits_beside_the_reply_gives_no_warning(caplog):
    with caplog.at_level("WARNING"):
        ReasoningBudget("a-model", 4096, 1024).tokens_for(100)

    assert caplog.text == ""


def test_a_thinking_budget_that_crowds_out_the_reply_warns_once(caplog):
    estimate = ReasoningBudget("a-model", 1000, 1024)

    with caplog.at_level("WARNING"):
        estimate.tokens_for(100)
        estimate.tokens_for(100)

    assert caplog.text.count("cut short or blank") == 1
    assert "1,000 of infer.max_new_tokens" in caplog.text


def test_a_thinking_budget_is_announced_as_an_upper_bound_when_first_estimated(caplog):
    with caplog.at_level("INFO"):
        ReasoningBudget("anthropic/claude-sonnet-4-5", 4096, 1024).tokens_for(100)

    assert (
        "each request to anthropic/claude-sonnet-4-5 is estimated to spend its whole thinking budget of 1,024 tokens"
        in caplog.text
    )
    assert "upper bound" in caplog.text


def test_a_total_with_a_thinking_budget_is_described_as_an_upper_bound():
    description = ReasoningBudget("a-model", 4096, 1024).describe_total(2048)

    assert "up to 2,048 reasoning tokens" in description
    assert "upper bound" in description


# --- a measured rate --------------------------------------------------------------------------


def test_measured_reasoning_scales_with_the_reply():
    assert MeasuredReasoning("a-model", 4096, 0.5, "a test").tokens_for(100) == 50


def test_measured_reasoning_gets_only_what_the_reply_leaves_of_max_new_tokens():
    assert MeasuredReasoning("a-model", 4096, 12.0, "a test").tokens_for(600) == 3496


def test_measured_reasoning_that_crowds_out_the_reply_warns(caplog):
    with caplog.at_level("WARNING"):
        MeasuredReasoning("a-model", 4096, 12.0, "a test").tokens_for(600)

    assert "cut short or blank" in caplog.text


def test_measured_reasoning_that_fits_beside_the_reply_gives_no_warning(caplog):
    with caplog.at_level("WARNING"):
        MeasuredReasoning("a-model", 4096, 0.5, "a test").tokens_for(600)

    assert caplog.text == ""


def test_measured_reasoning_is_announced_with_its_rate_and_where_it_was_measured_when_first_estimated(caplog):
    with caplog.at_level("INFO"):
        MeasuredReasoning("openai/gpt-6.1-sol", 4096, 0.28, "a test").tokens_for(100)

    assert (
        "openai/gpt-6.1-sol is estimated to reason for 0.28 tokens per reply token, as measured for a test"
        in caplog.text
    )


def test_measured_reasoning_is_announced_only_once(caplog):
    estimate = MeasuredReasoning("a-model", 4096, 0.28, "a test")

    with caplog.at_level("INFO"):
        estimate.tokens_for(100)
        estimate.tokens_for(100)

    assert caplog.text.count("tokens per reply token") == 1


def test_a_total_with_measured_reasoning_is_described_as_approximate():
    assert "about 1,234 reasoning tokens" in MeasuredReasoning("a-model", 4096, 0.5, "a test").describe_total(1234)


# --- reasoning until max_new_tokens runs out ---------------------------------------------------


def test_reasoning_to_the_limit_takes_all_the_reply_leaves_of_max_new_tokens():
    assert ReasoningToTheLimit("a-model", 4096, "a test").tokens_for(100) == 3996


def test_reasoning_to_the_limit_leaves_nothing_when_the_reply_fills_max_new_tokens():
    assert ReasoningToTheLimit("a-model", 1000, "a test").tokens_for(1200) == 0


def test_reasoning_to_the_limit_warns_once_that_replies_will_be_cut_short(caplog):
    estimate = ReasoningToTheLimit("gemini/gemini-3.7-flash", 4096, "a test")

    with caplog.at_level("WARNING"):
        estimate.tokens_for(100)
        estimate.tokens_for(100)

    assert caplog.text.count("cut short or blank") == 1
    assert "gemini/gemini-3.7-flash is estimated to reason until infer.max_new_tokens runs out" in caplog.text
    assert "Measured with max_new_tokens at 4,096 for a test." in caplog.text


def test_a_total_with_reasoning_to_the_limit_is_described_as_an_upper_bound():
    description = ReasoningToTheLimit("a-model", 4096, "a test").describe_total(2048)

    assert "up to 2,048 reasoning tokens" in description
    assert "upper bound" in description


# --- the rates measured for each provider --------------------------------------------------------


def _tokens_for_a_100_token_reply(model: str, litellm_options: Dict[str, Any]) -> int:
    return ReasoningRates.for_model(model).estimate_for(litellm_options, 100_000).tokens_for(100)


def _leaves_reasoning_out(model: str, litellm_options: Dict[str, Any], caplog) -> bool:
    with caplog.at_level("WARNING"):
        tokens = _tokens_for_a_100_token_reply(model, litellm_options)
    return tokens == 0 and "leaves out" in caplog.text


def _announcement(model: str, litellm_options: Dict[str, Any], caplog) -> str:
    with caplog.at_level("INFO"):
        _tokens_for_a_100_token_reply(model, litellm_options)
    return caplog.text


def test_gpt_6_at_its_default_effort_reasons_for_about_half_its_reply():
    assert _tokens_for_a_100_token_reply("openai/gpt-6.1-sol", {}) == 54


def test_gpt_6_at_minimal_effort_barely_reasons():
    assert _tokens_for_a_100_token_reply("openai/gpt-6.1-sol", {"reasoning_effort": "minimal"}) == 3


def test_gpt_6_at_high_effort_reasons_for_several_times_its_reply():
    assert _tokens_for_a_100_token_reply("openai/gpt-6.1-sol", {"reasoning_effort": "high"}) == 570


def test_gpt_6_at_extra_high_effort_reasons_for_twelve_times_its_reply():
    assert _tokens_for_a_100_token_reply("openai/gpt-6.1-sol", {"reasoning_effort": "xhigh"}) == 1200


def test_gpt_6_with_reasoning_turned_off_spends_nothing_on_it():
    assert _tokens_for_a_100_token_reply("openai/gpt-6.1-terra", {"reasoning_effort": "none"}) == 0


def test_every_gpt_6_tier_reasons_at_the_measured_rate():
    assert _tokens_for_a_100_token_reply("openrouter/openai/gpt-6-luna", {"reasoning_effort": "high"}) == 570


def test_openrouters_form_of_the_effort_setting_is_read():
    assert _tokens_for_a_100_token_reply("openrouter/openai/gpt-6.1-sol", {"reasoning": {"effort": "low"}}) == 6


def test_a_gpt_reasoning_budget_is_estimated_like_medium_effort():
    assert _tokens_for_a_100_token_reply("openrouter/openai/gpt-6.1-sol", {"reasoning": {"max_tokens": 3000}}) == 28


def test_litellms_effort_setting_wins_over_openrouters_budget():
    options = {"reasoning_effort": "high", "reasoning": {"max_tokens": 3000}}
    assert _tokens_for_a_100_token_reply("openrouter/openai/gpt-6.1-sol", options) == 570


def test_an_earlier_gpt_model_reasons_at_the_rate_measured_for_gpt_6_1_sol():
    assert _tokens_for_a_100_token_reply("openai/gpt-5", {"reasoning_effort": "high"}) == 570


def test_a_gpt_model_openrouter_lists_under_an_alias_reasons_at_the_measured_rate():
    assert _tokens_for_a_100_token_reply("openrouter/~openai/gpt-sol-latest", {"reasoning_effort": "low"}) == 6


def test_a_gpt_effort_that_was_not_measured_is_left_out_with_a_warning(caplog):
    assert _leaves_reasoning_out("openai/gpt-6.1-sol", {"reasoning_effort": "max"}, caplog)


def test_an_openai_model_not_named_gpt_is_left_out_with_a_warning(caplog):
    assert _leaves_reasoning_out("openai/o3", {"reasoning_effort": "high"}, caplog)


def test_a_gpt_rate_is_announced_as_measured_on_gpt_6_1_sol_at_its_effort(caplog):
    announcement = _announcement("openai/gpt-5", {"reasoning_effort": "medium"}, caplog)
    assert "0.28 tokens per reply token, as measured for GPT-6.1 Sol at reasoning effort 'medium'" in announcement


def test_claude_at_its_default_effort_reasons_for_nearly_three_times_its_reply():
    assert _tokens_for_a_100_token_reply("openrouter/anthropic/claude-sonnet-5.5", {}) == 270


def test_claude_with_reasoning_turned_off_spends_nothing_on_it():
    assert (
        _tokens_for_a_100_token_reply("openrouter/anthropic/claude-haiku-5.5", {"reasoning": {"effort": "none"}}) == 0
    )


def test_claude_at_low_effort_does_not_reason():
    assert _tokens_for_a_100_token_reply("openrouter/anthropic/claude-haiku-5.5", {"reasoning": {"effort": "low"}}) == 0


def test_claude_at_medium_effort_reasons_for_nearly_four_times_its_reply():
    options = {"reasoning": {"effort": "medium"}}
    assert _tokens_for_a_100_token_reply("openrouter/anthropic/claude-haiku-5.5", options) == 380


def test_claude_at_high_effort_reasons_for_sixteen_times_its_reply():
    assert (
        _tokens_for_a_100_token_reply("openrouter/anthropic/claude-haiku-5.5", {"reasoning": {"effort": "high"}})
        == 1600
    )


def test_claude_at_extra_high_effort_reasons_for_sixty_five_times_its_reply():
    options = {"reasoning": {"effort": "xhigh"}}
    assert _tokens_for_a_100_token_reply("openrouter/anthropic/claude-haiku-5.5", options) == 6500


def test_litellms_form_of_the_effort_setting_is_read_for_claude():
    assert _tokens_for_a_100_token_reply("anthropic/claude-opus-5-5", {"reasoning_effort": "medium"}) == 380


def test_a_claude_model_on_bedrock_reasons_at_the_measured_rate():
    assert (
        _tokens_for_a_100_token_reply("bedrock/anthropic.claude-sonnet-5-5-v1:0", {"reasoning_effort": "high"}) == 1600
    )


def test_a_claude_reasoning_budget_is_estimated_in_full():
    options = {"reasoning": {"max_tokens": 2000}}
    assert _tokens_for_a_100_token_reply("openrouter/anthropic/claude-haiku-5.5", options) == 2000


def test_a_claude_effort_that_was_not_measured_is_left_out_with_a_warning(caplog):
    assert _leaves_reasoning_out("anthropic/claude-opus-5-5", {"reasoning_effort": "max"}, caplog)


def test_a_claude_rate_is_announced_as_the_higher_of_the_two_models_measured(caplog):
    announcement = _announcement("anthropic/claude-opus-5-5", {"reasoning_effort": "high"}, caplog)
    assert "as measured for Claude Haiku 5.5 and Sonnet 5.5 at reasoning effort 'high', whichever reasoned more" in (
        announcement
    )


def test_gemini_at_its_default_effort_reasons_until_max_new_tokens_runs_out():
    assert _tokens_for_a_100_token_reply("gemini/gemini-3.7-flash", {}) == 100_000 - 100


def test_gemini_at_low_effort_reasons_for_about_four_times_its_reply():
    assert _tokens_for_a_100_token_reply("gemini/gemini-3.7-flash", {"reasoning_effort": "low"}) == 420


def test_gemini_at_minimal_effort_reasons_like_low_effort():
    assert _tokens_for_a_100_token_reply("gemini/gemini-3.7-flash", {"reasoning_effort": "minimal"}) == 420


def test_gemini_with_reasoning_turned_off_reasons_like_low_effort():
    assert _tokens_for_a_100_token_reply("gemini/gemini-3.7-flash", {"reasoning_effort": "none"}) == 420


def test_gemini_at_medium_effort_reasons_until_max_new_tokens_runs_out():
    assert _tokens_for_a_100_token_reply("gemini/gemini-3.1-pro-preview", {"reasoning_effort": "medium"}) == 99_900


def test_gemini_at_high_effort_reasons_until_max_new_tokens_runs_out():
    assert _tokens_for_a_100_token_reply("gemini/gemini-3.5-flash-lite", {"reasoning_effort": "high"}) == 99_900


def test_openrouters_form_of_the_effort_setting_is_read_for_gemini():
    assert _tokens_for_a_100_token_reply("openrouter/google/gemini-3.7-flash", {"reasoning": {"effort": "low"}}) == 420


def test_a_gemini_reasoning_budget_is_estimated_in_full():
    assert (
        _tokens_for_a_100_token_reply("openrouter/google/gemini-3.7-flash", {"reasoning": {"max_tokens": 1024}}) == 1024
    )


def test_a_gemini_effort_that_was_not_measured_is_left_out_with_a_warning(caplog):
    assert _leaves_reasoning_out("gemini/gemini-3.1-pro-preview", {"reasoning_effort": "xhigh"}, caplog)


def test_a_gemini_rate_is_announced_as_the_highest_of_the_three_models_measured(caplog):
    announcement = _announcement("gemini/gemini-3.7-flash", {"reasoning_effort": "low"}, caplog)
    assert "Gemini 3.7 Flash, 3.5 Flash Lite and 3.1 Pro at reasoning effort 'low', whichever reasoned most" in (
        announcement
    )


def test_another_providers_reasoning_is_left_out_with_a_warning(caplog):
    assert _leaves_reasoning_out("mistral/magistral-medium", {"reasoning_effort": "high"}, caplog)


def test_another_providers_reasoning_budget_is_estimated_in_full():
    assert _tokens_for_a_100_token_reply("openrouter/deepseek/deepseek-r1", {"reasoning": {"max_tokens": 1500}}) == 1500
