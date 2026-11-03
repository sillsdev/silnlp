from typing import Any, Dict, Optional

from silnlp.nmt.reasoning_estimate import (
    GptSixReasoningRates,
    MeasuredReasoning,
    NoReasoning,
    ReasoningBudget,
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


# --- the rates measured for GPT-6 -------------------------------------------------------------


def _gpt_six_tokens_for_a_100_token_reply(model: str, litellm_options: Dict[str, Any]) -> Optional[int]:
    estimate = GptSixReasoningRates().estimate_for(model, litellm_options, 100_000)
    return None if estimate is None else estimate.tokens_for(100)


def test_gpt_6_at_its_default_effort_reasons_for_about_half_its_reply():
    assert _gpt_six_tokens_for_a_100_token_reply("openai/gpt-6.1-sol", {}) == 54


def test_gpt_6_at_minimal_effort_barely_reasons():
    assert _gpt_six_tokens_for_a_100_token_reply("openai/gpt-6.1-sol", {"reasoning_effort": "minimal"}) == 3


def test_gpt_6_at_high_effort_reasons_for_several_times_its_reply():
    assert _gpt_six_tokens_for_a_100_token_reply("openai/gpt-6.1-sol", {"reasoning_effort": "high"}) == 570


def test_gpt_6_at_extra_high_effort_reasons_for_twelve_times_its_reply():
    assert _gpt_six_tokens_for_a_100_token_reply("openai/gpt-6.1-sol", {"reasoning_effort": "xhigh"}) == 1200


def test_gpt_6_with_reasoning_turned_off_spends_nothing_on_it():
    assert _gpt_six_tokens_for_a_100_token_reply("openai/gpt-6.1-terra", {"reasoning_effort": "none"}) == 0


def test_every_gpt_6_tier_reasons_at_the_measured_rate():
    assert _gpt_six_tokens_for_a_100_token_reply("openrouter/openai/gpt-6-luna", {"reasoning_effort": "high"}) == 570


def test_openrouters_form_of_the_effort_setting_is_read():
    assert _gpt_six_tokens_for_a_100_token_reply("openrouter/openai/gpt-6.1-sol", {"reasoning": {"effort": "low"}}) == 6


def test_an_effort_that_was_not_measured_has_no_gpt_6_rate():
    assert _gpt_six_tokens_for_a_100_token_reply("openai/gpt-6.1-sol", {"reasoning_effort": "max"}) is None


def test_an_earlier_openai_model_has_no_gpt_6_rate():
    assert _gpt_six_tokens_for_a_100_token_reply("openai/gpt-5", {"reasoning_effort": "high"}) is None


def test_a_model_from_another_provider_has_no_gpt_6_rate():
    assert _gpt_six_tokens_for_a_100_token_reply("anthropic/claude-sonnet-4-5", {}) is None


def test_a_gpt_6_rate_is_announced_with_the_effort_it_was_measured_at(caplog):
    estimate = GptSixReasoningRates().estimate_for("openai/gpt-6.1-sol", {"reasoning_effort": "medium"}, 4096)
    assert estimate is not None

    with caplog.at_level("INFO"):
        estimate.tokens_for(100)

    assert "0.28 tokens per reply token, as measured for GPT-6 models at reasoning effort 'medium'" in caplog.text
