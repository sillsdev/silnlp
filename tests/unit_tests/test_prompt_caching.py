from typing import Dict, List

from silnlp.nmt.prompt_caching import (
    AnthropicPromptCachingPolicy,
    PromptCachingPolicy,
    PromptTokenCount,
    SharedSystemMessageCacheStatus,
)


def _prompt() -> List[Dict[str, str]]:
    return [{"role": "system", "content": "translate"}, {"role": "user", "content": "hello"}]


def _caches(model: str, system_message_tokens: int) -> bool:
    cache_status = SharedSystemMessageCacheStatus(system_message_tokens, already_cached=True)
    return AnthropicPromptCachingPolicy(model).split_prompt(10000, cache_status).read_from_cache > 0


def test_a_claude_request_marks_its_system_message_for_caching():
    assert AnthropicPromptCachingPolicy("anthropic/claude-sonnet-5-5").mark(_prompt()) == [
        {"role": "system", "content": [{"type": "text", "text": "translate", "cache_control": {"type": "ephemeral"}}]},
        {"role": "user", "content": "hello"},
    ]


def test_marking_a_claude_request_leaves_the_callers_messages_as_they_were():
    messages = _prompt()

    AnthropicPromptCachingPolicy("anthropic/claude-sonnet-5-5").mark(messages)

    assert messages == _prompt()


def test_a_claude_model_is_recognized_through_any_provider():
    marked = PromptCachingPolicy.for_model("openrouter/anthropic/claude-haiku-5.5").mark(_prompt())
    assert marked == AnthropicPromptCachingPolicy("anthropic/claude-haiku-5-5").mark(_prompt())


def test_a_request_to_another_model_is_left_unmarked():
    assert PromptCachingPolicy.for_model("openrouter/openai/gpt-6.1-sol").mark(_prompt()) == _prompt()


def test_a_claude_request_writes_a_system_message_no_earlier_request_has_cached():
    caching_policy = AnthropicPromptCachingPolicy("anthropic/claude-sonnet-5-5")
    prompt_token_count = caching_policy.split_prompt(1000, SharedSystemMessageCacheStatus(800, already_cached=False))
    assert prompt_token_count == PromptTokenCount(1000, written_to_cache=800)


def test_a_claude_request_reads_a_system_message_an_earlier_request_has_cached():
    caching_policy = AnthropicPromptCachingPolicy("anthropic/claude-sonnet-5-5")
    prompt_token_count = caching_policy.split_prompt(1000, SharedSystemMessageCacheStatus(800, already_cached=True))
    assert prompt_token_count == PromptTokenCount(1000, read_from_cache=800)


def test_claude_is_estimated_alike_whether_or_not_the_provider_is_assumed_to_cache():
    caching_policy = AnthropicPromptCachingPolicy("anthropic/claude-sonnet-5-5")
    cache_status = SharedSystemMessageCacheStatus(800, already_cached=True)
    assert caching_policy.split_prompt_if_cached(1000, cache_status) == caching_policy.split_prompt(1000, cache_status)


def test_claude_haiku_5_5_caches_a_system_message_of_512_tokens():
    assert _caches("anthropic/claude-haiku-5-5", 512)


def test_claude_haiku_5_5_leaves_a_shorter_system_message_uncached():
    assert not _caches("anthropic/claude-haiku-5-5", 511)


def test_claude_sonnet_5_5_caches_a_system_message_shorter_than_claude_sonnet_5_needs():
    assert _caches("anthropic/claude-sonnet-5-5", 1023)


def test_claude_sonnet_5_leaves_a_system_message_under_1024_tokens_uncached():
    assert not _caches("anthropic/claude-sonnet-5", 1023)


def test_claude_opus_4_7_leaves_a_system_message_under_2048_tokens_uncached():
    assert not _caches("anthropic/claude-opus-4-7", 2047)


def test_claude_haiku_4_5_leaves_a_system_message_under_4096_tokens_uncached():
    assert not _caches("anthropic/claude-haiku-4-5-20251001", 4095)


def test_claude_haiku_4_5_caches_a_system_message_of_4096_tokens():
    assert _caches("anthropic/claude-haiku-4-5-20251001", 4096)


def test_a_dated_claude_opus_4_caches_a_system_message_of_1024_tokens():
    assert _caches("anthropic/claude-opus-4-20250514", 1024)


def test_a_dated_claude_opus_4_leaves_a_system_message_under_1024_tokens_uncached():
    assert not _caches("anthropic/claude-opus-4-20250514", 1023)


def test_an_openrouter_claude_name_is_matched_like_anthropics_own():
    assert not _caches("openrouter/anthropic/claude-haiku-4.5", 4095)


def test_a_bedrock_claude_name_is_matched_like_anthropics_own():
    assert not _caches("bedrock/anthropic.claude-haiku-4-5-20251001-v1:0", 4095)


def test_a_claude_model_none_of_the_known_ones_match_is_taken_to_cache_from_512_tokens():
    assert _caches("openrouter/~anthropic/claude-haiku-latest", 512)


def test_another_model_is_expected_to_write_its_whole_prompt_to_the_cache():
    caching_policy = PromptCachingPolicy.for_model("openrouter/openai/gpt-6.1-sol")
    prompt_token_count = caching_policy.split_prompt(2000, SharedSystemMessageCacheStatus(1500, already_cached=True))
    assert prompt_token_count == PromptTokenCount(2000, written_to_cache=2000)


def test_another_model_would_read_a_system_message_of_1024_tokens_if_the_provider_cached_it():
    caching_policy = PromptCachingPolicy.for_model("openrouter/openai/gpt-6.1-sol")
    prompt_token_count = caching_policy.split_prompt_if_cached(
        2000, SharedSystemMessageCacheStatus(1024, already_cached=True)
    )
    assert prompt_token_count == PromptTokenCount(2000, read_from_cache=1024, written_to_cache=976)


def test_another_model_would_not_read_a_shorter_system_message():
    caching_policy = PromptCachingPolicy.for_model("openrouter/openai/gpt-6.1-sol")
    prompt_token_count = caching_policy.split_prompt_if_cached(
        2000, SharedSystemMessageCacheStatus(1023, already_cached=True)
    )
    assert prompt_token_count == PromptTokenCount(2000, written_to_cache=2000)


def test_another_model_would_not_read_a_system_message_no_earlier_request_has_cached():
    caching_policy = PromptCachingPolicy.for_model("openrouter/openai/gpt-6.1-sol")
    prompt_token_count = caching_policy.split_prompt_if_cached(
        2000, SharedSystemMessageCacheStatus(1500, already_cached=False)
    )
    assert prompt_token_count == PromptTokenCount(2000, written_to_cache=2000)
