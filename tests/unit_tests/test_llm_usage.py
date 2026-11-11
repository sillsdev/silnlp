import json
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Tuple

from silnlp.nmt.llm_usage import RequestUsage, RunUsage
from silnlp.nmt.usage_summary_table import UsageSummaryTable


def _priced(prompt_tokens: int, completion_tokens: int, cost: float) -> RequestUsage:
    return RequestUsage.of_priced_request(prompt_tokens, completion_tokens, cost)


def _unpriced(prompt_tokens: int, completion_tokens: int) -> RequestUsage:
    return RequestUsage.of_unpriced_request(prompt_tokens, completion_tokens)


def rows_of(table: UsageSummaryTable) -> Dict[str, Tuple[str, str]]:
    rows = {}
    for line in table.render().splitlines()[1:]:
        label, value, *note = re.split(r"\s{2,}", line.strip(), maxsplit=2)
        rows[label] = (value, note[0] if note else "")
    return rows


def test_a_priced_request_is_described_with_its_cost():
    assert _priced(100, 10, 0.01).describe() == "1 requests, 100 prompt + 10 completion tokens, $0.0100"


def test_a_free_request_costs_nothing_rather_than_an_unknown_amount():
    assert _priced(100, 10, 0.0).describe().endswith(", $0.0000")


def test_an_unpriced_request_has_no_cost_rather_than_a_zero_one():
    described = _unpriced(100, 10).describe()

    assert described.endswith("; cost unavailable (no pricing for this model)")
    assert "$" not in described


def test_no_usage_is_no_requests_at_no_cost():
    assert RequestUsage().describe() == "0 requests, 0 prompt + 0 completion tokens, $0.0000"


def test_priced_requests_add_up():
    total = _priced(100, 10, 0.01) + _priced(200, 20, 0.02)

    assert total.describe() == "2 requests, 300 prompt + 30 completion tokens, $0.0300"


def test_a_sum_with_an_unpriced_request_gives_the_cost_of_the_priced_ones():
    total = _priced(100, 10, 0.01) + _unpriced(200, 20)

    assert total.describe() == "2 requests, 300 prompt + 30 completion tokens, $0.0100 excluding 1 unpriced requests"


def test_a_sum_of_unpriced_requests_has_no_cost():
    assert "cost unavailable" in (_unpriced(100, 10) + _unpriced(200, 20)).describe()


def test_a_retried_request_is_one_request_billed_for_both_attempts():
    usage = _priced(100, 2, 0.01).including_retried_attempt(_priced(100, 5, 0.02))

    assert usage.describe() == "1 requests, 200 prompt + 7 completion tokens, $0.0300"


def test_a_retried_request_with_an_unpriced_attempt_has_no_cost():
    usage = _priced(100, 2, 0.01).including_retried_attempt(_unpriced(100, 5))

    assert usage.describe() == (
        "1 requests, 200 prompt + 7 completion tokens; cost unavailable (no pricing for this model)"
    )


def test_a_priced_request_comes_back_from_its_record():
    usage = _priced(100, 10, 0.003)

    assert RequestUsage.from_record(json.loads(json.dumps(usage.to_record()))) == usage


def test_an_unpriced_request_comes_back_from_its_record():
    usage = _unpriced(100, 10)

    assert RequestUsage.from_record(json.loads(json.dumps(usage.to_record()))) == usage


def test_the_record_of_an_unpriced_request_has_a_null_cost():
    assert _unpriced(100, 10).to_record() == {"prompt_tokens": 100, "completion_tokens": 10, "cost": None}


def test_requests_with_different_costs_are_not_equal():
    assert _priced(100, 10, 0.01) != _priced(100, 10, 0.02)


def test_a_priced_and_an_unpriced_request_are_not_equal():
    assert _priced(100, 10, 0.0) != _unpriced(100, 10)


def test_no_usage_is_empty():
    assert RequestUsage().is_empty()


def test_a_request_is_not_empty():
    assert not _unpriced(0, 0).is_empty()


def test_a_cost_is_described_in_dollars_to_four_places():
    assert _priced(100, 10, 0.012345).describe_cost() == "$0.0123"


def test_a_priced_usage_costs_less_than_a_dearer_one():
    assert _priced(100, 10, 0.01).costs_less_than(_priced(100, 10, 0.02))


def test_a_priced_usage_does_not_cost_less_than_an_equal_one():
    assert not _priced(100, 10, 0.01).costs_less_than(_priced(100, 10, 0.01))


def test_a_partly_unpriced_usage_is_not_said_to_cost_less():
    assert not (_priced(100, 10, 0.01) + _unpriced(100, 10)).costs_less_than(_priced(100, 10, 0.05))


def _rows(usage: RequestUsage) -> Dict[str, Tuple[str, str]]:
    table = UsageSummaryTable("Usage")
    usage.add_requests_row_to(table)
    usage.add_tokens_and_cost_rows_to(table)
    return rows_of(table)


def test_usage_adds_its_requests_tokens_and_cost_to_a_table():
    assert _rows(_priced(1000, 10, 0.01) + _priced(2000, 20, 0.02)) == {
        "Requests": ("2", ""),
        "Prompt tokens": ("3,000", ""),
        "Completion tokens": ("30", ""),
        "Cost": ("$0.0300", ""),
    }


def test_no_usage_adds_a_zero_cost_to_a_table():
    assert _rows(RequestUsage())["Cost"] == ("$0.0000", "")


def test_unpriced_usage_adds_an_unavailable_cost_to_a_table():
    assert _rows(_unpriced(100, 10))["Cost"] == ("unavailable", "no pricing for this model")


def test_partly_priced_usage_notes_its_unpriced_requests_beside_its_cost():
    assert _rows(_priced(100, 10, 0.01) + _unpriced(200, 20))["Cost"] == ("$0.0100", "excluding 1 unpriced requests")


def test_a_runs_usage_adds_up_its_requests():
    run_usage = RunUsage()
    run_usage.add_request(_priced(100, 10, 0.01))
    run_usage.add_request(_priced(200, 20, 0.02))

    assert run_usage.describe() == "2 requests, 300 prompt + 30 completion tokens, $0.0300"


def test_a_runs_usage_reports_an_unpriced_model_rather_than_calling_it_free():
    run_usage = RunUsage()
    run_usage.add_request(_unpriced(100, 10))

    assert "cost unavailable" in run_usage.describe()
    assert "$" not in run_usage.describe()


def test_a_runs_usage_flags_a_partial_cost():
    run_usage = RunUsage()
    run_usage.add_request(_priced(100, 10, 0.01))
    run_usage.add_request(_unpriced(100, 10))

    assert "$0.0100 excluding 1 unpriced requests" in run_usage.describe()


def test_a_runs_usage_counts_reused_replies_without_billing_them():
    run_usage = RunUsage()
    run_usage.add_reused_reply()
    run_usage.add_request(_priced(100, 0, 0.01))

    described = run_usage.describe()
    assert "1 requests, 100 prompt" in described
    assert "$0.0100" in described
    assert "replies reused from an earlier run: 1" in described


def test_a_runs_usage_counts_translations_copied_from_the_training_corpus():
    run_usage = RunUsage()
    run_usage.add_copied_translation()

    assert run_usage.describe().endswith("; translations copied from the training corpus: 1")


def test_a_runs_usage_includes_the_usage_of_another_translation():
    run_usage = RunUsage()
    run_usage.add_request(_priced(100, 10, 0.01))
    other = RunUsage()
    other.add_request(_unpriced(200, 20))
    other.add_reused_reply()
    other.add_copied_translation()

    run_usage.include(other)

    assert (
        "2 requests, 300 prompt + 30 completion tokens, $0.0100 excluding 1 unpriced requests" in run_usage.describe()
    )
    assert "replies reused from an earlier run: 1" in run_usage.describe()
    assert "translations copied from the training corpus: 1" in run_usage.describe()


def test_a_runs_usage_is_thread_safe():
    run_usage = RunUsage()

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(lambda _: run_usage.add_request(_priced(1, 1, 0.001)), range(2000)))

    assert run_usage.describe().startswith("2,000 requests, 2,000 prompt + 2,000 completion tokens")


def test_a_runs_usage_adds_its_rows_to_a_table_with_how_to_avoid_reuse_and_copies():
    run_usage = RunUsage()
    run_usage.add_request(_priced(100, 10, 0.01))
    run_usage.add_reused_reply()
    run_usage.add_copied_translation()
    table = UsageSummaryTable("Usage")

    run_usage.add_rows_to(table)

    assert list(rows_of(table)) == [
        "Requests",
        "Saved replies reused",
        "Translations copied",
        "Prompt tokens",
        "Completion tokens",
        "Cost",
    ]
    assert rows_of(table)["Saved replies reused"] == (
        "1",
        "no requests; disable with --force-infer or infer.reuse_saved_replies: false",
    )
    assert rows_of(table)["Translations copied"] == (
        "1",
        "from training data, no requests; disable with infer.copy_exact_matches: false",
    )
