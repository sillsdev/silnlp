from silnlp.nmt.usage_summary_table import UsageSummaryTable


def test_a_table_starts_with_its_title_and_indents_its_rows_beneath_it():
    table = UsageSummaryTable("Totals")
    table.add_row("Requests", "2")

    assert table.render() == "Totals\n    Requests  2"


def test_a_tables_labels_line_up_on_the_left_and_its_values_on_the_right():
    table = UsageSummaryTable("Totals")
    table.add_row("Requests", "99")
    table.add_row("Prompt tokens", "1,206,330")

    assert table.render().splitlines()[1:] == [
        "    Requests              99",
        "    Prompt tokens  1,206,330",
    ]


def test_a_note_follows_its_rows_value():
    table = UsageSummaryTable("Totals")
    table.add_row("Cost", "$3.0432", "excluding 1 unpriced requests")
    table.add_row("Requests", "99")

    assert table.render().splitlines()[1:] == [
        "    Cost      $3.0432  excluding 1 unpriced requests",
        "    Requests       99",
    ]


def test_a_table_without_rows_is_only_its_title():
    assert UsageSummaryTable("Totals").render() == "Totals"
