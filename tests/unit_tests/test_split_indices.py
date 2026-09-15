import random

import pytest

from silnlp.nmt.scripture_data_set_writer import SplitIndices


def indices_of(count: int, start: int = 0):
    return set(range(start, start + count))


def test_no_test_verses_are_chosen_when_the_test_set_need_not_be_disjoint():
    split = SplitIndices(disjoint_test=False, disjoint_val=True, test_size=2, val_size=2)

    assert split.for_test(indices_of(10), 10) is None


def test_no_validation_verses_are_chosen_when_the_validation_set_need_not_be_disjoint():
    split = SplitIndices(disjoint_test=True, disjoint_val=False, test_size=2, val_size=2)

    assert split.for_validation(indices_of(10), 10) is None


def test_the_test_verses_are_taken_from_what_is_available():
    split = SplitIndices(disjoint_test=True, disjoint_val=True, test_size=3, val_size=2)

    chosen = split.for_test(indices_of(10), 10)

    assert len(chosen) == 3
    assert chosen.issubset(indices_of(10))


def test_the_first_file_pair_fixes_the_test_verses_for_the_rest():
    split = SplitIndices(disjoint_test=True, disjoint_val=True, test_size=3, val_size=2)

    first = split.for_test(indices_of(10), 10)
    second = split.for_test(indices_of(10, start=100), 10)

    assert second == first


def test_the_first_file_pair_fixes_the_validation_verses_for_the_rest():
    split = SplitIndices(disjoint_test=True, disjoint_val=True, test_size=3, val_size=2)

    first = split.for_validation(indices_of(10), 10)
    second = split.for_validation(indices_of(10, start=100), 10)

    assert second == first


@pytest.mark.parametrize(
    "size,expected",
    [(0.5, 5), (0.25, 2), (2.0, 2), (3, 3), (1.0, 10)],
)
def test_the_size_may_be_a_share_of_the_corpus_or_a_number_of_verses(size, expected):
    # A fraction at or below one is a share of the corpus; anything above one is a count of verses.
    split = SplitIndices(disjoint_test=True, disjoint_val=True, test_size=size, val_size=size)

    assert len(split.for_test(indices_of(10), 10)) == expected


def test_asking_for_more_verses_than_are_available_takes_them_all():
    split = SplitIndices(disjoint_test=True, disjoint_val=True, test_size=50, val_size=2)

    assert split.for_test(indices_of(4), 4) == indices_of(4)


def test_the_share_is_of_the_whole_corpus_rather_than_of_what_is_left():
    split = SplitIndices(disjoint_test=True, disjoint_val=True, test_size=0.5, val_size=0.5)

    assert len(split.for_test(indices_of(4), 20)) == 4


def test_the_validation_verses_avoid_the_test_verses():
    random.seed(4)
    split = SplitIndices(disjoint_test=True, disjoint_val=True, test_size=5, val_size=5)

    test = split.for_test(indices_of(10), 10)
    validation = split.for_validation(indices_of(10), 10)

    assert test.isdisjoint(validation)


def test_the_test_verses_avoid_the_validation_verses_when_validation_came_first():
    random.seed(4)
    split = SplitIndices(disjoint_test=True, disjoint_val=True, test_size=5, val_size=5)

    validation = split.for_validation(indices_of(10), 10)
    test = split.for_test(indices_of(10), 10)

    assert test.isdisjoint(validation)


def test_the_validation_verses_may_overlap_a_test_set_that_is_not_disjoint():
    split = SplitIndices(disjoint_test=False, disjoint_val=True, test_size=5, val_size=10)
    split.reserve_for_test(indices_of(10))

    assert split.for_validation(indices_of(10), 10) == indices_of(10)


def test_reserved_test_verses_are_used_instead_of_choosing_any():
    split = SplitIndices(disjoint_test=True, disjoint_val=True, test_size=3, val_size=2)
    split.reserve_for_test([7, 8])

    assert split.for_test(indices_of(10), 10) == [7, 8]


def test_only_the_first_reservation_counts():
    split = SplitIndices(disjoint_test=True, disjoint_val=True, test_size=3, val_size=2)
    split.reserve_for_test([7, 8])
    split.reserve_for_test([1, 2])

    assert split.for_test(indices_of(10), 10) == [7, 8]


def test_what_is_available_is_left_alone_by_the_subtraction():
    random.seed(4)
    split = SplitIndices(disjoint_test=True, disjoint_val=True, test_size=5, val_size=5)
    available = indices_of(10)
    split.for_test(available, 10)
    split.for_validation(available, 10)

    assert available == indices_of(10)
