"""
Unit tests for normalization logic defined in normalizer.py
See https://github.com/sillsdev/silnlp/issues/494 for more context,
in particular comment: https://github.com/sillsdev/silnlp/issues/494#issuecomment-2404328574
"""
import unittest

from .normalizer import standard_normalizer


class TestNormalize(unittest.TestCase):
    def run_test(self, unnormalized: str, expected_normalized: str) -> None:
        actual_normalized = standard_normalizer.normalize(unnormalized).normalized_sentence
        self.assertEqual(actual_normalized, expected_normalized)

    ### Left Clinging
    def test_left_clinging_typical_case(self):
        self.run_test(unnormalized="Hello ( there   <  fish", expected_normalized="Hello (there <fish")

    def test_left_clinging_on_left_boundary(self):
        self.run_test(unnormalized="( This has a leading bracket", expected_normalized="(This has a leading bracket")

    def test_left_clinging_on_right_boundary(self):
        self.run_test(
            unnormalized="This ends with an opening caret   <", expected_normalized="This ends with an opening caret <"
        )
        # TODO test for a warning

    def test_left_clinging_with_many_spaces_around(self):
        self.run_test(unnormalized="(  A  (  \t B < C    \t < D", expected_normalized="(A (B <C <D")

    def test_left_clinging_with_no_spaces_around(self):
        self.run_test(unnormalized="A very(compact<sentence.", expected_normalized="A very (compact <sentence.")

    ### Right Clinging
    def test_right_clinging_typical_case(self):
        self.run_test(unnormalized="Hello , how are you ? Good", expected_normalized="Hello, how are you? Good")

    def test_right_clinging_on_left_boundary(self):
        self.run_test(unnormalized=",   This has a leading comma", expected_normalized=", This has a leading comma")
        # TODO test for a warning

    def test_right_clinging_on_right_boundary(self):
        self.run_test(unnormalized="This has a closing period .", expected_normalized="This has a closing period.")

    def test_right_clinging_with_many_spaces_around(self):
        self.run_test(
            unnormalized="Captains log   .  We   \t ,     have  .  \r\n  Encountered   ,  a  . New species  .  Kirk  out.",
            expected_normalized="Captains log. We, have. Encountered, a. New species. Kirk out.",
        )

    def test_right_clinging_with_no_spaces_around(self):
        self.run_test(unnormalized="A very,compact,sentence.", expected_normalized="A very, compact, sentence.")

    ### Left-Right Clinging
    def test_left_right_clinging_typical_case(self):
        self.run_test(
            unnormalized="She said:   'he said:   \"Hi\"   he really did'",
            expected_normalized="She said: 'he said: \"Hi\" he really did'",
        )

    def test_left_right_clinging_surrounded_by_space(self):
        # For this case it's not clear which side the punctuation should cling to
        # so no changes are made
        self.run_test(
            unnormalized="A lonely quote ' in the middle of things",
            expected_normalized="A lonely quote ' in the middle of things",
        )
        # TODO test for a warning

    def test_left_right_clinging_on_left_boundary(self):
        # On the left boundary you know it's left clinging even if there's space to the right
        self.run_test(unnormalized="'   This has a leading quote", expected_normalized="'This has a leading quote")

    def test_left_right_clinging_on_right_boundary(self):
        # On the right boundary you know it's right clinging even if there's space to the right
        self.run_test(unnormalized="This has a closing quote '", expected_normalized="This has a closing quote'")

    def test_left_right_clinging_with_no_spaces_around(self):
        # For this case it's not clear which side the punctuation should cling to
        # so no changes are made
        self.run_test(unnormalized="A very'compact\"sentence.", expected_normalized="A very'compact\"sentence.")

    ### Unclinging
    def test_unclinging_typical_case(self):
        # This is the defined behavior but probably not what is wanted
        self.run_test(
            unnormalized="John-the-main - the tallest in town",
            expected_normalized="John - the - main - the tallest in town",
        )

    def test_unclinging_no_space_on_left(self):
        self.run_test(unnormalized="hyphen- in the middle", expected_normalized="hyphen - in the middle")

    def test_unclinging_no_space_on_right(self):
        self.run_test(unnormalized="hyphen -in the middle", expected_normalized="hyphen - in the middle")

    def test_unclinging_with_no_spaces_around(self):
        self.run_test(unnormalized="a-very-compact-sentence", expected_normalized="a - very - compact - sentence")

    def test_unclinging_left_boundary_no_spaces_to_boundary(self):
        self.run_test(unnormalized="-leading hyphen", expected_normalized="- leading hyphen")
        # TODO test for a warning

    def test_unclinging_right_boundary_no_spaces_to_boundary(self):
        self.run_test(unnormalized="trailing hyphen-", expected_normalized="trailing hyphen -")
        # TODO test for a warning

    def test_unclinging_left_boundary_spaces_to_boundary(self):
        self.run_test(unnormalized="  -leading hyphen", expected_normalized="- leading hyphen")
        # TODO test for a warning

    def test_unclinging_right_boundary_spaces_to_boundary(self):
        self.run_test(unnormalized="trailing hyphen-  \t", expected_normalized="trailing hyphen -")
        # TODO test for a warning

    ### Consecutive punctuation
    def test_consecutive_punctuation_is_ignored(self):
        self.run_test(unnormalized="Hello,. how are you !?", expected_normalized="Hello,. how are you !?")

    def test_consecutive_punctuation_is_ignored_with_whitespace_between(self):
        self.run_test(unnormalized="Hello, . there", expected_normalized="Hello, . there")

    def test_many_consecutive_punctuation_is_ignored(self):
        self.run_test(unnormalized='Hello , . - ! "" \' there', expected_normalized='Hello , . - ! "" \' there')

    def test_consecutive_punctuation_doesnt_prevent_normalizing_single_punctuation_and_consecutive_spaces(self):
        self.run_test(
            unnormalized="Hello,   . there !   How  are  things !?",
            expected_normalized="Hello, . there! How are things !?",
        )

    def test_consecutive_punctuation_doesnt_prevent_shrinking_of_consecutive_whitespace_around_it(self):
        self.run_test(unnormalized="Hello  ,. \t  there", expected_normalized="Hello ,. there")

    def test_consecutive_punctuation_doesnt_prevent_trimming_boundary_whitespace(self):
        self.run_test(unnormalized="  ., Hello ?! \t", expected_normalized="., Hello ?!")

    ### Boundary whitespace
    def test_left_boundary_whitespace_trimmed_off(self):
        self.run_test(unnormalized=" \t Hello", expected_normalized="Hello")

    def test_left_boundary_whitespace_trimmed_off_first_char_punctuation(self):
        self.run_test(unnormalized=" \t -Hello", expected_normalized="- Hello")

    def test_right_boundary_whitespace_trimmed_off(self):
        self.run_test(unnormalized="Hello \r\n", expected_normalized="Hello")

    def test_right_boundary_whitespace_trimmed_off_last_char_punctuation(self):
        self.run_test(unnormalized="Hello (  \t", expected_normalized="Hello (")

    def test_left_right_boundary_whitespace_trimmed_off(self):
        self.run_test(unnormalized=" \r ) Hello (  \t", expected_normalized=") Hello (")

    # TODO - tests for warnings around consecutive punctuation
    # TODO - test the Transformation objects created

    ### Misc
    def test_single_non_space_whitespace_characters_converted(self):
        # Regression test for a subtle bug
        self.run_test(unnormalized="A\tB", expected_normalized="A B")

    def test_complex_case(self):
        self.run_test(
            unnormalized=" \r  Hi there , you! (my -    friend )  How's   it\tgoing ?  \t",
            expected_normalized="Hi there, you! (my - friend) How's it going?",
        )


if __name__ == "__main__":
    unittest.main()
