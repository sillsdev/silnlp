"""
Unit tests for normalization logic defined in normalizer.py
See https://github.com/sillsdev/silnlp/issues/494 for more context,
in particular comment: https://github.com/sillsdev/silnlp/issues/494#issuecomment-2404328574
"""
import unittest

from .normalizer import NormalizationWarning, standard_normalizer, WarningCode
from typing import List


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

    def test_left_clinging_with_many_spaces_around(self):
        self.run_test(unnormalized="(  A  (  \t B < C    \t < D", expected_normalized="(A (B <C <D")

    def test_left_clinging_with_no_spaces_around(self):
        self.run_test(unnormalized="A very(compact<sentence.", expected_normalized="A very (compact <sentence.")

    ### Right Clinging
    def test_right_clinging_typical_case(self):
        self.run_test(unnormalized="Hello , how are you ? Good", expected_normalized="Hello, how are you? Good")

    def test_right_clinging_on_left_boundary(self):
        self.run_test(unnormalized=",   This has a leading comma", expected_normalized=", This has a leading comma")

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

    def test_unclinging_right_boundary_no_spaces_to_boundary(self):
        self.run_test(unnormalized="trailing hyphen-", expected_normalized="trailing hyphen -")

    def test_unclinging_left_boundary_spaces_to_boundary(self):
        self.run_test(unnormalized="  -leading hyphen", expected_normalized="- leading hyphen")

    def test_unclinging_right_boundary_spaces_to_boundary(self):
        self.run_test(unnormalized="trailing hyphen-  \t", expected_normalized="trailing hyphen -")

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

    ### Warnings
    def extract_warnings(self, sentence: str) -> List[NormalizationWarning]:
        """
        Returns all warnings found in the sentence, sorted by the start position in the sentence.
        """
        summary = standard_normalizer.normalize(sentence)
        return sorted(
            summary.warnings,
            key=lambda warning: warning.slice.start_index,
        )

    def extract_warnings_with_code(self, sentence: str, warning_code: WarningCode) -> List[NormalizationWarning]:
        """
        Returns all warnings found in the sentence with the code passed, sorted by the start position in the sentence.
        """
        return list(filter(lambda warning: warning.warning_code == warning_code, self.extract_warnings(sentence)))

    def assert_no_warnings(self, sentence: str) -> None:
        summary = standard_normalizer.normalize(sentence)
        self.assertEqual(len(summary.warnings), 0)

    def test_warnings_generated_for_complex_example(self):
        #           0         1         2         3         4         5
        #           012345678901234567890123456789012345678901234567890123456789
        sentence = ")Yikes !? 1 + 1 = so-many-warnings... ' It's too bad ("
        #           ^      ^^   ^   ^   ^    ^        ^^^^^   ^          ^
        #                                                 ^
        warnings = standard_normalizer.normalize(sentence).warnings
        warnings.sort(key=lambda warning: warning.slice.start_index)
        self.assertEqual(len(warnings), 10)
        self.assertEqual(warnings[0].warning_code, WarningCode.RIGHT_CLINGING_CHARACTER_STARTING_SENTENCE)
        self.assertEqual(warnings[0].slice.start_index, 0)
        self.assertEqual(warnings[0].slice.end_index, 1)
        self.assertEqual(warnings[1].warning_code, WarningCode.CONSECUTIVE_PUNCTUATION)
        self.assertEqual(warnings[1].slice.start_index, 7)
        self.assertEqual(warnings[1].slice.end_index, 9)
        self.assertEqual(warnings[2].warning_code, WarningCode.POTENTIAL_UNDEFINED_PUNCTUATION)
        self.assertEqual(warnings[2].slice.start_index, 12)
        self.assertEqual(warnings[2].slice.end_index, 13)
        self.assertEqual(warnings[3].warning_code, WarningCode.POTENTIAL_UNDEFINED_PUNCTUATION)
        self.assertEqual(warnings[3].slice.start_index, 16)
        self.assertEqual(warnings[3].slice.end_index, 17)
        self.assertEqual(warnings[4].warning_code, WarningCode.BORDERED)
        self.assertEqual(warnings[4].slice.start_index, 20)
        self.assertEqual(warnings[4].slice.end_index, 21)
        self.assertEqual(warnings[5].warning_code, WarningCode.BORDERED)
        self.assertEqual(warnings[5].slice.start_index, 25)
        self.assertEqual(warnings[5].slice.end_index, 26)
        self.assertEqual(warnings[6].warning_code, WarningCode.CONSECUTIVE_PUNCTUATION)
        self.assertEqual(warnings[6].slice.start_index, 34)
        self.assertEqual(warnings[6].slice.end_index, 39)
        self.assertEqual(
            warnings[7].warning_code, WarningCode.LEFT_RIGHT_CLINGING_NOT_TOUCHING_EXACTLY_ONE_NONWHITESPACE
        )
        self.assertEqual(warnings[7].slice.start_index, 38)
        self.assertEqual(warnings[7].slice.end_index, 39)
        self.assertEqual(warnings[8].warning_code, WarningCode.BORDERED)
        self.assertEqual(warnings[8].slice.start_index, 42)
        self.assertEqual(warnings[8].slice.end_index, 43)
        self.assertEqual(warnings[9].warning_code, WarningCode.LEFT_CLINGING_CHARACTER_ENDING_SENTENCE)
        self.assertEqual(warnings[9].slice.start_index, 53)
        self.assertEqual(warnings[9].slice.end_index, 54)

    def test_warnings_generated_for_multiple_consecutive_punctuation(self):
        consecutive_punctuation_warnings = self.extract_warnings_with_code(
            "Hello, . there ! How , are  things !? Good...", WarningCode.CONSECUTIVE_PUNCTUATION
        )
        self.assertEqual(len(consecutive_punctuation_warnings), 3)
        self.assertEqual(consecutive_punctuation_warnings[0].slice.start_index, 5)
        self.assertEqual(consecutive_punctuation_warnings[0].slice.end_index, 8)
        self.assertEqual(consecutive_punctuation_warnings[1].slice.start_index, 35)
        self.assertEqual(consecutive_punctuation_warnings[1].slice.end_index, 37)
        self.assertEqual(consecutive_punctuation_warnings[2].slice.start_index, 42)
        self.assertEqual(consecutive_punctuation_warnings[2].slice.end_index, 45)

    def test_warnings_generated_for_unrecognized_punctuation(self):
        false_negative_warnings = self.extract_warnings_with_code(
            "An arabic 3 ٣. Some angle brackets « and ».", WarningCode.POTENTIAL_UNDEFINED_PUNCTUATION
        )
        self.assertEqual(len(false_negative_warnings), 2)
        self.assertEqual(false_negative_warnings[0].slice.start_index, 35)
        self.assertEqual(false_negative_warnings[0].slice.end_index, 36)
        self.assertEqual(false_negative_warnings[1].slice.start_index, 41)
        self.assertEqual(false_negative_warnings[1].slice.end_index, 42)

    def test_warning_generated_left_clinging_bordered_by_nonwhitespace(self):
        bordered_warnings = self.extract_warnings_with_code(
            "the man(Mr Li) waved",
            #       ^
            WarningCode.BORDERED,
        )
        self.assertEqual(len(bordered_warnings), 1)
        self.assertEqual(bordered_warnings[0].slice.start_index, 7)
        self.assertEqual(bordered_warnings[0].slice.end_index, 8)

    def test_warning_generated_right_clinging_bordered_by_nonwhitespace(self):
        bordered_warnings = self.extract_warnings_with_code(
            "the man (Mr Li)waved",
            #              ^
            WarningCode.BORDERED,
        )
        self.assertEqual(len(bordered_warnings), 1)
        self.assertEqual(bordered_warnings[0].slice.start_index, 14)
        self.assertEqual(bordered_warnings[0].slice.end_index, 15)

    def test_warning_generated_left_right_clinging_bordered_by_nonwhitespace(self):
        bordered_warnings = self.extract_warnings_with_code(
            "it's Brian's right",
            #  ^       ^
            WarningCode.BORDERED,
        )
        self.assertEqual(len(bordered_warnings), 2)
        self.assertEqual(bordered_warnings[0].slice.start_index, 2)
        self.assertEqual(bordered_warnings[0].slice.end_index, 3)
        self.assertEqual(bordered_warnings[1].slice.start_index, 10)
        self.assertEqual(bordered_warnings[1].slice.end_index, 11)

    def test_warning_generated_unclinging_bordered_by_nonwhitespace(self):
        bordered_warnings = self.extract_warnings_with_code(
            "Han-the-man",
            #   ^   ^
            WarningCode.BORDERED,
        )
        self.assertEqual(len(bordered_warnings), 2)
        self.assertEqual(bordered_warnings[0].slice.start_index, 3)
        self.assertEqual(bordered_warnings[0].slice.end_index, 4)
        self.assertEqual(bordered_warnings[1].slice.start_index, 7)
        self.assertEqual(bordered_warnings[1].slice.end_index, 8)

    def test_warning_not_generated_for_punctuation_bordered_by_punctuation(self):
        # Consecutive punctuation like an elipsis (...) already triggers a warning.
        # So no warnings about them being bordered in are generated.
        bordered_warnings = self.extract_warnings_with_code("Hello...", WarningCode.BORDERED)
        self.assertEqual(len(bordered_warnings), 0)

    def test_warning_generated_interior_left_right_clinging_bordered_by_whitespace(self):
        bordered_warnings = self.extract_warnings_with_code(
            "She said '  how are you ' then walked off",
            #         ^              ^
            WarningCode.LEFT_RIGHT_CLINGING_NOT_TOUCHING_EXACTLY_ONE_NONWHITESPACE,
        )
        self.assertEqual(len(bordered_warnings), 2)
        self.assertEqual(bordered_warnings[0].slice.start_index, 9)
        self.assertEqual(bordered_warnings[0].slice.end_index, 10)
        self.assertEqual(bordered_warnings[1].slice.start_index, 24)
        self.assertEqual(bordered_warnings[1].slice.end_index, 25)

    def test_warning_generated_consecutive_interior_left_right_clinging(self):
        # Regression test
        bordered_warnings = self.extract_warnings_with_code(
            "She said ' ' hmm",
            #         ^ ^
            WarningCode.LEFT_RIGHT_CLINGING_NOT_TOUCHING_EXACTLY_ONE_NONWHITESPACE,
        )
        self.assertEqual(len(bordered_warnings), 2)
        self.assertEqual(bordered_warnings[0].slice.start_index, 9)
        self.assertEqual(bordered_warnings[0].slice.end_index, 10)
        self.assertEqual(bordered_warnings[1].slice.start_index, 11)
        self.assertEqual(bordered_warnings[1].slice.end_index, 12)

    def test_warning_not_generated_left_right_clinging_on_left_boundary_with_right_whitespace(self):
        # Usually when a left-right clinging character isn't touching other non-whitespace characters,
        # a warning is generated because it's ambiguous as to which character it should cling to.
        # However in boundary cases, no warning is generated as it's not ambiguous - it clings away from the boundary.
        # This is to prevent making the warnings too noisy.
        self.assert_no_warnings("' Hi there' she said")

    def test_warning_not_generated_left_right_clinging_on_left_boundary_with_left_and_right_whitespace(self):
        # Ditto above
        self.assert_no_warnings(" ' Hi there' she said")

    def test_warning_not_generated_left_right_clinging_on_right_boundary_with_left_whitespace(self):
        # Ditto above
        self.assert_no_warnings("They yelled 'stop '")

    def test_warning_not_generated_left_right_clinging_on_right_boundary_with_left_and_right_whitespace(self):
        # Ditto above
        self.assert_no_warnings("They yelled 'stop ' ")

    def test_warning_generated_right_clinging_on_left_boundary_no_whitespace(self):
        boundary_warnings = self.extract_warnings_with_code(
            ")Hello", WarningCode.RIGHT_CLINGING_CHARACTER_STARTING_SENTENCE
        )
        self.assertEqual(len(boundary_warnings), 1)
        self.assertEqual(boundary_warnings[0].slice.start_index, 0)
        self.assertEqual(boundary_warnings[0].slice.end_index, 1)

    def test_warning_generated_right_clinging_on_left_boundary_left_whitespace(self):
        boundary_warnings = self.extract_warnings_with_code(
            "   )Hello", WarningCode.RIGHT_CLINGING_CHARACTER_STARTING_SENTENCE
        )
        self.assertEqual(len(boundary_warnings), 1)
        self.assertEqual(boundary_warnings[0].slice.start_index, 3)
        self.assertEqual(boundary_warnings[0].slice.end_index, 4)

    def test_warning_generated_right_clinging_on_left_boundary_right_whitespace(self):
        boundary_warnings = self.extract_warnings_with_code(
            ")\tHello", WarningCode.RIGHT_CLINGING_CHARACTER_STARTING_SENTENCE
        )
        self.assertEqual(len(boundary_warnings), 1)
        self.assertEqual(boundary_warnings[0].slice.start_index, 0)
        self.assertEqual(boundary_warnings[0].slice.end_index, 1)

    def test_warning_generated_right_clinging_on_left_boundary_left_and_right_whitespace(self):
        boundary_warnings = self.extract_warnings_with_code(
            " \t  )  Hello", WarningCode.RIGHT_CLINGING_CHARACTER_STARTING_SENTENCE
        )
        self.assertEqual(len(boundary_warnings), 1)
        self.assertEqual(boundary_warnings[0].slice.start_index, 4)
        self.assertEqual(boundary_warnings[0].slice.end_index, 5)

    def test_warning_generated_left_clinging_on_right_boundary_no_whitespace(self):
        boundary_warnings = self.extract_warnings_with_code(
            "Hello(", WarningCode.LEFT_CLINGING_CHARACTER_ENDING_SENTENCE
        )
        self.assertEqual(len(boundary_warnings), 1)
        self.assertEqual(boundary_warnings[0].slice.start_index, 5)
        self.assertEqual(boundary_warnings[0].slice.end_index, 6)

    def test_warning_generated_left_clinging_on_right_boundary_left_whitespace(self):
        boundary_warnings = self.extract_warnings_with_code(
            "Hello \t(", WarningCode.LEFT_CLINGING_CHARACTER_ENDING_SENTENCE
        )
        self.assertEqual(len(boundary_warnings), 1)
        self.assertEqual(boundary_warnings[0].slice.start_index, 7)
        self.assertEqual(boundary_warnings[0].slice.end_index, 8)

    def test_warning_generated_left_clinging_on_right_boundary_right_whitespace(self):
        boundary_warnings = self.extract_warnings_with_code(
            "Hello (\n", WarningCode.LEFT_CLINGING_CHARACTER_ENDING_SENTENCE
        )
        self.assertEqual(len(boundary_warnings), 1)
        self.assertEqual(boundary_warnings[0].slice.start_index, 6)
        self.assertEqual(boundary_warnings[0].slice.end_index, 7)

    def test_warning_generated_left_clinging_on_right_boundary_left_and_right_whitespace(self):
        boundary_warnings = self.extract_warnings_with_code(
            " Hi \n  (  \r", WarningCode.LEFT_CLINGING_CHARACTER_ENDING_SENTENCE
        )
        self.assertEqual(len(boundary_warnings), 1)
        self.assertEqual(boundary_warnings[0].slice.start_index, 7)
        self.assertEqual(boundary_warnings[0].slice.end_index, 8)


if __name__ == "__main__":
    unittest.main()
