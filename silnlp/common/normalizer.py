"""
Normalization tooling for cleaning up whitespace and punctuation in extract sentences
See normalize_extracts.py for context
"""

import logging
from dataclasses import dataclass
from enum import Enum, IntEnum
from typing import Dict, List, Optional, Set, Tuple

import regex


class PunctuationCategory(Enum):
    LEFT_CLINGING = "LEFT_CLINGING"
    RIGHT_CLINGING = "RIGHT_CLINGING"
    LEFT_RIGHT_CLINGING = "LEFT_RIGHT_CLINGING"
    UNCLINGING = "UNCLINGING"


logger = logging.getLogger(__package__ + ".normalizer")


@dataclass(frozen=True)
class PunctuationNormalizationRule:
    character: str  # length 1
    category: PunctuationCategory


@dataclass(frozen=True)
class StringSlice:
    """
    Represents a section of a string
    """

    start_index: int
    end_index: int
    slice: str
    # The string the slice came from
    outer: str

    def length(self) -> int:
        return self.end_index - self.start_index


def shift_slice(slice: StringSlice, offset: int, new_outer: str) -> StringSlice:
    """Shifts a slice that is from a substring to a new outer string"""
    return StringSlice(
        start_index=slice.start_index + offset, end_index=slice.end_index + offset, slice=slice.slice, outer=new_outer
    )


def find_slices(reg: regex.Pattern, text: str) -> List[StringSlice]:
    return [
        StringSlice(start_index=match.span()[0], end_index=match.span()[1], slice=match.group(), outer=text)
        for match in regex.finditer(reg, text)
    ]


def build_slice(start_index: int, end_index: int, outer: str) -> StringSlice:
    """
    Convenience method to make building slices easier
    The .slice field can always be derived from the outer text and the indexes,
    but the dataclass doesn't allow making the slice field a method, hence this method
    """
    return StringSlice(start_index=start_index, end_index=end_index, slice=outer[start_index:end_index], outer=outer)


def slice_contains(outer: StringSlice, inner: StringSlice) -> bool:
    """
    Returns whether the alleged outer slice contains the inner.
    It's implicitly assumed the slices correspond to the same string
    """
    return (outer.start_index <= inner.start_index) and (outer.end_index >= inner.end_index)


# TODO - delete when you're confident it's not going to be used
def pretty_print_slice(slice: StringSlice) -> None:
    logger.debug(slice.outer)
    logger.debug("012345678901234567890123456789")
    logger.debug("0         1         2         ")
    logger.debug(slice.start_index * " " + len(slice.slice) * "*")
    logger.debug(slice.start_index * " " + f"({slice.start_index},{slice.end_index})")


@dataclass(frozen=True)
class SentenceTransformation:
    """
    A representation of a delta applied to a sentence at a particular position
    """

    slice: StringSlice
    replacement: str
    description: str


class WarningCode(IntEnum):
    CONSECUTIVE_PUNCTUATION = 0
    POTENTIAL_UNDEFINED_PUNCTUATION = 1
    BORDERED = 2
    RIGHT_CLINGING_CHARACTER_STARTING_SENTENCE = 3
    LEFT_CLINGING_CHARACTER_ENDING_SENTENCE = 4
    LEFT_RIGHT_CLINGING_NOT_TOUCHING_EXACTLY_ONE_NONWHITESPACE = 5


@dataclass(frozen=True)
class NormalizationWarning:
    """
    A warning of a potential false negative/positive spotted during normalization
    For example it may notice a character that looks like punctuation,
    but it's not included in the list of punctuation characters specified by the user
    """

    slice: StringSlice
    warning_code: WarningCode
    description: str


@dataclass(frozen=True)
class SentenceNormalizationSummary:
    """
    A summary of the normalization process for a particular sentence.
    Warnings about potential issues are also included.
    """

    original_sentence: str
    normalized_sentence: str
    transformations: List[SentenceTransformation]
    warnings: List[NormalizationWarning]


def unicode_hex(character: str) -> str:
    """
    Returns a unicode representation of the character passed, e.g. U+04FA
    """
    code = hex(ord(character))[2:]
    # Left pad with 0
    code = code.zfill(4)
    return "U+" + code.upper()


class Normalizer:
    """
    Encapsulates the state required to normalize sentences

    See corresponding tests in normalize_tests.py
    """

    def __init__(self, punctuation_normalization_rules: List[PunctuationNormalizationRule]):
        self.validate_normalization_rules(punctuation_normalization_rules)
        self.punctuation_normalization_rules = punctuation_normalization_rules

        # The constructor defines many complex regexes which are compiled once and stored
        # inside the class to prevent recompilation on every sentence normalized

        self.punctuation_char_2_normalization_rule: Dict[str, PunctuationNormalizationRule] = {
            rule.character: rule for rule in punctuation_normalization_rules
        }
        self.supported_punctuation: Set[str] = set(self.punctuation_char_2_normalization_rule.keys())

        self.consecutive_spaces = regex.compile("\\s+")

        # Regex escape all the punctuation characters defined in the rules so that they be shoved into a regex
        escaped_punctuation_chars = "".join(regex.escape(rule.character) for rule in punctuation_normalization_rules)
        # Matches a single punctuation character with optional whitespace before and after
        # Negative look behind and ahead is used to stop it matching punctuation that's part of a multiple punctuation group
        self.single_punctuation_with_optional_whitespace_regex = regex.compile(
            f"(?<![{escaped_punctuation_chars}\s])\s*[{escaped_punctuation_chars}]\s*(?![{escaped_punctuation_chars}\s])"
        )
        # Matches a starting punctuation char, then any number of punctuation/whitespace, then a closing punctuation character
        self.multiple_punctuation_regex = regex.compile(
            f"[{escaped_punctuation_chars}][{escaped_punctuation_chars}\s]*[{escaped_punctuation_chars}]"
        )

        self.not_letters_or_numbers_or_whitespace_regex = regex.compile("""[^\p{N}\p{L}\s]""")

        self.single_punctuation_surrounded_by_nonwhitepace_nonpunctuation_regex = regex.compile(
            f"(?<=[^\s{escaped_punctuation_chars}])[{escaped_punctuation_chars}](?=[^\s{escaped_punctuation_chars}])"
        )

        right_clinging_punctuation: Set[str] = set(
            rule.character
            for rule in self.punctuation_normalization_rules
            if rule.category == PunctuationCategory.RIGHT_CLINGING
        )
        escaped_right_clinging_punctuation_chars = "".join(
            regex.escape(character) for character in right_clinging_punctuation
        )
        self.right_clinging_character_starting_sentence_regex = regex.compile(
            f"(?<=^\s*)[{escaped_right_clinging_punctuation_chars}]"
        )

        left_clinging_punctuation: Set[str] = set(
            rule.character
            for rule in self.punctuation_normalization_rules
            if rule.category == PunctuationCategory.LEFT_CLINGING
        )
        escaped_left_clinging_punctuation_chars = "".join(
            regex.escape(character) for character in left_clinging_punctuation
        )
        self.left_clinging_character_ending_sentence_regex = regex.compile(
            f"[{escaped_left_clinging_punctuation_chars}](?=\s*$)"
        )

        left_right_clinging_punctuation: Set[str] = set(
            rule.character
            for rule in self.punctuation_normalization_rules
            if rule.category == PunctuationCategory.LEFT_RIGHT_CLINGING
        )
        escaped_left_right_clinging_punctuation_chars = "".join(
            regex.escape(character) for character in left_right_clinging_punctuation
        )
        self.left_right_clinging_character_not_touching_anything_regex = regex.compile(
            f"(?<=\S\s+)[{escaped_left_right_clinging_punctuation_chars}](?=\s+\S)"
        )

    def validate_normalization_rules(self, punctuation_normalization_rules: List[PunctuationNormalizationRule]) -> None:
        """
        Does basic checks on the punctuation normalization rules to make sure they are conceptually sound
        For example:
        - ensure punctuation characters don't have more than 1 rule
        - ensure punctuation characters are indeed single characters and not multi-character strings
        - ensure that whitespace characters aren't being used as punctuation characters
        """

        supported_punctuation: List[str] = [rule.character for rule in punctuation_normalization_rules]

        # Ensure there are no duplicates punctuation characters in the rules
        # Conceptually it doesn't make sense for a character to have more than one rule
        supported_punctuation_distinct: Set[str] = set(supported_punctuation)
        characters_and_counts = [(p, supported_punctuation.count(p)) for p in supported_punctuation_distinct]
        duplicates_and_counts = list(filter(lambda character_and_count: character_and_count[1] > 1, characters_and_counts))
        for char, count in duplicates_and_counts:
            print(f"[char={char}][count={count}]")
        if duplicates_and_counts:
            logger.error(f"{len(duplicates_and_counts)} punctuation character(s) are found in multiple rules")
            for punctuation_char, count in duplicates_and_counts:
                logger.error(f"  [punctuation='{punctuation_char}'][count='{count}'] Punctuation character occurs in multiple rules")
            raise Exception(f"Invalid punctuation normalization rules - {len(duplicates_and_counts)} character(s) appear in multiple rules")

        # Ensure punctuation characters are indeed single character strings
        multiple_char_punctuations = list(filter(lambda punctuation: len(punctuation) > 1, supported_punctuation))
        if multiple_char_punctuations:
            logger.error(f"{len(multiple_char_punctuations)} punctuation character(s) found containing more than 1 character")
            for multiple_char_punctuation in multiple_char_punctuations:
                logger.error(f"  [punctuation='{multiple_char_punctuation}'][length='{len(multiple_char_punctuation)}'] Punctuation character contains multiple characters")
            raise Exception(f"Invalid punctuation normalization rules - {len(multiple_char_punctuations)} character(s) have multiple characters")

        # Ensure punctuation characters aren't whitespace
        whitespace_regex = regex.compile("\\s")
        whitespace_punctuations = list(filter(lambda punctuation: whitespace_regex.match(punctuation), supported_punctuation))
        if whitespace_punctuations:
            logger.error(f"{len(whitespace_punctuations)} whitespace punctuation character(s) found")
            for whitespace in whitespace_punctuations:
                encoded = whitespace.encode("unicode_escape")
                logger.error(f"  [punctuation={encoded!r}'][ord='{ord(whitespace)}'] Punctuation character is whitespace")
            raise Exception(f"Invalid punctuation normalization rules - {len(whitespace_punctuations)} character(s) are whitespace")



    def normalize(self, sentence: str) -> SentenceNormalizationSummary:
        """
        Generates a series of transformations and warnings for normalizing the string passed.
        """
        logger.debug("======================")
        logger.debug(f"Normalizing '{sentence}'")

        all_transformations: List[SentenceTransformation] = self.find_transformations_sorted(sentence)

        multiple_punctuation_warnings: List[NormalizationWarning] = [
            NormalizationWarning(
                slice=slice,
                warning_code=WarningCode.CONSECUTIVE_PUNCTUATION,
                description="Multiple consecutive punctuation characters (ignoring whitespace) - currently this is not normalized",
            )
            for slice in find_slices(self.multiple_punctuation_regex, sentence)
        ]

        all_warnings = (
            multiple_punctuation_warnings
            + self.search_false_negatives(sentence)
            + self.search_false_positives(sentence)
        )

        logger.debug(f"[#transformations={len(all_transformations)}][#warnings={len(all_warnings)}]")

        # Pretty print out all the transformation relative to the original string
        if all_transformations:
            num_blocks_of_10 = len(sentence) // 10 + 1
            tens_row = (" " * 9).join([str(i) for i in range(0, num_blocks_of_10)])
            logger.debug("" + tens_row)
            logger.debug("" + "0123456789" * num_blocks_of_10)
            logger.debug("" + sentence)
            for transformation in all_transformations:
                slice = transformation.slice
                indent = slice.start_index * " "
                back_padding = (len(sentence) + 5 - slice.start_index - slice.length()) * " "
                logger.debug(
                    indent
                    + "^" * slice.length()
                    + back_padding
                    + f"({slice.start_index},{slice.end_index})"
                    + " "
                    + transformation.description
                )

        # Rebuild the string by applying all the transformations
        # We extract the parts unaffected by normalization, then rebuild by interleaving them with the normalized parts
        # Example:
        #   Hello , there you  !Bye
        #   -----   ---------   ---     <-- original parts
        #        ---         ---        <-- parts to normalize
        #        ", "         "! "      <-- normalized replacements
        if all_transformations:
            logger.debug(f"Reconstructing normalized string from {len(all_transformations)} transformations")
            parts = []
            last_part_end_index = 0
            for transformation in all_transformations:
                # The part prior to this normalization segment
                parts.append(sentence[last_part_end_index : transformation.slice.start_index])
                last_part_end_index = transformation.slice.end_index
                parts.append(transformation.replacement)
            # Deal with the ending original part
            parts.append(sentence[last_part_end_index:])

            normalized = "".join(parts)
            logger.debug(f"* Original:   '{sentence}'")
            logger.debug(f"* Normalized: '{normalized}'" + (" (unchanged)" if sentence == normalized else ""))
        else:
            logger.debug("No transformations applied, normalization has no effect")
            normalized = sentence

        return SentenceNormalizationSummary(
            original_sentence=sentence,
            normalized_sentence=normalized,
            transformations=all_transformations,
            warnings=all_warnings,
        )

    def find_transformations_sorted(self, sentence: str) -> List[SentenceTransformation]:
        """
        Searches the sentence passed for slices of text that need normalization.
        Example:
          My friend(John) said  to me...
                   ^          ^^

        In the example a left clinging character '(' is found which needs space added before it,
        and a double space is between two words.

        For each case found, a SentenceTransformation is generated which describes the change, i
        - the area of the string to be changed represented as a slice
        - what to change it to
        - why it's being changed to

        Note the changes are applied elsewhere - this method only generates a description of the changes.
        """
        # Boundary whitespace is dealt with specially first and later logic analyses the trimmed input
        # Later searching uses sentence_trimmed hence won't refind the boundary cases
        # It will right shift the slices found by the amount left trimmed
        boundary_trim_transformations, sentence_trimmed, trim_offset = self.compute_boundary_transformations(sentence)

        # NOTE: This method only handles single consecutive punctuation.
        # In this context "consecutive" means two punctuations next to each other ignoring whitespace.
        # Example:
        #       Hi there, . How are you ?
        #               ^^^^           ^^
        #               multiple       single
        # Multiple punctuation groups aren't transformed, but are instead captured as warnings (see related warning code).
        #
        # In this context "punctuation" refers only to the characters defined in the normalization rules.
        # The search regex is built by regex escaping the punctuation characters and combining them.
        #
        # Note the regex captures surrounding whitespace to analyze whether it is consistent with the normalization rules
        # for the punctuation.
        # For example in the example above, if '?' is right clinging, then the whitespace before it needs to be removed.
        single_punctuation_transformations: List[SentenceTransformation] = []
        for slice in find_slices(self.single_punctuation_with_optional_whitespace_regex, sentence_trimmed):
            # Figure out the punctuation character that was found in the slice and find the associated normalization rule for it
            punctuation_char = regex.sub(self.consecutive_spaces, "", slice.slice)
            rule = self.punctuation_char_2_normalization_rule[punctuation_char]
            normalized = self.normalize_single_punctuation_slice(rule, slice)
            # Some normalizations couldn't be applied or don't actually transform the text
            if normalized is not None and normalized != slice.slice:
                single_punctuation_transformations.append(
                    SentenceTransformation(
                        slice=shift_slice(slice, trim_offset, sentence),
                        replacement=normalized,
                        description=f"Punctuation '{punctuation_char}' normalized by rule {rule.category}",
                    )
                )

        # Search for blocks of consecutive whitespace that should be shrunk to a single space character
        consecutive_spaces_transformations = [
            SentenceTransformation(
                slice=shift_slice(slice, trim_offset, sentence),
                replacement=" ",
                description="Whitespace normalized to a single space",
            )
            for slice in find_slices(self.consecutive_spaces, sentence_trimmed)
            # Don't create transformations for a single space as the before and after are identical
            if slice.slice != " "
        ]
        # Some of these will overlap with the single punctuation transformations which are already going to normalize whitespace
        # so those need to be knocked out
        # Note this needs to be done _after_ shifting the slice
        consecutive_spaces_transformations = list(
            filter(
                lambda consecutive_space_transformation: not any(
                    slice_contains(
                        outer=single_punctuation_transformation.slice, inner=consecutive_space_transformation.slice
                    )
                    for single_punctuation_transformation in single_punctuation_transformations
                ),
                consecutive_spaces_transformations,
            )
        )

        transformations_sorted = sorted(
            boundary_trim_transformations + consecutive_spaces_transformations + single_punctuation_transformations,
            key=lambda transformation: transformation.slice.start_index,
        )

        # Assert that none of the transformations are overlapping (they _shouldn't_ if everything is programmed correctly).
        # Ensuring non-overlapping transformations guarantees that no matter what order you apply them,
        # the normalized sentence is the same.
        # This gives more flexibility to the consumer to what order they apply the transformations
        # (in reality it will probably always be first to last though)
        for index in range(0, len(transformations_sorted) - 1):
            current = transformations_sorted[index]
            next = transformations_sorted[index + 1]
            assert current.slice.end_index <= next.slice.start_index, f"Transformation at index {index} overlaps following transformation"

        return transformations_sorted

    def normalize_single_punctuation_slice(
        self, punctuation_rule: PunctuationNormalizationRule, slice: StringSlice
    ) -> Optional[str]:
        """
        Normalizes the portion of text found that contains a single punctuation character
        If the normalized text is the same as the original text, None is returned to indicate no replacement is needed
        """
        punctuation_char = punctuation_rule.character
        if punctuation_rule.category == PunctuationCategory.LEFT_CLINGING:
            if slice.start_index != 0:
                return " " + punctuation_char
            else:
                return punctuation_char
        elif punctuation_rule.category == PunctuationCategory.RIGHT_CLINGING:
            if slice.end_index != len(slice.outer):
                return punctuation_char + " "
            else:
                return punctuation_char
        elif punctuation_rule.category == PunctuationCategory.LEFT_RIGHT_CLINGING:
            # Get the boundary cases out of the way first
            if slice.start_index == 0 or slice.end_index == len(slice.outer):
                # If the group is at the boundary, then in all cases you just remove all whitespace
                return punctuation_char
            elif slice.slice == punctuation_char:
                # The punctuation char has no surrounding whitespace
                # That means it's either boxed in, e.g. abc"def or it's at the boundary, e.g. abc"
                # In the first case we don't know which way it should cling and can't do anything
                # (separate warning code picks this up)
                # In the second case there's nothing to do
                # So in both cases there's nothing to do, just return the original punctuation
                return None
            elif slice.slice[0] != punctuation_char and slice.slice[-1] != punctuation_char:
                # The punctuation char has whitespace on both sides
                # In this case we don't know which way it should cling so all we can do is shrink
                # the punctuation on both sides to a single space
                # (separate warning code picks this up)
                return " " + punctuation_char + " "
            elif slice.slice[0] == punctuation_char:
                # The punctuation char has space to the right, e.g. `abc" def`
                # so it's right clinging
                return punctuation_char + " "
            else:
                # Ditto above, but it's left clinging
                # Shrink that space to 1 character
                return " " + punctuation_char
        elif punctuation_rule.category == PunctuationCategory.UNCLINGING:
            if slice.start_index == 0:
                return punctuation_char + " "
            elif slice.end_index == len(slice.outer):
                return " " + punctuation_char
            else:
                return " " + punctuation_char + " "
        else:
            logger.warn(f"Punctuation '{punctuation_char}' detected with unhandled punctuation category: '{punctuation_rule.category}'")
            logger.warn("Ignoring punctuation, however a handler should always be defined")
            return None

    def compute_boundary_transformations(self, sentence: str) -> Tuple[List[SentenceTransformation], str, int]:
        boundary_trim_transformations: List[SentenceTransformation] = []

        left_trimmed = sentence.lstrip()
        if sentence != left_trimmed:
            slice_length = len(sentence) - len(left_trimmed)
            boundary_trim_transformations.append(
                SentenceTransformation(
                    slice=build_slice(start_index=0, end_index=slice_length, outer=sentence),
                    replacement="",
                    description="Removing left boundary whitespace",
                )
            )

        right_trimmed = sentence.rstrip()
        if sentence != right_trimmed:
            slice_length = len(sentence) - len(right_trimmed)
            boundary_trim_transformations.append(
                SentenceTransformation(
                    slice=build_slice(start_index=len(right_trimmed), end_index=len(sentence), outer=sentence),
                    replacement="",
                    description="Removing right boundary whitespace",
                )
            )

        return (boundary_trim_transformations, sentence.strip(), len(sentence) - len(left_trimmed))

    def search_false_negatives(self, sentence: str) -> List[NormalizationWarning]:
        """
        Searches the sentence passed for characters that look like punctuation but aren't being normalized by the script
        These characters are defined as non-letter, non-numbers according the broad unicode categories for Letter and Number respectively.
        """
        potential_false_negatives = find_slices(self.not_letters_or_numbers_or_whitespace_regex, sentence)
        return [
            NormalizationWarning(
                slice,
                WarningCode.POTENTIAL_UNDEFINED_PUNCTUATION,
                f"Character '{slice.slice}' ({unicode_hex(slice.slice)}) is not a letter or digit or whitespace and is not listed as punctuation. Potential false negative.",
            )
            for slice in potential_false_negatives
            if slice.slice not in self.supported_punctuation
        ]

    def search_false_positives(self, sentence: str) -> List[NormalizationWarning]:
        """
        Searches the sentence passed for punctuation characters defined in the normaliztion rules that are potentially
        not acting in the way intended by the normalization rules.

        e.g.
            It's Sam-the-man
              ^     ^   ^

        In the above example,
        - a single quote is being used for contraction, but it was intended for quoting,
          e.g. He said 'hi'
        - the hyphen is being used to join words which is different to the intended usage of acting like a semi-colon,
          e.g. That person over there - the one wearing the hat

        The worry is that the normalization process will misunderstand the role of these characters and normalize them
        when they are correct as is.
        With standard English normalization rules, the above could be incorrectly changed to something like:

           It 's Sam - the - man
        """

        # For all 4 punctuation categories, any punctuation character bordered by non-whitespace characters on both sides is suspicious
        bordered_warnings = [
            NormalizationWarning(
                slice,
                WarningCode.BORDERED,
                "Punctuation character is surrounded by non-whitespace on both sides. "
                + "Potentially it is being used in a different way to the punctuation character defined in the normalization rules.",
            )
            for slice in find_slices(self.single_punctuation_surrounded_by_nonwhitepace_nonpunctuation_regex, sentence)
        ]

        # Right clinging characters shouldn't be the first non-whitespace character in a sentence
        # e.g. ") hi there"
        right_clinging_sentence_start_warnings = [
            NormalizationWarning(
                slice,
                WarningCode.RIGHT_CLINGING_CHARACTER_STARTING_SENTENCE,
                f"Punctuation character '{slice.slice}' is right clinging, but is starting a sentence. "
                + "Usually it is expected to have text preceeding it. This could indicate it is playing a different role to what is expected.",
            )
            for slice in find_slices(self.right_clinging_character_starting_sentence_regex, sentence)
        ]

        # Left clinging characters shouldn't be the last non-whitespace character in a sentence
        # e.g. "hi there ("
        left_clinging_sentence_end_warnings = [
            NormalizationWarning(
                slice,
                WarningCode.LEFT_CLINGING_CHARACTER_ENDING_SENTENCE,
                f"Punctuation character '{slice.slice}' is left clinging, but is ending a sentence. "
                + "Usually it is expected to have text following it. This could indicate it is playing a different role to what is expected.",
            )
            for slice in find_slices(self.left_clinging_character_ending_sentence_regex, sentence)
        ]

        # Left-right clinging characters should always be touching exactly one non-whitespace character,
        # otherwise it's not possible to determine whether it's acting as left or right clinging
        # In some cases this could just be the user accidentally adding in an extra whitespace character, e.g. She said ' hi'
        # which is different to a "false positive" (ie. in the sense of us attributing it a function it's not performing), e.g. it's
        # NOTE - this overlaps with the bordered_warnings above.
        # This block only checks for characters with whitespace on both sides to prevent duplicate warnings.
        left_right_clinging_not_touching_exactly_one_warnings = [
            NormalizationWarning(
                slice,
                WarningCode.LEFT_RIGHT_CLINGING_NOT_TOUCHING_EXACTLY_ONE_NONWHITESPACE,
                "Punctuation character is not touching exactly one non-whitespace character. "
                + "This could indicate it's playing a different role to what is expected (false positive), "
                + "or it could indicate user error. "
                + "In either case normalization is unable to determine whether the character is acting in a left or right clinging role.",
            )
            for slice in find_slices(self.left_right_clinging_character_not_touching_anything_regex, sentence)
        ]

        return (
            bordered_warnings
            + right_clinging_sentence_start_warnings
            + left_clinging_sentence_end_warnings
            + left_right_clinging_not_touching_exactly_one_warnings
        )


# Used for testing, examples etc...
# Not to be thought of as a canonical normalizer
standard_normalizer = Normalizer(
    [
        PunctuationNormalizationRule(".", PunctuationCategory.RIGHT_CLINGING),
        PunctuationNormalizationRule(",", PunctuationCategory.RIGHT_CLINGING),
        PunctuationNormalizationRule("!", PunctuationCategory.RIGHT_CLINGING),
        PunctuationNormalizationRule("?", PunctuationCategory.RIGHT_CLINGING),
        PunctuationNormalizationRule("(", PunctuationCategory.LEFT_CLINGING),
        PunctuationNormalizationRule(")", PunctuationCategory.RIGHT_CLINGING),
        PunctuationNormalizationRule("<", PunctuationCategory.LEFT_CLINGING),
        PunctuationNormalizationRule(">", PunctuationCategory.RIGHT_CLINGING),
        PunctuationNormalizationRule("'", PunctuationCategory.LEFT_RIGHT_CLINGING),
        PunctuationNormalizationRule('"', PunctuationCategory.LEFT_RIGHT_CLINGING),
        PunctuationNormalizationRule("-", PunctuationCategory.UNCLINGING),
    ]
)
