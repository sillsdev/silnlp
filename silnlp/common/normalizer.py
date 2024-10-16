"""
Normalization tooling for cleaning up whitespace and punctuation in extract sentences
See normalize_extracts.py for context
"""
import logging
import regex

from dataclasses import dataclass
from enum import Enum, IntEnum
from typing import Dict, List, Optional, Set, Tuple


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


def shift_slice(slice: StringSlice, offset: int, new_outer: str) -> StringSlice:
    """Shifts a slice that is from a substring to a new outer string"""
    return StringSlice(
        start_index=slice.start_index + offset, end_index=slice.end_index + offset, slice=slice.slice, outer=new_outer
    )


def find_slices(reg: regex.Regex, text: str) -> List[StringSlice]:
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
    MULTIPLE_PUNCTUATION = 0
    FALSE_NEGATIVE_CANDIDATE = 1


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
        self.punctuation_normalization_rules = punctuation_normalization_rules
        # TODO - ensure no duplicate characters
        # TODO - ensure strings are length 1
        # TODO - ensure no whitespace

        self.punctuation_char_2_normalization_rule: Dict[str, PunctuationNormalizationRule] = {
            rule.character: rule for rule in punctuation_normalization_rules
        }
        self.supported_punctuation: Set[str] = set(self.punctuation_char_2_normalization_rule.keys())

        self.consecutive_spaces = regex.compile("\\s+")
        self.single_whitespace = regex.compile("\\s")

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

    def normalize(self, sentence: str) -> SentenceNormalizationSummary:
        """
        Generates a series of transformations and warnings for normalizing the string passed.
        """
        logger.debug(f"Normalizing '{sentence}'")

        all_transformations: List[SentenceTransformation] = self.find_transformations_sorted(sentence)

        multiple_punctuation_warnings: List[NormalizationWarning] = [
            NormalizationWarning(
                slice=slice,
                warning_code=WarningCode.MULTIPLE_PUNCTUATION,
                description="Multiple consecutive punctuation characters (ignoring whitespace) - currently this is not normalized",
            )
            for slice in find_slices(self.multiple_punctuation_regex, sentence)
        ]

        false_negative_warnings: List[NormalizationWarning] = self.search_false_negatives(sentence)

        # TODO - add other kinds of warnings
        all_warnings = multiple_punctuation_warnings + false_negative_warnings

        logger.debug(f"#transformations={len(all_transformations)}")
        logger.debug(f"#warnings={len(all_warnings)}")

        # Pretty print out all the transformation relative to the original string
        # TODO This is just for debugging and will be replaced by better reporting
        num_blocks_of_10 = len(sentence) // 10 + 1
        tens_row = (" " * 9).join([str(i) for i in range(0, num_blocks_of_10)])
        logger.debug(">>> " + tens_row)
        logger.debug(">>> " + "0123456789" * num_blocks_of_10)
        logger.debug(">>> " + sentence)
        for transformation in all_transformations:
            slice = transformation.slice
            indent = ">>> " + slice.start_index * " "
            back_padding = (len(sentence) + 5 - slice.start_index - len(slice.slice)) * " "
            logger.debug(
                indent
                + regex.sub(self.single_whitespace, "~", slice.slice)
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
        logger.debug(f"Normalized string is: '{normalized}'")
        # TODO - test an example where there's no normalization
        # TODO - shortcircuit this if no normalization? I guess it will be pretty fast

        return SentenceNormalizationSummary(
            original_sentence=sentence,
            normalized_sentence=normalized,
            transformations=all_transformations,
            warnings=all_warnings,
        )

    def find_transformations_sorted(self, sentence: str) -> List[SentenceTransformation]:
        # Boundary whitespace is dealt with specially first and later logic analyses the trimmed input
        boundary_trim_transformations, sentence_trimmed, trim_offset = self.compute_boundary_transformations(sentence)

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
                        description=f"Punctuation ({punctuation_char}) normalized by rule {rule.category}",
                    )
                )

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

        # TODO - put in a general check that the transformations aren't overlapping
        return sorted(
            boundary_trim_transformations + consecutive_spaces_transformations + single_punctuation_transformations,
            key=lambda transformation: transformation.slice.start_index,
        )

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
                # In the second case there's nothing to do
                # So in both cases there's nothing to do, just return the original punctuation
                # TODO - warn on the first case
                return None
            elif slice.slice[0] != punctuation_char and slice.slice[-1] != punctuation_char:
                # The punctuation char has whitespace on both sides
                # In this case we don't know which way it should cling so all we can do is shrink
                # the punctuation on both sides to a single space
                # TODO - warn on this case
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
            # TODO warn for this case - we don't recognize the rule category
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
                WarningCode.FALSE_NEGATIVE_CANDIDATE,
                f"Character '{slice.slice}' ({unicode_hex(slice.slice)}) is not a letter or digit or whitespace and is not listed as punctuation. Potential false negative.",
            )
            for slice in potential_false_negatives
            if slice.slice not in self.supported_punctuation
        ]


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
