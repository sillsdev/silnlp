"""
Normalization tooling for cleaning up whitespace and punctuation in extract sentences
See normalize_extracts.py for context
"""
import logging
import regex

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple


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


class WarningCode(Enum):
    multiple_punctuation = 0


@dataclass(frozen=True)
class NormalizationWarning:
    """
    A warning of a potential false negative/positive spotted during normalization
    For example it may notice a character that looks like punctuation,
    but it's not included in the list of punctuation characters specified by the user
    """

    # TODO - use a slice
    start_index: int
    end_index: int
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

        self.consecutive_spaces = regex.compile("\\s+")
        self.single_whitespace = regex.compile("\\s")

        escaped_punctuation_chars = "".join(regex.escape(rule.character) for rule in punctuation_normalization_rules)
        self.punctuation_regex = regex.compile(f"[{escaped_punctuation_chars}\s]+")

    def normalize(self, sentence: str) -> SentenceNormalizationSummary:
        """
        Generates a series of transformations and warnings for normalizing the string passed.
        """
        logger.debug(f"Normalizing '{sentence}'")

        # Boundary whitespace is dealt with specially first and later logic analyses the trimmed input
        boundary_trim_transformations, sentence_trimmed, trim_offset = self.compute_boundary_transformations(sentence)

        # Find groups of punctuation within the trimmed string
        # Any results have to be shifted back to the coordinate system of the original sentence
        punctuation_slices = find_slices(self.punctuation_regex, sentence_trimmed)

        # Categorize each slice found
        consecutive_spaces_slices: List[StringSlice] = []
        single_punctuation_slices: List[StringSlice] = []
        multiple_punctuation_warnings: List[NormalizationWarning] = []
        for slice in punctuation_slices:
            whitespace_removed = regex.sub(self.consecutive_spaces, "", slice.slice)
            # match is completely whitespace
            if len(whitespace_removed) == 0:
                consecutive_spaces_slices.append(slice)
            # match has one punctuation character
            elif len(whitespace_removed) == 1:
                single_punctuation_slices.append(slice)
            # match has 2+ punctuation character
            else:
                # For this case, we don't transform the punctuation, but there's still potentially
                # consecutive spaces that can be shrunk down to a single space
                # We search within the current slice for consecutive spaces - the coordinate systems
                # within those slices need to be shifted back to the main slice
                consecutive_spaces_sub_slices = [
                    shift_slice(spaces_slice, slice.start_index, slice.outer)
                    for spaces_slice in find_slices(self.consecutive_spaces, slice.slice)
                ]
                consecutive_spaces_slices.extend(consecutive_spaces_sub_slices)

                # Generate a warning that spans just the punctuation characters in the slice,
                # and not boundary whitespace
                left_trimmed = slice.slice.lstrip()
                offset = len(slice.slice) - len(left_trimmed)
                all_punctuation_trimmed = left_trimmed.rstrip()
                multiple_punctuation_warnings.append(
                    NormalizationWarning(
                        start_index=slice.start_index + offset,
                        end_index=slice.start_index + offset + len(all_punctuation_trimmed),
                        warning_code=WarningCode.multiple_punctuation,
                        description="Multiple consecutive punctuation characters (ignoring whitespace) - currently this is not normalized",
                    )
                )
        logger.debug(f"   #consecutive space slices={len(consecutive_spaces_slices)}")
        logger.debug(f"  #single punctuation slices={len(single_punctuation_slices)}")
        logger.debug(f"#multiple punctuation slices={len(multiple_punctuation_warnings)}")

        # Convert consecutive space slices into transformations
        consecutive_spaces_transformations = [
            SentenceTransformation(
                slice=shift_slice(slice, trim_offset, sentence),
                replacement=" ",
                description="Consecutive whitespace found",
            )
            for slice in consecutive_spaces_slices
            # Don't generate transformations for single spaces
            if slice.slice != " "
        ]

        # Convert single punctuation slices into transformations
        single_punctuation_transformations: List[SentenceTransformation] = []
        for slice in single_punctuation_slices:

            # Figure out the punctuation character that was found in the slice and find the associated normalization rule for it
            punctuation_char = regex.sub(self.consecutive_spaces, "", slice.slice)
            rule = next(filter(lambda rule: rule.character == punctuation_char, self.punctuation_normalization_rules))
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

        # TODO - add other kinds of warnings
        all_warnings = multiple_punctuation_warnings

        all_transformations = sorted(
            consecutive_spaces_transformations + single_punctuation_transformations + boundary_trim_transformations,
            key=lambda transformation: transformation.slice.start_index,
        )

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

        # TODO - check the transformations aren't overlapping

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
