import unicodedata
from abc import abstractmethod
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import List, Tuple

from machine.annotations import Range
from machine.corpora import ScriptureRef, UsfmStylesheet, UsfmStyleType, UsfmToken, UsfmTokenizer, UsfmTokenType
from machine.tokenization import LatinWordTokenizer
from machine.translation import TranslationResult, WordAlignmentMatrix

from ..alignment.utils import compute_alignment_scores
from .corpus import load_corpus, write_corpus

PUNCT_CATEGORIES = ["Pi", "Pf", "Po", "Ps", "Pe"]


def all_punct(string: str) -> bool:
    for c in string:
        if unicodedata.category(c) not in PUNCT_CATEGORIES:
            return False
    return True


class UsfmPreserver:
    _src_sents: List[str]  # w/o markers
    _trg_sents: List[str]
    _vrefs: List[ScriptureRef]
    _stylesheet: UsfmStylesheet
    _markers: List[Tuple[int, int, str]] = []  # (sent_idx, start idx in text_only_sent, tok (inc. \ and spaces))
    _ignored_segments: List[Tuple[ScriptureRef, str]] = []

    def __init__(self, src_sents: List[str], vrefs: List[ScriptureRef], stylesheet: UsfmStylesheet):
        usfm_tokenizer = UsfmTokenizer(stylesheet)
        sentence_toks = []
        self._vrefs = []
        for sent, ref in zip(src_sents, vrefs):
            if len(ref.path) == 0 or ref.path[-1].name != "rem":
                sentence_toks.append(usfm_tokenizer.tokenize(sent))
                self._vrefs.append(ref)
        self._src_sents = self._extract_markers(sentence_toks)
        self._stylesheet = stylesheet

    @property
    def text_only_sents(self) -> List[str]:  # TODO: is it bad to have mismatching names?
        return self._src_sents

    def _extract_markers(self, sentence_toks: List[List[UsfmToken]]) -> List[str]:
        to_delete = ["fig"]
        text_only_sents = ["" for _ in sentence_toks]
        for i, (toks, ref) in enumerate(zip(sentence_toks, self._vrefs)):
            ignored_segment = ""
            ignore_scope = None
            for tok in toks:
                if ignore_scope is not None:
                    ignored_segment += tok.to_usfm()
                    if tok.type == UsfmTokenType.END and tok.marker[:-1] == ignore_scope.marker:
                        self._ignored_segments.append((ref, ignored_segment))
                        ignored_segment = ""
                        ignore_scope = None
                elif tok.type == UsfmTokenType.NOTE or tok.marker in to_delete:
                    ignored_segment += tok.to_usfm()
                    ignore_scope = tok
                elif tok.type in [UsfmTokenType.PARAGRAPH, UsfmTokenType.CHARACTER, UsfmTokenType.END]:
                    self._markers.append((i, len(text_only_sents[i]), tok.to_usfm()))
                elif tok.type == UsfmTokenType.TEXT:
                    text_only_sents[i] += tok.text

        return text_only_sents

    def construct_rows(self, translation_results: List[TranslationResult]) -> List[Tuple[List[ScriptureRef], str]]:
        self._trg_sents = [tr.translation for tr in translation_results]

        # Map each token to a character range in the original strings
        src_tok_ranges, trg_tok_ranges = self._create_tok_ranges()

        # Get index of the text token immediately following each marker and predict the corresponding token on the target side
        adj_src_toks = []
        for sent_idx, start_idx, _ in self._markers:
            for i, tok_range in reversed(list(enumerate(src_tok_ranges[sent_idx]))):
                if tok_range.start < start_idx:
                    adj_src_toks.append(i + 1)
                    break
                if i == 0:
                    adj_src_toks.append(i)
        adj_trg_toks = self._predict_marker_locations(adj_src_toks)

        # Collect the markers to be inserted
        to_insert = [[] for _ in trg_tok_ranges]
        for i, ((sent_idx, _, marker), adj_trg_tok) in enumerate(zip(self._markers, adj_trg_toks)):
            trg_str_idx = (
                trg_tok_ranges[sent_idx][adj_trg_tok].start
                if adj_trg_tok < len(trg_tok_ranges[sent_idx])
                else len(self._trg_sents[sent_idx])
            )

            # Determine the order of the markers in the sentence to handle ambiguity for directly adjacent markers
            insert_pos = 0
            while insert_pos < len(to_insert[sent_idx]) and to_insert[sent_idx][insert_pos][0] <= trg_str_idx:
                insert_pos += 1
            to_insert[sent_idx].insert(insert_pos, (trg_str_idx, marker))

        # Construct rows for the USFM file
        # Insert character markers back into text and create new rows at each paragraph marker
        rows = []
        for ref, translation, inserts in zip(self._vrefs, self._trg_sents, to_insert):
            if len(inserts) == 0:
                rows.append(([ref], translation))
                continue

            row_texts = [translation[: inserts[0][0]]]
            for i, (insert_idx, marker) in enumerate(inserts):
                is_para_marker = self._stylesheet.get_tag(marker.strip(" \\+*")).style_type == UsfmStyleType.PARAGRAPH
                if is_para_marker:
                    row_texts.append("")

                row_text = (
                    ("" if is_para_marker else marker)  # Paragraph markers are inserted by the USFM updater
                    + (
                        " "  # Extra space if inserting an end marker before a non-punctuation character
                        if "*" in marker and insert_idx < len(translation) and translation[insert_idx].isalpha()
                        # and not all_punct(
                        #     translation[insert_idx]
                        # )  # different from before, previously was "all alpha", now is "not all punct"
                        else ""
                    )
                    + (
                        translation[insert_idx : inserts[i + 1][0]]
                        if i + 1 < len(inserts)
                        else translation[insert_idx:]
                    )
                )
                # Prevent spaces before end markers
                if i + 1 < len(inserts) and "*" in inserts[i + 1][1] and len(row_text) > 0 and row_text[-1] == " ":
                    row_text = row_text[:-1]
                row_texts[-1] += row_text

            # One row_text for each paragraph in a sentence
            for row_text in row_texts:
                rows.append(([ref], row_text))

        # Add any note-type segments to the ends of their verses
        for i, row in enumerate(rows):
            ref = row[0][0]
            # Only add segments to the last row of a given vref
            if i + 1 < len(rows) and rows[i + 1][0][0].verse_ref == ref.verse_ref:
                continue
            # Add the text all segments with a matching vref
            while len(self._ignored_segments) > 0 and self._ignored_segments[0][0].verse_ref == ref.verse_ref:
                rows[i] = ([ref], rows[i][1] + self._ignored_segments[0][1])
                self._ignored_segments.pop(0)
        return rows

    @abstractmethod
    def _create_tok_ranges(self): ...

    @abstractmethod
    def _get_alignment_matrices(self): ...

    def _predict_marker_locations(self, adj_src_toks: List[int]) -> List[int]:
        alignment_matrices: List[WordAlignmentMatrix] = self._get_alignment_matrices()

        # Gets the number of alignment pairs that "cross the line" between
        # the src marker position and the potential trg marker position, (src_idx - .5) and (trg_idx - .5)
        def num_align_crossings(sent_idx: int, src_idx: int, trg_idx: int) -> int:
            crossings = 0
            alignment = alignment_matrices[sent_idx]
            for i in range(alignment.row_count):
                for j in range(alignment.column_count):
                    if alignment[i, j] and ((i < src_idx and j >= trg_idx) or (i >= src_idx and j < trg_idx)):
                        crossings += 1
            return crossings

        adj_trg_toks = []
        for (sent_idx, _, _), adj_src_tok in zip(self._markers, adj_src_toks):
            # If the token on either side of a potential target location is punctuation,
            # use it as the basis for deciding the target marker location
            trg_hyp = -1
            punct_hyps = [-1, 0]
            for punct_hyp in punct_hyps:
                src_hyp = adj_src_tok + punct_hyp
                if src_hyp < 0 or src_hyp >= len(self._src_toks[sent_idx]):
                    continue
                # only accept aligned pairs where both the src and trg token are punct
                # if len(self._src_toks[sent_idx][src_hyp]) > 0 and all_punct(self._src_toks[sent_idx][src_hyp]):  # should be the same as before
                if len(self.src_toks[sent_idx][src_hyp]) > 0 and not any(
                    c.isalpha() for c in self.src_toks[sent_idx][src_hyp]
                ):
                    aligned_trg_toks = list(alignment_matrices[sent_idx].get_row_aligned_indices(src_hyp))
                    # if aligning to a token that precedes that marker,
                    # the trg token predicted to be closest to the marker is the last token aligned to the src rather than the first
                    if punct_hyp < 0:
                        aligned_trg_toks.reverse()

                    for trg_tok in aligned_trg_toks:
                        # if all_punct(self._trg_toks[sent_idx][trg_tok]):  # should be the same as before
                        if not any(c.isalpha() for c in self.trg_toks[sent_idx][trg_tok]):
                            trg_hyp = trg_tok
                            break
                if trg_hyp != -1:
                    # since adj_trg_toks points to the token after the marker,
                    # adjust the index when aligning to punctuation that precedes the token
                    adj_trg_toks.append(trg_hyp - punct_hyp)
                    break
            if trg_hyp != -1:
                continue

            hyps = [0, 1, 2]
            # TODO: set max crossings more intelligently
            best_hyp = (
                -1,
                None,
                200**2,
            )  # trg token index, offset of corresponding src token from the marker, num crossings
            checked = set()  # to prevent checking the same idx twice
            for hyp in hyps:
                src_hyp = adj_src_tok + hyp
                if src_hyp in checked:
                    continue
                trg_hyp = -1
                while trg_hyp == -1 and src_hyp >= 0 and src_hyp < len(self._src_toks[sent_idx]):
                    checked.add(src_hyp)
                    aligned_trg_toks = list(alignment_matrices[sent_idx].get_row_aligned_indices(src_hyp))
                    if len(aligned_trg_toks) > 0:
                        # if aligning with a source token that precedes the marker,
                        # the target token predicted to be closest to the marker is the last aligned token rather than the first
                        trg_hyp = aligned_trg_toks[0 if hyp >= 0 else -1]
                    else:  # continue the search outwards
                        src_hyp += -1 if hyp < 0 else 1
                if trg_hyp != -1:
                    num_crossings = num_align_crossings(sent_idx, src_hyp, trg_hyp)
                    if num_crossings < best_hyp[2]:
                        best_hyp = (trg_hyp, hyp, num_crossings)

            # if no alignments found, insert at the end of the sentence
            if best_hyp[0] == -1:
                adj_trg_toks.append(len(self._trg_toks[sent_idx]))
                continue

            adj_trg_toks.append(best_hyp[0])

        return adj_trg_toks


class StatisticalUsfmPreserver(UsfmPreserver):
    def __init__(self, src_sentences, vrefs, stylesheet, aligner="eflomal"):
        self._aligner = aligner
        super().__init__(src_sentences, vrefs, stylesheet)

    def _create_tok_ranges(self) -> Tuple[List[List[Range[int]]], List[List[Range[int]]]]:
        tokenizer = LatinWordTokenizer()
        src_tok_ranges = [list(tokenizer.tokenize_as_ranges(sent)) for sent in self._src_sents]
        # self._src_toks = [list(tokenizer.tokenize(sent)) for sent in self._src_sents]
        self._src_toks = [
            [sent[r.start : r.end] for r in ranges] for sent, ranges in zip(self._src_sents, src_tok_ranges)
        ]

        trg_tok_ranges = [list(tokenizer.tokenize_as_ranges(sent)) for sent in self._trg_sents]
        # self._trg_toks = [list(tokenizer.tokenize(sent)) for sent in self._trg_sents]
        self._trg_toks = [
            [sent[r.start : r.end] for r in ranges] for sent, ranges in zip(self._trg_sents, trg_tok_ranges)
        ]

        return src_tok_ranges, trg_tok_ranges

    def _get_alignment_matrices(self) -> List[WordAlignmentMatrix]:
        alignments = []
        with TemporaryDirectory() as td:
            align_path = Path(td, "sym-align.txt")
            write_corpus(Path(td, "src_align.txt"), self._src_sents)
            write_corpus(Path(td, "trg_align.txt"), self._trg_sents)
            compute_alignment_scores(Path(td, "src_align.txt"), Path(td, "trg_align.txt"), self._aligner, align_path)

            for i, line in enumerate(load_corpus(align_path)):
                pairs = []
                for pair in line.split():
                    pair = pair.split("-") if self._aligner == "eflomal" else pair.split(":")[0].split("-")
                    pairs.append((int(pair[0]), int(pair[1])))
                alignments.append(
                    WordAlignmentMatrix.from_word_pairs(len(self._src_toks[i]), len(self._trg_toks[i]), pairs)
                )

        return alignments


class AttentionUsfmPreserver(UsfmPreserver):
    def construct_rows(self, translation_results: List[TranslationResult]) -> List[Tuple[List[ScriptureRef], str]]:
        self._translation_results = translation_results
        super().construct_rows(translation_results)

    # NOTE: the "▁" characters in this function are from the NllbTokenizer and are not the same character as the standard underscore
    def _create_tok_ranges(self) -> Tuple[List[List[Range[int]]], List[List[Range[int]]]]:
        self._src_toks = []
        self._trg_toks = []
        src_tok_ranges = []
        trg_tok_ranges = []
        for tr in self._translation_results:
            src_sent_tok_ranges = [Range.create(0, len(tr.source_tokens[0]) - 1)]
            for tok in tr.source_tokens[1:]:
                src_sent_tok_ranges.append(
                    Range.create(
                        src_sent_tok_ranges[-1].end + (1 if tok[0] == "▁" else 0),
                        src_sent_tok_ranges[-1].end + len(tok),
                    )
                )
            self._src_toks.append(tr.source_tokens)
            src_tok_ranges.append(src_sent_tok_ranges)
            trg_sent_tok_ranges = [Range.create(0, len(tr.target_tokens[0]) - 1)]
            for tok in tr.target_tokens[1:]:
                trg_sent_tok_ranges.append(
                    Range.create(
                        trg_sent_tok_ranges[-1].end + (1 if tok[0] == "▁" else 0),
                        trg_sent_tok_ranges[-1].end + len(tok),
                    )
                )
            self._trg_toks.append(tr.target_tokens)
            trg_tok_ranges.append(trg_sent_tok_ranges)
        return src_tok_ranges, trg_tok_ranges

    def _get_alignment_matrices(self) -> List[WordAlignmentMatrix]:
        return [tr.alignment for tr in self._translation_results]
