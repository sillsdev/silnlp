from abc import abstractmethod
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import List, Tuple

from machine.annotations import Range
from machine.corpora import ScriptureRef, UsfmStylesheet, UsfmStyleType, UsfmTokenizer, UsfmTokenType
from machine.tokenization import LatinWordTokenizer
from machine.translation import TranslationResult, WordAlignmentMatrix

from ..alignment.utils import compute_alignment_scores
from .corpus import load_corpus, write_corpus


class UsfmInserter:
    orig_src_sents: List[str] = []  # w/ markers
    vrefs: List[ScriptureRef] = []
    stylesheet: UsfmStylesheet

    markers: List[Tuple[int, int, str]] = []  # (sent_idx, start idx in text_only_sent, tok (inc. \s and spaces))
    ignored_segments: List[str] = []
    src_sents: List[str]  # w/o markers
    trg_sents: List[str]
    translation_results: List[TranslationResult]

    def __init__(self, orig_src_sents: List[str], vrefs: List[ScriptureRef], stylesheet: UsfmStylesheet):
        for sent, ref in zip(orig_src_sents, vrefs):
            if len(ref.path) > 0 and ref.path[-1].name == "rem":
                continue
            self.orig_src_sents.append(sent)
            self.vrefs.append(ref)
        self.stylesheet = stylesheet

    def extract_markers(self):
        usfm_tokenizer = UsfmTokenizer(self.stylesheet)
        sentence_toks = [usfm_tokenizer.tokenize(sent) for sent in self.orig_src_sents]

        to_delete = ["fig"]
        text_only_sents = ["" for _ in sentence_toks]
        for i, (toks, ref) in enumerate(zip(sentence_toks, self.vrefs)):
            ignored_segment = ""
            ignore_scope = None
            for tok in toks:
                if ignore_scope is not None:
                    ignored_segment += tok.to_usfm()
                    if tok.type == UsfmTokenType.END and tok.marker[:-1] == ignore_scope.marker:
                        self.ignored_segments.append((ref, ignored_segment))
                        ignored_segment = ""
                        ignore_scope = None
                elif tok.type == UsfmTokenType.NOTE or (
                    tok.type == UsfmTokenType.CHARACTER and tok.marker in to_delete
                ):
                    ignore_scope = tok
                    ignored_segment += tok.to_usfm()
                elif tok.type in [UsfmTokenType.PARAGRAPH, UsfmTokenType.CHARACTER, UsfmTokenType.END]:
                    self.markers.append((i, len(text_only_sents[i]), tok.to_usfm()))
                elif tok.type == UsfmTokenType.TEXT:
                    text_only_sents[i] += tok.text

        self.src_sents = text_only_sents
        return text_only_sents

    @abstractmethod
    def create_tok_ranges(self): ...

    @abstractmethod
    def get_alignment_matrices(self): ...

    # @abstractmethod
    def predict_marker_locations(self, src_toks_after_markers: List[int]):
        print("here new")
        alignment_matrices: List[WordAlignmentMatrix] = self.get_alignment_matrices()

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

        trg_toks_after_markers = []
        for marker, tok_after_marker in zip(self.markers, src_toks_after_markers):
            sent_idx = marker[0]
            mark = marker[2].strip(" \\+")

            # If the token on either side of a hypothesis is punctuation, use that
            trg_hyp = -1
            punct_hyps = [-1, 0]
            for punct_hyp in punct_hyps:
                src_hyp = tok_after_marker + punct_hyp
                if src_hyp < 0 or src_hyp >= len(self.src_toks[sent_idx]):
                    continue
                # only accept pairs where both the src and trg token are punct
                # can define more specifically what the punct tokens can look like later
                if len(self.src_toks[sent_idx][src_hyp]) > 0 and not any(
                    c.isalpha() for c in self.src_toks[sent_idx][src_hyp]
                ):
                    aligned_trg_toks = list(alignment_matrices[sent_idx].get_row_aligned_indices(src_hyp))
                    # if aligning to a token that precedes that marker,
                    # the trg token predicted to be closest to the marker is the last token aligned to the src rather than the first
                    if punct_hyp < 0:
                        aligned_trg_toks.reverse()

                    for trg_tok in aligned_trg_toks:
                        if not any(c.isalpha() for c in self.trg_toks[sent_idx][trg_tok]):
                            trg_hyp = trg_tok
                            break
                if trg_hyp != -1:
                    # since trg_tokens_after_markers points to the token after the marker,
                    # adjust the index when aligning to punctuation that precedes the token
                    insert_idx = trg_hyp - punct_hyp
                    trg_toks_after_markers.append(insert_idx)
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
            # TODO: might need to change how the "checked" set works, if a hyp has been checked, it doesn't necessarily mean that any hyp
            # that would follow in the searching sequence would have also been checked.
            # it may be worth while to redo the hyps overall, i.e. the next couple of tokens will already have been checked
            # during "hyp 0" if needed, so it may be simpler to ALWAYS check 0, 1, -1, 2, -2, and then only go further if none found yet
            # this would probably be about the same practically, but it might make the code more straightforward
            checked = set()  # to prevent checking the same idx twice
            for hyp in hyps:
                src_hyp = tok_after_marker + hyp
                if src_hyp in checked:
                    continue
                # TODO: I think this will stop the search prematurely when the initial hyp is at the edge of a sentence
                # e.g. if the hyp is index 0, it will search at 0, 1, and stop at -1, even though it could still search further forward
                trg_hyp = -1
                while trg_hyp == -1 and src_hyp >= 0 and src_hyp < len(self.src_toks[sent_idx]):
                    checked.add(src_hyp)
                    aligned_trg_toks = list(alignment_matrices[sent_idx].get_row_aligned_indices(src_hyp))
                    # if aligning to a token that precedes that marker,
                    # the trg token predicted to be closest to the marker is the last token aligned to the src rather than the first
                    if hyp < 0:
                        aligned_trg_toks.reverse()

                    if len(aligned_trg_toks) > 0:
                        trg_hyp = aligned_trg_toks[0]
                    else:  # continue the search outwards
                        src_hyp += -1 if hyp < 0 else 1
                if trg_hyp != -1:
                    num_crossings = num_align_crossings(sent_idx, src_hyp, trg_hyp)
                    if num_crossings < best_hyp[2]:
                        best_hyp = (trg_hyp, hyp, num_crossings)

            if best_hyp[0] == -1:
                trg_toks_after_markers.append(len(self.trg_toks[sent_idx]))  # insert at the end of the sentence
                continue

            trg_toks_after_markers.append(best_hyp[0])
            # for end markers, preventing double adding (from punct first) was almost the same but slightly worse

        return trg_toks_after_markers

    def construct_rows(self, translation_results: List[TranslationResult]):
        self.translation_results = translation_results
        self.trg_sents = [tr.translation for tr in translation_results]

        # Map each token to a character range in the original strings
        src_tok_ranges, trg_tok_ranges = self.create_tok_ranges()

        # Match markers to their closest token idx
        src_toks_after_markers = []
        for sent_idx, start_idx, _ in self.markers:
            for i, tok_range in reversed(list(enumerate(src_tok_ranges[sent_idx]))):
                if tok_range.start < start_idx:
                    src_toks_after_markers.append(i + 1)
                    break
                if i == 0:
                    src_toks_after_markers.append(i)

        trg_toks_after_markers = self.predict_marker_locations(src_toks_after_markers)

        # # to check that the string indices match up with the corresponding tokens
        # for mark, next_trg_tok in zip(self.markers, trg_toks_after_markers):
        #     print(mark[2])
        #     print(translation_results[mark[0]].target_tokens[next_trg_tok] if next_trg_tok < len(translation_results[mark[0]].target_tokens) else "")
        #     print(translation_results[mark[0]].translation)

        # Collect the markers to be inserted
        to_insert = [[] for _ in trg_tok_ranges]
        for i, (mark, next_trg_tok) in enumerate(zip(self.markers, trg_toks_after_markers)):
            sent_idx, _, marker = mark
            if next_trg_tok == len(trg_tok_ranges[sent_idx]):
                trg_str_idx = len(self.trg_sents[sent_idx])
            else:
                try:
                    trg_str_idx = trg_tok_ranges[sent_idx][next_trg_tok].start
                except Exception:
                    print(i)
                    print(sent_idx, len(trg_tok_ranges))
                    if sent_idx < len(trg_tok_ranges):
                        print(self.trg_sents[sent_idx])
                        print(trg_tok_ranges[sent_idx])
                        print(next_trg_tok, len(trg_tok_ranges[sent_idx]))
                        if next_trg_tok < len(trg_tok_ranges[sent_idx]):
                            print(trg_tok_ranges[sent_idx][next_trg_tok])
                    raise

            # figure out the order of the markers in the sentence to handle ambiguity for directly adjacent markers
            insert_place = 0
            while insert_place < len(to_insert[sent_idx]) and to_insert[sent_idx][insert_place][0] <= trg_str_idx:
                insert_place += 1

            to_insert[sent_idx].insert(insert_place, (trg_str_idx, marker))

        # Construct rows for the USFM file
        # Insert character markers back into text and create new rows at each paragraph marker
        rows = []
        for ref, translation, inserts in zip(self.vrefs, self.trg_sents, to_insert):
            if len(inserts) == 0:
                rows.append(([ref], translation))
                continue

            row_texts = [translation[: inserts[0][0]]]
            for i, (insert_idx, marker) in enumerate(inserts):
                is_para_marker = self.stylesheet.get_tag(marker.strip(" \\+*")).style_type == UsfmStyleType.PARAGRAPH
                if is_para_marker:
                    row_texts.append("")

                row_text = (
                    ("" if is_para_marker else marker)  # paragraph markers are inserted by the USFM updater
                    + (
                        " "
                        if "*" in marker and insert_idx < len(translation) and translation[insert_idx].isalpha()
                        else ""
                    )
                    + (
                        translation[insert_idx : inserts[i + 1][0]]
                        if i + 1 < len(inserts)
                        else translation[insert_idx:]
                    )
                )
                # don't want a space before an end marker
                if i + 1 < len(inserts) and "*" in inserts[i + 1][1] and len(row_text) > 0 and row_text[-1] == " ":
                    row_text = row_text[:-1]
                row_texts[-1] += row_text

            for row_text in row_texts:
                rows.append(([ref], row_text))

        # add footnotes to ends of versess
        for i, row in enumerate(rows):
            ref = row[0][0]
            if i < len(rows) - 1 and rows[i + 1][0][0].verse_ref == ref.verse_ref:
                continue
            while len(self.ignored_segments) > 0 and self.ignored_segments[0][0].verse_ref == ref.verse_ref:
                rows[i] = ([ref], rows[i][1] + self.ignored_segments[0][1])
                self.ignored_segments.pop(0)
        return rows


class StatisticalUsfmInserter(UsfmInserter):
    def __init__(self, src_sentences, vrefs, stylesheet, aligner="eflomal"):
        self.aligner = aligner
        super().__init__(src_sentences, vrefs, stylesheet)

    def create_tok_ranges(self):
        tokenizer = LatinWordTokenizer()
        src_tok_ranges = [list(tokenizer.tokenize_as_ranges(sent)) for sent in self.src_sents]
        self.src_toks = [
            [sent[r.start : r.end] for r in ranges] for sent, ranges in zip(self.src_sents, src_tok_ranges)
        ]

        trg_tok_ranges = [list(tokenizer.tokenize_as_ranges(sent)) for sent in self.trg_sents]
        self.trg_toks = [
            [sent[r.start : r.end] for r in ranges] for sent, ranges in zip(self.trg_sents, trg_tok_ranges)
        ]

        return src_tok_ranges, trg_tok_ranges

    def get_alignment_matrices(self):
        alignments = []
        with TemporaryDirectory() as td:
            # eflomal
            align_path = Path(td, "sym-align.txt")
            write_corpus(Path(td, "src_align.txt"), self.src_sents)
            write_corpus(Path(td, "trg_align.txt"), self.trg_sents)
            compute_alignment_scores(Path(td, "src_align.txt"), Path(td, "trg_align.txt"), self.aligner, align_path)

            for i, line in enumerate(load_corpus(align_path)):
                pairs = []
                for pair in line.split():
                    pair = pair.split("-") if self.aligner == "eflomal" else pair.split(":")[0].split("-")
                    pairs.append((int(pair[0]), int(pair[1])))
                alignments.append(
                    WordAlignmentMatrix.from_word_pairs(len(self.src_toks[i]), len(self.trg_toks[i]), pairs)
                )

        return alignments


class AttentionUsfmInserter(UsfmInserter):
    # NOTE: the "▁" characters in this function are from the NllbTokenizer and are not the same character as the standard underscore
    def create_tok_ranges(self):
        self.src_toks = []
        self.trg_toks = []
        src_tok_ranges = []
        trg_tok_ranges = []
        for tr in self.translation_results:
            src_sent_tok_ranges = [Range.create(0, len(tr.source_tokens[0]) - 1)]
            for tok in tr.source_tokens[1:]:
                src_sent_tok_ranges.append(
                    Range.create(
                        src_sent_tok_ranges[-1].end + (1 if tok[0] == "▁" else 0),
                        src_sent_tok_ranges[-1].end + len(tok),
                    )
                )
            self.src_toks.append(tr.source_tokens)
            src_tok_ranges.append(src_sent_tok_ranges)
            trg_sent_tok_ranges = [Range.create(0, len(tr.target_tokens[0]) - 1)]
            for tok in tr.target_tokens[1:]:
                trg_sent_tok_ranges.append(
                    Range.create(
                        trg_sent_tok_ranges[-1].end + (1 if tok[0] == "▁" else 0),
                        trg_sent_tok_ranges[-1].end + len(tok),
                    )
                )
            self.trg_toks.append(tr.target_tokens)
            trg_tok_ranges.append(trg_sent_tok_ranges)
        return src_tok_ranges, trg_tok_ranges

    def get_alignment_matrices(self):
        return [tr.alignment for tr in self.translation_results]
