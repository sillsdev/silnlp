from abc import abstractmethod
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import List, Tuple

from machine.annotations import Range
from machine.corpora import ScriptureRef, UsfmStylesheet, UsfmStyleType, UsfmTokenizer, UsfmTokenType
from machine.tokenization import LatinWordTokenizer
from machine.translation import TranslationResult

from ..alignment.utils import compute_alignment_scores
from .corpus import load_corpus, write_corpus


class UsfmInserter:
    orig_src_sents: List[str]  # w/ markers
    vrefs: List[ScriptureRef]
    stylesheet: UsfmStylesheet
    markers: List[Tuple[int, int, str]] = []  # (sent_idx, start idx in text_only_sent, tok (inc. \s and spaces))
    ignored_segments = List[str] = []
    src_sents: List[str]  # w/o markers
    trg_sents: List[str]
    translation_results: List[TranslationResult]

    def __init__(self, orig_src_sents: List[str], vrefs: List[ScriptureRef], stylesheet: UsfmStylesheet):
        self.orig_src_sents = orig_src_sents
        self.vrefs = vrefs
        self.stylesheet = stylesheet

    def extract_markers(self):
        tokenizer = UsfmTokenizer(self.stylesheet)
        sentence_toks = [tokenizer.tokenize(sent) for sent in self.orig_src_sents]

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
    def predict_marker_locations(self, src_toks_after_markers: List[int]): ...

    def construct_rows(self, translation_results: List[TranslationResult], vrefs: List[ScriptureRef]):
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

        # Collect the markers to be inserted
        to_insert = [[] for _ in trg_tok_ranges]
        for mark, next_trg_tok in zip(self.markers, trg_toks_after_markers):
            sent_idx, _, marker = mark
            trg_str_idx = trg_tok_ranges[sent_idx][next_trg_tok].start

            # figure out the order of the markers in the sentence to handle ambiguity for directly adjacent markers
            insert_place = 0
            while insert_place < len(to_insert[sent_idx]) and to_insert[sent_idx][insert_place][0] <= trg_str_idx:
                insert_place += 1

            to_insert[sent_idx].insert(insert_place, (trg_str_idx, marker))

        # Construct rows for the USFM file
        # Insert character markers back into text and create new rows at each paragraph marker
        rows = []
        for ref, translation, inserts in zip(vrefs, self.trg_sents, to_insert):
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
        src_tok_ranges = [list(tokenizer.tokenize_as_ranges(sent)) for sent in self.text_only_src_sents]
        self.src_toks = [
            [sent[r.start : r.end] for r in ranges] for sent, ranges in zip(self.text_only_src_sents, src_tok_ranges)
        ]

        trg_tok_ranges = [list(tokenizer.tokenize_as_ranges(sent)) for sent in self.trg_sents]
        self.trg_toks = [
            [sent[r.start : r.end] for r in ranges] for sent, ranges in zip(self.trg_sents, trg_tok_ranges)
        ]

        return src_tok_ranges, trg_tok_ranges

    def predict_marker_locations(self, src_toks_after_markers):
        # Align sentences
        with TemporaryDirectory() as td:
            # align_sents(toks, trg_toks, aligner, align_path) # Path(f"zzz_PN_KTs/tpi_aps/train.src.detok.txt"),Path(f"zzz_PN_KTs/tpi_aps/train.trg.detok.txt")
            # eflomal
            align_path = Path(td, "sym-align.txt")
            write_corpus(Path(td, "src_align.txt"), self.src_sents)
            write_corpus(Path(td, "trg_align.txt"), self.trg_sents)
            compute_alignment_scores(Path(td, "src_align.txt"), Path(td, "trg_align.txt"), self.aligner, align_path)

            if self.aligner == "eflomal":
                align_lines = [
                    [(lambda x: (int(x[0]), int(x[1])))(pair.split("-")) for pair in line.split()]
                    for line in load_corpus(align_path)
                ]
            else:
                align_lines = [
                    [(lambda x: (int(x[0]), int(x[1])))(pair.split(":")[0].split("-")) for pair in line.split()]
                    for line in load_corpus(align_path)
                ]

        # Gets the number of alignment pairs that "cross the line" between (src_idx - .5) and (trg_idx - .5)
        def num_align_crossings(sent_idx: int, src_idx: int, trg_idx: int) -> int:
            crossings = 0
            for i, j in align_lines[sent_idx]:
                if i < src_idx and j >= trg_idx:
                    crossings += 1
                if i >= src_idx and j < trg_idx:
                    crossings += 1
            return crossings

        SLCM = ["qt"]  # sentence level character markers
        trg_toks_after_markers = []
        for marker, tok_after_marker in zip(self.markers, src_toks_after_markers):
            sent_idx = marker[0]
            mark = marker[2].strip(" \\+")

            # If the token on either side of a hypothesis is punctuation, use that
            # can also try a less extreme version where the punct hyps are still subject to having the least crossings, but they are still always checked first
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
                    align_pairs = reversed(align_lines[sent_idx]) if punct_hyp < 0 else align_lines[sent_idx]
                    for s, t in align_pairs:
                        if s == src_hyp and not any(c.isalpha() for c in self.trg_toks[sent_idx][t]):
                            trg_hyp = t
                            break
                if trg_hyp != -1:
                    # if this search gets expanded beyond [-1,0] can do insert_idx -= punct_hyp
                    insert_idx = trg_hyp + 1 if punct_hyp < 0 else trg_hyp
                    # push end markers one further unless the next token is punctuation
                    end_shift = (
                        1
                        if (
                            "*" in mark
                            and (
                                any(mark.strip(" \\*+") == m for m in SLCM)
                                or (
                                    insert_idx < len(self.trg_toks[sent_idx])
                                    and any(char.isalpha() for char in self.trg_toks[sent_idx][insert_idx])
                                )
                            )
                        )
                        else 0
                    )
                    trg_toks_after_markers.append(insert_idx + end_shift)  # TODO: hacky
                    break
            if trg_hyp != -1:
                continue

            hyps = [0, 1, 2]
            best_hyp = (-1, None, len(align_lines[sent_idx]))
            checked = set()  # to prevent checking the same idx twice
            for hyp in hyps:
                align_pairs = reversed(align_lines[sent_idx]) if hyp < 0 else align_lines[sent_idx]
                src_hyp = tok_after_marker + hyp
                if src_hyp in checked:
                    continue
                trg_hyp = -1
                while trg_hyp == -1 and src_hyp >= 0 and src_hyp < len(self.src_toks[sent_idx]):
                    checked.add(src_hyp)
                    trg_hyps = [t for (s, t) in align_pairs if s == src_hyp]
                    if len(trg_hyps) > 0:
                        trg_hyp = trg_hyps[0]
                    else:
                        src_hyp += -1 if hyp < 0 else 1
                if trg_hyp != -1:
                    num_crossings = num_align_crossings(sent_idx, src_hyp, trg_hyp)
                    if num_crossings < best_hyp[2]:
                        # replace hyp with src_hyp - adj_src_tok
                        # above is what I was trying to do before with offsetting the farther away hyps, but it's worse
                        # am I doing this below now with "insert_idx = insert_idx - best_hyp[1]"?
                        best_hyp = (trg_hyp, hyp, num_crossings)

            if best_hyp[0] == -1:
                trg_toks_after_markers.append(len(self.trg_toks[sent_idx]))  # insert at the end of the sentence
            else:
                insert_idx = best_hyp[0]
                # do this before or after subtracting the offset? only subtract offset if not adjacent to punct? check before and after subtraction?
                if self.trg_toks[sent_idx][insert_idx] in [",", ".", "!", "?"]:
                    insert_idx += 1
                # subtracting hyp at the end may be bad in other cases, or make wrong answers worse/more confusing
                else:
                    insert_idx = insert_idx - best_hyp[1]  # - best_hyp[1]
                    if self.trg_toks[sent_idx][insert_idx] in [",", ".", "!", "?"]:
                        insert_idx += 1

                # push end markers one further unless the next token is punctuation
                end_shift = (
                    1
                    if (
                        "*" in mark
                        and (
                            any(mark.strip(" \\*+") == m for m in SLCM)
                            or (
                                insert_idx < len(self.trg_toks[sent_idx])
                                and any(char.isalpha() for char in self.trg_toks[sent_idx][insert_idx])
                            )
                        )
                    )
                    else 0
                )
                trg_toks_after_markers.append(insert_idx + end_shift)  # TODO: hacky
                # for end markers, preventing double adding (from punct first) was almost the same but slightly worse

        return trg_toks_after_markers


class AttentionUsfmInserter(UsfmInserter):
    # NOTE: the "▁" characters in this function are from the NllbTokenizer and are not the same character as the standard underscore
    def create_tok_ranges(self):
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
            src_tok_ranges.append(src_sent_tok_ranges)
            trg_sent_tok_ranges = [Range.create(0, len(tr.target_tokens[0]) - 1)]
            for tok in tr.target_tokens[1:]:
                trg_sent_tok_ranges.append(
                    Range.create(
                        trg_sent_tok_ranges[-1].end + (1 if tok[0] == "▁" else 0),
                        trg_sent_tok_ranges[-1].end + len(tok),
                    )
                )
            trg_tok_ranges.append(trg_sent_tok_ranges)
        return src_tok_ranges, trg_tok_ranges

    def predict_marker_locations(self, src_toks_after_markers: List[int]):
        trg_toks_after_markers = []
        for idx, (sent_idx, _, _) in zip(src_toks_after_markers, self.markers):
            offset = 0
            while True:
                try:
                    # if aligning with a token before the marker, get the last target token it aligns to
                    # if aligning with a token after the marker, get the first target token it aligns to
                    pos = -1 if offset < 0 else 0
                    adjacent_tok = list(
                        self.translation_results[sent_idx].alignment.get_row_aligned_indices(idx + offset)
                    )[pos]
                    trg_toks_after_markers.append(adjacent_tok + (1 if pos == -1 else 0))
                    # TODO: could try adjusting trg idx to be "closer" to the marker (i.e. subtracting for 3 for offset 3)
                    break
                except:
                    # outward expanding search: offset = 0, -1, 1, -2, 2, ...
                    offset = -offset - (1 if offset >= 0 else 0)

        return trg_toks_after_markers
