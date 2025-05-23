from abc import abstractmethod
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import List, Tuple

from machine.annotations import Range
from machine.corpora import ScriptureRef, UsfmStylesheet, UsfmToken, UsfmTokenizer, UsfmTokenType
from machine.tokenization import LatinWordTokenizer
from machine.translation import WordAlignmentMatrix

from ..alignment.eflomal import to_word_alignment_matrix
from ..alignment.utils import compute_alignment_scores
from .corpus import load_corpus, write_corpus

# Marker "type" is as defined by the UsfmTokenType given to tokens by the UsfmTokenizer,
# which mostly aligns with a marker's StyleType in the USFM stylesheet
CHARACTER_TYPE_EMBEDS = ["fig", "fm", "jmp", "rq", "va", "vp", "xt", "xtSee", "xtSeeAlso"]
PARAGRAPH_TYPE_EMBEDS = ["lit", "r", "rem"]
NON_NOTE_TYPE_EMBEDS = CHARACTER_TYPE_EMBEDS + PARAGRAPH_TYPE_EMBEDS


class UsfmPreserver:
    _src_sents: List[str]
    _vrefs: List[ScriptureRef]

    # (sent_idx, start idx in text_only_sent, is_paragraph_marker, tok (inc. \ and spaces))
    _markers: List[Tuple[int, int, str]]
    # (sent_idx, embed)
    _char_embeds: List[Tuple[int, str]]
    # (sent_idx, ref, embed contents)
    _para_embeds: List[Tuple[int, ScriptureRef, str]]

    def __init__(
        self,
        src_sents: List[str],
        vrefs: List[ScriptureRef],
        stylesheet: UsfmStylesheet,
        include_paragraph_markers: bool = False,
        include_style_markers: bool = False,
        include_embeds: bool = False,
    ):
        # Remove sentences that are paragraph-type embeds
        # NOTE: when only dealing with inserting back into the same USFM structure, i.e. translate_usfm's trg_project is None,
        # paragraph-type embeds can be handled more simply with the updater's preserve_paragraph_styles argument,
        # but because this approach is necessary when updating a project different from the source, we use it for both cases
        src_sents, self._vrefs = self._remove_para_embeds(src_sents, vrefs, include_embeds)

        usfm_tokenizer = UsfmTokenizer(stylesheet)
        sentence_toks = []
        for sent in src_sents:
            sentence_toks.append(usfm_tokenizer.tokenize(sent))

        # Take markers and character-type embeds out of sentences
        self._src_sents = self._extract_markers(
            sentence_toks, include_paragraph_markers, include_style_markers, include_embeds
        )
        self._src_tok_ranges = self._tokenize_sents(self._src_sents)

    # Source sentences without USFM markers or embeds, to be used as input to an MT model
    @property
    def src_sents(self) -> List[str]:
        return self._src_sents

    @property
    def vrefs(self) -> List[ScriptureRef]:
        return self._vrefs

    def _remove_para_embeds(
        self, sents: List[str], vrefs: List[ScriptureRef], include_embeds: bool
    ) -> List[ScriptureRef]:
        para_embeds = []
        for i, (sent, ref) in reversed(list(enumerate(zip(sents, vrefs)))):
            if (ref.path[-1].name if len(ref.path) > 0 else "") in PARAGRAPH_TYPE_EMBEDS:
                para_embeds.append((i, ref, sent if include_embeds else ""))
                sents.pop(i)
                vrefs.pop(i)

        self._para_embeds = list(reversed(para_embeds))
        return sents, vrefs

    def _extract_markers(
        self,
        sentence_toks: List[List[UsfmToken]],
        include_paragraph_markers: bool,
        include_style_markers: bool,
        include_embeds: bool,
    ) -> List[str]:
        markers = []
        char_embeds = []
        text_only_sents = ["" for _ in sentence_toks]
        for i, toks in enumerate(sentence_toks):
            embed_usfm = ""
            curr_embed = None
            for tok in toks:
                if curr_embed is not None:
                    embed_usfm += tok.to_usfm()
                    if tok.type == UsfmTokenType.END and tok.marker[:-1] == curr_embed.marker:
                        if include_embeds:
                            char_embeds.append((i, embed_usfm))
                        embed_usfm = ""
                        curr_embed = None
                elif tok.type == UsfmTokenType.NOTE or tok.marker in CHARACTER_TYPE_EMBEDS:
                    embed_usfm += tok.to_usfm()
                    curr_embed = tok
                elif tok.type == UsfmTokenType.PARAGRAPH and include_paragraph_markers:
                    markers.append((i, len(text_only_sents[i]), True, tok.to_usfm()))
                elif tok.type in [UsfmTokenType.CHARACTER, UsfmTokenType.END] and include_style_markers:
                    markers.append((i, len(text_only_sents[i]), False, tok.to_usfm()))
                elif tok.type == UsfmTokenType.TEXT:
                    text_only_sents[i] += tok.text

        self._markers = markers
        self._char_embeds = char_embeds
        return text_only_sents

    def construct_rows(self, translations: List[str]) -> List[Tuple[List[ScriptureRef], str]]:
        # Map each token to a character range in the original strings
        trg_tok_ranges = self._tokenize_sents(translations)

        # Get index of the text token immediately following each marker and predict the corresponding token on the target side
        adj_src_toks = []
        for sent_idx, start_idx, _, _ in self._markers:
            for i, tok_range in reversed(list(enumerate(self._src_tok_ranges[sent_idx]))):
                if tok_range.start < start_idx:
                    adj_src_toks.append(i + 1)
                    break
                if i == 0:
                    adj_src_toks.append(i)
        adj_trg_toks = self._predict_marker_locations(adj_src_toks, translations, trg_tok_ranges)

        # Collect the markers to be inserted
        to_insert = [[] for _ in trg_tok_ranges]
        for i, ((sent_idx, _, is_para_marker, marker), adj_trg_tok) in enumerate(zip(self._markers, adj_trg_toks)):
            trg_str_idx = (
                trg_tok_ranges[sent_idx][adj_trg_tok].start
                if adj_trg_tok < len(trg_tok_ranges[sent_idx])
                else len(translations[sent_idx])
            )

            # Determine the order of the markers in the sentence to handle ambiguity for directly adjacent markers
            insert_pos = 0
            while insert_pos < len(to_insert[sent_idx]) and to_insert[sent_idx][insert_pos][0] <= trg_str_idx:
                insert_pos += 1
            to_insert[sent_idx].insert(insert_pos, (trg_str_idx, is_para_marker, marker))

        # Construct rows for the USFM file
        embed_idx = 0
        para_embed_idx = 0
        rows = []

        # Add any paragraph-style embeds that come before the main sentences
        while para_embed_idx < len(self._para_embeds) and self._para_embeds[para_embed_idx][0] == para_embed_idx:
            rows.append(([self._para_embeds[para_embed_idx][1]], self._para_embeds[para_embed_idx][2]))
            para_embed_idx += 1

        for i, (ref, translation, inserts) in enumerate(zip(self._vrefs, translations, to_insert)):
            # row_text = translation[: inserts[0][0]] if len(inserts) > 0 else translation
            row_texts = [translation[: inserts[0][0]]] if len(inserts) > 0 else [translation]

            for j, (insert_idx, is_para_marker, marker) in enumerate(inserts):
                if is_para_marker:
                    row_texts.append("")

                # row_text += (
                row_texts[-1] += (
                    #     ("\n" if is_para_marker else "")
                    #     + marker
                    (marker if not is_para_marker else "")
                    + (
                        " "  # Extra space if inserting an end marker before a non-punctuation character
                        if "*" in marker and insert_idx < len(translation) and translation[insert_idx].isalpha()
                        else ""
                    )
                    + (
                        translation[insert_idx : inserts[j + 1][0]]
                        if j + 1 < len(inserts)
                        else translation[insert_idx:]
                    )
                )
                # Prevent spaces before end markers
                # if j + 1 < len(inserts) and "*" in inserts[j + 1][2] and len(row_text) > 0 and row_text[-1] == " ":
                #     row_text = row_text[:-1]
                if (
                    j + 1 < len(inserts)
                    and "*" in inserts[j + 1][2]
                    and len(row_texts[-1]) > 0
                    and row_texts[-1][-1] == " "
                ):
                    row_texts[-1] = row_texts[-1][:-1]

            # Append any transferred embeds that match the current ScriptureRef
            while embed_idx < len(self._char_embeds) and self._char_embeds[embed_idx][0] == i:
                # row_text += self._char_embeds[embed_idx][1]
                row_texts[-1] += self._char_embeds[embed_idx][1]
                embed_idx += 1

            # rows.append(([ref], row_text))
            for row_text in row_texts:
                rows.append(([ref], row_text))

            # (sent_idx, ref, embed contents)
            # sent_idx == orig idx, in order
            while (
                para_embed_idx < len(self._para_embeds)
                and self._para_embeds[para_embed_idx][0] == i + 1 + para_embed_idx
            ):
                rows.append(([self._para_embeds[para_embed_idx][1]], self._para_embeds[para_embed_idx][2]))
                para_embed_idx += 1

        # # Add transferred paragraph-type embeds
        # for sent_idx, ref, sent in self._para_embeds:
        #     rows.insert(sent_idx, ([ref], sent))

        return rows

    @abstractmethod
    def _tokenize_sents(self, sents: List[str]) -> List[List[Range[int]]]: ...

    @abstractmethod
    def _get_alignment_matrices(self, trg_sents: List[str]) -> List[WordAlignmentMatrix]: ...

    def _predict_marker_locations(
        self, adj_src_toks: List[int], trg_sents: List[str], trg_tok_ranges: List[List[Range[int]]]
    ) -> List[int]:
        if len(adj_src_toks) == 0:
            return []

        alignment_matrices: List[WordAlignmentMatrix] = self._get_alignment_matrices(trg_sents)

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
        for (sent_idx, _, _, _), adj_src_tok in zip(self._markers, adj_src_toks):
            # If the token on either side of a potential target location is punctuation,
            # use it as the basis for deciding the target marker location
            trg_hyp = -1
            punct_hyps = [-1, 0]
            for punct_hyp in punct_hyps:
                src_hyp = adj_src_tok + punct_hyp
                if src_hyp < 0 or src_hyp >= len(self._src_tok_ranges[sent_idx]):
                    continue
                # only accept aligned pairs where both the src and trg token are punct
                src_hyp_range = self._src_tok_ranges[sent_idx][src_hyp]
                if (
                    src_hyp_range.length > 0
                    and not any(self._src_sents[sent_idx][char_idx].isalpha() for char_idx in src_hyp_range)
                    and src_hyp < alignment_matrices[sent_idx].row_count
                ):
                    aligned_trg_toks = list(alignment_matrices[sent_idx].get_row_aligned_indices(src_hyp))
                    # if aligning to a token that precedes that marker,
                    # the trg token predicted to be closest to the marker is the last token aligned to the src rather than the first
                    for trg_tok in reversed(aligned_trg_toks) if punct_hyp < 0 else aligned_trg_toks:
                        trg_tok_range = trg_tok_ranges[sent_idx][trg_tok]
                        if not any(trg_sents[sent_idx][char_idx].isalpha() for char_idx in trg_tok_range):
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
            best_hyp = -1
            best_num_crossings = 200**2  # mostly meaningless, a big number
            checked = set()
            for hyp in hyps:
                src_hyp = adj_src_tok + hyp
                if src_hyp in checked:
                    continue
                trg_hyp = -1
                while trg_hyp == -1 and src_hyp >= 0 and src_hyp < alignment_matrices[sent_idx].row_count:
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
                    if num_crossings < best_num_crossings:
                        best_hyp = trg_hyp
                        best_num_crossings = num_crossings

            # if no alignments found, insert at the end of the sentence
            if best_hyp == -1:
                adj_trg_toks.append(len(trg_tok_ranges[sent_idx]))
                continue

            adj_trg_toks.append(best_hyp)

        return adj_trg_toks


class StatisticalUsfmPreserver(UsfmPreserver):
    def __init__(
        self,
        src_sentences,
        vrefs,
        stylesheet,
        include_paragraph_markers,
        include_style_markers,
        include_embeds,
        aligner="eflomal",
    ):
        self._aligner = aligner
        super().__init__(
            src_sentences, vrefs, stylesheet, include_paragraph_markers, include_style_markers, include_embeds
        )

    def _tokenize_sents(self, sents: List[str]) -> List[List[Range[int]]]:
        tokenizer = LatinWordTokenizer()
        return [list(tokenizer.tokenize_as_ranges(sent)) for sent in sents]

    def _get_alignment_matrices(self, trg_sents: List[str]) -> List[WordAlignmentMatrix]:
        with TemporaryDirectory() as td:
            align_path = Path(td, "sym-align.txt")
            write_corpus(Path(td, "src_align.txt"), self._src_sents)
            write_corpus(Path(td, "trg_align.txt"), trg_sents)
            compute_alignment_scores(Path(td, "src_align.txt"), Path(td, "trg_align.txt"), self._aligner, align_path)

            return [to_word_alignment_matrix(line) for line in load_corpus(align_path)]


"""
Necessary changes to use AttentionUsfmPreserver:
* Use machine.py's _TranslationPipeline (in translation.huggingface.hugging_face_nmt_engine)
  to output attentions along with the translations
* Build TranslationResults based on the outputs of the pipeline, using the logic from
  HuggingFaceNmtEngine._try_translate_n_batch (also in translation.huggingface.hugging_face_nmt_engine)
"""


class AttentionUsfmPreserver(UsfmPreserver):
    def __init__(self, src_sents, vrefs, stylesheet, include_paragraph_markers, include_style_markers, include_embeds):
        raise NotImplementedError(
            "AttentionUsfmPreserver is not a supported class. See class definition for more information about the work needed to use."
        )

    """
    def construct_rows(self, translation_results: List[TranslationResult]) -> List[Tuple[List[ScriptureRef], str]]:
        self._translation_results = translation_results
        # TODO: do source token ranges need to be reconstructed for each draft?
        self._src_tok_ranges = self._construct_tok_ranges([tr.source_tokens for tr in translation_results])

        super().construct_rows(translation_results)

    # NOTE: only used for target side
    # NOTE: _tokenize_sents is called for the source side in UsfmPreserver.__init__, but it will get overwritten
    #       with the correct tokens when construct_rows is called
    def _tokenize_sents(self, sents: List[str]) -> Tuple[List[List[Range[int]]], List[List[str]]]:
        return self._construct_tok_ranges([tr.target_tokens for tr in self._translation_results])

    # NOTE: the "▁" characters in this function are from the NllbTokenizer and are not the same character as the typical underscore
    def _construct_tok_ranges(self, toks: List[str]) -> List[List[Range[int]]]:
        tok_ranges = []
        for sent_toks in toks:
            sent_tok_ranges = [Range.create(0, len(sent_toks[0]) - 1)]
            for tok in sent_toks[1:]:
                sent_tok_ranges.append(
                    Range.create(
                        sent_tok_ranges[-1].end + (1 if tok[0] == "▁" else 0),
                        sent_tok_ranges[-1].end + len(tok),
                    )
                )
            tok_ranges.append(sent_tok_ranges)
        return tok_ranges

    def _get_alignment_matrices(self) -> List[WordAlignmentMatrix]:
        return [tr.alignment for tr in self._translation_results]
    """
