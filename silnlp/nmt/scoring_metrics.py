from typing import Dict, Iterator, List, Optional, Set, TextIO, Tuple

import sacrebleu
from sacrebleu.metrics import BLEUScore

CORPUS_SCORERS = [
    "bleu",
    "chrf3",
    "chrf3+",
    "chrf3++",
    "spbleu",
    "m-bleu",
    "m-chrf3",
    "m-chrf3+",
    "m-chrf3++",
    "ter",
]

SENTENCE_SCORERS = [
    "bleu",
    "chrf3",
    "chrf3+",
    "chrf3++",
    "spbleu",
    "ter",
]

DEFAULT_SACREBLEU_TOKENIZE = "13a"


class PairScore:
    def __init__(
        self,
        book: str,
        src_iso: str,
        trg_iso: str,
        bleu: Optional[BLEUScore],
        sent_len: int,
        projects: Set[str],
        other_scores: Dict[str, float] = {},
        draft_index: int = 1,
    ) -> None:
        self.src_iso = src_iso
        self.trg_iso = trg_iso
        self.bleu = bleu
        self.sent_len = sent_len
        self.num_refs = len(projects)
        self.refs = "_".join(sorted(projects))
        self.other_scores = other_scores
        self.book = book
        self.draft_index = draft_index

    def writeHeader(self, file: TextIO) -> None:
        header = (
            "book,draft_index,src_iso,trg_iso,num_refs,references,sent_len"
            + (
                ",BLEU,BLEU_1gram_prec,BLEU_2gram_prec,BLEU_3gram_prec,BLEU_4gram_prec,BLEU_brevity_penalty,BLEU_total_sys_len,BLEU_total_ref_len"
                if self.bleu is not None
                else ""
            )
            + ("," if len(self.other_scores) > 0 else "")
            + ",".join(self.other_scores.keys())
            + "\n"
        )
        file.write(header)

    def write(self, file: TextIO) -> None:
        file.write(
            f"{self.book},{self.draft_index},{self.src_iso},{self.trg_iso},"
            f"{self.num_refs},{self.refs},{self.sent_len:d}"
        )
        if self.bleu is not None:
            file.write(
                f",{self.bleu.score:.2f},{self.bleu.precisions[0]:.2f},{self.bleu.precisions[1]:.2f}"
                f",{self.bleu.precisions[2]:.2f},{self.bleu.precisions[3]:.2f},{self.bleu.bp:.3f}"
                f",{self.bleu.sys_len:d},{self.bleu.ref_len:d}"
            )
        for scorer, val in self.other_scores.items():
            if scorer.lower() == "confidence":
                file.write(f",{val:.8f}")
            else:
                file.write(f",{val:.2f}")
        file.write("\n")


def compute_corpus_scores(
    pair_sys: List[str],
    pair_refs: List[List[str]],
    scorers: Set[str],
    sacrebleu_tokenize: Optional[str] = None,
) -> Tuple[Optional[BLEUScore], Dict[str, float]]:
    sacrebleu_tokenize = sacrebleu_tokenize or DEFAULT_SACREBLEU_TOKENIZE
    bleu_score = None
    if "bleu" in scorers:
        bleu_score = sacrebleu.corpus_bleu(
            pair_sys,
            pair_refs,
            lowercase=True,
            tokenize=sacrebleu_tokenize,
        )

    other_scores: Dict[str, float] = {}
    if "chrf3" in scorers:
        chrf3_score = sacrebleu.corpus_chrf(pair_sys, pair_refs, char_order=6, beta=3, remove_whitespace=True)
        other_scores["chrF3"] = chrf3_score.score

    if "chrf3+" in scorers:
        chrfp_score = sacrebleu.corpus_chrf(
            pair_sys, pair_refs, char_order=6, beta=3, word_order=1, remove_whitespace=True, eps_smoothing=True
        )
        other_scores["chrF3+"] = chrfp_score.score

    if "chrf3++" in scorers:
        chrfpp_score = sacrebleu.corpus_chrf(
            pair_sys, pair_refs, char_order=6, beta=3, word_order=2, remove_whitespace=True, eps_smoothing=True
        )
        other_scores["chrF3++"] = chrfpp_score.score

    if "spbleu" in scorers:
        spbleu_score = sacrebleu.corpus_bleu(
            pair_sys,
            pair_refs,
            lowercase=True,
            tokenize="flores200",
        )
        other_scores["spBLEU"] = spbleu_score.score

    # m-bleu, m-chrf3, m-chrf3+, and m-chrf3++ are from the paper https://arxiv.org/pdf/2407.12832
    # These metrics are implemented at the verse-level, rather than the sentence-level
    if "m-bleu" in scorers:
        sentence_bleu_scores: List[float] = []
        for sentence_i, sentence in enumerate(pair_sys):
            references = [reference[sentence_i] for reference in pair_refs]
            sentence_bleu_score = sacrebleu.sentence_bleu(
                sentence,
                references,
                lowercase=True,
                tokenize=sacrebleu_tokenize,
            )
            sentence_bleu_scores.append(sentence_bleu_score.score)
        if len(sentence_bleu_scores) == 0:
            other_scores["m-BLEU"] = 0
        else:
            other_scores["m-BLEU"] = sum(sentence_bleu_scores) / len(sentence_bleu_scores)

    if "m-chrf3" in scorers:
        sentence_chrf3_scores: List[float] = []
        for sentence_i, sentence in enumerate(pair_sys):
            references = [reference[sentence_i] for reference in pair_refs]
            sentence_chrf3_score = sacrebleu.sentence_chrf(
                sentence, references, char_order=6, beta=3, remove_whitespace=True
            )
            sentence_chrf3_scores.append(sentence_chrf3_score.score)
        if len(sentence_chrf3_scores) == 0:
            other_scores["m-chrf3"] = 0
        else:
            other_scores["m-chrf3"] = sum(sentence_chrf3_scores) / len(sentence_chrf3_scores)

    if "m-chrf3+" in scorers:
        sentence_chrfp_scores: List[float] = []
        for sentence_i, sentence in enumerate(pair_sys):
            references = [reference[sentence_i] for reference in pair_refs]
            sentence_chrfp_score = sacrebleu.sentence_chrf(
                sentence, references, char_order=6, beta=3, word_order=1, remove_whitespace=True, eps_smoothing=True
            )
            sentence_chrfp_scores.append(sentence_chrfp_score.score)
        if len(sentence_chrfp_scores) == 0:
            other_scores["m-chrf3+"] = 0
        else:
            other_scores["m-chrf3+"] = sum(sentence_chrfp_scores) / len(sentence_chrfp_scores)

    if "m-chrf3++" in scorers:
        sentence_chrfpp_scores: List[float] = []
        for sentence_i, sentence in enumerate(pair_sys):
            references = [reference[sentence_i] for reference in pair_refs]
            sentence_chrfpp_score = sacrebleu.sentence_chrf(
                sentence, references, char_order=6, beta=3, word_order=2, remove_whitespace=True, eps_smoothing=True
            )
            sentence_chrfpp_scores.append(sentence_chrfpp_score.score)
        if len(sentence_chrfpp_scores) == 0:
            other_scores["m-chrf3++"] = 0
        else:
            other_scores["m-chrf3++"] = sum(sentence_chrfpp_scores) / len(sentence_chrfpp_scores)

    if "ter" in scorers:
        ter_score = sacrebleu.corpus_ter(pair_sys, pair_refs)
        if ter_score.score >= 0:
            other_scores["TER"] = ter_score.score

    return bleu_score, other_scores


def iter_verse_scores(
    pair_sys: List[str],
    pair_refs: List[List[str]],
    scorers: Set[str],
    sacrebleu_tokenize: Optional[str] = None,
) -> Iterator[Tuple[int, str, List[str], Optional[BLEUScore], Dict[str, float]]]:
    sacrebleu_tokenize = sacrebleu_tokenize or DEFAULT_SACREBLEU_TOKENIZE
    spbleu_metric = sacrebleu.metrics.BLEU(tokenize="flores200", lowercase=True) if "spbleu" in scorers else None
    for index, pred in enumerate(pair_sys):
        sentences: List[str] = []
        for ref in pair_refs:
            sentences.append(ref[index])

        bleu_verse_score = None
        if "bleu" in scorers:
            bleu_verse_score = sacrebleu.sentence_bleu(
                pred,
                sentences,
                lowercase=True,
                tokenize=sacrebleu_tokenize,
            )

        other_verse_scores: Dict[str, float] = {}
        if "chrf3" in scorers:
            chrf3_verse_score = sacrebleu.sentence_chrf(pred, sentences, char_order=6, beta=3, remove_whitespace=True)
            other_verse_scores["chrF3"] = chrf3_verse_score.score

        if "chrf3+" in scorers:
            chrfp_verse_score = sacrebleu.sentence_chrf(
                pred, sentences, char_order=6, beta=3, word_order=1, remove_whitespace=True, eps_smoothing=True
            )
            other_verse_scores["chrF3+"] = chrfp_verse_score.score

        if "chrf3++" in scorers:
            chrfpp_verse_score = sacrebleu.sentence_chrf(
                pred, sentences, char_order=6, beta=3, word_order=2, remove_whitespace=True, eps_smoothing=True
            )
            other_verse_scores["chrF3++"] = chrfpp_verse_score.score

        if "spbleu" in scorers and spbleu_metric is not None:
            spbleu_verse_score = spbleu_metric.sentence_score(pred, sentences)
            other_verse_scores["spBLEU"] = spbleu_verse_score.score

        if "ter" in scorers:
            ter_verse_score = sacrebleu.sentence_ter(pred, sentences)
            if ter_verse_score.score >= 0:
                other_verse_scores["TER"] = ter_verse_score.score

        yield index, pred, sentences, bleu_verse_score, other_verse_scores
