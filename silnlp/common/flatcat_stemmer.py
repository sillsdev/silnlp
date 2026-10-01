import logging
from typing import Any, Iterable, List, Sequence, Tuple

import flatcat
import morfessor

from ..common.stemmer import Stemmer


def convert_verses_to_morfessor_data(verses: Iterable[Sequence[str]]) -> Iterable[Tuple[int, Any]]:
    for verse in verses:
        for word in verse:
            yield (1, word)
        yield (0, ())


def setup_morfessor_logging() -> None:
    plain_formatter = logging.Formatter("%(message)s")
    morfessor._logger.propagate = False
    flatcat._logger.propagate = False

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(plain_formatter)

    morfessor._logger.addHandler(ch)
    flatcat._logger.addHandler(ch)


setup_morfessor_logging()


class FlatCatStemmer(Stemmer):
    def __init__(self, corpus_weight: float = 1.0, ppl_threshold: float = 100) -> None:
        self.morfessor_model = morfessor.BaselineModel(corpusweight=corpus_weight, forcesplit_list=["-"])

        props = flatcat.MorphUsageProperties(ppl_threshold=ppl_threshold)
        self.flatcat_model = flatcat.FlatcatModel(
            props, corpusweight=corpus_weight, forcesplit=["-"], ml_emissions_epoch=0
        )
        self.flatcat_model.postprocessing.append(flatcat.HeuristicPostprocessor())

    def train(self, corpus: Iterable[Sequence[str]]) -> None:
        print("Training Morfessor model...")
        print("Hyperparameters:")
        print(f"- corpusweight: {self.morfessor_model.get_corpus_coding_weight()}")
        self.morfessor_model.load_data(convert_verses_to_morfessor_data(corpus), count_modifier=lambda x: 1)
        self.morfessor_model.train_batch()

        print("Training FlatCat model...")
        print("Hyperparameters:")
        params = self.flatcat_model.get_params()
        for key, value in params.items():
            if isinstance(value, float):
                value = round(value, 5)
            print(f"- {key}: {value}")
        self.flatcat_model.add_corpus_data(map(lambda s: (s[0], tuple(s[2])), self.morfessor_model.get_segmentations()))
        self.flatcat_model.initialize_hmm()
        self.flatcat_model.train_batch(max_epochs=4, min_epoch_cost_gain=None, max_resegment_iterations=2)
        print("Done.")

    def stem(self, words: Sequence[str]) -> Sequence[str]:
        stems: List[str] = []
        for word in words:
            constructions, _ = self.flatcat_model.viterbi_analyze(word)
            for processor in self.flatcat_model.postprocessing:
                constructions = processor.apply_to(constructions, self.flatcat_model)
            stem = ""
            for construct in constructions:
                if construct.category == "STM":
                    stem += construct.morph
            assert len(stem) > 0
            stems.append(stem)
        return stems
