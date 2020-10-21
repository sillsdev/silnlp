import logging
from typing import Any, Iterable, List, Tuple

import flatcat
import morfessor

from nlp.common.stemmer import Stemmer


def convert_verses_to_morfessor_data(verses: Iterable[List[str]]) -> Iterable[Tuple[int, Any]]:
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
    def __init__(self, ppl_threshold: float = 100) -> None:
        props = flatcat.MorphUsageProperties(ppl_threshold=ppl_threshold)
        self.model = flatcat.FlatcatModel(props, forcesplit=["-"], ml_emissions_epoch=0)
        self.model.postprocessing.append(flatcat.HeuristicPostprocessor())

    def train(self, corpus: Iterable[List[str]]) -> None:
        print("Training Morfessor model...")
        morfessor_model = morfessor.BaselineModel(forcesplit_list=["-"])
        morfessor_model.load_data(convert_verses_to_morfessor_data(corpus), count_modifier=lambda x: 1)
        morfessor_model.train_batch()

        print("Training FlatCat model...")
        print("Hyperparameters:")
        params = self.model.get_params()
        for key, value in params.items():
            if isinstance(value, float):
                value = round(value, 5)
            print(f"- {key}: {value}")
        self.model.add_corpus_data(map(lambda s: (s[0], tuple(s[2])), morfessor_model.get_segmentations()))
        self.model.initialize_hmm()
        self.model.train_batch(max_epochs=4, min_epoch_cost_gain=None, max_resegment_iterations=2)
        print("Done.")

    def stem(self, words: List[str]) -> List[str]:
        stems: List[str] = []
        for word in words:
            constructions, _ = self.model.viterbi_analyze(word)
            for processor in self.model.postprocessing:
                constructions = processor.apply_to(constructions, self.model)
            stem = ""
            for construct in constructions:
                if construct.category == "STM":
                    stem += construct.morph
            assert len(stem) > 0
            stems.append(stem)
        return stems
