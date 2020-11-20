from typing import Iterable, List

from nltk import download, pos_tag
from nltk.stem import WordNetLemmatizer

from ..common.stemmer import Stemmer


def convert_treebank_tag_to_wordnet_tag(tag: str) -> str:
    if tag.startswith("NN") or tag == "CD":
        return "n"
    if tag.startswith("JJ") or tag in {"PDT", "RP"}:
        return "a"
    if tag.startswith("VB"):
        return "v"
    if tag.startswith("RB") or tag in {"EX", "IN"}:
        return "r"
    return "n"


class WordNetStemmer(Stemmer):
    def __init__(self) -> None:
        download("wordnet")
        download("averaged_perceptron_tagger")
        self.lemmatizer = WordNetLemmatizer()

    def train(self, corpus: Iterable[List[str]]) -> None:
        pass

    def stem(self, words: List[str]) -> List[str]:
        tagged = pos_tag(words)
        stems: List[str] = []
        for word, tag in tagged:
            stem = self.lemmatizer.lemmatize(word, convert_treebank_tag_to_wordnet_tag(tag))
            stems.append(stem)
        return stems
