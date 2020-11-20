from typing import Iterable, List

from nltk.stem import SnowballStemmer as NltkSnowballStemmer

from ..common.stemmer import Stemmer

SUPPORTED_LANGS = {
    "ar": "arabic",
    "da": "danish",
    "nl": "dutch",
    "en": "english",
    "fi": "finnish",
    "fr": "french",
    "de": "german",
    "hu": "hungarian",
    "it": "italian",
    "no": "norwegian",
    "pt": "portuguese",
    "ro": "romanian",
    "ru": "russian",
    "es": "spanish",
    "sv": "swedish",
}


class SnowballStemmer(Stemmer):
    def __init__(self, lang: str = "en", ignore_stopwords: bool = False) -> None:
        language_name = SUPPORTED_LANGS.get(lang)
        if language_name is None:
            raise RuntimeError("The specified language is not supported by the Snowball stemmer.")
        self.stemmer = NltkSnowballStemmer(language_name, ignore_stopwords=ignore_stopwords)

    def train(self, corpus: Iterable[List[str]]) -> None:
        pass

    def stem(self, words: List[str]) -> List[str]:
        return list(map(lambda w: self.stemmer.stem(w), words))
