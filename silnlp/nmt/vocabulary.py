import json
from pathlib import Path
from typing import Dict, List, Set, Union

import pandas as pd
from tokenizers import AddedToken
from tokenizers.implementations import SentencePieceBPETokenizer, SentencePieceUnigramTokenizer
from tokenizers.normalizers import Normalizer
from transformers.models.m2m_100.tokenization_m2m_100 import M2M100Tokenizer
from transformers.models.mbart.tokenization_mbart import MBartTokenizer
from transformers.models.mbart50.tokenization_mbart50 import MBart50Tokenizer
from transformers.models.nllb.tokenization_nllb import NllbTokenizer
from transformers.models.t5.tokenization_t5 import T5Tokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from ..common.utils import Side
from .corpus_inventory import CorpusInventory
from .huggingface_tokenizer import CustomNormalizerWrapper, HuggingFaceTokenizer
from .model_name import ModelName
from .pretrained_tokenizer import PretrainedTokenizer
from .token_occurrence_logger import TokenOccurrenceLogger
from .tokenizer import Tokenizer
from .tokenizer_settings import TokenizerSettings, TokenizerSource
from .vocabulary_builder import VocabularyBuilder

TrainedTokenizer = Union[SentencePieceBPETokenizer, SentencePieceUnigramTokenizer]


class VocabularyExtension:
    """The tokens an experiment's corpora need beyond the vocabulary its pretrained tokenizer came with."""

    def __init__(
        self,
        tokens: List[str],
        source_tokens: List[str],
        target_tokens: List[str],
        trained_tokenizers: List[TrainedTokenizer],
        shared: bool,
    ) -> None:
        self._tokens = tokens
        self._source_tokens = source_tokens
        self._target_tokens = target_tokens
        self._trained_tokenizers = trained_tokenizers
        self._shared = shared

    def add_to(self, vocabulary: "TokenizerVocabulary") -> None:
        if self._tokens:
            vocabulary.add(self._tokens, self._trained_tokenizers)

    def counts_by_side(self) -> List[List]:
        if self._shared:
            # TODO: Calculate representative split of tokens for shared vocab case
            source_share = int(len(self._tokens) / 2)
            return [["Source", source_share], ["Target", len(self._tokens) - source_share]]
        return [["Source", len(self._source_tokens)], ["Target", len(self._target_tokens)]]


class MissingTokens:
    """Finds what an experiment's corpora need that its vocabulary does not cover, either by training a
    SentencePiece model on them or by collecting the characters the vocabulary has no token for."""

    def __init__(
        self,
        pretrained: PretrainedTokenizer,
        tokenizer: HuggingFaceTokenizer,
        source: TokenizerSource,
        settings: TokenizerSettings,
        inventory: CorpusInventory,
        model_name: ModelName,
        exp_dir: Path,
    ) -> None:
        self._pretrained = pretrained
        self._tokenizer = tokenizer
        self._source = source
        self._settings = settings
        self._inventory = inventory
        self._model_name = model_name
        self._exp_dir = exp_dir

    def find(self) -> VocabularyExtension:
        settings = self._settings
        shared = settings.shares_vocab() and settings.updates_both()
        if not settings.updates_either():
            return VocabularyExtension([], [], [], [], shared)
        if settings.trains_tokens() and self._source.has_tokenizer_assets():
            return self._trained(shared)
        return self._characters(shared)

    def _trained(self, shared: bool) -> VocabularyExtension:
        settings = self._settings
        source_paths = list(self._inventory.source_file_paths())
        target_paths = list(self._inventory.target_file_paths())
        if not settings.shares_vocab() and settings.updates_both():
            source_tokens, source_tokenizer = self._train(source_paths, settings.source_vocab_size())
            target_tokens, target_tokenizer = self._train(target_paths, settings.target_vocab_size())
            target_tokens = sorted(set(target_tokens) - set(source_tokens))
            return VocabularyExtension(
                source_tokens + target_tokens,
                source_tokens,
                target_tokens,
                [source_tokenizer, target_tokenizer],
                shared,
            )
        if settings.shares_vocab() and settings.updates_both():
            tokens, trained = self._train(source_paths + target_paths, settings.shared_vocab_size())
            return VocabularyExtension(tokens, [], [], [trained], shared)
        if settings.updates_source():
            tokens, trained = self._train(source_paths, settings.source_vocab_size())
            return VocabularyExtension(tokens, tokens, [], [trained], shared)
        tokens, trained = self._train(target_paths, settings.target_vocab_size())
        return VocabularyExtension(tokens, [], tokens, [trained], shared)

    def _characters(self, shared: bool) -> VocabularyExtension:
        settings = self._settings
        tokens: List[str] = []
        source_tokens: List[str] = []
        target_tokens: List[str] = []
        if settings.updates_source():
            tokens = source_tokens = self._uncovered_characters(list(self._inventory.source_file_paths()))
        if settings.updates_target():
            tokens = target_tokens = self._uncovered_characters(list(self._inventory.target_file_paths()))
        if settings.updates_both():
            target_tokens = sorted(set(target_tokens) - set(source_tokens))
            tokens = source_tokens + target_tokens
        return VocabularyExtension(tokens, source_tokens, target_tokens, [], shared)

    def _train(self, file_paths: List[Path], vocab_size: int):
        trained = self._train_sentence_piece([str(path) for path in file_paths], vocab_size)
        trained_keys = trained.get_vocab().keys()
        known_keys = self._pretrained.build().get_vocab().keys()
        missing = sorted(set(trained_keys) - set(known_keys))
        self._log_occurrences(missing, file_paths)
        return missing, trained

    def _train_sentence_piece(self, files: List[str], vocab_size: int) -> TrainedTokenizer:
        tokenizer = self._pretrained.build()
        settings = self._model_name.sentence_piece_settings()
        trained = SentencePieceBPETokenizer() if settings["type"] == "BPE" else SentencePieceUnigramTokenizer()
        trained.normalizer = Normalizer.custom(CustomNormalizerWrapper(self._tokenizer))

        if settings["type"] == "BPE":
            trained.train(files, vocab_size=vocab_size, min_frequency=2, special_tokens=settings["special_tokens"])
        elif settings["type"] == "Unigram":
            trained.train(
                files,
                vocab_size=vocab_size,
                special_tokens=settings["special_tokens"],
                unk_token=settings["unk_token"],
            )
        trained.normalizer = tokenizer.backend_tokenizer.normalizer
        return trained

    def _uncovered_characters(self, corpus: List[Path]) -> List[str]:
        vocab = self._pretrained.build().get_vocab().keys()
        charset: Set[str] = set()
        for path in corpus:
            with path.open("r", encoding="utf-8-sig") as file:
                for line in file:
                    charset = charset | set(self._tokenizer.normalize(Side.TARGET, line))

        charset = set(filter(None, {character.strip() for character in charset}))
        missing = sorted(charset - vocab)
        self._log_occurrences(missing, corpus)
        return missing

    def _log_occurrences(self, tokens: List[str], corpus: List[Path]) -> None:
        with TokenOccurrenceLogger(corpus, self._exp_dir) as logger:
            logger.log(tokens)


class TokenizerVocabularyFile:
    """The model vocabulary inside a saved tokenizer.json. New tokens are written into it directly, because
    the tokenizer's own add_tokens would make them added tokens rather than part of the model."""

    def __init__(self, path: Path) -> None:
        self._path = path

    def add(self, tokens: List[str], trained_models: List[dict]) -> None:
        with open(self._path, "r+", encoding="utf-8") as file:
            data = json.load(file)
            if data["model"]["type"] == "BPE":
                self._add_to_byte_pair_model(data, tokens, trained_models)
            elif data["model"]["type"] == "Unigram":
                self._add_to_unigram_model(data, tokens, trained_models)
            file.seek(0)
            json.dump(data, file, ensure_ascii=False, indent=4)
            file.truncate()

    def _add_to_byte_pair_model(self, data: dict, tokens: List[str], trained_models: List[dict]) -> None:
        vocab_len = len(data["model"]["vocab"].keys())
        for index, token in enumerate(tokens):
            data["model"]["vocab"][token] = vocab_len + index
        for trained in trained_models:
            data["model"]["merges"] = trained["model"]["merges"] + data["model"]["merges"]

    def _add_to_unigram_model(self, data: dict, tokens: List[str], trained_models: List[dict]) -> None:
        if not trained_models:
            for token in tokens:
                data["model"]["vocab"].append([token, -18])
            return
        for trained in trained_models:
            # Use the probability from the base tokenizer for tokens already in the base tokenizer
            base_tokens = [entry[0] for entry in data["model"]["vocab"]]
            trained_vocab = list(trained["model"]["vocab"])
            for index in reversed(range(len(trained_vocab))):
                if trained_vocab[index][0] in base_tokens:
                    del trained_vocab[index]
            data["model"]["vocab"] = data["model"]["vocab"] + trained_vocab


class TokenizerVocabulary:
    """An experiment's saved tokenizer, reloaded once its vocabulary has been extended on disk."""

    def __init__(self, pretrained: PretrainedTokenizer, exp_dir: Path) -> None:
        self._pretrained = pretrained
        self._exp_dir = exp_dir

    def add(self, tokens: List[str], trained_tokenizers: List[TrainedTokenizer]) -> None:
        self._pretrained.build().save_pretrained(str(self._exp_dir))
        TokenizerVocabularyFile(self._exp_dir / "tokenizer.json").add(
            tokens, [self._saved_model_of(trained) for trained in trained_tokenizers]
        )
        self._pretrained.reload_from_experiment()

    def _saved_model_of(self, trained: TrainedTokenizer) -> dict:
        path = self._exp_dir / "tokenizer_trained.json"
        trained.save(str(path))
        with open(path, "r", encoding="utf-8") as file:
            return json.load(file)


class LanguageCodes:
    """The language codes an experiment's corpora need among the tokenizer's special tokens."""

    def __init__(self, lang_codes: Dict[str, str], inventory: CorpusInventory, exp_dir: Path) -> None:
        self._lang_codes = lang_codes
        self._inventory = inventory
        self._exp_dir = exp_dir

    def add_to(self, tokenizer: PreTrainedTokenizerBase) -> None:
        updated = False
        for iso in self._inventory.source_isos() | self._inventory.target_isos():
            lang_code = self._lang_codes.get(iso, iso)
            if isinstance(tokenizer, T5Tokenizer):
                if lang_code not in tokenizer.all_special_tokens and iso in self._inventory.target_isos():
                    self._add(tokenizer, lang_code)
                    updated = True
            elif isinstance(tokenizer, MBartTokenizer):
                if lang_code not in tokenizer.lang_code_to_id:
                    self._add(tokenizer, lang_code)
                    updated = True
            elif isinstance(tokenizer, NllbTokenizer):
                self._add(tokenizer, lang_code)
                updated = True
            elif lang_code not in tokenizer.lang_code_to_id:
                self._add(tokenizer, lang_code)
                updated = True
        if updated:
            tokenizer.save_pretrained(self._exp_dir)

    def _add(self, tokenizer: PreTrainedTokenizerBase, lang_code: str) -> None:
        tokenizer.add_special_tokens({"extra_special_tokens": [lang_code]}, replace_extra_special_tokens=False)
        lang_id = tokenizer.convert_tokens_to_ids(lang_code)
        if isinstance(tokenizer, (MBart50Tokenizer, MBartTokenizer)):
            tokenizer.id_to_lang_code[lang_id] = lang_code
            tokenizer.fairseq_tokens_to_ids[lang_code] = lang_id
            tokenizer.fairseq_ids_to_tokens[lang_id] = lang_code
        elif isinstance(tokenizer, M2M100Tokenizer):
            tokenizer.lang_code_to_token[lang_code] = lang_code
            tokenizer.lang_token_to_id[lang_code] = lang_id
            tokenizer.id_to_lang_token[lang_id] = lang_code


class VocabularyStatistics:
    """How many tokens each side of the corpus added to the vocabulary."""

    def __init__(self, exp_dir: Path) -> None:
        self._exp_dir = exp_dir

    def write(self, counts_by_side: List[List]) -> None:
        columns = pd.MultiIndex.from_tuples([(" ", "Translation Side"), (" ", "Num Tokens Added to Vocab")])
        report = pd.DataFrame(counts_by_side, columns=columns)
        report.to_csv(self._exp_dir / "tokenization_stats.csv", index=False)
        report.to_excel(self._exp_dir / "tokenization_stats.xlsx")


class TokenizerVocabularyBuilder(VocabularyBuilder):
    """Extends an experiment's pretrained tokenizer with the tokens, language codes and tags its corpora need."""

    def __init__(
        self,
        pretrained: PretrainedTokenizer,
        tokenizer: Tokenizer,
        missing_tokens: MissingTokens,
        inventory: CorpusInventory,
        lang_codes: LanguageCodes,
        exp_dir: Path,
        add_new_lang_code: bool,
    ) -> None:
        self._pretrained = pretrained
        self._tokenizer = tokenizer
        self._missing_tokens = missing_tokens
        self._inventory = inventory
        self._lang_codes = lang_codes
        self._exp_dir = exp_dir
        self._add_new_lang_code = add_new_lang_code

    def build(self, stats: bool = False) -> Tokenizer:
        # Unconditional, so that everything downstream shares this tokenizer rather than loading its own.
        self._pretrained.build()
        extension = self._missing_tokens.find()
        extension.add_to(TokenizerVocabulary(self._pretrained, self._exp_dir))

        if stats:
            VocabularyStatistics(self._exp_dir).write(extension.counts_by_side())

        if self._add_new_lang_code:
            self._lang_codes.add_to(self._pretrained.build())

        tags = self._inventory.tags()
        if len(tags) > 0:
            self._pretrained.build().add_tokens([AddedToken(tag, rstrip=True, special=True) for tag in tags])

        return self._tokenizer
