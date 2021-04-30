import argparse
import logging
import os
import shutil
from abc import ABC, abstractmethod
from enum import Enum, Flag, auto
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union, cast

logging.basicConfig()

import sentencepiece as sp
import tensorflow as tf
import yaml
from opennmt import END_OF_SENTENCE_TOKEN, PADDING_TOKEN, START_OF_SENTENCE_TOKEN
from opennmt.data import Noise, Vocab, WordDropout, WordNoiser, tokens_to_words
from opennmt.models import Model, get_model_from_catalog

from ..common.corpus import load_corpus
from ..common.utils import get_git_revision_hash, get_mt_exp_dir, merge_dict, set_seed
from .runner import SILRunner
from .transformer import SILTransformer
from .utils import encode_sp_lines, get_best_model_dir, get_last_checkpoint

_PYTHON_TO_TENSORFLOW_LOGGING_LEVEL: Dict[int, int] = {
    logging.CRITICAL: 3,
    logging.ERROR: 2,
    logging.WARNING: 1,
    logging.INFO: 0,
    logging.DEBUG: 0,
    logging.NOTSET: 0,
}


_DEFAULT_NEW_CONFIG: dict = {
    "data": {
        "share_vocab": False,
        "character_coverage": 1.0,
        "mirror": False,
        "mixed_src": False,
        "seed": 111,
        "test_size": 1,
        "val_size": 1,
        "disjoint_test": False,
        "disjoint_val": False,
        "score_threshold": 0,
    },
    "train": {"maximum_features_length": 150, "maximum_labels_length": 150},
    "eval": {"multi_ref_eval": False},
    "params": {
        "length_penalty": 0.2,
        "dropout": 0.2,
        "transformer_dropout": 0.1,
        "transformer_attention_dropout": 0.1,
        "transformer_ffn_dropout": 0.1,
        "word_dropout": 0.1,
    },
}


# Different types of parent model checkpoints (last, best, average)
class CheckpointType(Enum):
    LAST = 1
    BEST = 2
    AVERAGE = 3


class DataFileType(Flag):
    NONE = 0
    TRAIN = auto()
    TEST = auto()
    VAL = auto()


def convert_vocab(sp_vocab_path: Path, onmt_vocab_path: Path, tag_langs: Set[str] = None) -> None:
    special_tokens = [PADDING_TOKEN, START_OF_SENTENCE_TOKEN, END_OF_SENTENCE_TOKEN]
#    if tag_langs is not None:
#        special_tokens.extend(map(lambda l: f"<2{l}>", tag_langs))

    vocab = Vocab(special_tokens)
    with open(sp_vocab_path, "r", encoding="utf-8") as vocab_file:
        for line in vocab_file:
            token = line.rstrip("\r\n")
            index = token.rindex("\t")
            token = token[:index]
            if token in ("<unk>", "<s>", "</s>", "<range>"):  # Ignore special tokens
                continue
            vocab.add(token)
    vocab.pad_to_multiple(8)
    vocab.serialize(onmt_vocab_path)


def build_vocab(
    file_paths: Iterable[Path],
    vocab_size: int,
    casing: str,
    character_coverage: float,
    model_prefix: Path,
    vocab_path: Path,
    tag_langs: Set[str] = None,
) -> None:
    joined_file_paths = ",".join(str(fp) for fp in file_paths)
    user_defined_symbols = "<blank>"
    if tag_langs is not None:
        for tag in tag_langs:
            user_defined_symbols += "," + f"<2{tag}>"

    casing = casing.lower()
    normalization: str
    if casing == "lower":
        normalization = "nmt_nfkc_cf"
    elif casing == "preserve":
        normalization = "nmt_nfkc"
    else:
        raise RuntimeError("Invalid casing was specified in the config.")

    # use custom normalization that does not convert ZWJ and ZWNJ to spaces
    # allows properly handling of scripts like Devanagari
    normalization_path = Path(__file__).parent / f"{normalization}.tsv"
    sp_train_params = (
        f"--normalization_rule_tsv={normalization_path} --input={joined_file_paths} --model_prefix={model_prefix}"
        f" --vocab_size={vocab_size} --character_coverage={character_coverage:.4f} --input_sentence_size=1000000"
        f" --shuffle_input_sentence=true --control_symbols=<range> --user_defined_symbols={user_defined_symbols}"
    )

    sp.SentencePieceTrainer.Train(sp_train_params)

    convert_vocab(model_prefix.with_suffix(".vocab"), vocab_path, tag_langs)


def get_checkpoint_path(model_dir: Path, checkpoint_type: CheckpointType) -> Tuple[Optional[Path], Optional[int]]:
    if checkpoint_type == CheckpointType.AVERAGE:
        # Get the checkpoint path and step count for the averaged checkpoint
        return get_last_checkpoint(model_dir / "avg")
    elif checkpoint_type == CheckpointType.BEST:
        # Get the checkpoint path and step count for the best checkpoint
        best_model_dir, step = get_best_model_dir(model_dir)
        return (best_model_dir / "ckpt", step)
    elif checkpoint_type == CheckpointType.LAST:
        return (None, None)
    else:
        raise RuntimeError(f"Unsupported checkpoint type: {checkpoint_type}")


class Config(ABC):
    def __init__(
        self,
        exp_dir: Path,
        config: dict,
        src_isos: Set[str],
        trg_isos: Set[str],
        src_file_paths: Set[Path],
        trg_file_paths: Set[Path],
        src_tags: Set[str],
    ) -> None:
        config = merge_dict(
            {
                "model": "SILTransformerBase",
                "model_dir": str(exp_dir / "run"),
                "data": {
                    "train_features_file": str(exp_dir / "train.src.txt"),
                    "train_labels_file": str(exp_dir / "train.trg.txt"),
                    "eval_features_file": str(exp_dir / "val.src.txt"),
                    "eval_labels_file": str(exp_dir / "val.trg.txt"),
                    "share_vocab": True,
                    "character_coverage": 1.0,
                    "mirror": False,
                    "parent_use_best": False,
                    "parent_use_average": False,
                    "parent_use_vocab": False,
                    "seed": 111,
                    "tokenize": True,
                },
                "train": {
                    "average_last_checkpoints": 0,
                    "maximum_features_length": 150,
                    "maximum_labels_length": 150,
                    "keep_checkpoint_max": 3,
                    "save_checkpoints_steps": 1000,
                },
                "eval": {
                    "external_evaluators": "bleu_multi_ref",
                    "steps": 1000,
                    "early_stopping": {"metric": "bleu", "min_improvement": 0.2, "steps": 4},
                    "export_on_best": "bleu",
                    "export_format": "checkpoint",
                    "max_exports_to_keep": 1,
                    "multi_ref_eval": False,
                },
                "params": {
                    "length_penalty": 0.2,
                    "transformer_dropout": 0.1,
                    "transformer_attention_dropout": 0.1,
                    "transformer_ffn_dropout": 0.1,
                    "word_dropout": 0,
                },
            },
            config,
        )
        data_config: dict = config["data"]
        eval_config: dict = config["eval"]
        multi_ref_eval: bool = eval_config["multi_ref_eval"]
        if multi_ref_eval:
            data_config["eval_labels_file"] = str(exp_dir / "val.trg.txt.0")
        if data_config["share_vocab"]:
            data_config["source_vocabulary"] = str(exp_dir / "onmt.vocab")
            data_config["target_vocabulary"] = str(exp_dir / "onmt.vocab")
            if (
                "src_vocab_size" not in data_config
                and "trg_vocab_size" not in data_config
                and "vocab_size" not in data_config
            ):
                data_config["vocab_size"] = 24000
            if "src_casing" not in data_config and "trg_casing" not in data_config and "casing" not in data_config:
                data_config["casing"] = "lower"
        else:
            data_config["source_vocabulary"] = str(exp_dir / "src-onmt.vocab")
            data_config["target_vocabulary"] = str(exp_dir / "trg-onmt.vocab")
            if "vocab_size" not in data_config:
                if "src_vocab_size" not in data_config:
                    data_config["src_vocab_size"] = 8000
                if "trg_vocab_size" not in data_config:
                    data_config["trg_vocab_size"] = 8000
            if "casing" not in data_config:
                if "src_casing" not in data_config:
                    data_config["src_casing"] = "lower"
                if "trg_casing" not in data_config:
                    data_config["trg_casing"] = "lower"

        self.exp_dir = exp_dir
        self.root = config
        self.src_isos = src_isos
        self.trg_isos = trg_isos
        self.src_file_paths = src_file_paths
        self.trg_file_paths = trg_file_paths
        self.src_tags = src_tags

        parent: Optional[str] = self.data.get("parent")
        self.parent_config: Optional[Config] = None
        if parent is not None:
            self.parent_config = load_config(parent)
            freeze_layers: Optional[List[str]] = self.parent_config.params.get("freeze_layers")
            # do not freeze any word embeddings layer, because we will update them when we create the parent model
            if freeze_layers is not None:
                self.parent_config.params["freeze_layers"] = list()

        self.write_trg_tag: bool = (
            len(self.trg_isos) > 1
            or self.mirror
            or (self.parent_config is not None and self.parent_config.write_trg_tag)
        )

    @property
    def default_src_iso(self) -> str:
        return next(iter(self.src_isos))

    @property
    def default_trg_iso(self) -> str:
        return next(iter(self.trg_isos))

    @property
    def model(self) -> str:
        return self.root["model"]

    @property
    def model_dir(self) -> Path:
        return Path(self.root["model_dir"])

    @property
    def params(self) -> dict:
        return self.root["params"]

    @property
    def data(self) -> dict:
        return self.root["data"]

    @property
    def mirror(self) -> bool:
        return self.data["mirror"]

    @property
    def share_vocab(self) -> bool:
        return self.data["share_vocab"]

    @property
    def has_parent(self) -> bool:
        return "parent" in self.data

    def set_seed(self) -> None:
        seed = self.data["seed"]
        set_seed(seed)
        tf.random.set_seed(seed)

    def preprocess(self, stats: bool) -> None:
        self._build_vocabs()
        src_spp, trg_spp = self.create_sp_processors()
        self._build_corpora(src_spp, trg_spp, stats)

    def create_sp_processors(self) -> Tuple[Optional[sp.SentencePieceProcessor], Optional[sp.SentencePieceProcessor]]:
        if not self.data["tokenize"]:
            return (None, None)
        if self.share_vocab:
            model_prefix = self.exp_dir / "sp"
            src_spp = sp.SentencePieceProcessor()
            src_spp.Load(str(model_prefix.with_suffix(".model")))

            trg_spp = src_spp
        else:
            src_spp = sp.SentencePieceProcessor()
            src_spp.Load(str(self.exp_dir / "src-sp.model"))

            trg_spp = sp.SentencePieceProcessor()
            trg_spp.Load(str(self.exp_dir / "trg-sp.model"))
        return (src_spp, trg_spp)

    def create_src_sp_processor(self) -> Optional[sp.SentencePieceProcessor]:
        if not self.data["tokenize"]:
            return None
        src_spp = sp.SentencePieceProcessor()
        src_spp.Load(str(self.exp_dir / "sp.model" if self.share_vocab else self.exp_dir / "src-sp.model"))
        return src_spp

    @abstractmethod
    def _build_corpora(
        self, src_spp: Optional[sp.SentencePieceProcessor], trg_spp: Optional[sp.SentencePieceProcessor], stats: bool
    ) -> None:
        pass

    def _build_vocabs(self) -> None:
        if not self.data["tokenize"]:
            return

        tag_isos: Optional[Set[str]] = set()
        if self.write_trg_tag:
            tag_isos = self.trg_isos | self.src_isos if self.mirror else set(self.trg_isos)
        if self.src_tags is not None:
            for tag in self.src_tags:
                tag_isos.add(tag)

        if self.share_vocab:
            print("Building shared vocabulary...")
            vocab_size: Optional[int] = self.data.get("vocab_size")
            if vocab_size is None:
                vocab_size = self.data.get("src_vocab_size")
                if vocab_size is None:
                    vocab_size = self.data["trg_vocab_size"]
                elif self.data.get("trg_vocab_size", vocab_size) != vocab_size:
                    raise RuntimeError(
                        "The source and target vocab sizes cannot be different when creating a shared vocab."
                    )

            casing: Optional[str] = self.data.get("casing")
            if casing is None:
                casing = self.data.get("src_casing")
                if casing is None:
                    casing = self.data["trg_casing"]
                elif self.data.get("trg_casing", casing) != casing:
                    raise RuntimeError("The source and target casing cannot be different when creating a shared vocab.")

            model_prefix = self.exp_dir / "sp"
            vocab_path = self.exp_dir / "onmt.vocab"
            share_vocab_file_paths: Set[Path] = self.src_file_paths | self.trg_file_paths
            character_coverage = self.data.get("character_coverage", 1.0)
            build_vocab(
                share_vocab_file_paths, vocab_size, casing, character_coverage, model_prefix, vocab_path, tag_isos
            )

            self._update_vocab(vocab_path, vocab_path)
        else:
            src_vocab_file_paths: Set[Path] = set(self.src_file_paths)
            if self.mirror:
                src_vocab_file_paths.update(self.trg_file_paths)
            self._create_unshared_vocab(
                self.src_isos,
                src_vocab_file_paths,
                "source",
                tag_langs=tag_isos,
            )

            trg_vocab_file_paths: Set[Path] = set(self.trg_file_paths)
            if self.mirror:
                trg_vocab_file_paths.update(self.src_file_paths)
            self._create_unshared_vocab(
                self.trg_isos,
                trg_vocab_file_paths,
                "target",
            )

            self._update_vocab(self.exp_dir / "src-onmt.vocab", self.exp_dir / "trg-onmt.vocab")

    def _update_vocab(self, src_vocab_path: Path, trg_vocab_path: Path) -> None:
        if self.parent_config is None:
            return

        model_dir = self.parent_config.model_dir
        parent_model_to_use = (
            CheckpointType.BEST
            if self.data["parent_use_best"]
            else CheckpointType.AVERAGE
            if self.data["parent_use_average"]
            else CheckpointType.LAST
        )
        checkpoint_path, step = get_checkpoint_path(model_dir, parent_model_to_use)
        parent_runner = create_runner(self.parent_config)
        parent_runner.update_vocab(
            str(self.exp_dir / "parent"),
            str(src_vocab_path),
            str(trg_vocab_path),
            None if checkpoint_path is None else str(checkpoint_path),
            step,
        )

    def _create_unshared_vocab(
        self,
        isos: Set[str],
        vocab_file_paths: Set[Path],
        side: str,
        tag_langs: Set[str] = None,
    ) -> None:
        prefix = "src" if side == "source" else "trg"
        model_prefix = self.exp_dir / f"{prefix}-sp"
        vocab_path = self.exp_dir / f"{prefix}-onmt.vocab"
        if self.parent_config is not None:
            parent_isos = self.parent_config.src_isos if side == "source" else self.parent_config.trg_isos
            if isos == parent_isos:
                if self.parent_config.share_vocab:
                    parent_sp_prefix_path = self.parent_config.exp_dir / "sp"
                    parent_vocab_path = self.parent_config.exp_dir / "onmt.vocab"
                else:
                    parent_sp_prefix_path = self.parent_config.exp_dir / f"{prefix}-sp"
                    parent_vocab_path = self.parent_config.exp_dir / f"{prefix}-onmt.vocab"

                parent_vocab: Optional[Vocab] = None
                child_tokens: Optional[Set[str]] = None
                parent_use_vocab: bool = self.data["parent_use_vocab"]
                if not parent_use_vocab:
                    parent_spp = sp.SentencePieceProcessor()
                    parent_spp.Load(str(parent_sp_prefix_path.with_suffix(".model")))

                    parent_vocab = Vocab()
                    parent_vocab.load(str(parent_vocab_path))

                    child_tokens = set()
                    for vocab_file_path in vocab_file_paths:
                        for line in encode_sp_lines(parent_spp, load_corpus(vocab_file_path)):
                            child_tokens.update(line.split())
                    parent_use_vocab = child_tokens.issubset(parent_vocab.words)

                # all tokens in the child corpora are in the parent vocab, so we can just use the parent vocab
                # or, the user wants to reuse the parent vocab for this child experiment
                if parent_use_vocab:
                    sp_vocab_path = self.exp_dir / f"{prefix}-sp.vocab"
                    onmt_vocab_path = self.exp_dir / f"{prefix}-onmt.vocab"
                    shutil.copy2(parent_sp_prefix_path.with_suffix(".model"), self.exp_dir / f"{prefix}-sp.model")
                    shutil.copy2(parent_sp_prefix_path.with_suffix(".vocab"), sp_vocab_path)
                    convert_vocab(sp_vocab_path, onmt_vocab_path, tag_langs)
                    return
                elif child_tokens is not None and parent_vocab is not None:
                    onmt_delta_vocab_path = self.exp_dir / f"{prefix}-onmt-delta.vocab"
                    vocab_delta = child_tokens.difference(parent_vocab.words)
                    with open(onmt_delta_vocab_path, "w", encoding="utf-8") as f:
                        [f.write(f"{token}\n") for token in vocab_delta]

        print(f"Building {side} vocabulary...")
        vocab_size: int = self.data.get(f"{prefix}_vocab_size", self.data.get("vocab_size"))
        casing: str = self.data.get(f"{prefix}_casing", self.data.get("casing"))
        character_coverage: float = self.data.get(f"{prefix}_character_coverage", self.data.get("character_coverage"))
        build_vocab(vocab_file_paths, vocab_size, casing, character_coverage, model_prefix, vocab_path, tag_langs)


def set_transformer_dropout(
    root_layer: SILTransformer,
    dropout: float = 0.1,
    attention_dropout: float = 0.1,
    ffn_dropout: float = 0.1,
):
    for layer in (root_layer,) + root_layer.submodules:
        name: str = layer.name
        if name == "self_attention_encoder":
            layer.dropout = dropout
        elif name == "self_attention_decoder":
            layer.dropout = dropout
        elif name.startswith("transformer_layer_wrapper"):
            layer.output_dropout = dropout
        elif name.startswith("multi_head_attention"):
            layer.dropout = attention_dropout
        elif name.startswith("feed_forward_network"):
            layer.dropout = ffn_dropout


class SILWordNoiser(WordNoiser):
    def __init__(
        self,
        noises: Optional[List[Noise]] = None,
        subword_token: str = "￭",
        is_spacer: Optional[bool] = None,
        has_lang_tag: bool = False,
    ) -> None:
        super().__init__(noises=noises, subword_token=subword_token, is_spacer=is_spacer)
        self.has_lang_tag = has_lang_tag

    def _call(
        self, tokens: tf.Tensor, sequence_length: Optional[Union[int, tf.Tensor]], keep_shape: bool
    ) -> Tuple[tf.Tensor, Union[int, tf.Tensor]]:
        rank = tokens.shape.rank
        if rank == 1:
            input_length = tf.shape(tokens)[0]
            if sequence_length is not None:
                tokens = tokens[:sequence_length]
            else:
                tokens = tokens[: tf.math.count_nonzero(tokens)]
            words = tokens_to_words(tokens, subword_token=self.subword_token, is_spacer=self.is_spacer)
            words = cast(tf.Tensor, words.to_tensor())
            if self.has_lang_tag:
                tag = words[:1]
                words = words[1:]
            for noise in self.noises:
                words = noise(words)
            if self.has_lang_tag:
                words = tf.concat([tag, words], axis=0)
            outputs = tf.RaggedTensor.from_tensor(words, padding="").flat_values
            output_length = tf.shape(outputs)[0]
            if keep_shape:
                outputs = tf.pad(outputs, [[0, input_length - output_length]])
            return outputs, output_length
        else:
            return super()._call(tokens, sequence_length=sequence_length, keep_shape=keep_shape)


def create_model(config: Config) -> Model:
    model_name = config.model
    if model_name.startswith("Transformer"):
        model_name = "SIL" + model_name
    model = get_model_from_catalog(model_name)
    if isinstance(model, SILTransformer):
        dropout = config.params["transformer_dropout"]
        attention_dropout = config.params["transformer_attention_dropout"]
        ffn_dropout = config.params["transformer_ffn_dropout"]
        if dropout != 0.1 or attention_dropout != 0.1 or ffn_dropout != 0.1:
            set_transformer_dropout(
                model, dropout=dropout, attention_dropout=attention_dropout, ffn_dropout=ffn_dropout
            )

    word_dropout: float = config.params["word_dropout"]
    if word_dropout > 0:
        source_noiser = SILWordNoiser(subword_token="▁", has_lang_tag=config.write_trg_tag)
        source_noiser.add(WordDropout(word_dropout))
        model.features_inputter.set_noise(source_noiser, probability=1.0)

        target_noiser = SILWordNoiser(subword_token="▁")
        target_noiser.add(WordDropout(word_dropout))
        model.labels_inputter.set_noise(target_noiser, probability=1.0)
    return model


def set_log_level(log_level: int) -> None:
    tf.get_logger().setLevel(log_level)
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = str(_PYTHON_TO_TENSORFLOW_LOGGING_LEVEL[log_level])


def create_runner(
    config: Config, mixed_precision: bool = False, log_level: int = logging.INFO, memory_growth: bool = False
) -> SILRunner:
    set_log_level(log_level)

    if memory_growth:
        gpus = tf.config.list_physical_devices(device_type="GPU")
        for device in gpus:
            tf.config.experimental.set_memory_growth(device, enable=True)

    model = create_model(config)

    return SILRunner(model, config.root, auto_config=True, mixed_precision=mixed_precision)


def load_config(exp_name: str) -> Config:
    exp_dir = get_mt_exp_dir(exp_name)
    config_path = exp_dir / "config.yml"

    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    from .corpus_pairs_config import CorpusPairsConfig
    from .langs_config import LangsConfig

    config_class: Any = CorpusPairsConfig if "corpus_pairs" in config["data"] else LangsConfig
    return config_class(exp_dir, config)


def copy_config_value(src: dict, trg: dict, key: str) -> None:
    if key in src:
        trg[key] = src[key]


def main() -> None:
    parser = argparse.ArgumentParser(description="Creates a NMT experiment config file")
    parser.add_argument("experiment", help="Experiment name")
    parser.add_argument("--src-langs", nargs="*", metavar="lang", default=[], help="Source language")
    parser.add_argument("--trg-langs", nargs="*", metavar="lang", default=[], help="Target language")
    parser.add_argument("--vocab-size", type=int, help="Shared vocabulary size")
    parser.add_argument("--src-vocab-size", type=int, help="Source vocabulary size")
    parser.add_argument("--trg-vocab-size", type=int, help="Target vocabulary size")
    parser.add_argument("--parent", type=str, help="Parent experiment name")
    parser.add_argument("--mirror", default=False, action="store_true", help="Mirror train and validation data sets")
    parser.add_argument("--force", default=False, action="store_true", help="Overwrite existing config file")
    parser.add_argument("--seed", type=int, help="Randomization seed")
    parser.add_argument("--model", type=str, help="The neural network model")
    args = parser.parse_args()

    print("Git commit:", get_git_revision_hash())

    exp_dir = get_mt_exp_dir(args.experiment)
    config_path = exp_dir / "config.yml"
    if config_path.is_file() and not args.force:
        print('The experiment config file already exists. Use "--force" if you want to overwrite the existing config.')
        return

    exp_dir.mkdir(exist_ok=True, parents=True)

    config = _DEFAULT_NEW_CONFIG.copy()
    if args.model is not None:
        config["model"] = args.model
    data_config: dict = config["data"]
    data_config["src_langs"] = args.src_langs
    data_config["trg_langs"] = args.trg_langs
    if args.parent is not None:
        data_config["parent"] = args.parent
        parent_config = load_config(args.parent)
        for key in [
            "share_vocab",
            "vocab_size",
            "src_vocab_size",
            "trg_vocab_size",
            "casing",
            "src_casing",
            "trg_casing",
        ]:
            if key in parent_config.data:
                data_config[key] = parent_config.data[key]
    if args.vocab_size is not None:
        data_config["vocab_size"] = args.vocab_size
    elif args.src_vocab_size is not None or args.trg_vocab_size is not None:
        data_config["share_vocab"] = False
        if args.src_vocab_size is not None:
            data_config["src_vocab_size"] = args.src_vocab_size
        elif "vocab_size" in data_config:
            data_config["src_vocab_size"] = data_config["vocab_size"]
            del data_config["vocab_size"]
        if args.trg_vocab_size is not None:
            data_config["trg_vocab_size"] = args.trg_vocab_size
        elif "vocab_size" in data_config:
            data_config["trg_vocab_size"] = data_config["vocab_size"]
            del data_config["vocab_size"]
    if data_config["share_vocab"]:
        if "vocab_size" not in data_config:
            data_config["vocab_size"] = 24000
        if "casing" not in data_config:
            data_config["casing"] = "lower"
    else:
        if "src_vocab_size" not in data_config:
            data_config["src_vocab_size"] = 8000
        if "trg_vocab_size" not in data_config:
            data_config["trg_vocab_size"] = 8000
        if "src_casing" not in data_config:
            data_config["src_casing"] = data_config.get("casing", "lower")
        if "trg_casing" not in data_config:
            data_config["trg_casing"] = data_config.get("casing", "lower")
        if "casing" in data_config:
            del data_config["casing"]
    if args.seed is not None:
        data_config["seed"] = args.seed
    if args.mirror:
        data_config["mirror"] = True
    with open(config_path, "w", encoding="utf-8") as file:
        yaml.dump(config, file)
    print("Config file created")


if __name__ == "__main__":
    main()
