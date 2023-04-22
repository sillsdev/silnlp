import logging
import os
import shutil
import tempfile
from contextlib import ExitStack
from pathlib import Path
from typing import IO, Iterable, List, Optional, Sequence, Set, Tuple, Union, cast

import numpy as np
import sacrebleu
import sentencepiece as sp
import tensorflow as tf
import yaml
from machine.scripture import VerseRef, get_books
from opennmt import END_OF_SENTENCE_TOKEN, PADDING_TOKEN, START_OF_SENTENCE_TOKEN
from opennmt.data import Vocab
from opennmt.models.catalog import list_model_names_from_catalog
from opennmt.utils import Scorer, register_scorer

from ..alignment.config import get_aligner, get_aligner_name
from ..alignment.machine_aligner import MachineAligner
from ..common.corpus import load_corpus, split_corpus
from ..common.environment import SIL_NLP_ENV, download_if_s3_paths
from ..common.tf_utils import set_tf_log_level
from ..common.utils import Side, get_mt_exp_dir, merge_dict
from .config import CheckpointType, Config, CorpusPair, NMTModel
from .opennmt.runner import create_runner
from .sp_utils import decode_sp, decode_sp_lines
from .tokenizer import NullTokenizer, Tokenizer

LOGGER = logging.getLogger(__package__ + ".open_nmt_config")


def is_open_nmt_model(model_name: str) -> bool:
    if model_name.startswith("Transformer"):
        model_name = "SIL" + model_name
    open_nmt_model_names = list_model_names_from_catalog()
    return model_name in open_nmt_model_names


def convert_vocab(sp_vocab_path: Path, onmt_vocab_path: Path, tags: Set[str]) -> None:
    special_tokens = [START_OF_SENTENCE_TOKEN, END_OF_SENTENCE_TOKEN, PADDING_TOKEN] + list(tags)

    vocab = Vocab(special_tokens)
    with sp_vocab_path.open("r", encoding="utf-8") as vocab_file:
        for line in vocab_file:
            token = line.rstrip("\r\n")
            index = token.rindex("\t")
            token = token[:index]
            if token in ("<unk>", "<s>", "</s>", "<blank>"):  # Ignore special tokens
                continue
            vocab.add(token)
    vocab.pad_to_multiple(8)
    vocab.serialize(onmt_vocab_path)


def build_vocab(
    file_paths: Iterable[Path],
    vocab_size: int,
    vocab_type: str,
    vocab_seed: int,
    casing: str,
    vocab_split_by_unicode_script: bool,
    character_coverage: float,
    model_prefix: Path,
    vocab_path: Path,
    tags: Set[str],
    max_train_size: int,
) -> None:
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
    file_paths = [fp for fp in file_paths]
    file_paths = download_if_s3_paths(file_paths)
    file_paths.sort()

    tags_str = ",".join(tags)
    if len(tags_str) > 0:
        tags_str = "," + tags_str

    if vocab_seed is not None:
        sp.set_random_generator_seed(vocab_seed)
    sp.SentencePieceTrainer.Train(
        normalization_rule_tsv=normalization_path,
        input=file_paths,
        model_prefix=model_prefix,
        model_type=vocab_type,
        vocab_size=vocab_size,
        user_defined_symbols="<blank>" + tags_str,
        character_coverage="%.4f" % character_coverage,
        input_sentence_size=max_train_size,
        shuffle_input_sentence=True,
        split_by_unicode_script=vocab_split_by_unicode_script,
    )

    convert_vocab(model_prefix.with_suffix(".vocab"), vocab_path, tags)


def _get_checkpoint_path(model_dir: Path, checkpoint: Union[CheckpointType, str, int]) -> Tuple[Path, int]:
    step: Optional[int] = None
    if isinstance(checkpoint, str):
        checkpoint = checkpoint.lower()
        if "avg" in checkpoint:
            checkpoint = CheckpointType.AVERAGE
        elif "best" in checkpoint:
            checkpoint = CheckpointType.BEST
        elif "last" in checkpoint:
            checkpoint = CheckpointType.LAST
        else:
            step = int(checkpoint)
            checkpoint = CheckpointType.OTHER
    if checkpoint is CheckpointType.AVERAGE:
        # Get the checkpoint path and step count for the averaged checkpoint
        ckpt, _ = get_last_checkpoint(model_dir / "avg")
        step = -1
    elif checkpoint is CheckpointType.BEST:
        # Get the checkpoint path and step count for the best checkpoint
        best_model_dir, step = get_best_model_dir(model_dir)
        ckpt, step = (best_model_dir / "ckpt", step)
    elif checkpoint is CheckpointType.LAST:
        ckpt, step = get_last_checkpoint(model_dir)
    elif checkpoint is CheckpointType.OTHER and step is not None:
        ckpt = model_dir / f"ckpt-{step}"
    else:
        raise RuntimeError(f"Unsupported checkpoint type: {checkpoint}")
    return ckpt, step


def get_best_model_dir(model_dir: Path) -> Tuple[Path, int]:
    export_path = model_dir / "export"
    models = list(d.name for d in export_path.iterdir())
    best_model_dir: Optional[Path] = None
    step = 0
    for model in sorted(models, key=lambda m: int(m), reverse=True):
        path = export_path / model
        if path.is_dir():
            best_model_dir = path
            step = int(model)
            break
    if best_model_dir is None:
        raise RuntimeError("There are no exported models.")
    return best_model_dir, step


def get_last_checkpoint(model_dir: Path) -> Tuple[Path, int]:
    with (model_dir / "checkpoint").open("r", encoding="utf-8") as file:
        checkpoint_config = yaml.safe_load(file)
        checkpoint_prefix = Path(checkpoint_config["model_checkpoint_path"])
        parts = checkpoint_prefix.name.split("-")
        checkpoint_path = model_dir / checkpoint_prefix
        step = int(parts[-1])
        return checkpoint_path, step


@register_scorer(name="bleu_sp")
class BLEUSentencepieceScorer(Scorer):
    def __init__(self):
        super().__init__("bleu")

    def __call__(self, ref_path: str, hyp_path: str) -> float:
        with tf.io.gfile.GFile(ref_path) as ref_stream, tf.io.gfile.GFile(hyp_path) as sys_stream:
            sys = decode_sp_lines(sys_stream)
            ref = decode_sp_lines(ref_stream)
            bleu = sacrebleu.corpus_bleu(cast(Sequence[str], sys), cast(Sequence[Sequence[str]], [ref]), lowercase=True)
            return bleu.score


def load_ref_streams(ref_path: str, detok: bool = False) -> List[List[str]]:
    ref_streams: List[IO] = []
    try:
        if ref_path.endswith(".0"):
            prefix = ref_path[:-2]
            i = 0
            while os.path.isfile(f"{prefix}.{i}"):
                ref_streams.append(tf.io.gfile.GFile(f"{prefix}.{i}"))
                i += 1
        else:
            ref_streams.append(tf.io.gfile.GFile(ref_path))
        refs: List[List[str]] = []
        for lines in zip(*ref_streams):
            for ref_index in range(len(ref_streams)):
                if not detok:
                    ref_line = lines[ref_index].strip()
                else:
                    ref_line = decode_sp(lines[ref_index].strip())
                if len(refs) == ref_index:
                    refs.append([])
                refs[ref_index].append(ref_line)
        return refs
    finally:
        for ref_stream in ref_streams:
            ref_stream.close()


def load_sys_stream(hyp_path: str, detok: bool = False) -> List[str]:
    sys_stream = []
    with tf.io.gfile.GFile(hyp_path) as f:
        for line in f:
            if not detok:
                sys_stream.append(line.rstrip())
            else:
                sys_stream.append(decode_sp(line.rstrip()))
    return sys_stream


@register_scorer(name="bleu_multi_ref")
class BLEUMultiRefScorer(Scorer):
    def __init__(self):
        super().__init__("bleu")

    def __call__(self, ref_path: str, hyp_path: str) -> float:
        ref_streams = load_ref_streams(ref_path)
        sys_stream = load_sys_stream(hyp_path)
        bleu = sacrebleu.corpus_bleu(sys_stream, cast(Sequence[Sequence[str]], ref_streams), force=True)
        return bleu.score


@register_scorer(name="bleu_multi_ref_detok")
class BLEUMultiRefDetokScorer(Scorer):
    def __init__(self):
        super().__init__("bleu")

    def __call__(self, ref_path: str, hyp_path: str) -> float:
        ref_streams = load_ref_streams(ref_path, detok=True)
        sys_stream = load_sys_stream(hyp_path, detok=True)
        bleu = sacrebleu.corpus_bleu(sys_stream, cast(Sequence[Sequence[str]], ref_streams), force=True)
        return bleu.score


@register_scorer(name="chrf3")
class chrF3Scorer(Scorer):
    def __init__(self):
        super().__init__("chrf3")

    def __call__(self, ref_path: str, hyp_path: str) -> float:
        ref_streams = load_ref_streams(ref_path)
        sys_stream = load_sys_stream(hyp_path)
        chrf3_score = sacrebleu.corpus_chrf(sys_stream, ref_streams, char_order=6, beta=3, remove_whitespace=True)
        return np.round(float(chrf3_score.score), 2)


@register_scorer(name="chrf3_detok")
class chrF3DetokScorer(Scorer):
    def __init__(self):
        super().__init__("chrf3_detok")

    def __call__(self, ref_path: str, hyp_path: str) -> float:
        ref_streams = load_ref_streams(ref_path, detok=True)
        sys_stream = load_sys_stream(hyp_path, detok=True)
        chrf3_score = sacrebleu.corpus_chrf(sys_stream, ref_streams, char_order=6, beta=3, remove_whitespace=True)
        return np.round(float(chrf3_score.score), 2)


class OpenNMTConfig(Config):
    def __init__(self, exp_dir: Path, config: dict) -> None:
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
                    "sp_max_train_size": 1000000,
                    "mirror": False,
                    "parent_use_best": False,
                    "parent_use_average": False,
                    "parent_use_vocab": False,
                    "seed": 111,
                    "tokenize": True,
                    "aligner": "fast_align",
                    "guided_alignment": False,
                    "guided_alignment_train_size": 1000000,
                    "stats_max_size": 100000,  # a little over the size of the bible
                    "terms": {
                        "train": True,
                        "dictionary": False,
                        "categories": "PN",
                        "include_glosses": True,
                    },
                    "transfer_alignment_heads": True,
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
                    "use_dictionary": True,
                },
                "params": {
                    "length_penalty": 0.2,
                    "transformer_dropout": 0.1,
                    "transformer_attention_dropout": 0.1,
                    "transformer_ffn_dropout": 0.1,
                    "word_dropout": 0,
                    "guided_alignment_type": "mse",
                    "guided_alignment_weight": 0.3,
                },
                "infer": {},
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
            if (
                "src_vocab_type" not in data_config
                and "trg_vocab_type" not in data_config
                and "vocab_type" not in data_config
            ):
                data_config["vocab_type"] = "unigram"
            if "src_casing" not in data_config and "trg_casing" not in data_config and "casing" not in data_config:
                data_config["casing"] = "lower"
            if (
                "src_vocab_split_by_unicode_script" not in data_config
                and "trg_vocab_split_by_unicode_script" not in data_config
                and "vocab_split_by_unicode_script" not in data_config
            ):
                data_config["vocab_split_by_unicode_script"] = True
        else:
            data_config["source_vocabulary"] = str(exp_dir / "src-onmt.vocab")
            data_config["target_vocabulary"] = str(exp_dir / "trg-onmt.vocab")
            if "vocab_size" not in data_config:
                if "src_vocab_size" not in data_config:
                    data_config["src_vocab_size"] = 8000
                if "trg_vocab_size" not in data_config:
                    data_config["trg_vocab_size"] = 8000
            if "vocab_type" not in data_config:
                if "src_vocab_type" not in data_config:
                    data_config["src_vocab_type"] = "unigram"
                if "trg_vocab_type" not in data_config:
                    data_config["trg_vocab_type"] = "unigram"
            if "casing" not in data_config:
                if "src_casing" not in data_config:
                    data_config["src_casing"] = "lower"
                if "trg_casing" not in data_config:
                    data_config["trg_casing"] = "lower"
            if "vocab_split_by_unicode_script" not in data_config:
                if "src_vocab_split_by_unicode_script" not in data_config:
                    data_config["src_vocab_split_by_unicode_script"] = True
                if "trg_vocab_split_by_unicode_script" not in data_config:
                    data_config["trg_vocab_split_by_unicode_script"] = True

        model: str = config["model"]
        if model.endswith("AlignmentEnhanced"):
            data_config["guided_alignment"] = True

        if data_config["guided_alignment"]:
            if config["params"]["word_dropout"] > 0:
                raise RuntimeError("Guided alignment will not work with word dropout enabled.")
            data_config["train_alignments"] = str(exp_dir / "train.alignments.txt")

        super().__init__(exp_dir, config)

        if any(
            p.is_dictionary or (len(p.src_terms_files) > 0 and data_config["terms"]["dictionary"])
            for p in self.corpus_pairs
        ):
            data_config["source_dictionary"] = str(exp_dir / self.dict_src_filename())
            data_config["target_dictionary"] = str(exp_dir / self.dict_trg_filename())
            data_config["ref_dictionary"] = str(exp_dir / self.dict_vref_filename())

        if self.has_scripture_data:
            data_config["eval_features_file"] = [
                str(exp_dir / self.val_src_filename()),
                str(exp_dir / self.val_vref_filename()),
            ]

        parent: Optional[str] = self.data.get("parent")
        self.parent_config: Optional[Config] = None
        if parent is not None:
            SIL_NLP_ENV.copy_experiment_from_bucket(parent, patterns=("config.yml", "*.model", "*.vocab"))
            parent_exp_dir = get_mt_exp_dir(parent)
            parent_config_path = parent_exp_dir / "config.yml"
            with parent_config_path.open("r", encoding="utf-8") as file:
                parent_config: dict = yaml.safe_load(file)
            self.parent_config = OpenNMTConfig(parent_exp_dir, parent_config)
            freeze_layers: Optional[List[str]] = self.parent_config.params.get("freeze_layers")
            # do not freeze any word embeddings layer, because we will update them when we create the parent model
            if freeze_layers is not None:
                self.parent_config.params["freeze_layers"] = list()

        self.write_trg_tag: bool = (
            len(self.trg_isos) > 1
            or self.mirror
            or (self.parent_config is not None and self.parent_config.write_trg_tag)
        )

        if self.write_trg_tag:
            self._tags.update(f"<2{trg_iso}>" for trg_iso in self.trg_isos)
            if self.mirror:
                self._tags.update(f"<2{src_iso}>" for src_iso in self.src_isos)

    @property
    def model_dir(self) -> Path:
        return Path(self.root["model_dir"])

    def create_model(self, mixed_precision: bool = False, num_devices: int = 1) -> NMTModel:
        return OpenNMTModel(self, mixed_precision, num_devices)

    def create_tokenizer(self) -> Tokenizer:
        if not self.data["tokenize"]:
            return NullTokenizer()

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

        return OpenNMTTokenizer(src_spp, trg_spp, self.write_trg_tag)

    def _build_vocabs(self) -> None:
        if not self.data["tokenize"]:
            return

        if self.share_vocab:
            LOGGER.info("Building shared vocabulary...")
            vocab_size: Optional[int] = self.data.get("vocab_size")
            if vocab_size is None:
                vocab_size = self.data.get("src_vocab_size")
                if vocab_size is None:
                    vocab_size = self.data["trg_vocab_size"]
                elif self.data.get("trg_vocab_size", vocab_size) != vocab_size:
                    raise RuntimeError(
                        "The source and target vocab sizes cannot be different when creating a shared vocab."
                    )
            assert vocab_size is not None

            vocab_type: Optional[str] = self.data.get("vocab_type")
            if vocab_type is None:
                vocab_type = self.data.get("src_vocab_type")
                if vocab_type is None:
                    vocab_type = self.data["trg_vocab_type"]
                elif self.data.get("trg_vocab_type", vocab_type) != vocab_type:
                    raise RuntimeError(
                        "The source and target vocab types cannot be different when creating a shared vocab."
                    )
            assert vocab_type is not None

            vocab_seed: Optional[int] = self.data.get("vocab_seed")
            if vocab_seed is None:
                vocab_seed = self.data.get("src_vocab_seed")
                if vocab_seed is None:
                    vocab_seed = self.data["trg_vocab_seed"]

            casing: Optional[str] = self.data.get("casing")
            if casing is None:
                casing = self.data.get("src_casing")
                if casing is None:
                    casing = self.data["trg_casing"]
                elif self.data.get("trg_casing", casing) != casing:
                    raise RuntimeError("The source and target casing cannot be different when creating a shared vocab.")
            assert casing is not None

            vocab_split_by_unicode_script: Optional[bool] = self.data.get("vocab_split_by_unicode_script")
            if vocab_split_by_unicode_script is None:
                vocab_split_by_unicode_script = self.data.get("src_vocab_split_by_unicode_script")
                if vocab_split_by_unicode_script is None:
                    vocab_split_by_unicode_script = self.data["trg_vocab_split_by_unicode_script"]
                elif (
                    self.data.get("trg_vocab_split_by_unicode_script", vocab_split_by_unicode_script)
                    != vocab_split_by_unicode_script
                ):
                    raise RuntimeError(
                        "The source and target cannot split tokens differently when creating a shared vocab."
                    )
            assert vocab_split_by_unicode_script is not None

            model_prefix = self.exp_dir / "sp"
            vocab_path = self.exp_dir / "onmt.vocab"
            share_vocab_file_paths: Set[Path] = self.src_file_paths | self.trg_file_paths
            character_coverage: float = self.data["character_coverage"]
            max_train_size: int = self.data["sp_max_train_size"]
            build_vocab(
                share_vocab_file_paths,
                vocab_size,
                vocab_type,
                vocab_seed,
                casing,
                vocab_split_by_unicode_script,
                character_coverage,
                model_prefix,
                vocab_path,
                self._tags,
                max_train_size,
            )

            self._update_vocab(vocab_path, vocab_path)
        else:
            src_vocab_file_paths: Set[Path] = set(self.src_file_paths)
            if self.mirror:
                src_vocab_file_paths.update(self.trg_file_paths)
            self._create_unshared_vocab(self.src_isos, src_vocab_file_paths, Side.SOURCE)

            trg_vocab_file_paths: Set[Path] = set(self.trg_file_paths)
            if self.mirror:
                trg_vocab_file_paths.update(self.src_file_paths)
            self._create_unshared_vocab(self.trg_isos, trg_vocab_file_paths, Side.TARGET)

            self._update_vocab(self.exp_dir / "src-onmt.vocab", self.exp_dir / "trg-onmt.vocab")

    def _update_vocab(self, src_vocab_path: Path, trg_vocab_path: Path) -> None:
        if self.parent_config is None:
            return

        parent_model_to_use = (
            CheckpointType.BEST
            if self.data["parent_use_best"]
            else CheckpointType.AVERAGE
            if self.data["parent_use_average"]
            else CheckpointType.LAST
        )
        checkpoint_path, step = _get_checkpoint_path(self.parent_config.model_dir, parent_model_to_use)
        parent_config = cast(OpenNMTConfig, self.parent_config)
        parent_runner = create_runner(parent_config.model, parent_config.root, parent_config.write_trg_tag)
        parent_runner.update_vocab(
            str(self.exp_dir / "parent"),
            str(src_vocab_path),
            str(trg_vocab_path),
            None if checkpoint_path is None else str(checkpoint_path),
            step,
            transfer_alignment_heads=self.data["transfer_alignment_heads"],
        )

    def _create_unshared_vocab(self, isos: Set[str], vocab_file_paths: Set[Path], side: Side) -> None:
        prefix = "src" if side == Side.SOURCE else "trg"
        model_prefix = self.exp_dir / f"{prefix}-sp"
        vocab_path = self.exp_dir / f"{prefix}-onmt.vocab"
        tags = self._tags if side == Side.SOURCE else set()
        if self.parent_config is not None:
            parent_isos = self.parent_config.src_isos if side == Side.SOURCE else self.parent_config.trg_isos
            if isos.issubset(parent_isos):
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
                    parent_tokenizer = self.parent_config.create_tokenizer()

                    parent_vocab = Vocab()
                    parent_vocab.load(str(parent_vocab_path))

                    child_tokens = set(tags)
                    for vocab_file_path in vocab_file_paths:
                        for line in parent_tokenizer.tokenize_all(side, load_corpus(vocab_file_path)):
                            child_tokens.update(line.split())
                    parent_use_vocab = child_tokens.issubset(parent_vocab.words)

                # all tokens in the child corpora are in the parent vocab, so we can just use the parent vocab
                # or, the user wants to reuse the parent vocab for this child experiment
                if parent_use_vocab:
                    sp_vocab_path = self.exp_dir / f"{prefix}-sp.vocab"
                    onmt_vocab_path = self.exp_dir / f"{prefix}-onmt.vocab"
                    shutil.copy2(parent_sp_prefix_path.with_suffix(".model"), self.exp_dir / f"{prefix}-sp.model")
                    shutil.copy2(parent_sp_prefix_path.with_suffix(".vocab"), sp_vocab_path)
                    convert_vocab(sp_vocab_path, onmt_vocab_path, tags)
                    return
                elif child_tokens is not None and parent_vocab is not None:
                    onmt_delta_vocab_path = self.exp_dir / f"{prefix}-onmt-delta.vocab"
                    vocab_delta = child_tokens.difference(parent_vocab.words)
                    with onmt_delta_vocab_path.open("w", encoding="utf-8", newline="\n") as f:
                        [f.write(f"{token}\n") for token in vocab_delta]

        LOGGER.info(f"Building {side.name.lower()} vocabulary...")
        vocab_size: int = self.data.get(f"{prefix}_vocab_size", self.data.get("vocab_size"))
        vocab_type: str = self.data.get(f"{prefix}_vocab_type", self.data.get("vocab_type"))
        vocab_seed: int = self.data.get(f"{prefix}_vocab_seed", self.data.get("vocab_seed"))
        casing: str = self.data.get(f"{prefix}_casing", self.data.get("casing"))
        vocab_split_by_unicode_script: bool = self.data.get(
            f"{prefix}_vocab_split_by_unicode_script", self.data.get("vocab_split_by_unicode_script")
        )
        character_coverage: float = self.data.get(f"{prefix}_character_coverage", self.data.get("character_coverage"))
        max_train_size: int = self.data["sp_max_train_size"]
        build_vocab(
            vocab_file_paths,
            vocab_size,
            vocab_type,
            vocab_seed,
            casing,
            vocab_split_by_unicode_script,
            character_coverage,
            model_prefix,
            vocab_path,
            tags,
            max_train_size,
        )

    def _build_corpora(self, tokenizer: Tokenizer, stats: bool) -> int:
        train_count = super()._build_corpora(tokenizer, stats)
        if self.data["guided_alignment"]:
            self._create_train_alignments(train_count)
        return train_count

    def _create_train_alignments(self, train_count: int) -> None:
        with tempfile.TemporaryDirectory() as td:
            temp_dir = Path(td)
            aligner = get_aligner(self.data["aligner"], temp_dir)
            aligner_name = get_aligner_name(aligner.id)
            if not isinstance(aligner, MachineAligner):
                raise RuntimeError(f"{aligner_name} is not supported for generating train alignments.")
            aligner.lowercase = True

            LOGGER.info(f"Generating train alignments using {aligner_name}")
            src_align_path = self.exp_dir / "train.src.txt"
            trg_align_path = self.exp_dir / "train.trg.txt"
            train_size: Union[int, float] = self.data["guided_alignment_train_size"]
            split_indices = split_corpus(train_count, train_size)
            if split_indices is not None:
                # reduce size of alignment training data
                src_train_path = temp_dir / "train.src.align.txt"
                trg_train_path = temp_dir / "train.trg.align.txt"

                with src_align_path.open("r", encoding="utf-8-sig") as src_in_file, trg_align_path.open(
                    "r", encoding="utf-8-sig"
                ) as trg_in_file, src_train_path.open(
                    "w", encoding="utf-8", newline="\n"
                ) as src_out_file, trg_train_path.open(
                    "w", encoding="utf-8", newline="\n"
                ) as trg_out_file:
                    i = 0
                    for src_sentence, trg_sentence in zip(src_in_file, trg_in_file):
                        if i in split_indices:
                            src_out_file.write(src_sentence)
                            trg_out_file.write(trg_sentence)
                        i += 1
            else:
                src_train_path = src_align_path
                trg_train_path = trg_align_path

            aligner.train(src_train_path, trg_train_path)
            aligner.force_align(src_align_path, trg_align_path, self.exp_dir / "train.alignments.txt")

    def _write_dictionary(self, tokenizer: Tokenizer, pair: CorpusPair) -> int:
        terms_config = self.data["terms"]
        dict_books = get_books(terms_config["dictionary_books"]) if "dictionary_books" in terms_config else None
        terms = self._collect_terms(pair, filter_books=dict_books)

        dict_count = 0
        with ExitStack() as stack:
            dict_src_file = stack.enter_context(self._open_append(self.dict_src_filename()))
            dict_trg_file = stack.enter_context(self._open_append(self.dict_trg_filename()))
            dict_vref_file = stack.enter_context(self._open_append(self.dict_vref_filename()))

            if terms is not None:
                for _, term in terms.iterrows():
                    src_term = term["source"]
                    trg_term = term["target"]
                    vrefs = term["vrefs"]
                    tokenizer.set_src_lang(term["source_lang"])
                    tokenizer.set_trg_lang(term["target_lang"])

                    src_term_variants = [
                        tokenizer.tokenize(Side.SOURCE, src_term, add_dummy_prefix=True, add_special_tokens=False),
                        tokenizer.tokenize(Side.SOURCE, src_term, add_dummy_prefix=False, add_special_tokens=False),
                    ]
                    trg_term_variants = [
                        tokenizer.tokenize(Side.TARGET, trg_term, add_dummy_prefix=True, add_special_tokens=False),
                        tokenizer.tokenize(Side.TARGET, trg_term, add_dummy_prefix=False, add_special_tokens=False),
                    ]
                    dict_src_file.write("\t".join(src_term_variants) + "\n")
                    dict_trg_file.write("\t".join(trg_term_variants) + "\n")
                    dict_vref_file.write(vrefs + "\n")
                    dict_count += 1
        return dict_count


class OpenNMTModel(NMTModel):
    def __init__(self, config: OpenNMTConfig, mixed_precision: bool, num_devices: int) -> None:
        set_tf_log_level()
        self._config = config
        self._runner = create_runner(config.model, config.root, config.write_trg_tag, mixed_precision)
        self._num_devices = num_devices

    def train(self) -> None:
        checkpoint_path: Optional[str] = None
        if not (self._config.exp_dir / "run").is_dir() and self._config.has_parent:
            checkpoint_path = str(self._config.exp_dir / "parent")

        self._runner.train(num_devices=self._num_devices, with_eval=True, checkpoint_path=checkpoint_path)

    def save_effective_config(self, path: Path) -> None:
        self._runner.save_effective_config(str(path), training=True)

    def translate_test_files(
        self,
        input_paths: List[Path],
        translation_paths: List[Path],
        vref_paths: Optional[List[Path]] = None,
        checkpoint: Union[CheckpointType, str, int] = CheckpointType.LAST,
    ) -> None:
        features_paths: List[Union[str, Sequence[str]]]
        if vref_paths is None:
            features_paths = [str(ip) for ip in input_paths]
        else:
            features_paths = [[str(ip), str(vp)] for ip, vp in zip(input_paths, vref_paths)]
        predictions_paths = [str(p) for p in translation_paths]
        checkpoint_path, _ = self.get_checkpoint_path(checkpoint)
        self._runner.infer_multiple(features_paths, predictions_paths, str(checkpoint_path))

    def translate(
        self,
        sentences: Iterable[str],
        src_iso: str,
        trg_iso: str,
        vrefs: Optional[Iterable[VerseRef]] = None,
        checkpoint: Union[CheckpointType, str, int] = CheckpointType.LAST,
    ) -> Iterable[str]:
        tokenizer = self._config.create_tokenizer()
        tokenizer.set_trg_lang(trg_iso)
        features_list: List[List[str]] = [[tokenizer.tokenize(Side.SOURCE, s) for s in sentences]]
        if vrefs is not None:
            features_list.append([str(vref) if vref.verse_num != 0 else "" for vref in vrefs])
        checkpoint_path, _ = self.get_checkpoint_path(checkpoint)
        translations = self._runner.infer_list(features_list, str(checkpoint_path))
        return (decode_sp(t[0]) for t in translations)

    def get_checkpoint_path(self, checkpoint: Union[CheckpointType, str, int]) -> Tuple[Path, int]:
        return _get_checkpoint_path(self._config.model_dir, checkpoint)


class OpenNMTTokenizer(Tokenizer):
    def __init__(
        self,
        src_spp: sp.SentencePieceProcessor,
        trg_spp: sp.SentencePieceProcessor,
        write_trg_tag: bool = False,
    ) -> None:
        self._src_spp = src_spp
        self._trg_spp = trg_spp
        self._write_trg_tag = write_trg_tag
        self._src_lang: Optional[str] = None
        self._trg_lang: Optional[str] = None

    def set_src_lang(self, src_lang: str) -> None:
        self._src_lang = src_lang

    def set_trg_lang(self, trg_lang: str) -> None:
        self._trg_lang = trg_lang

    def tokenize(
        self,
        side: Side,
        line: str,
        add_dummy_prefix: bool = True,
        sample_subwords: bool = False,
        add_special_tokens: bool = True,
    ) -> str:
        spp = self._src_spp if side is Side.SOURCE else self._trg_spp
        if not add_dummy_prefix:
            line = "\ufffc" + line
        if sample_subwords:
            pieces = spp.Encode(line, out_type=str, enable_sampling=True, alpha=0.1, nbest_size=-1)
        else:
            pieces = spp.EncodeAsPieces(line)
        if not add_dummy_prefix:
            pieces = pieces[2:]
        line = " ".join(pieces)
        prefix = ""
        if add_special_tokens and side is Side.SOURCE and self._write_trg_tag and self._trg_lang is not None:
            prefix = f"<2{self._trg_lang}> "
        return prefix + line

    def normalize(self, side: Side, line: str) -> str:
        return self.detokenize(self.tokenize(side, line))

    def detokenize(self, line: str) -> str:
        return decode_sp(line)
