import argparse
import logging
import os
from enum import Flag, auto
from typing import Dict, Iterable, List, Optional, Set, Union

logging.basicConfig()

import opennmt
import opennmt.data
import opennmt.inputters
import opennmt.models
import opennmt.utils
import tensorflow as tf
import yaml

from ..common.corpus import get_corpus_path
from ..common.utils import get_git_revision_hash, get_mt_root_dir
from .noise import WordDropout
from .runner import RunnerEx

_PYTHON_TO_TENSORFLOW_LOGGING_LEVEL: Dict[int, int] = {
    logging.CRITICAL: 3,
    logging.ERROR: 2,
    logging.WARNING: 1,
    logging.INFO: 0,
    logging.DEBUG: 0,
    logging.NOTSET: 0,
}

DEFAULT_NEW_CONFIG: dict = {
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


@opennmt.models.register_model_in_catalog
class TransformerMedium(opennmt.models.Transformer):
    def __init__(self):
        super().__init__(
            source_inputter=opennmt.inputters.WordEmbedder(embedding_size=512),
            target_inputter=opennmt.inputters.WordEmbedder(embedding_size=512),
            num_layers=3,
            num_units=512,
            num_heads=8,
            ffn_inner_dim=2048,
            dropout=0.1,
            attention_dropout=0.1,
            ffn_dropout=0.1,
        )


def set_log_level(log_level: int) -> None:
    tf.get_logger().setLevel(log_level)
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = str(_PYTHON_TO_TENSORFLOW_LOGGING_LEVEL[log_level])


def set_transformer_dropout(
    root_layer: opennmt.models.Transformer,
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


def load_config(exp_name: str) -> dict:
    root_dir = get_mt_root_dir(exp_name)
    config_path = os.path.join(root_dir, "config.yml")

    config: dict = {
        "model": "TransformerBase",
        "model_dir": os.path.join(root_dir, "run"),
        "data": {
            "train_features_file": os.path.join(root_dir, "train.src.txt"),
            "train_labels_file": os.path.join(root_dir, "train.trg.txt"),
            "eval_features_file": os.path.join(root_dir, "val.src.txt"),
            "eval_labels_file": os.path.join(root_dir, "val.trg.txt"),
            "share_vocab": True,
            "character_coverage": 1.0,
            "mirror": False,
            "mixed_src": False,
            "parent_use_best": False,
            "parent_use_average": False,
            "parent_use_vocab": False,
            "seed": 111,
            "val_size": 250,
            "disjoint_test": False,
            "disjoint_val": False,
            "score_threshold": 0,
            "scripture": True,
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
    }

    config = opennmt.load_config([config_path], config)
    data_config: dict = config["data"]
    eval_config: dict = config["eval"]
    multi_ref_eval: bool = eval_config["multi_ref_eval"]
    if multi_ref_eval:
        data_config["eval_labels_file"] = os.path.join(root_dir, "val.trg.txt.0")
    if data_config["share_vocab"]:
        data_config["source_vocabulary"] = os.path.join(root_dir, "onmt.vocab")
        data_config["target_vocabulary"] = os.path.join(root_dir, "onmt.vocab")
        if (
            "src_vocab_size" not in data_config
            and "trg_vocab_size" not in data_config
            and "vocab_size" not in data_config
        ):
            data_config["vocab_size"] = 24000
        if "src_casing" not in data_config and "trg_casing" not in data_config and "casing" not in data_config:
            data_config["casing"] = "lower"
    else:
        data_config["source_vocabulary"] = os.path.join(root_dir, "src-onmt.vocab")
        data_config["target_vocabulary"] = os.path.join(root_dir, "trg-onmt.vocab")
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
    if "test_size" not in data_config:
        data_config["test_size"] = 0 if "test_books" in data_config else 250
    return config


def create_model(config: dict) -> opennmt.models.Model:
    data_config: dict = config["data"]
    params_config: dict = config["params"]

    parent: Optional[str] = data_config.get("parent")
    parent_data_config = {}
    if parent:
        parent_config = load_config(parent)
        parent_data_config = parent_config["data"]

    model = opennmt.models.get_model_from_catalog(config["model"])
    if isinstance(model, opennmt.models.Transformer):
        dropout = params_config["transformer_dropout"]
        attention_dropout = params_config["transformer_attention_dropout"]
        ffn_dropout = params_config["transformer_ffn_dropout"]
        if dropout != 0.1 or attention_dropout != 0.1 or ffn_dropout != 0.1:
            set_transformer_dropout(
                model, dropout=dropout, attention_dropout=attention_dropout, ffn_dropout=ffn_dropout
            )

    word_dropout: float = params_config["word_dropout"]
    if word_dropout > 0:
        write_trg_tag = (
            len(data_config["trg_langs"]) > 1
            or len(parent_data_config.get("trg_langs", [])) > 1
            or data_config["mirror"]
            or parent_data_config.get("mirror", False)
        )
        source_noiser = opennmt.data.WordNoiser(subword_token="▁", is_spacer=True)
        source_noiser.add(WordDropout(word_dropout, skip_first_word=write_trg_tag))
        model.features_inputter.set_noise(source_noiser, probability=1.0)

        target_noiser = opennmt.data.WordNoiser(subword_token="▁", is_spacer=True)
        target_noiser.add(WordDropout(word_dropout))
        model.labels_inputter.set_noise(target_noiser, probability=1.0)
    return model


def create_runner(
    config: dict, mixed_precision: bool = False, log_level: int = logging.INFO, memory_growth: bool = False
) -> RunnerEx:
    set_log_level(log_level)

    if memory_growth:
        gpus = tf.config.list_physical_devices(device_type="GPU")
        for device in gpus:
            tf.config.experimental.set_memory_growth(device, enable=True)

    model = create_model(config)

    return RunnerEx(model, config, auto_config=True, mixed_precision=mixed_precision)


class DataFileType(Flag):
    NONE = 0
    TRAIN = auto()
    TEST = auto()
    VAL = auto()
    SYNTH = auto()


class DataFile:
    def __init__(self, path: str, type: DataFileType):
        self.path = path
        self.type = type
        file_name = os.path.splitext(os.path.basename(path))[0]
        index = file_name.index("-")
        self.iso = file_name[:index]
        self.project = file_name[index + 1 :]

    @property
    def is_train(self):
        return (self.type & DataFileType.TRAIN) == DataFileType.TRAIN

    @property
    def is_test(self):
        return (self.type & DataFileType.TEST) == DataFileType.TEST

    @property
    def is_val(self):
        return (self.type & DataFileType.VAL) == DataFileType.VAL

    @property
    def is_synth(self):
        return (self.type & DataFileType.SYNTH) == DataFileType.SYNTH


class Language:
    def __init__(self, iso: str):
        self.iso = iso
        self.data_files: List[DataFile] = []


def parse_projects(projects_value: Optional[Union[str, List[str]]], default: Set[str] = set()) -> Set[str]:
    if projects_value is None:
        return default
    if isinstance(projects_value, str):
        return set(map(lambda p: p.strip(), projects_value.split(",")))
    return set(projects_value)


def parse_langs(langs: Iterable[Union[str, dict]]) -> Dict[str, Language]:
    lang_infos: Dict[str, Language] = {}
    for lang in langs:
        if isinstance(lang, str):
            index = lang.find("-")
            if index == -1:
                raise RuntimeError("A language project is not fully specified.")
            iso = lang[:index]
            projects_str = lang[index + 1 :]
            data_files: List[DataFile] = []
            for project in projects_str.split(","):
                project = project.strip()
                project_path = get_corpus_path(iso, project)
                data_files.append(DataFile(project_path, DataFileType.TRAIN | DataFileType.TEST | DataFileType.VAL))

        else:
            iso = lang["iso"]
            train_projects = parse_projects(lang.get("train"))
            test_projects = parse_projects(lang.get("test"), default=train_projects)
            val_projects = parse_projects(lang.get("val"), default=train_projects)
            synth_projects = parse_projects(lang.get("synth"))
            data_files: List[DataFile] = []
            for project in train_projects | test_projects | val_projects | synth_projects:
                file_path = get_corpus_path(iso, project)
                file_type = DataFileType.NONE
                if project in train_projects:
                    file_type |= DataFileType.TRAIN
                if project in test_projects:
                    file_type |= DataFileType.TEST
                if project in val_projects:
                    file_type |= DataFileType.VAL
                if project in synth_projects:
                    file_type |= DataFileType.SYNTH | DataFileType.TRAIN
                data_files.append(DataFile(file_path, file_type))

        lang_info = lang_infos.get(iso)
        if lang_info is None:
            lang_info = Language(iso)
            lang_infos[iso] = lang_info
        lang_info.data_files.extend(data_files)
    return lang_infos


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

    root_dir = get_mt_root_dir(args.experiment)
    config_path = os.path.join(root_dir, "config.yml")
    if os.path.isfile(config_path) and not args.force:
        print('The experiment config file already exists. Use "--force" if you want to overwrite the existing config.')
        return

    os.makedirs(root_dir, exist_ok=True)

    config = DEFAULT_NEW_CONFIG.copy()
    if args.model is not None:
        config["model"] = args.model
    data_config: dict = config["data"]
    data_config["src_langs"] = args.src_langs
    data_config["trg_langs"] = args.trg_langs
    if args.parent is not None:
        data_config["parent"] = args.parent
        parent_config = load_config(args.parent)
        parent_data_config: dict = parent_config["data"]
        for key in [
            "share_vocab",
            "vocab_size",
            "src_vocab_size",
            "trg_vocab_size",
            "casing",
            "src_casing",
            "trg_casing",
        ]:
            if key in parent_data_config:
                data_config[key] = parent_data_config[key]
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
