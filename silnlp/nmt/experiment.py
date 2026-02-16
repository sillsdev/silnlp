import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Set, Union

import yaml

from ..common.environment import SIL_NLP_ENV
from ..common.postprocesser import PostprocessConfig, PostprocessHandler
from ..common.utils import get_git_revision_hash, show_attrs
from ..common.clearml_connection import TAGS_LIST, SILClearML
from .config import Config, get_mt_exp_dir
from .test import SUPPORTED_SCORERS, test
from .translate import TranslationTask


@dataclass
class SILExperiment:
    name: str
    make_stats: bool = False
    force_align: bool = False
    mixed_precision: bool = True
    num_devices: int = 1
    clearml_queue: Optional[str] = None
    run_prep: bool = False
    run_train: bool = False
    run_test: bool = False
    run_translate: bool = False
    produce_multiple_translations: bool = False
    save_confidences: bool = False
    scorers: Optional[Set[str]] = None
    score_by_book: bool = False
    commit: Optional[str] = None
    clearml_tag: Optional[str] = None

    def __post_init__(self):
        self.clearml = SILClearML(
            self.name,
            self.clearml_queue,
            commit=self.commit,
            tag=self.clearml_tag,
        )
        self.name: str = self.clearml.name
        self.config: Config = self.clearml.config
        self.rev_hash = get_git_revision_hash()
        self.config.set_seed()

        if self.scorers is None:
            self.scorers = set()

    def run(self):
        if self.run_prep:
            self.preprocess()
        if self.run_train:
            self.train()
        if self.run_test:
            self.test()
        if self.run_translate:
            self.translate()

    def preprocess(self):
        # Do some basic checks before starting the experiment
        exp_dir = Path(get_mt_exp_dir(self.name))
        if not exp_dir.exists():
            raise RuntimeError(f"ERROR: Experiment folder {exp_dir} does not exist.")
        config_file = Path(exp_dir, "config.yml")
        if not config_file.exists():
            raise RuntimeError(f"ERROR: Config file does not exist in experiment folder {exp_dir}.")
        self.config.preprocess(self.make_stats, self.force_align)

    def train(self):
        os.system("nvidia-smi")
        model = self.config.create_model(self.mixed_precision, self.num_devices)
        model.save_effective_config(self.config.exp_dir / f"effective-config-{self.rev_hash}.yml")
        print(f"=== Training ({self.name}) ===")
        model.train()
        print("Training completed")

    def test(self):
        assert self.scorers is not None
        test(
            experiment=self.name,
            last=self.config.model_dir.exists(),
            best=self.config.model_dir.exists(),
            by_book=self.score_by_book,
            scorers=self.scorers,
            produce_multiple_translations=self.produce_multiple_translations,
            save_confidences=self.save_confidences,
        )

    def translate(self):
        with (self.config.exp_dir / "translate_config.yml").open("r", encoding="utf-8") as file:
            translate_configs = yaml.safe_load(file)

        postprocess_configs = translate_configs.get("postprocess", [])
        postprocess_handler = PostprocessHandler([PostprocessConfig(pc) for pc in postprocess_configs])

        for translate_config in translate_configs.get("translate", []):
            checkpoint: Union[str, int] = translate_config.get("checkpoint", "last") or "last"
            translator = TranslationTask(
                name=self.name,
                checkpoint=checkpoint,
                commit=self.commit,
            )

            if not postprocess_configs:
                postprocess_handler = PostprocessHandler([])

            if "tags" in translate_config and isinstance(translate_config["tags"], list):
                translate_config["tags"] = ",".join(translate_config["tags"])

            if len(translate_config.get("books", [])) > 0:
                if isinstance(translate_config["books"], list):
                    translate_config["books"] = ";".join(translate_config["books"])
                translator.translate_books(
                    translate_config["books"],
                    translate_config.get("src_project"),
                    translate_config.get("trg_project"),
                    translate_config.get("trg_iso"),
                    self.produce_multiple_translations,
                    self.save_confidences,
                    postprocess_handler,
                    translate_config.get("tags"),
                )
            elif translate_config.get("src_prefix"):
                if translate_config.get("trg_prefix") is None:
                    raise RuntimeError("A target file prefix must be specified.")
                if translate_config.get("start_seq") is None or translate_config.get("end_seq") is None:
                    raise RuntimeError("Start and end sequence numbers must be specified.")

                translator.translate_text_files(
                    translate_config.get("src_prefix"),
                    translate_config.get("trg_prefix"),
                    translate_config.get("start_seq"),
                    translate_config.get("end_seq"),
                    translate_config.get("src_iso"),
                    translate_config.get("trg_iso"),
                    self.produce_multiple_translations,
                    self.save_confidences,
                    translate_config.get("tags"),
                )
            elif translate_config.get("src"):
                translator.translate_files(
                    translate_config.get("src"),
                    translate_config.get("trg"),
                    translate_config.get("src_iso"),
                    translate_config.get("trg_iso"),
                    self.produce_multiple_translations,
                    self.save_confidences,
                    postprocess_handler,
                    translate_config.get("tags"),
                )
            else:
                raise RuntimeError("A Scripture book, file, or file prefix must be specified for translation.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run experiment - preprocess, train, and test")
    parser.add_argument("experiment", help="Experiment name")
    parser.add_argument("--stats", default=False, action="store_true", help="Compute tokenization statistics")
    parser.add_argument(
        "--force-align", default=False, action="store_true", help="Force recalculation of all alignment scores"
    )
    parser.add_argument("--disable-mixed-precision", default=False, action="store_true", help="Disable mixed precision")
    parser.add_argument("--num-devices", type=int, default=1, help="Number of devices to train on")
    parser.add_argument(
        "--clearml-queue",
        default=None,
        type=str,
        help="Run remotely on ClearML queue.  Default: None - don't register with ClearML.  The queue 'local' will run "
        + "it locally and register it with ClearML.",
    )
    parser.add_argument(
        "--clearml-tag",
        metavar="tag",
        choices=TAGS_LIST,
        default=None,
        type=str,
        help=f"Tag to add to the ClearML Task - {TAGS_LIST}",
    )
    parser.add_argument(
        "--commit", type=str, default=None, help="The silnlp git commit id with which to run a remote job"
    )
    parser.add_argument(
        "--save-checkpoints",
        default=False,
        action="store_true",
        help="Save checkpoints to bucket. Only used if running the train step.",
    )
    parser.add_argument("--preprocess", default=False, action="store_true", help="Run the preprocess step.")
    parser.add_argument("--train", default=False, action="store_true", help="Run the train step.")
    parser.add_argument("--test", default=False, action="store_true", help="Run the test step.")
    parser.add_argument("--translate", default=False, action="store_true", help="Create drafts.")
    parser.add_argument(
        "--multiple-translations",
        default=False,
        action="store_true",
        help='Produce multiple translations of each verse. These will be saved in separate files with suffixes like ".1.txt", ".2.txt", etc.',
    )
    parser.add_argument(
        "--save-confidences",
        default=False,
        action="store_true",
        help="Generate confidence files for test and/or translate step.",
    )
    parser.add_argument("--score-by-book", default=False, action="store_true", help="Score individual books")
    parser.add_argument("--mt-dir", default=None, type=str, help="The machine translation directory.")
    parser.add_argument(
        "--debug",
        default=False,
        action="store_true",
        help="Show information about the environment variables and arguments.",
    )
    parser.add_argument(
        "--scorers",
        nargs="*",
        metavar="scorer",
        choices=SUPPORTED_SCORERS,
        default=[
            "bleu",
            "sentencebleu",
            "chrf3",
            "chrf3+",
            "chrf3++",
            "m-bleu",
            "m-chrf3",
            "m-chrf3+",
            "m-chrf3++",
            "spbleu",
        ],
        help=f"List of scorers - {SUPPORTED_SCORERS}",
    )

    args = parser.parse_args()

    if args.clearml_queue is not None and args.clearml_tag is None:
        parser.error("Missing ClearML tag. Add a tag using --clearml-tag. Possible tags: " + f"{TAGS_LIST}")

    if args.mt_dir is not None:
        SIL_NLP_ENV.set_machine_translation_dir(SIL_NLP_ENV.data_dir / args.mt_dir)

    if args.debug:
        show_attrs(cli_args=args)
        exit()

    if not (args.preprocess or args.train or args.test):
        args.preprocess = True
        args.train = True
        args.test = True

    exp = SILExperiment(
        name=args.experiment,
        make_stats=args.stats,
        force_align=args.force_align,
        mixed_precision=not args.disable_mixed_precision,
        num_devices=args.num_devices,
        clearml_queue=args.clearml_queue,
        clearml_tag=args.clearml_tag,
        commit=args.commit,
        run_prep=args.preprocess,
        run_train=args.train,
        run_test=args.test,
        run_translate=args.translate,
        produce_multiple_translations=args.multiple_translations,
        save_confidences=args.save_confidences,
        scorers=set(s.lower() for s in args.scorers),
        score_by_book=args.score_by_book,
    )

    if not args.save_checkpoints:
        SIL_NLP_ENV.delete_path_on_exit(get_mt_exp_dir(args.experiment) / "run")
    exp.run()


if __name__ == "__main__":
    main()
