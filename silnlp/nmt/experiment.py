import argparse
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Set

import yaml

from ..common.environment import SIL_NLP_ENV
from ..common.postprocesser import PostprocessConfig, PostprocessHandler
from ..common.utils import get_git_revision_hash, show_attrs
from .clearml_connection import SILClearML
from .config import Config, get_mt_exp_dir
from .test import _SUPPORTED_SCORERS, test
from .translate import TranslationTask


@dataclass
class SILExperiment:
    name: str
    make_stats: bool = False
    force_align: bool = False
    mixed_precision: bool = True
    num_devices: int = 1
    clearml_queue: Optional[str] = None
    save_checkpoints: bool = False
    run_prep: bool = False
    run_train: bool = False
    run_test: bool = False
    run_translate: bool = False
    produce_multiple_translations: bool = False
    save_confidences: bool = False
    scorers: Set[str] = field(default_factory=set)
    score_by_book: bool = False
    commit: Optional[str] = None

    def __post_init__(self):
        self.clearml = SILClearML(
            self.name, self.clearml_queue, commit=self.commit, use_default_model_dir=self.save_checkpoints
        )
        self.name: str = self.clearml.name
        self.config: Config = self.clearml.config
        self.rev_hash = get_git_revision_hash()
        self.config.set_seed()

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
        test(
            experiment=self.name,
            last=self.config.model_dir.exists(),
            best=self.config.model_dir.exists(),
            by_book=self.score_by_book,
            scorers=self.scorers,
            produce_multiple_translations=self.produce_multiple_translations,
            save_confidences=self.save_confidences,
            use_default_model_dir=self.save_checkpoints,
        )

    def translate(self):
        with (self.config.exp_dir / "translate_config.yml").open("r", encoding="utf-8") as file:
            translate_configs = yaml.safe_load(file)

        postprocess_configs = translate_configs.get("postprocess", [])
        postprocess_handler = PostprocessHandler([PostprocessConfig(pc) for pc in postprocess_configs])

        for translate_config in translate_configs.get("translate", []):
            translator = TranslationTask(
                name=self.name,
                checkpoint=config.get("checkpoint", "last"),
                use_default_model_dir=self.save_checkpoints,
                commit=self.commit,
            )

            if not postprocess_configs:
                postprocess_handler = PostprocessHandler([])

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
                )
            elif translate_config.get("src_prefix"):
                translator.translate_text_files(
                    translate_config.get("src_prefix"),
                    translate_config.get("trg_prefix"),
                    translate_config.get("start_seq"),
                    translate_config.get("end_seq"),
                    translate_config.get("src_iso"),
                    translate_config.get("trg_iso"),
                    self.produce_multiple_translations,
                    self.save_confidences,
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
        "--commit", type=str, default=None, help="The silnlp git commit id with which to run a remote job"
    )
    parser.add_argument(
        "--scorers",
        nargs="*",
        metavar="scorer",
        choices=_SUPPORTED_SCORERS,
        default=["bleu", "sentencebleu", "chrf3", "chrf3+", "chrf3++", "spbleu"],
        help=f"List of scorers - {_SUPPORTED_SCORERS}",
    )

    args = parser.parse_args()

    if args.mt_dir is not None:
        SIL_NLP_ENV.set_machine_translation_dir(SIL_NLP_ENV.data_dir / args.mt_dir)

    if args.debug:
        show_attrs(cli_args=args)
        exit()

    if not (args.preprocess or args.train or args.test):
        args.preprocess = True
        args.train = True
        args.test = True

    if not args.train:
        args.save_checkpoints = True

    exp = SILExperiment(
        name=args.experiment,
        make_stats=args.stats,
        force_align=args.force_align,
        mixed_precision=not args.disable_mixed_precision,
        num_devices=args.num_devices,
        clearml_queue=args.clearml_queue,
        save_checkpoints=args.save_checkpoints,
        run_prep=args.preprocess,
        run_train=args.train,
        run_test=args.test,
        run_translate=args.translate,
        produce_multiple_translations=args.multiple_translations,
        save_confidences=args.save_confidences,
        scorers=set(s.lower() for s in args.scorers),
        score_by_book=args.score_by_book,
        commit=args.commit,
    )
    exp.run()


if __name__ == "__main__":
    main()
