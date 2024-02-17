import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from ..common.environment import SIL_NLP_ENV
from ..common.tf_utils import enable_memory_growth
from ..common.utils import get_git_revision_hash, show_attrs
from .clearml_connection import SILClearML
from .config import Config, get_mt_exp_dir
from .test import test
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
    score_by_book: bool = False
    commit: Optional[str] = None

    def __post_init__(self):
        self.clearml = SILClearML(self.name, self.clearml_queue, commit=self.commit)
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
        SIL_NLP_ENV.copy_experiment_to_bucket(self.name)

    def train(self):
        os.system("nvidia-smi")
        SIL_NLP_ENV.copy_experiment_from_bucket(self.name)

        model = self.config.create_model(self.mixed_precision, self.num_devices)
        model.save_effective_config(self.config.exp_dir / f"effective-config-{self.rev_hash}.yml")
        SIL_NLP_ENV.copy_experiment_to_bucket(self.name, patterns=f"effective-config-{self.rev_hash}.yml")
        print(f"=== Training ({self.name}) ===")
        model.train()
        if self.save_checkpoints:
            SIL_NLP_ENV.copy_experiment_to_bucket(self.name)
        print("Training completed")

    def test(self):
        SIL_NLP_ENV.copy_experiment_from_bucket(self.name)
        test(
            experiment=self.name,
            last=self.config.model_dir.exists(),
            best=self.config.model_dir.exists(),
            by_book=self.score_by_book,
            scorers={"bleu", "sentencebleu", "chrf3", "wer", "ter", "spbleu"},
        )
        SIL_NLP_ENV.copy_experiment_to_bucket(
            self.name, patterns=("scores-*.csv", "test.*trg-predictions.*"), overwrite=True
        )

    def translate(self):
        translate_configs = self.config.translate
        for translate_config in translate_configs:
            translator = TranslationTask(
            name=self.name,
            checkpoint=translate_config.get("checkpoint", "last"),
            commit=self.commit
            )

            if len(translate_config.get("books", [])) > 0:
                if isinstance(translate_config["books"], list):
                    translate_config["books"] = ";".join(translate_config["books"])
                translator.translate_books(
                    translate_config["books"],
                    translate_config.get("src_project"),
                    translate_config.get("trg_project"),
                    translate_config.get("trg_iso"),
                    translate_config.get("include_inline_elements", False),
                    translate_config.get("stylesheet_field_update", "merge"),
                )
            elif translate_config.get("src_prefix"):
                translator.translate_text_files(
                    translate_config.get("src_prefix"), 
                    translate_config.get("trg_prefix"), 
                    translate_config.get("start_seq"), 
                    translate_config.get("end_seq"), 
                    translate_config.get("src_iso"), 
                    translate_config.get("trg_iso")
                )
            elif translate_config.get("src"):
                translator.translate_files(translate_config.get("src"), 
                                        translate_config.get("trg"), 
                                        translate_config.get("src_iso"), 
                                        translate_config.get("trg_iso"), 
                                        translate_config.get("include_inline_elements", False))
            else:
                raise RuntimeError("A Scripture book, file, or file prefix must be specified for translation.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run experiment - preprocess, train, and test")
    parser.add_argument("experiment", help="Experiment name")
    parser.add_argument("--stats", default=False, action="store_true", help="Output corpus statistics")
    parser.add_argument("--force-align", default=False, action="store_true", help="Force recalculation of all alignment scores")
    parser.add_argument("--disable-mixed-precision", default=False, action="store_true", help="Disable mixed precision")
    parser.add_argument("--memory-growth", default=False, action="store_true", help="Enable memory growth")
    parser.add_argument("--num-devices", type=int, default=1, help="Number of devices to train on")
    parser.add_argument(
        "--clearml-queue",
        default=None,
        type=str,
        help="Run remotely on ClearML queue.  Default: None - don't register with ClearML.  The queue 'local' will run "
        + "it locally and register it with ClearML.",
    )
    parser.add_argument("--save-checkpoints", default=False, action="store_true", help="Save checkpoints to S3 bucket")
    parser.add_argument("--preprocess", default=False, action="store_true", help="Run the preprocess step.")
    parser.add_argument("--train", default=False, action="store_true", help="Run the train step.")
    parser.add_argument("--test", default=False, action="store_true", help="Run the test step.")
    parser.add_argument("--translate", default=False, action="store_true", help="Create drafts.")
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

    args = parser.parse_args()

    if args.mt_dir is not None:
        SIL_NLP_ENV.set_machine_translation_dir(SIL_NLP_ENV.data_dir / args.mt_dir)

    if args.debug:
        show_attrs(cli_args=args)
        exit()

    if args.memory_growth:
        enable_memory_growth()

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
        save_checkpoints=args.save_checkpoints,
        run_prep=args.preprocess,
        run_train=args.train,
        run_test=args.test,
        run_translate=args.translate,
        score_by_book=args.score_by_book,
        commit=args.commit,
    )
    exp.run()


if __name__ == "__main__":
    main()
