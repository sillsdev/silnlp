import argparse
import logging
from pathlib import Path

import yaml

from ..nmt.clearml_connection import SILClearML
from ..nmt.config_utils import load_config
from ..nmt.postprocess import get_draft_paths_from_exp, postprocess_draft
from .paratext import get_project_dir
from .postprocesser import PostprocessConfig, PostprocessHandler
from .utils import get_mt_exp_dir

LOGGER = logging.getLogger(__package__ + ".postprocess_draft")


def main() -> None:
    parser = argparse.ArgumentParser(description="Applies draft postprocessing steps to a draft.")
    parser.add_argument(
        "--experiment",
        default=None,
        help="Name of an experiment directory in MT/experiments. \
        If this option is used, the experiment's translate config will be used to find source and draft files.",
    )
    parser.add_argument(
        "--source",
        default=None,
        help="Path of the source USFM file. \
        If in a Paratext project, the project settings will be used when reading the files.",
    )
    parser.add_argument(
        "--draft",
        default=None,
        help="Path of the draft USFM file that postprocessing will be applied to. \
        Must have the exact same USFM structure as 'source', which it will if it is a draft from that source.",
    )
    parser.add_argument(
        "--book",
        default=None,
        help="3-letter book id of book being evaluated, e.g. MAT. \
        Only necessary if the source file is not in a Paratext project directory.",
    )
    parser.add_argument(
        "--output-folder",
        default=None,
        help="Output folder for the postprocessed draft. Defaults to the folder of the original draft.",
    )
    parser.add_argument(
        "--include-paragraph-markers",
        default=False,
        action="store_true",
        help="Attempt to place paragraph markers in translated verses based on the source project's markers",
    )
    parser.add_argument(
        "--include-style-markers",
        default=False,
        action="store_true",
        help="Attempt to place style markers in translated verses based on the source project's markers",
    )
    parser.add_argument(
        "--include-embeds",
        default=False,
        action="store_true",
        help="Carry over embeds from the source project to the output without translating them",
    )
    parser.add_argument(
        "--clearml-queue",
        default=None,
        type=str,
        help="Run remotely on ClearML queue.  Default: None - don't register with ClearML.  The queue 'local' will run "
        + "it locally and register it with ClearML.",
    )
    args = parser.parse_args()

    if args.experiment and (args.source or args.draft or args.book):
        LOGGER.info("--experiment option used. --source, --draft, and --book will be ignored.")
    if not (args.experiment or (args.source and args.draft)):
        raise ValueError("Not enough options used. Please use --experiment OR --source and --draft.")

    experiment = args.experiment.replace("\\", "/") if args.experiment else None
    if experiment and get_mt_exp_dir(experiment).exists():
        if args.clearml_queue is not None:
            if "cpu" not in args.clearml_queue:
                raise ValueError("Running this script on a GPU queue will not speed it up. Please only use CPU queues.")
            clearml = SILClearML(experiment, args.clearml_queue)
            config = clearml.config
        else:
            config = load_config(experiment)

        if not (config.exp_dir / "translate_config.yml").exists():
            raise ValueError(
                "Experiment translate_config.yml not found. Please use --source and --draft options instead."
            )
        src_paths, draft_paths = get_draft_paths_from_exp(config)
    elif args.clearml_queue is not None:
        raise ValueError("Must use --experiment option to use ClearML.")
    else:
        src_paths = [Path(args.source.replace("\\", "/"))]
        draft_paths = [Path(args.draft.replace("\\", "/"))]
        if not str(src_paths[0]).startswith(str(get_project_dir(""))) and args.book is None:
            raise ValueError(
                "--book argument must be passed if the source file is not in a Paratext project directory."
            )

    # If no postprocessing options are used, use any postprocessing requests in the experiment's translate config
    if args.include_paragraph_markers or args.include_style_markers or args.include_embeds:
        postprocess_configs = [
            {
                "include_paragraph_markers": args.include_paragraph_markers,
                "include_style_markers": args.include_style_markers,
                "include_embeds": args.include_embeds,
            }
        ]
    else:
        if args.experiment:
            LOGGER.info("No postprocessing options used. Applying postprocessing requests from translate config.")
            with (config.exp_dir / "translate_config.yml").open("r", encoding="utf-8") as file:
                postprocess_configs = yaml.safe_load(file).get("postprocess", [])
            if len(postprocess_configs) == 0:
                LOGGER.info("No postprocessing requests found in translate config.")
                exit()
        else:
            LOGGER.info("Please use at least one postprocessing option.")
            exit()
    postprocess_handler = PostprocessHandler([PostprocessConfig(pc) for pc in postprocess_configs], include_base=False)

    if args.output_folder:
        args.output_folder = Path(args.output_folder.replace("\\", "/"))
    for src_path, draft_path in zip(src_paths, draft_paths):
        postprocess_draft(src_path, draft_path, postprocess_handler, args.book, args.output_folder)


if __name__ == "__main__":
    main()
