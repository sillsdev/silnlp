import argparse
import logging
from dataclasses import dataclass

logging.basicConfig()

from .experiment import SILExperiment
from ..common.environment import SIL_NLP_ENV


@dataclass
class SILExperimentCML(SILExperiment):
    def __post_init__(self):
        from clearml import Task
        import datetime

        self.task = Task.init(
            project_name="LangTech_" + self.name,
            task_name=self.name + "_" + str(datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S")),
        )
        return super().__post_init__()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run experiment - preprocesses, train and test")
    parser.add_argument("experiment", help="Experiment name")
    args = parser.parse_args()

    exp = SILExperimentCML(
        name=args.experiment,
        make_stats=True,  # limited by stats_max_size to process only Bibles
        mixed_precision=True,  # clearML GPU's can handle mixed precision
        memory_growth=False,  # we can allocate all memory all the time
        num_devices=-1,  # get all devices
    )
    exp.run()


if __name__ == "__main__":
    main()
