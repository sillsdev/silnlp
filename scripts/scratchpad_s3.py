from silnlp.common.environment import SIL_NLP_ENV
import logging

LOGGER = logging.getLogger("silnlp")
LOGGER.setLevel(logging.DEBUG)

(SIL_NLP_ENV.mt_experiments_dir / "hello").mkdir()
(SIL_NLP_ENV.mt_experiments_dir / "hello" / "exp").mkdir()
(SIL_NLP_ENV.mt_experiments_dir / "hello" / "exp" / "world.txt").open("w+").write("hello world!")

SIL_NLP_ENV.copy_experiment_to_bucket("hello")
