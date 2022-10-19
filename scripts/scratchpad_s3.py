from silnlp.common.tf_utils import enable_memory_growth
from silnlp.nmt.experiment import SILExperiment
from silnlp.nmt.translate import TranslationTask

enable_memory_growth()

exp = SILExperiment(
    name="BT-Swahili/en-swh-1",
    make_stats=True,  # limited by stats_max_size to process only Bibles
    mixed_precision=True,  # clearML GPU's can handle mixed precision
    clearml_queue="langtech_10gb",
)
exp.run()

# tlr = TranslationTask(name=r"BT-Swahili/child_model", clearml_queue="langtech_10gb")
# tlr.translate_book("JUD")
