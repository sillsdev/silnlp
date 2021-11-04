from silnlp.nmt.experiment import SILExperiment
from silnlp.nmt.translate import NMTTranslator

tlr = NMTTranslator(
    name="de-to-en-WMT2020+Bibles_AE/bch-en",
    memory_growth=False,
    clearml_queue=None,
    experiment_suffix="_2PE_2nd",
)
tlr.translate_book_by_step("2PE")
