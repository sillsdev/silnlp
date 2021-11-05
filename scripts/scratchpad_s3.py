from silnlp.nmt.experiment import SILExperiment
from silnlp.nmt.translate import TranslationTask

tlr = TranslationTask(name="de-to-en-WMT2020+Bibles_AE/bch-en", clearml_queue="langtech_10gb")
tlr.translate_book("ACT")
