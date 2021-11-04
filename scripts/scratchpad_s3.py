from silnlp.nmt.experiment import SILExperiment
from silnlp.nmt.translate import NMTTranslator

book = "GAL"
tlr = NMTTranslator(
    name="de-to-en-WMT2020+Bibles_AE/bch-en",
    clearml_queue="langtech_10gb",
)
tlr.translate_book_by_step(book)
