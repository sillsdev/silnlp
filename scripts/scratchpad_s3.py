from silnlp.nmt.experiment import SILExperiment
from silnlp.nmt.translate import TranslationTask
from silnlp.nmt.utils import enable_memory_growth

enable_memory_growth()

#exp = SILExperiment(
#    name="BT-Swahili/en-swh-1",
#    make_stats=True,  # limited by stats_max_size to process only Bibles
#    mixed_precision=True,  # clearML GPU's can handle mixed precision
#    clearml_queue="langtech_10gb",
#)
#exp.run()

exp_name = r'BT-Multilingual/AE.10SL.10TL.2M.kpz_en.GEN.112'
src_file = "MT/scripture/kpz-KPZ-111-fix.txt"
trg_file = "MT/scripture/en-KpzBT-111-112.txt"

tlr = TranslationTask()
tlr.translate_text_file(src_file_path=src_file, trg_iso='en', trg_file_path=trg_file)
