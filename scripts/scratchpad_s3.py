from silnlp.nmt.clearml_experiment import SILExperimentCML

exp = SILExperimentCML(
    name="de-to-en-WMT2020+Bibles_AE/abp-en",
    make_stats=True,  # limited by stats_max_size to process only Bibles
    mixed_precision=True,  # clearML GPU's can handle mixed precision
    memory_growth=False,  # we can allocate all memory all the time
    queue_name="langtech_40gb",
    remote_execution=True,
)
exp.test()
