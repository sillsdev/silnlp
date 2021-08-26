from silnlp.nmt.clearml_experiment import SILExperimentCML

exp = SILExperimentCML(
    name="parent-alignment",
    make_stats=True,  # limited by stats_max_size to process only Bibles
    mixed_precision=True,  # clearML GPU's can handle mixed precision
    memory_growth=False,  # we can allocate all memory all the time
    num_devices=-1,  # get all devices
)
exp.run()
