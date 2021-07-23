import os
import gc
import logging


def init_file_logger(exp_folder: str):
    LOGGER = logging.getLogger()
    LOGGER.setLevel(logging.INFO)
    LOGGER.handlers = []  # removes all handlers
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    fh = logging.FileHandler(os.path.join(exp_folder, "log.txt"))
    fh.setFormatter(formatter)
    LOGGER.addHandler(fh)


def compare_folders(truth_folder: str, computed_folder: str):
    truth_files = os.listdir(truth_folder)
    for tf in truth_files:
        tfp = os.path.join(truth_folder, tf)
        cfp = os.path.join(computed_folder, tf)
        assert os.path.isfile(cfp), "The file " + tf + " should have been but was not created."
        if tf == "log.txt":
            with open(tfp, "r", encoding="utf-8") as f:
                tf_content = f.readlines()
            with open(cfp, "r", encoding="utf-8") as f:
                cf_content = f.readlines()
            # remove the timestamp from the logfile
            tf_content = [l[26:] for l in tf_content]
            cf_content = [l[26:] for l in cf_content]
            assert len(cf_content) == len(
                tf_content
            ), f"Log entry has {len(cf_content)} lines but should have {len(tf_content)} lines"
            for i in range(len(tf_content)):
                assert tf_content[i] == cf_content[i], (
                    "Log entry line " + str(i) + " should be:\n  " + tf_content[i] + "\nbut is:\n  " + cf_content[i]
                )
        elif tf.endswith((".xml", ".txt", ".csv", ".json", ".vocab")):
            with open(tfp, "r", encoding="utf-8") as f:
                tf_content = f.readlines()
            with open(cfp, "r", encoding="utf-8") as f:
                cf_content = f.readlines()
            for i in range(len(tf_content)):
                # normalize unix and PC endings
                assert (
                    tf_content[i].strip() == cf_content[i].strip()
                ), f"line {i} in {tf} should be:\n  {tf_content[i]}\nbut is:\n  {cf_content[i]}"
        else:
            with open(tfp, "rb") as f:
                tf_content = f.readlines()
            with open(cfp, "rb") as f:
                cf_content = f.readlines()
            assert tf_content == cf_content, f"The file {tf} was created differently."