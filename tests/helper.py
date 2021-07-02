import os
import gc
import logging


def init_file_logger(exp_folder: str):
    LOGGER = logging.getLogger()
    LOGGER.setLevel(logging.INFO)
    LOGGER.handlers = []  # removes all handlers
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    LOGGER.addHandler(ch)
    fh = logging.FileHandler(os.path.join(exp_folder, "log.txt"))
    fh.setFormatter(formatter)
    LOGGER.addHandler(fh)


def remove_previously_created_files(folder: str, extensions_to_delete=("txt", "json")):
    gc.collect()
    for file in os.listdir(folder):
        if file.lower().endswith(extensions_to_delete):
            os.remove(folder / file)


def compare_folders(truth_folder: str, computed_folder: str):
    truth_files = os.listdir(truth_folder)
    for tf in truth_files:
        tfp = os.path.join(truth_folder, tf)
        cfp = os.path.join(computed_folder, tf)
        assert os.path.isfile(cfp), "The file " + tf + " should have been but was not created."
        if tf == "log.txt":
            tf_content = open(tfp, "r", encoding="utf-8").readlines()
            cf_content = open(cfp, "r", encoding="utf-8").readlines()
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
        elif tf.endswith((".src.txt", ".trg.txt")):
            tf_content = ["".join(l.split()) for l in open(tfp, "r", encoding="utf-8").readlines()]
            cf_content = ["".join(l.split()) for l in open(cfp, "r", encoding="utf-8").readlines()]
            for i in range(len(tf_content)):
                # remove all whitespace, as it changes as per sentencepiece and is hard to control.
                assert (
                    tf_content[i] == cf_content[i]
                ), f"line {i} in {tf} should be:\n  {tf_content[i]}\nbut is:\n  {cf_content[i]}"
        elif tf.endswith((".xml", ".txt", ".csv", ".json")):
            tf_content = open(tfp, "r", encoding="utf-8").readlines()
            cf_content = open(cfp, "r", encoding="utf-8").readlines()
            for i in range(len(tf_content)):
                # normalize unix and PC endings
                assert (
                    tf_content[i].strip() == cf_content[i].strip()
                ), f"line {i} in {tf} should be:\n  {tf_content[i]}\nbut is:\n  {cf_content[i]}"
        elif tf.endswith((".vocab")):
            # make sure it's not too different.
            tf_set = set([l.split()[0] for l in open(tfp, "r", encoding="utf-8").readlines()])
            cf_set = set([l.split()[0] for l in open(cfp, "r", encoding="utf-8").readlines()])
            diff_set = (tf_set - cf_set) | (cf_set - tf_set)
            # Sentencepiece may not always make it excaclty the same.  If there is a big change, flag it.
            assert (
                len(diff_set) < len(tf_set) * 0.01
            ), f"There are {len(diff_set)} differences in the file {tf}.  They are: {diff_set}"
        elif tf.endswith((".model")):
            # there is no good way to compare models, other than that they are created.
            continue
        else:
            tf_content = open(tfp, "rb").read()
            cf_content = open(cfp, "rb").read()
            assert tf_content == cf_content, f"The file {tf} was created differently."