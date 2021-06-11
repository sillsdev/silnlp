import os
from pathlib import Path
import gc


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
        assert cfp.is_file(), "The file " + tf + " should have been but was not created."
        if tf == "log.txt":
            tf_content = tfp.open("r", encoding="utf-8").readlines()
            cf_content = cfp.open("r", encoding="utf-8").readlines()
            # remove the timestamp from the logfile
            tf_content = [l[26:] for l in tf_content]
            cf_content = [l[26:] for l in cf_content]
            for i in range(len(tf_content)):
                assert tf_content[i] == cf_content[i], (
                    "Log entry line " + str(i) + " should be:\n  " + tf_content[i] + "\nbut is:\n  " + cf_content[i]
                )
        elif tf.endswith((".xml", ".txt", ".csv", ".json")):
            tf_content = open(tfp, "r", encoding="utf-8").readlines()
            cf_content = open(cfp, "r", encoding="utf-8").readlines()
            for i in range(len(tf_content)):
                # normalize unix and PC endings
                assert tf_content[i].strip() == cf_content[i].strip(), (
                    "line " + str(i) + " in " + tf + " should be:\n  " + tf_content[i] + "\nbut is:\n  " + cf_content[i]
                )
        else:
            tf_content = open(tfp, "rb").read()
            cf_content = open(cfp, "rb").read()
            assert tf_content == cf_content, "The file " + tf + " was created differently."