import logging
import os
from typing import List


def init_file_logger(exp_folder: str) -> None:
    LOGGER = logging.getLogger()
    LOGGER.setLevel(logging.INFO)
    LOGGER.handlers = []  # removes all handlers
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    fh = logging.FileHandler(os.path.join(exp_folder, "log.txt"))
    fh.setFormatter(formatter)
    LOGGER.addHandler(fh)


def read_text_file(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return f.readlines()


def read_binary_file(path: str) -> List[bytes]:
    with open(path, "rb") as f:
        return f.readlines()


def compare_folders(truth_folder: str, computed_folder: str) -> None:
    truth_filenames = os.listdir(truth_folder)
    for truth_filename in truth_filenames:
        truth_file_path = os.path.join(truth_folder, truth_filename)
        computed_file_path = os.path.join(computed_folder, truth_filename)
        assert os.path.isfile(computed_file_path), (
            "The file " + truth_filename + " should have been but was not created."
        )
        if truth_filename == "log.txt":
            truth_file_text_content = read_text_file(truth_file_path)
            computed_file_text_content = read_text_file(computed_file_path)
            # remove the timestamp from the logfile
            truth_file_text_content = [l[26:] for l in truth_file_text_content]
            computed_file_text_content = [l[26:] for l in computed_file_text_content]
            assert len(computed_file_text_content) == len(
                truth_file_text_content
            ), f"Log entry has {len(computed_file_text_content)} lines but should have {len(truth_file_text_content)} lines"
            for i in range(len(truth_file_text_content)):
                assert truth_file_text_content[i] == computed_file_text_content[i], (
                    "Log entry line "
                    + str(i)
                    + " should be:\n  "
                    + truth_file_text_content[i]
                    + "\nbut is:\n  "
                    + computed_file_text_content[i]
                )
        elif truth_filename.endswith((".xml", ".txt", ".csv", ".json", ".vocab")):
            truth_file_text_content = read_text_file(truth_file_path)
            computed_file_text_content = read_text_file(computed_file_path)
            for i in range(len(truth_file_text_content)):
                # normalize unix and PC endings
                assert (
                    truth_file_text_content[i].strip() == computed_file_text_content[i].strip()
                ), f"line {i} in {truth_filename} should be:\n  {truth_file_text_content[i]}\nbut is:\n  {computed_file_text_content[i]}"
        elif truth_filename.endswith(".model"):
            # don't compare models - just keep going.
            pass
        else:
            truth_file_bin_content = read_binary_file(truth_file_path)
            computed_file_bin_content = read_binary_file(computed_file_path)
            assert (
                truth_file_bin_content == computed_file_bin_content
            ), f"The file {truth_filename} was created differently."
