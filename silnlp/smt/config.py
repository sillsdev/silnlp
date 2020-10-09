import os
import yaml

from nlp.common.utils import get_mt_root_dir, merge_dict


def load_config(exp_name: str) -> dict:
    root_dir = get_mt_root_dir(exp_name)
    config_path = os.path.join(root_dir, "config.yml")

    config: dict = {"test_size": 250}

    with open(config_path, "r", encoding="utf-8") as file:
        loaded_config = yaml.safe_load(file)
        return merge_dict(config, loaded_config)
