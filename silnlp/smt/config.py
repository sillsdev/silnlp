import os
import yaml

from nlp.common.utils import get_root_dir


def merge_dict(dict1: dict, dict2: dict) -> dict:
    for key, value in dict2.items():
        if isinstance(value, dict):
            dict1[key] = merge_dict(dict1.get(key, {}), value)
        else:
            dict1[key] = value
    return dict1


def load_config(exp_name: str) -> dict:
    root_dir = get_root_dir(exp_name)
    config_path = os.path.join(root_dir, "config.yml")

    config: dict = {"test_size": 250}

    with open(config_path, "r", encoding="utf-8") as file:
        loaded_config = yaml.safe_load(file)
        return merge_dict(config, loaded_config)
