from pathlib import Path
import yaml
from .environment import SIL_NLP_ENV
from typing import Dict, List, Set, Tuple, Union  

def choose_yes_no_cancel(prompt: str) -> bool:
    prompt = "\n" + prompt + "\nChoose Y, N, or Q to quit: "
    while True:
        choice = input(prompt).strip().lower()
        if choice in ["y", "n", "q"]:
            break
        print("Invalid choice, please choose Y, N, or Q.")
    
    if choice == "y":
        return True
    elif choice == "n":
        return False
    else:
        sys.exit(0)

def confirm_write_config(config, config_file, message):
    if not config_file.is_file():
        with open(config_file, "w", encoding="utf-8") as f:
            yaml.dump(config, f, default_flow_style=False)
        print(f"{message} {config_file} was created.")
    else:
        if choose_yes_no_cancel(f"Would you like to overwrite the current {message} at {config_file}"):
            with open(config_file, "w") as f:
                yaml.dump(config, f, default_flow_style=False)
            print(f"{message} {config_file} was overwritten.")
        else:
            print(f"{message} {config_file} was not overwritten.")


def create_experiment_config(experiment_dir: Path, target: str, source: str, lang_codes: Dict[str, str]):
    config = {
        "data": {
            "corpus_pairs": [
                {
                    "corpus_books": "GEN;MAT;MRK;LUK;JHN;ACT;GAL;EPH;PHP;COL;1TH;2TH;1TI;2TI;TIT;PHM;JAS;1PE;2PE;1JN;2JN;3JN;JUD",
                    "mapping": "mixed_src",
                    "src": source,
                    "test_size": 250,
                    "trg": target,
                    "type": "train,test,val",
                    "val_size": 250,
                }
            ],
            "lang_codes": lang_codes,
            "seed": 111,
        },
        "eval": {
            "detokenize": False,
            "early_stopping": {"min_improvement": 0.1, "steps": 5},
            "per_device_eval_batch_size": 16,
        },
        "infer": {"infer_batch_size": 16, "num_beams": 2},
        "model": "facebook/nllb-200-distilled-1.3B",
        "params": {"label_smoothing_factor": 0.2, "learning_rate": 0.0001, "warmup_steps": 4000},
        "train": {"gradient_accumulation_steps": 4, "per_device_train_batch_size": 16},
    }

    config_file = experiment_dir / "config.yml"
    #confirm_write_config(config, config_file, message="Experiment config file")
    return config_file



def main():
    experiments = {
        'Arop_Mal': ('aps-aArp_2024_07_04', 'mbk-aMal_2024_07_04'),
        'Arop_Sno': ('aps-aArp_2024_07_04', 'sso-aSno_2024_07_04'),
        'tpi_AropFT_Goi': ('tpi-aArpBT_2024_07_04', 'onr-cGoi_2024_07_04'),
        'tpi_AropFT_Wol': ('tpi-aArpBT_2024_07_04', 'onr-cWol_2024_07_04'),
        'Arop_AropFT': ('aps-aArp_2024_07_04', 'tpi-aArpBT_2024_07_04'),
        'tpi_AropFT_Rbr': ('tpi-aArpBT_2024_07_04', 'onr-cRbr_2024_07_04'),
        'tpi_AropFT_Mal': ('tpi-aArpBT_2024_07_04', 'mbk-aMal_2024_07_04'),
        'Arop_Goi': ('aps-aArp_2024_07_04', 'onr-cGoi_2024_07_04'),
        'Arop_Wol': ('aps-aArp_2024_07_04', 'onr-cWol_2024_07_04'),
        'tpi_AropFT_Sno': ('tpi-aArpBT_2024_07_04', 'sso-aSno_2024_07_04'),
        'Arop_Rbr': ('aps-aArp_2024_07_04', 'onr-cRbr_2024_07_04'),
        'tpi_AropFT_bar': ('tpi-aArpBT_2024_07_04', 'bar-bBar_2024_07_04'),
        'tpi_AropFT_sum': ('tpi-aArpBT_2024_07_04', 'sum-bSum_2024_07_04'),
        'Arop_bar': ('aps-aArp_2024_07_04', 'bar-bBar_2024_07_04'),
        'tpi_AropFT_pou': ('tpi-aArpBT_2024_07_04', 'pou-bPou_2024_07_04'),
        'Arop_sum': ('aps-aArp_2024_07_04', 'sum-bSum_2024_07_04'),
        'tpi_AropFT_ram': ('tpi-aArpBT_2024_07_04', 'ram-bRam_2024_07_04'),
        'Arop_pou': ('aps-aArp_2024_07_04', 'pou-bPou_2024_07_04'),
        'Arop_ram': ('aps-aArp_2024_07_04', 'ram-bRam_2024_07_04'),
    }

    lang_codes = {
    'aps': 'aps_Latn',
    'bar': 'bar_Latn',
    'en': 'eng_Latn',
    'eng': 'eng_Latn',
    'mbk': 'mbk_Latn',
    'onr': 'onr_Latn',
    'pou': 'pou_Latn',
    'ram': 'ram_Latn',
    'sso': 'sso_Latn',
    'sum': 'sum_Latn',
    'tpi': 'tpi_Latn',
    }


    experiment_series_dir = SIL_NLP_ENV.mt_experiments_dir / 'FT-Arop'

    prep_commands = []
    train_commands = []


    for item in experiments.items():
        #print(item, type(item))
        experiment, (source, target) = item
        #print(f"Experiment {experiment} has source: {source} and target: {target}")
        print(f"{experiment}")
        experiment_dir = experiment_series_dir / experiment
        
        # Create experiment folder
        experiment_dir.mkdir(parents=False, exist_ok=True)
        create_experiment_config(experiment_dir, target=target, source=source, lang_codes=lang_codes)
        prep_commands.append(f"poetry run python -m silnlp.nmt.experiment --preprocess FT-Arop/{experiment}\n")
        train_commands.append(f"poetry run python -m silnlp.nmt.experiment --save-checkpoints --memory-growth --clearml-queue jobs_backlog FT-Arop/{experiment}\n")

    with open(experiment_series_dir /"Notes2.txt" , "w", encoding='utf-8') as note_file:
        note_file.writelines(prep_commands)
        note_file.write('\n')
        note_file.writelines(train_commands)

if __name__ == "__main__":
    main()