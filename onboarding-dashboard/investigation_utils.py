import json
import os
import sys
from pathlib import Path
from urllib import request

import gspread_dataframe as gd
import pandas as pd

from clowder import functions
from clowder.environment import ClowderEnvironment
from clowder.investigation import Investigation

parent_dir = Path(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(str(parent_dir.parent))
from consts import BOOKS_ABBREVS
from utils import simplify_books

template_df = pd.read_csv("onboarding-dashboard/config-templates/investigation.csv")

config_data = ""
with open("onboarding-dashboard/config-templates/config.jinja-yml", "r") as f:
    config_data = f.read()


def get_lang_tags_mapping_data():
    data = {}
    with request.urlopen("https://ldml.api.sil.org/index.html?query=langtags&ext=json") as r:
        data: dict = json.loads(r.read())
    return data


def get_lang_tag_mapping(tag: str):
    if tag == "cmn":
        tag = "zh"  # Follow Serval behavior
    data = get_lang_tags_mapping_data()
    for obj in data:
        if "iso639_3" not in obj:
            continue
        if obj["tag"].split("-")[0] == tag or (
            "tags" in obj and tag in list(map(lambda t: t.split("-")[0], obj["tags"]))
        ):
            script = obj["script"]
            if script == "Kore":
                script = "Hang"  # Follow Serval behavior
            return f'{obj["iso639_3"]}_{script}'
    raise ValueError("Language tag does not exist")


def add_experiment_ext(values: dict, clowder_env: ClowderEnvironment, investigation: Investigation):
    try:
        with functions._lock:
            import clowder.investigation as inv

            inv.ENV = clowder_env
            functions.ENV = clowder_env
            sheet = functions.ENV.gc.open_by_key(functions.ENV.get_investigation(investigation.name).sheet_id)
            df: pd.DataFrame = gd.get_as_dataframe(sheet.sheet1)
    except Exception as e:
        st.error(f"Something went wrong while adding experiment. Please try again. Error: {e}")
        return
    df = df.dropna(how="all")
    df = df.dropna(how="all", axis="columns")
    if len(df) == 0:
        df = pd.DataFrame(columns=template_df.columns)
    df = df[df["name"] != values["name"]]
    if "draft" in values["name"]:
        df = df[df["type"] != "draft"]
    temp_df = pd.DataFrame(columns=template_df.columns)
    for name, val in values.items():
        temp_df.loc[0, name] = val
    df = pd.concat([df, temp_df], ignore_index=True)
    gd.set_with_dataframe(sheet.sheet1, df)


def get_results_ext(
    results_name: str, investigation_name: str, keep_name: bool, clowder_env: ClowderEnvironment
) -> pd.DataFrame:
    with functions._lock:
        import clowder.investigation as inv

        inv.ENV = clowder_env
        functions.ENV = clowder_env
        sheet = functions.ENV.gc.open_by_key(functions.ENV.get_investigation(investigation_name).sheet_id)
        result_sheet = list(filter(lambda w: w.title == results_name, sheet.worksheets()))[0]
        df: pd.DataFrame = gd.get_as_dataframe(result_sheet)
    df = df.dropna(how="all")
    df = df.dropna(how="any", axis="columns")
    if not keep_name:
        df.drop(axis="columns", labels="name", inplace=True)
    return df


def set_config_ext(clowder_env: ClowderEnvironment, investigation: Investigation):
    with functions._lock:
        import clowder.investigation as inv

        inv.ENV = clowder_env
        functions.ENV = clowder_env
        functions.ENV._write_gdrive_file_in_folder(
            functions.ENV.get_investigation(investigation.name).id,
            "config.yml",
            config_data,
        )


def sync_ext(clowder_env: ClowderEnvironment, investigation):
    with functions._lock:
        import clowder.investigation as inv

        inv.ENV = clowder_env
        functions.ENV = clowder_env
        return functions.ENV.get_investigation(investigation.name)


def get_drafts_ext(clowder_env: ClowderEnvironment, investigation_name):
    with functions._lock:
        import clowder.investigation as inv

        inv.ENV = clowder_env
        functions.ENV = clowder_env
        investigation = functions.ENV.get_investigation(investigation_name)
        drafts_folder_id = functions.ENV._dict_of_gdrive_files(investigation.experiments_folder_id)["drafts"]["id"]
        drafts = dict()
        for name, folder in functions.ENV._dict_of_gdrive_files(drafts_folder_id).items():
            drafts[name] = {}
            for filename, file in functions.ENV._dict_of_gdrive_files(folder["id"]).items():
                drafts[name][filename] = functions.ENV._read_gdrive_file_as_string(file["id"])
    return drafts


def create_stats_experiment(texts: list[str], clowder_env: ClowderEnvironment, inv: Investigation):
    stats_row = template_df[template_df["name"] == "stats"]
    stats_setup = stats_row.to_dict(orient="records")[0]
    stats_setup["entrypoint"] = stats_setup["entrypoint"].replace(
        "$FILES", ";".join(list(map(lambda t: f"{t}.txt", texts)))
    )
    add_experiment_ext(stats_setup, clowder_env, investigation=inv)


def split_target_sources(results_stats: pd.DataFrame):
    if results_stats is None or "file" not in results_stats.columns:
        return None, None
    OT_score = results_stats["OT"].mean()
    NT_score = results_stats["NT"].mean()
    targ_index = None
    if NT_score == 100 or (OT_score > NT_score and OT_score != 100):
        targ_index = results_stats["OT"].idxmin()
    else:
        targ_index = results_stats["NT"].idxmin()
    return (
        [f for f in results_stats.drop(targ_index)["file"]],
        results_stats.iloc[targ_index]["file"],
    )


def get_default_books(training_sources, training_target, results_stats):
    default_books = None
    if results_stats is not None and "file" in results_stats.columns:
        texts_series = pd.DataFrame(list(set(training_sources) | set([training_target])), columns=["file"])
        results_stats_df_copy = results_stats.copy()
        results_stats_df_copy["file"] = results_stats_df_copy["file"].apply(lambda f: f[:-4])
        rel_stats = pd.merge(texts_series, results_stats_df_copy, how="inner")
        default_books = BOOKS_ABBREVS.copy()
        for _, row in rel_stats.iterrows():
            to_remove = set(["OT", "NT", "DT"])
            for book in default_books:
                if row[book] < 93:  # TODO more robust
                    to_remove.add(book)
            for book in to_remove:
                if book in default_books:
                    default_books.remove(book)
        default_books = simplify_books(default_books)
        if len(default_books) == 0:
            default_books = None
    return default_books


def alignments_is_running_ext(clowder_env, investigation):
    with functions._lock:
        import clowder.investigation as inv

        inv.ENV = clowder_env
        functions.ENV = clowder_env
        return (
            len(
                list(
                    filter(
                        lambda kv: "align" in kv[0] and kv[1].get("status", None) in ["in_progress", "queued"],
                        functions.ENV.get_investigation(investigation.name).experiments.items(),
                    )
                )
            )
            > 0
        )


def create_alignment_experiment(training_sources, training_target, books, clowder_env, investigation):
    align_row = template_df[template_df["name"] == "align"]
    align_setup = align_row.to_dict(orient="records")[0]
    align_setup["src_texts"] = ";".join(training_sources)
    align_setup["trg_texts"] = training_target
    if len(books) > 0:
        align_setup["align_set"] = ",".join(books)
    add_experiment_ext(align_setup, clowder_env, investigation)


def model_is_running_ext(investigation, clowder_env):
    with functions._lock:
        import clowder.investigation as inv

        inv.ENV = clowder_env
        functions.ENV = clowder_env
        return (
            len(
                list(
                    filter(
                        lambda kv: "NLLB" in kv[0]
                        and "draft" not in kv[0]
                        and kv[1].get("status", None) in ["in_progress", "queued"],
                        functions.ENV.get_investigation(investigation.name).experiments.items(),
                    )
                )
            )
            > 0
        )


def create_model_experiments(models, training_target, clowder_env, investigation):
    exps = []
    model_row = template_df[template_df["name"] == "model"]
    for model in models:
        model_setup = model_row.to_dict(orient="records")[0]
        exps.append(model["name"])
        model_setup["name"] = model["name"]
        model_setup["src"] = model["training_source"]
        model_setup["trg"] = model["training_target"]
        src_lang = model["training_source"].split("-")[0]
        trg_lang = training_target.split("-")[0]
        src_lang_tag_mapping = get_lang_tag_mapping(src_lang)
        trg_lang_tag_mapping = get_lang_tag_mapping(trg_lang)
        model_setup["src_lang"] = f"{src_lang}: {src_lang_tag_mapping}"
        model_setup["trg_lang"] = f"{trg_lang}: {trg_lang_tag_mapping}"
        model_setup["train_set"] = ",".join(model["books"])
        if "bt_books" in model:
            model_setup["bt_books"] = ",".join(model["bt_books"])
            model_setup["bt_src"] = model["bt_src"]
        add_experiment_ext(model_setup, clowder_env, investigation)
    return exps


def draft_is_running(clowder_env, investigation):
    with functions._lock:
        import clowder.investigation as inv

        inv.ENV = clowder_env
        functions.ENV = clowder_env
        _running_draft_list = list(
            filter(
                lambda kv: "draft" in kv[0] and kv[1].get("status", None) in ["in_progress", "queued"],
                functions.ENV.get_investigation(investigation.name).experiments.items(),
            )
        )
        return len(_running_draft_list) > 0


def create_draft_experiment(books, model, drafting_source, clowder_env, investigation):
    books_string = ";".join(books)
    draft_row = template_df[template_df["name"] == "draft"]
    draft_setup = draft_row.to_dict(orient="records")[0]
    draft_name = f"{model}_draft_{books_string}_{drafting_source}"
    draft_setup["name"] = draft_name
    draft_setup["entrypoint"] = (
        draft_setup["entrypoint"]
        .replace("$SRC_ISO", get_lang_tag_mapping(model.split("-")[0].split("NLLB.1.3B.")[1]))
        .replace("$SRC", "".join(drafting_source.split("-")[1:]))
        .replace("$TRG_ISO", get_lang_tag_mapping(model.split("-")[2]))
        .replace("$BOOKS", books_string)
    )
    add_experiment_ext(draft_setup, clowder_env, investigation)
    return draft_name
