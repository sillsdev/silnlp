import os
import sys
from pathlib import Path

parent_dir = Path(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(str(parent_dir.parent))
import json
from time import sleep
from urllib import request

import gspread_dataframe as gd
import investigation_utils as iu
import pandas as pd
import streamlit as st
from consts import BOOKS_ABBREVS
from models import Investigation, Status
from utils import check_error, check_required, check_success, expand_books, set_success, simplify_books

from clowder import functions

st.markdown(
    """
<style>
    [data-testid="collapsedControl"] {
        display: none
    }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def get_resources(env):
    with st.spinner("Fetching resources. This might take a few minutes..."):
        try:
            return list(map(lambda fn: fn[:-4], functions.list_resources()))
        except Exception as e:
            import traceback

            traceback.print_exc()
            st.error(f"Something went wrong while fetching resource data. Please try again. Error: {e}")


@st.cache_data(show_spinner=False)
def get_template():
    return pd.read_csv("onboarding-dashboard/config-templates/investigation.csv")


template_df = get_template()


def add_experiment(values: dict) -> None:
    iu.add_experiment_ext(values, st.session_state.clowder_env, st.session_state.current_investigation, template_df)


@st.cache_data(show_spinner=False)
def get_results(results_name: str, investigation_name: str = None, keep_name: bool = False) -> pd.DataFrame:
    if investigation_name is None:
        investigation_name = (
            st.session_state.current_investigation.name
        )  # Don't use defaults to avoid uninitialized current_investigation
    try:
        return iu.get_results_ext(results_name, investigation_name, keep_name, st.session_state.clowder_env)
    except Exception as e:
        import traceback

        traceback.print_exc()
        st.error(f"Something went wrong while fetching results data. Please try again. Error: {e}")
        return pd.DataFrame()


def set_config():
    print(f"Writing config for {st.session_state.current_investigation}")
    try:
        iu.set_config_ext(st.session_state.clowder_env, st.session_state.current_investigation)
    except Exception as e:
        import traceback

        traceback.print_exc()
        st.error(f"Something went wrong while setting up configuration. Please try again. Error: {e}")


@st.cache_data(show_spinner=False)
def get_lang_tags_mapping_data():
    data = {}
    with request.urlopen("https://ldml.api.sil.org/index.html?query=langtags&ext=json") as r:
        data: dict = json.loads(r.read())
    return data


@st.cache_data(show_spinner=False)
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


def sync(rerun: bool = True, container=None):
    try:
        functions.sync(st.session_state.current_investigation.name, env=st.session_state.clowder_env)
        if st.session_state.current_investigation in st.session_state.investigations:
            st.session_state.investigations.remove(st.session_state.current_investigation)
            st.session_state.current_investigation = Investigation.from_clowder(
                iu.sync_ext(st.session_state.clowder_env, st.session_state.current_investigation)
            )
            st.session_state.investigations.append(st.session_state.current_investigation)
            if rerun:
                st.rerun()
    except Exception as e:
        import traceback

        traceback.print_exc()
        if container is not None:
            container.error(f"Something went wrong while syncing. Please try again. Error: {e}")
        else:
            st.error(f"Something went wrong while syncing. Please try again. Error: {e}")


@st.cache_data(show_spinner=False)
def get_drafts(investigation_name: str):
    try:
        return iu.get_drafts_ext(st.session_state.clowder_env, investigation_name)
    except Exception as e:
        import traceback

        traceback.print_exc()
        st.error(f"Something went wrong while fetching drafts. Please try again. Error: {e}")


@st.experimental_fragment
def render_stats_section():
    if st.session_state.current_investigation.status.value >= Status.GatheredStats.value:
        st.session_state.results_stats = get_results(
            "verse_percentages.csv", st.session_state.current_investigation.name
        )
        if len(st.session_state.results_stats.index) > 0:
            st.dataframe(st.session_state.results_stats.style.format(precision=0))
        else:
            st.write("**No results found**")
    with st.form(key=f"{st.session_state.current_investigation.id}-gather-stats"):
        texts = st.multiselect("Texts", resources)
        check_error("stats")
        if st.form_submit_button("Run", type="primary"):
            get_results.clear("verse_percentages.csv", st.session_state.current_investigation.name)
            check_required("stats", texts)
            iu.create_stats_experiment(texts, st.session_state.clowder_env, st.session_state.current_investigation)
            with st.spinner("This might take a few minutes..."):
                try:
                    functions.run(st.session_state.current_investigation.name, experiments=["stats"], force_rerun=True)
                    set_success("stats", "Gathering stats has been successfully run!")
                except Exception as e:
                    import traceback

                    traceback.print_exc()
                    st.error(f"Something went wrong while attempting to run experiment. Please try again. Error: {e}")
                sync()
        check_success("stats")


@st.experimental_fragment
def render_alignment_section():
    if st.session_state.current_investigation.status.value >= Status.Aligned.value:
        st.session_state.results_align = get_results("corpus-stats.csv", st.session_state.current_investigation.name)
        if len(st.session_state.results_align.index) > 0:
            st.dataframe(
                st.session_state.results_align.style.highlight_max(
                    subset=st.session_state.results_align.select_dtypes(include="float64").columns, color="green"
                ).format(
                    lambda s: (
                        round(s, 3)
                        if not isinstance(s, str) and s // 1 != s
                        else int(s) if not isinstance(s, str) else s
                    ),
                    precision=3,
                )
            )
        else:
            st.write("**No results found**")
    with st.form(key=f"{st.session_state.current_investigation.id}-run-alignments"):
        default_sources, target_name = iu.split_target_sources(st.session_state.results_stats)
        training_sources = st.multiselect(
            "Alignment sources",
            resources,
            default=default_sources,
        )
        target_index = resources.index(target_name) if target_name is not None else None
        training_target = st.selectbox("Alignment target", resources, index=target_index)
        default_books = iu.get_default_books(training_sources, training_target, st.session_state.results_stats)
        books = st.multiselect("Books to align on", BOOKS_ABBREVS, default=default_books)
        alignments_already_running = iu.alignments_is_running_ext(
            st.session_state.clowder_env, st.sesion_state.current_investigation
        )
        if alignments_already_running:
            st.write("Your alignments are running. Check back in 15 minutes.")
        check_error("align")
        _button_text = "Run" if not alignments_already_running else "Cancel"
        _button_type = "primary" if not alignments_already_running else "secondary"
        if st.form_submit_button(
            _button_text,
            type=_button_type,
        ):
            if alignments_already_running:
                with st.spinner("This might take a few minutes..."):
                    try:
                        functions.cancel(st.session_state.current_investigation.name)
                        set_success("align", "Alignments have been successfully canceled!")
                    except Exception as e:
                        import traceback

                        traceback.print_exc()
                        st.error(
                            f"Something went wrong while attempting to cancel experiment. Please try again. Error: {e}"
                        )
                    sync()
            check_required("align", training_sources, training_target, books)
            iu.create_alignment_experiment(
                training_sources,
                training_target,
                books,
                st.session_state.clowder_env,
                st.session_state.current_investigation,
            )
            with st.spinner("This might take a few minutes..."):
                get_results.clear("corpus-stats.csv", st.session_state.current_investigation.name)
                get_results.clear("tokenization_stats.csv", st.session_state.current_investigation.name, keep_name=True)
                try:
                    functions.run(st.session_state.current_investigation.name, experiments=["align"], force_rerun=True)
                    set_success("align", "Alignments are successfully running! Check back after 15 minutes.")
                except Exception as e:
                    import traceback

                    traceback.print_exc()
                    st.error(f"Something went wrong while attempting to run experiment. Please try again. Error: {e}")
                sync()
        check_success("align")


@st.experimental_fragment
def render_model_section():
    if st.session_state.current_investigation.status.value >= Status.RanModels.value:
        st.session_state.results_models = get_results(
            "scores-best", st.session_state.current_investigation.name, keep_name=True
        )
        if len(st.session_state.results_models) > 0:
            st.dataframe(
                st.session_state.results_models.style.highlight_max(
                    subset=st.session_state.results_models.select_dtypes(include="number").columns, color="green"
                ).format(precision=2)
            )
        else:
            st.write("**No results found**")
    st.write("Models")
    models = st.session_state.models if "models" in st.session_state else []
    if len(models) == 0:
        st.write("*No models added*")
    for model in models:
        with st.container(border=True):
            c1, c2 = st.columns([5, 1])
            with c1:
                st.write(f'**{model["name"]}**')  # TODO kill more options at rerun
            with c2:
                if st.button("Remove", type="primary", key=f'remove_{model["name"]}'):
                    st.session_state.models.remove(model)
                    st.rerun()
            training_source_display = st.selectbox(
                "Training source",
                resources,
                key=f"ts_{model['name']}",
                index=resources.index(model["training_source"]),
                disabled=True,
            )
            training_target_display = st.selectbox(
                "Training target",
                resources,
                key=f"tt_{model['name']}",
                index=resources.index(model["training_target"]),
                disabled=True,
            )
            if "bt_src" in model:
                backtranslation_source_display = st.selectbox(
                    "Secondary training source",
                    resources,
                    key=f"bts_{model['name']}",
                    index=resources.index(model["bt_src"]),
                    disabled=True,
                )
                backtranslation_books_display = st.multiselect(
                    "Books to train on from secondary source",
                    BOOKS_ABBREVS,
                    key=f"btb_{model['name']}",
                    disabled=True,
                    default=model["bt_books"],
                )
            books = st.multiselect(
                "Books to train on",
                BOOKS_ABBREVS,
                key=f"b_{model['name']}",
                default=model["books"],
                disabled=True,
            )
    st.write("Add a model" if len(models) == 0 else "Add another model")
    models_form = st.form(key=f"{st.session_state.current_investigation.id}-run-models")
    with models_form:
        _source_index = (
            resources.index(
                st.session_state.results_align.loc[
                    [st.session_state.results_align["align_score"].idxmax()], ["src_project"]
                ].values[0]
            )
            if st.session_state.results_align is not None and len(st.session_state.results_align.index) > 0
            else None
        )
        training_source = st.selectbox(
            "Training source",
            resources,
            index=_source_index,
        )
        _target_index = (
            resources.index(st.session_state.results_align.loc[[0], ["trg_project"]].values[0])
            if st.session_state.results_align is not None and len(st.session_state.results_align.index) > 0
            else None
        )
        training_target = st.selectbox(
            "Training target",
            resources,
            index=_target_index,
        )
        default_books = []
        if (
            st.session_state.results_stats is not None
            and len(st.session_state.results_stats.index) > 0
            and st.session_state.results_align is not None
            and len(st.session_state.results_align.index) > 0
        ):
            if (
                st.session_state.results_align.loc[[0], ["trg_project"]].values[0]
                in st.session_state.results_stats["file"].values
            ):
                name = st.session_state.results_align.loc[[0], ["trg_project"]].values[0]
                name = name[0]
                for book in set(BOOKS_ABBREVS) - set(["OT", "NT", "DT"]):
                    if (
                        st.session_state.results_stats[st.session_state.results_stats["file"] == name][book] >= 93
                    ).all():  # TODO more robust
                        default_books.append(book)
        default_books = simplify_books(default_books)
        books = st.multiselect(
            "Books to train on", BOOKS_ABBREVS, default=default_books if len(default_books) > 0 else None
        )

    backtranslation_source = None
    backtranslation_books = None
    if st.toggle("More options"):
        res_index = (
            resources.index(
                st.session_state.results_align.loc[
                    [st.session_state.results_align["align_score"].idxmax()], ["src_project"]
                ].values[0]
            )
            if st.session_state.results_align is not None and len(st.session_state.results_align.index) > 0
            else None
        )
        backtranslation_source = models_form.selectbox(
            "Secondary training source",
            resources,
            index=res_index,
        )
        backtranslation_books = models_form.multiselect(
            "Books to train on from secondary source", BOOKS_ABBREVS
        )  # TODO smart defaults

    check_error("models")
    if models_form.form_submit_button("Add model"):
        check_required("models", training_source, training_target, books)
        if "models" not in st.session_state:
            st.session_state.models = []
        is_mixed_src = backtranslation_source not in ["", None, []] and backtranslation_books not in [
            "",
            None,
            [],
        ]
        model = dict()
        if not is_mixed_src:
            model["training_source"] = training_source
            model["training_target"] = training_target
            model["books"] = books
            model["name"] = f"NLLB.1.3B.{training_source}-{training_target}.[{','.join(books)}]"
        else:
            model["training_source"] = training_source
            model["training_target"] = training_target
            model["books"] = simplify_books(list(set(expand_books(books)) - set(expand_books(backtranslation_books))))
            model["bt_src"] = backtranslation_source
            model["bt_books"] = backtranslation_books
            model["name"] = (
                f"NLLB.1.3B.{training_source}+{backtranslation_source}-{training_target}.[{','.join(books)}]"
            )
        if len(list(filter(lambda m: m["name"] == model["name"], st.session_state.models))) > 0:
            st.session_state.errors["models"] = "A model with this configuration has already been added"
            st.rerun()
        st.session_state.models.append(model)
        st.rerun()
    models_already_running = iu.model_is_running_ext(
        st.session_state.current_investigation, st.session_state.clowder_env
    )
    if models_already_running:
        st.toast("Your models are training. Check back after a few hours.")
        st.write("Your models are training. Check back after a few hours.")
    if st.button(
        "Run models" if not models_already_running else "Cancel",
        type="primary" if not models_already_running else "secondary",
        disabled=not models_already_running and len(models) == 0,
    ):
        if models_already_running:
            with st.spinner("This might take a few minutes..."):
                try:
                    functions.cancel(st.session_state.current_investigation.name)
                    set_success("models", "Models have been successfully canceled!")
                except Exception as e:
                    import traceback

                    traceback.print_exc()
                    st.error(
                        f"Something went wrong while attempting to cancel experiment. Please try again. Error: {e}"
                    )
                sync()
        exps = iu.create_model_experiments(
            models, training_target, st.session_state.clowder_env, st.session_state.current_investigation
        )
        with st.spinner("This might take a few minutes..."):
            get_results.clear("scores-best", st.session_state.current_investigation.name, keep_name=True)
            try:
                functions.run(st.session_state.current_investigation.name, experiments=exps, force_rerun=True)
                set_success("models", "Models are successfully running. Check back after a few hours!")
            except Exception as e:
                import traceback

                traceback.print_exc()
                st.error(f"Something went wrong while attempting to run experiment. Please try again. Error: {e}")
            sync()
    check_success("models")


@st.experimental_fragment
def render_draft_section():
    if st.session_state.current_investigation.status.value >= Status.Drafted.value:
        drafts = get_drafts(st.session_state.current_investigation.name)
        st.write("*Available Drafts*")
        i = 0
        for model_name, model_drafts in drafts.items():
            i += 1
            with st.container(border=True):
                st.write(f"**{model_name}**")
                with st.container(border=True):
                    for draft_name, draft_content in model_drafts.items():
                        c1, c2 = st.columns([4, 1])
                        with c1:
                            st.write(draft_name)
                        with c2:
                            print(f"Key = ", f"{model_name}_{draft_name}_{i}")
                            st.download_button(
                                "Download",
                                data=draft_content,
                                file_name=draft_name,
                                key=f"{model_name}_{draft_name}_{i}",
                            )
    with st.form(key=f"{st.session_state.current_investigation.id}-draft-books"):
        drafting_source = st.selectbox("Drafting source", resources, index=None)
        model_options = []
        idx_of_best_model = None
        if st.session_state.results_models is not None and len(st.session_state.results_models.index) > 0:
            model_options = list(st.session_state.results_models["name"])
            idx_of_best_model = int(st.session_state.results_models["CHRF3"].idxmax())
        model = st.selectbox("Model", model_options, index=idx_of_best_model)
        books = st.multiselect("Books to draft", BOOKS_ABBREVS)
        draft_already_running = iu.draft_is_running(
            st.session_state.clowder_env, st.session_state.current_investigation
        )
        if draft_already_running:
            st.write("Your drafting job is running. Check back after a few hours.")
        check_error("drafts")
        _button_text = "Run" if not draft_already_running else "Cancel"
        _button_type = "primary" if not draft_already_running else "secondary"
        if st.form_submit_button(
            _button_text,
            type=_button_type,
        ):
            if draft_already_running:
                with st.spinner("This might take a few minutes..."):
                    try:
                        functions.cancel(st.session_state.current_investigation.name)
                        set_success("drafts", "Drafting jobs have been successfully canceled!")
                    except Exception as e:
                        import traceback

                        traceback.print_exc()
                        st.error(
                            f"Something went wrong while attempting to cancel drafting job. Please try again. Error: {e}"
                        )
                    sync()
            check_required("draft", drafting_source, model, books)
            draft_name = iu.create_draft_experiment(
                books, model, drafting_source, st.session_state.clowder_env, st.session_state.current_investigation
            )
            with st.spinner("This might take a few minutes..."):
                try:
                    functions.run(
                        st.session_state.current_investigation.name, experiments=[draft_name], force_rerun=True
                    )
                    set_success("drafts", "Drafting jobs are successfully running! Check back after a few hours.")
                except Exception as e:
                    import traceback

                    traceback.print_exc()
                    st.error(f"Something went wrong while attempting to run experiment. Please try again. Error: {e}")
                sync()
        check_success("drafts")


# TODO DESCRIPTIVE TEXT
if "current_investigation" in st.session_state:
    if st.session_state.google_auth is not None and st.session_state.google_auth.access_token_expired:
        # st.session_state.google_auth.Refresh() TODO
        st.session_state.set_up = False
        del st.session_state.google_auth
        st.switch_page("pages/LogIn.py")
    if (
        "synced_dict" not in st.session_state
        or st.session_state.current_investigation.name not in st.session_state.synced_dict
        or not st.session_state.synced_dict[st.session_state.current_investigation.name]
    ):
        with st.spinner(f"Fetching up-to-date data on {st.session_state.current_investigation.name}..."):
            sync(rerun=False)
        if "synced_dict" not in st.session_state:
            st.session_state.synced_dict = {}
        st.session_state.synced_dict[st.session_state.current_investigation.name] = True
    if st.session_state.current_investigation.status == Status.Created:
        set_config()
    resources = get_resources(str(st.session_state.clowder_env))
    st.session_state.results_stats = None
    st.session_state.results_align = None
    st.session_state.results_models = None
    st.page_link("Home.py", label="Back")
    c1, c2 = st.columns(2)
    st.title(st.session_state.current_investigation.name)
    st.progress(
        st.session_state.current_investigation.status.to_percent(), st.session_state.current_investigation.status.name
    )
    with st.container():
        with st.expander(
            "**Gather Stats**", st.session_state.current_investigation.status in [Status.Created, Status.GatheredStats]
        ):
            render_stats_section()
        if st.session_state.current_investigation.status.value >= Status.GatheredStats.value:
            with st.expander(
                "**Run Alignments**",
                st.session_state.current_investigation.status in [Status.GatheredStats, Status.Aligned],
            ):
                render_alignment_section()
        if st.session_state.current_investigation.status.value >= Status.Aligned.value:
            with st.expander(
                "**Run Models**", st.session_state.current_investigation.status in [Status.Aligned, Status.RanModels]
            ):
                render_model_section()
        if st.session_state.current_investigation.status.value >= Status.RanModels.value:
            with st.expander("**Draft Books**", st.session_state.current_investigation.status in [Status.RanModels]):
                render_draft_section()
        c1, c2, _ = st.columns([1, 1, 15])
        error_container = st.container()
        with c1:
            if st.button("⟳"):
                sync(container=error_container)
        with c2:
            if st.button("🗑️", type="primary", key=f"{st.session_state.current_investigation.id}_delete_button"):
                st.session_state.investigation_to_delete = st.session_state.current_investigation
                st.switch_page("pages/DeleteInvestigation.py")
        st.write(f"More detailed information at <{functions.urlfor(st.session_state.current_investigation.name)}>")
else:
    st.switch_page("Home.py")
