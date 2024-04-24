import os
import sys
from pathlib import Path
parent_dir = Path(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(str(parent_dir.parent))
from clowder import functions
from models import Investigation, Status
from consts import BOOKS_ABBREVS
import streamlit as st
from time import sleep, time
import pandas as pd
import gspread_dataframe as gd
from pprint import pprint
import numpy as np
from urllib import request
import json
from utils import check_error, check_required, simplify_books, expand_books

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
def get_resources():
    return list(map(lambda fn: fn[:-4], functions.list_resources()))

@st.cache_data(show_spinner=False)
def get_template():
    return pd.read_csv('onboarding-dashboard/config-templates/investigation.csv')

template_df = get_template()

def add_experiment(values: dict) -> None:
    sheet = functions.ENV.gc.open_by_key(functions.ENV.get_investigation(st.session_state.current_investigation.name).sheet_id)
    df: pd.DataFrame = gd.get_as_dataframe(sheet.sheet1)
    df = df.dropna(how='all')
    df = df.dropna(how='all', axis='columns')
    if len(df) == 0:
        df = pd.DataFrame(columns=template_df.columns)
    df = df[df['name'] != values['name']]
    if 'draft' in values['name']:
        df = df[df['type'] != 'draft']
    temp_df = pd.DataFrame(columns=template_df.columns)
    for name, val in values.items():
        temp_df.loc[0, name] = val
    df = pd.concat([df, temp_df], ignore_index=True)
    gd.set_with_dataframe(sheet.sheet1, df)

@st.cache_data(show_spinner=False)
def get_results(results_name:str, investigation_name:str = None, keep_name:bool = False) -> pd.DataFrame:
    if investigation_name is None:
        investigation_name = st.session_state.current_investigation.name #Don't use defaults to avoid uninitialized current_investigation
    sheet = functions.ENV.gc.open_by_key(functions.ENV.get_investigation(investigation_name).sheet_id)
    df: pd.DataFrame = gd.get_as_dataframe(list(filter(lambda w: w.title == results_name, sheet.worksheets()))[0])
    df = df.dropna(how='all')
    df = df.dropna(how='any', axis='columns')
    if not keep_name:
        df.drop(axis='columns', labels='name', inplace=True)
    return df

@st.cache_data(show_spinner=False)
def set_config(): #TODO streamline
    config_data = ""
    with open('onboarding-dashboard/config-templates/config.jinja-yml', 'r') as f:
        config_data = f.read()
    functions.ENV._write_gdrive_file_in_folder(functions.ENV.get_investigation(st.session_state.current_investigation.name).id, 'config.yml', config_data)
    return None

@st.cache_data(show_spinner=False)
def get_lang_tags_mapping_data():
    data = {}
    with request.urlopen("https://ldml.api.sil.org/index.html?query=langtags&ext=json") as r:
        data: dict = json.loads(r.read())
    return data

@st.cache_data(show_spinner=False)
def get_lang_tag_mapping(tag: str):
    if tag == 'cmn':
        tag = 'zh' #Follow Serval behavior
    data = get_lang_tags_mapping_data()
    for obj in data:
        if 'iso639_3' not in obj:
            continue
        if obj['tag'].split('-')[0] == tag:
            script = obj["script"]
            if script == 'Kore':
                script = 'Hang' #Follow Serval behavior
            return f'{obj["iso639_3"]}_{script}'
    raise ValueError("Language tag does not exist")

def sync():
    functions.sync(st.session_state.current_investigation.name)
    if st.session_state.current_investigation in st.session_state.investigations:
        st.session_state.investigations.remove(st.session_state.current_investigation)
        st.session_state.current_investigation = Investigation.from_clowder(functions.ENV.get_investigation(st.session_state.current_investigation.name))
        st.session_state.investigations.append(st.session_state.current_investigation)
    st.rerun()

def get_drafts(investigation_name: str):
    investigation = functions.ENV.get_investigation(investigation_name)
    experiment_folder_id = functions.ENV._dict_of_gdrive_files(investigation.experiments_folder_id)[list((filter(lambda i: 'draft' in i,investigation.experiments.keys())))[0]]["id"]
    drafts_folder_id = functions.ENV._dict_of_gdrive_files(experiment_folder_id)['drafts']["id"]
    return dict(map(lambda kv: (kv[0], functions.ENV._read_gdrive_file_as_string(kv[1]['id'])),functions.ENV._dict_of_gdrive_files(drafts_folder_id).items()))

#TODO DESCRIPTIVE TEXT

if 'current_investigation' in st.session_state:
    if st.session_state.current_investigation.status == Status.Created:
        set_config()
    resources = get_resources()
    results_stats = None
    results_align = None
    results_models = None
    c1,c2 = st.columns(2)
    st.title(st.session_state.current_investigation.name)
    st.progress(st.session_state.current_investigation.status.to_percent(), st.session_state.current_investigation.status.name)
    with st.container():
        with st.expander("**Gather Stats**", st.session_state.current_investigation.status in [Status.Created, Status.GatheredStats]):
            if st.session_state.current_investigation.status.value >= Status.GatheredStats.value:
                results_stats = get_results('verse_percentages.csv')
                st.dataframe(results_stats.style.format(precision=0))
            with st.form(key=f'{st.session_state.current_investigation.id}-gather-stats'):
                texts = st.multiselect("Texts", resources)
                check_error('stats')
                if st.form_submit_button("Run", type='primary'):
                    check_required('stats', texts)
                    stats_row = template_df[template_df["name"] == "stats"]
                    stats_setup = stats_row.to_dict(orient='records')[0]
                    stats_setup['entrypoint'] = stats_setup['entrypoint'].replace('$FILES', ';'.join(list(map(lambda t: f"{t}.txt", texts))))
                    add_experiment(stats_setup)
                    with st.spinner("This might take a few minutes..."):
                        functions.run(st.session_state.current_investigation.name, experiments=["stats"], force_rerun=True)
                        sync()
        with st.expander("**Run Alignments**", st.session_state.current_investigation.status in [Status.GatheredStats, Status.Aligned]):
            if st.session_state.current_investigation.status.value >= Status.Aligned.value:
                results_align = get_results('corpus-stats.csv')
                st.dataframe(results_align.style.highlight_max(subset=results_align.select_dtypes(include='number').columns, color='green').format(precision=3))
                results_tokenization = get_results('tokenization_stats.csv', keep_name=True)
                st.dataframe(results_tokenization)
            with st.form(key=f'{st.session_state.current_investigation.id}-run-alignments'):
                training_sources = st.multiselect("Training sources", resources, default=list(results_stats['file'].apply(lambda f: f[:-4])) if results_stats is not None else None)
                training_target = st.selectbox("Training target", resources, index=None)
                default_books = None
                if results_stats is not None:
                    texts_series = pd.DataFrame(list(set(training_sources) | set([training_target])), columns=['file'])
                    results_stats['file'] = results_stats['file'].apply(lambda f: f[:-4])
                    rel_stats = pd.merge(texts_series, results_stats, how='inner')
                    default_books = BOOKS_ABBREVS.copy()
                    for _, row in rel_stats.iterrows():
                        to_remove = set(['OT','NT','DT'])
                        for book in default_books:
                            if row[book] < 93: #TODO more robust
                                to_remove.add(book)
                        for book in to_remove:
                            if book in default_books:
                                default_books.remove(book)
                    default_books = simplify_books(default_books)
                    if len(default_books) == 0:
                        default_books = None
                books = st.multiselect("Books to align on (Optional)", BOOKS_ABBREVS, default=default_books)
                check_error('align')
                if st.form_submit_button("Run", type='primary'):
                    check_required('align', training_sources, training_target)
                    align_row = template_df[template_df["name"] == "align"]
                    align_setup = align_row.to_dict(orient='records')[0]
                    align_setup['src_texts'] = ';'.join(training_sources)
                    align_setup['trg_texts'] = training_target
                    if len(books) > 0:
                        align_setup['align_set'] = ','.join(books)
                    add_experiment(align_setup)
                    with st.spinner("This might take a few minutes..."):
                        functions.run(st.session_state.current_investigation.name, experiments=["align"], force_rerun=True)
                        sync()
        with st.expander("**Run Models**", st.session_state.current_investigation.status in [Status.Aligned, Status.RanModels]):
            if st.session_state.current_investigation.status.value >= Status.RanModels.value:
                results_models = get_results('scores-best', keep_name=True)
                st.dataframe(results_models.style.highlight_max(subset=results_models.select_dtypes(include='number').columns, color='green').format(precision=2))
            st.write("Models")
            models = st.session_state.models if 'models' in st.session_state else []
            if len(models) == 0:
                st.write("*No models added*")
            for model in models:
                with st.container(border=True):
                    c1,c2 = st.columns([5,1])
                    with c1:
                        st.write(f'**{model["name"]}**') #TODO kill more options at rerun
                    with c2:
                        if st.button("Remove", type='primary', key=f'remove_{model["name"]}'):
                            st.session_state.models.remove(model)
                            st.rerun()
                    training_source_display = st.selectbox("Training source", resources,key=f"ts_{model['name']}",index=resources.index(model['training_source']), disabled=True)
                    training_target_display = st.selectbox("Training target", resources,key=f"tt_{model['name']}", index=resources.index(model["training_target"]), disabled=True)
                    if 'bt_src' in model:
                        backtranslation_source_display = st.selectbox("Back translation training source",resources, key=f"bts_{model['name']}", index=resources.index(model['bt_src']), disabled=True)
                        backtranslation_books_display = st.multiselect("Books to train on from back translation as well", BOOKS_ABBREVS, key=f"btb_{model['name']}", disabled=True, default=model['bt_books'])
                    books = st.multiselect("Books to train on", BOOKS_ABBREVS,key=f"b_{model['name']}", default=model["books"], disabled=True)
            st.write("Add a model" if len(models) == 0 else "Add another model")
            models_form = st.form(key=f'{st.session_state.current_investigation.id}-run-models')
            with models_form:    
                training_source = st.selectbox("Training source", resources, index=resources.index(results_align.loc[[results_align['align_score'].idxmax()],['src_project']].values[0]) if results_align is not None and len(results_align.index) > 0 else None)
                training_target = st.selectbox("Training target", resources, index=resources.index(results_align.loc[[0],['trg_project']].values[0]) if results_align is not None and len(results_align.index) > 0 else None)
                default_books = []
                if results_stats is not None and len(results_stats.index) > 0 and results_align is not None and len(results_align.index) > 0:
                    if results_align.loc[[0],['trg_project']].values[0] in results_stats['file'].values:
                        name = results_align.loc[[0],['trg_project']].values[0]
                        name = name[0]
                        for book in set(BOOKS_ABBREVS) - set(['OT','NT','DT']):
                            if (results_stats[results_stats['file'] == name][book] >= 93).all(): #TODO more robust
                                default_books.append(book)
                default_books = simplify_books(default_books)
                books = st.multiselect("Books to train on", BOOKS_ABBREVS, default=default_books if len(default_books) > 0 else None)
            
            backtranslation_source = None
            backtranslation_books = None
            if st.toggle('More options'):
                backtranslation_source = models_form.selectbox("Back translation training source", resources, index=resources.index(results_align.loc[[results_align['align_score'].idxmax()],['src_project']].values[0]) if results_align is not None and len(results_align.index) > 0 else None)
                backtranslation_books = models_form.multiselect("Books to train on from back translation as well", BOOKS_ABBREVS) #TODO smart defaults

            check_error('models')
            if models_form.form_submit_button('Add model'):
                check_required('models',training_source, training_target,books) #TODO change + template mixed src
                if 'models' not in st.session_state:
                    st.session_state.models = []
                is_mixed_src = backtranslation_source not in ['',None,[]] and backtranslation_books not in ['',None,[]]
                model = dict()
                if not is_mixed_src:
                    model['training_source'] = training_source
                    model['training_target'] = training_target
                    model['books'] = books
                    model['name'] = f"NLLB.1.3B.{training_source}-{training_target}.[{','.join(books)}]"
                else:
                    model['training_source'] = training_source
                    model['training_target'] = training_target
                    model['books'] = simplify_books(list(set(expand_books(books)) - set(expand_books(backtranslation_books))))
                    model['bt_src'] = backtranslation_source
                    model['bt_books'] = backtranslation_books
                    model['name'] = f"NLLB.1.3B.{training_source}+{backtranslation_source}-{training_target}.[{','.join(books)}]"
                if len(list(filter(lambda m: m["name"] == model["name"], st.session_state.models))) > 0:
                    st.session_state.errors['models'] = 'A model with this configuration has already been added'
                    st.rerun()
                st.session_state.models.append(model)
                st.rerun()
                


            models_already_running = len(list(filter(lambda kv: 'NLLB' in kv[0] and 'draft' not in kv[0] and kv[1]['status'] in ['in_progress','queued'],functions.ENV.get_investigation(st.session_state.current_investigation.name).experiments.items()))) > 0
            if models_already_running:
                st.write('Your models are training. Check back after a few hours.')
            if st.button("Run models" if not models_already_running else "Cancel", type='primary' if not models_already_running else 'secondary'):
                if models_already_running:
                    with st.spinner("This might take a few minutes..."):
                        functions.cancel(st.session_state.current_investigation.name)
                        sync()
                exps = []
                model_row = template_df[template_df["name"] == "model"]
                for model in models:
                    model_setup = model_row.to_dict(orient="records")[0]
                    exps.append(model["name"])
                    model_setup['name'] = model["name"]
                    model_setup['src'] = model["training_source"]
                    model_setup['trg'] = model["training_target"]
                    src_lang = model["training_source"].split('-')[0]
                    trg_lang = training_target.split('-')[0]
                    src_lang_tag_mapping = get_lang_tag_mapping(src_lang)
                    trg_lang_tag_mapping = get_lang_tag_mapping(trg_lang)
                    model_setup['src_lang'] = f'{src_lang}: {src_lang_tag_mapping}'
                    model_setup['trg_lang'] = f'{trg_lang}: {trg_lang_tag_mapping}'
                    model_setup['train_set'] = ','.join(model['books'])
                    if 'bt_books' in model:
                        model_setup['bt_books'] = ','.join(model['bt_books'])
                        model_setup['bt_src'] = model['bt_src']
                    add_experiment(model_setup)
                with st.spinner("This might take a few minutes..."):
                    functions.run(st.session_state.current_investigation.name, experiments=exps)       
                    sync()
        with st.expander("**Draft Books**", st.session_state.current_investigation.status in [Status.RanModels]):
            #TODO draft per model name
            if st.session_state.current_investigation.status.value >= Status.Drafted.value:
                drafts = get_drafts(st.session_state.current_investigation.name)
                st.write('*Available Drafts*')
                with st.container(border=True):
                    for draft_name, draft_content in drafts.items():
                        st.write(draft_name)
                        st.download_button(f'Download', data=draft_content, file_name=draft_name)
            with st.form(key=f'{st.session_state.current_investigation.id}-draft-books'):
                drafting_source = st.selectbox("Drafting source", resources, index=None)
                model_options = []
                idx_of_best_model = None
                if results_models is not None and len(results_models.index) > 0:
                    model_options = list(results_models["name"])
                    idx_of_best_model = int(results_models['CHRF3'].idxmax())
                model = st.selectbox("Model", model_options, index=idx_of_best_model)
                book = st.selectbox("Book to draft", BOOKS_ABBREVS, index=None)
                check_error('draft')
                if st.form_submit_button("Run", type='primary'):
                    check_required('draft',drafting_source, model, book)
                    draft_row = template_df[template_df["name"] == "draft"]
                    draft_setup = draft_row.to_dict(orient='records')[0]
                    draft_name = f"{model}_draft_{book}_{drafting_source}"
                    draft_setup['name'] = draft_name
                    draft_setup['entrypoint'] = draft_setup['entrypoint'].replace('$SRC', ''.join(drafting_source.split('-')[1:])).replace('$BOOK',book)
                    add_experiment(draft_setup)
                    with st.spinner("This might take a few minutes..."):
                        functions.run(st.session_state.current_investigation.name, experiments=[draft_name])       
                        st.toast("Check back in a few hours!")
                        sync()        
        c1,c2,_ = st.columns([1,1,15])
        with c1:
            if st.button('⟳'):
                sync()
        with c2:
            if st.button("🗑️", type='primary', key=f"{st.session_state.current_investigation.id}_delete_button"):  
                st.session_state.investigation_to_delete = st.session_state.current_investigation      
                st.switch_page("pages/DeleteInvestigation.py")
        st.write(f"More detailed information at {functions.urlfor(st.session_state.current_investigation.name)}")
else:
    st.switch_page("Home.py")