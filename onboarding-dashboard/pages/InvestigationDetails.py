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

def get_template():
    return pd.read_csv('onboarding-dashboard/config-templates/investigation.csv')

template_df = get_template()

def add_experiment(values: dict) -> None:
    sheet = functions.ENV.gc.open_by_key(functions.ENV.get_investigation(st.session_state.current_investigation.name).sheet_id)
    df: pd.DataFrame = gd.get_as_dataframe(sheet.sheet1)
    df = df.dropna(how='all')
    df = df.dropna(how='any', axis='columns')
    if len(df) == 0:
        df = pd.DataFrame(columns=template_df.columns)
    df = df[df['name'] != values['name']]
    temp_df = pd.DataFrame(columns=template_df.columns)
    for name, val in values.items():
        temp_df.loc[0, name] = val
    df = pd.concat([df, temp_df], ignore_index=True)
    gd.set_with_dataframe(sheet.sheet1, df)

@st.cache_data
def get_results(results_name:str) -> pd.DataFrame:
    sheet = functions.ENV.gc.open_by_key(functions.ENV.get_investigation(st.session_state.current_investigation.name).sheet_id)
    df: pd.DataFrame = gd.get_as_dataframe(list(filter(lambda w: w.title == results_name, sheet.worksheets()))[0])
    df = df.dropna(how='all')
    df = df.dropna(how='any', axis='columns')
    df.drop(axis='columns', labels='name', inplace=True)
    return df

@st.cache_data #TODO remove cache running messages
def set_config(): #TODO streamline
    config_data = ""
    with open('onboarding-dashboard/config-templates/config.jinja-yml', 'r') as f:
        config_data = f.read()
    functions.ENV._write_gdrive_file_in_folder(functions.ENV.get_investigation(st.session_state.current_investigation.name).id, 'config.yml', config_data)
    return None

@st.cache_data
def get_resources():
    return list(map(lambda fn: fn[:-4], functions.list_resources()))

@st.cache_data
def get_lang_tags_mapping_data():
    data = {}
    with request.urlopen("https://ldml.api.sil.org/index.html?query=langtags&ext=json") as r:
        data: dict = json.loads(r.read())
    return data

@st.cache_data
def get_lang_tag_mapping(tag: str):
    data = get_lang_tags_mapping_data()
    for obj in data:
        if 'iso639_3' not in obj:
            continue
        if obj['tag'] == tag:
            return f'{obj["iso639_3"]}_{obj["script"]}'
    raise ValueError("Language tag does not exist")

#TODO DESCRIPTIVE TEXT

#TODO update investigation status


#TODO form validation
if 'current_investigation' in st.session_state:
    set_config()
    resources = get_resources()
    st.title(st.session_state.current_investigation.name)
    st.progress(st.session_state.current_investigation.status.to_percent(), st.session_state.current_investigation.status.name)
    tag = st.text_input('tag')
    if st.button("TEST"):
        ret = get_lang_tag_mapping(tag)
        print(ret)
        st.write(ret)
    with st.container():
        with st.expander("**Gather Stats**", st.session_state.current_investigation.status in [Status.Created, Status.GatheredStats]):
            #TODO what if already run?
            if st.session_state.current_investigation.status.value >= Status.GatheredStats.value:
                results = get_results('verse_percentages.csv')
                st.dataframe(results.style.format(precision=2))
            with st.form(key=f'{st.session_state.current_investigation.id}-gather-stats'):
                texts = st.multiselect("Texts", resources)
                #TODO set it up
                if st.form_submit_button("Run"):
                    stats_row = template_df[template_df["name"] == "stats"]
                    stats_setup = stats_row.to_dict(orient='records')[0]
                    stats_setup['entrypoint'] = stats_setup['entrypoint'].replace('$FILES', ';'.join(texts))
                    add_experiment(stats_setup)
                    with st.spinner("This might take a few minutes..."):
                        functions.run(st.session_state.current_investigation.name, experiments=["stats"], force_rerun=True)
                        functions.sync(st.session_state.current_investigation.name, gather_results=True)
                        st.rerun()
        with st.expander("**Run Alignments**", st.session_state.current_investigation.status in [Status.GatheredStats, Status.Aligned]):
            #TODO what if already run?
            if st.session_state.current_investigation.status.value >= Status.Aligned.value:
                results_align = get_results('corpus-stats.csv')
                st.dataframe(results_align.style.highlight_max(subset=results_align.select_dtypes(include='number').columns, color='green').format(precision=2))
                # results_tokenization = get_results('tokenization_stats.csv') #TODO multiheader
                # st.dataframe(results_tokenization)
            with st.form(key=f'{st.session_state.current_investigation.id}-run-alignments'):
                training_sources = st.multiselect("Training sources", resources)
                training_target = st.selectbox("Training target", resources, index=None)
                books = st.multiselect("Books to align on (Optional)", BOOKS_ABBREVS)
                #TODO set it up
                if st.form_submit_button("Run"):
                    align_row = template_df[template_df["name"] == "align"]
                    align_setup = align_row.to_dict(orient='records')[0]
                    align_setup['src_texts'] = ';'.join(training_sources)
                    align_setup['trg_texts'] = training_target
                    if len(books) > 0:
                        align_setup['align_set'] = ','.join(books)
                    add_experiment(align_setup)
                    with st.spinner("This might take a few minutes..."):
                        functions.run(st.session_state.current_investigation.name, experiments=["align"], force_rerun=True)
                        functions.sync(st.session_state.current_investigation.name, gather_results=True)
                        st.rerun()
        with st.expander("**Run Models**", st.session_state.current_investigation.status in [Status.Aligned, Status.RanModels]):
            #TODO what if already run?
            if st.session_state.current_investigation.status.value >= Status.RanModels.value:
                results_align = get_results('scores-best')
                st.dataframe(results_align.style.highlight_max(subset=results_align.select_dtypes(include='number').columns, color='green').format(precision=2))
            with st.form(key=f'{st.session_state.current_investigation.id}-run-models'):
                training_sources = st.multiselect("Training sources", resources)
                training_target = st.selectbox("Training target", resources, index=None)
                books = st.multiselect("Books to train on", BOOKS_ABBREVS)
                #TODO set it up
                exps = []
                if st.form_submit_button("Run"):
                    model_row = template_df[template_df["name"] == "model"]
                    for training_source in training_sources:
                        model_setup = model_row.to_dict(orient="records")[0]
                        model_name = f"NLLB.1.3B.{training_source}-{training_target}.[{','.join(books)}]"
                        exps.append(model_name)
                        model_setup['name'] = model_name
                        model_setup['src'] = training_source
                        model_setup['trg'] = training_target
                        src_lang = training_source.split('-')[0]
                        trg_lang = training_target.split('-')[0]
                        src_lang_tag_mapping = get_lang_tag_mapping(src_lang)
                        trg_lang_tag_mapping = get_lang_tag_mapping(trg_lang)
                        model_setup['src_lang'] = f'{src_lang}: {src_lang_tag_mapping}'
                        model_setup['trg_lang'] = f'{trg_lang}: {trg_lang_tag_mapping}'
                        model_setup['train_set'] = ','.join(books)
                        add_experiment(model_setup)
                    with st.spinner("This might take a few minutes..."):
                        functions.run(st.session_state.current_investigation.name, experiments=exps)       
                        st.toast("Check back in 8-12 hours!")
                        sleep(4)
        with st.expander("**Draft Books**", st.session_state.current_investigation.status == Status.RanModels):
            #TODO what if already run?
            if st.session_state.current_investigation.status.value >= Status.Drafted.value:
                st.write("STUFF")
            with st.form(key=f'{st.session_state.current_investigation.id}-draft-books'):
                drafting_source = st.multiselect("Drafting source", resources)
                model = st.selectbox("Model", [], index=None)
                book = st.selectbox("Book to draft", BOOKS_ABBREVS, index=None)
                #TODO set it up
                if st.form_submit_button("Run"):
                    with st.spinner("This might take a few minutes..."):
                        functions.run(st.session_state.current_investigation.name, experiments=["draft"])       
                        st.toast("Check back in a few hours!")
                        sleep(4)        
        
        if st.button("Delete", key=f"{st.session_state.current_investigation.id}_delete_button"):  
            st.session_state.investigation_to_delete = st.session_state.current_investigation      
            st.switch_page("pages/DeleteInvestigation.py")
        st.write(f"More detailed information at {functions.urlfor(st.session_state.current_investigation.name)}")
else:
    st.switch_page("Investigations.py")