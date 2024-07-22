import os
import sys
from enum import Enum
from pathlib import Path

import streamlit as st

parent_dir = Path(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(str(parent_dir.parent))
from clowder import functions, investigation, status


class Status(Enum):
    Created = 0
    GatheredStats = 1
    Aligned = 2
    RanModels = 3
    Drafted = 4

    def to_percent(self):
        return self.value / 4

    @staticmethod
    def from_clowder_investigation(i: investigation.Investigation):
        with functions._lock:
            import clowder.investigation as inv

            inv.ENV = st.session_state.clowder_env
            functions.ENV = st.session_state.clowder_env
            experiments = list(i.experiments.items())

        drafting_exps = list(filter(lambda exp: "draft" in exp[0], experiments))
        if len(drafting_exps) > 0 and len(
            list(filter(lambda exp: exp[1].get("status", None) == "completed", drafting_exps))
        ) == len(drafting_exps):
            return Status.Drafted
        model_exps = list(filter(lambda exp: "NLLB" in exp[0] and "draft" not in exp[0], experiments))
        if len(model_exps) > 0 and len(
            list(filter(lambda exp: exp[1].get("status", None) == "completed", model_exps))
        ) == len(model_exps):
            return Status.RanModels
        align_exps = list(filter(lambda exp: "align" in exp[0], experiments))
        if len(align_exps) > 0 and len(
            list(filter(lambda exp: exp[1].get("status", None) == "completed", align_exps))
        ) == len(align_exps):
            return Status.Aligned
        stats_exp = list(filter(lambda exp: "stats" in exp[0], experiments))
        if len(stats_exp) > 0 and len(
            list(filter(lambda exp: exp[1].get("status", None) == "completed", stats_exp))
        ) == len(stats_exp):
            return Status.GatheredStats
        return Status.Created


@st.experimental_dialog("Delete Investigation?")
def delete_investigation(investigation_to_delete):
    st.write(
        f'<div style="text-align: center"> Are you sure you want to delete {investigation_to_delete.name}? </div>',
        unsafe_allow_html=True,
    )
    _, c1, c2, _ = st.columns(4)
    with c1:
        if st.button("Yes"):
            try:
                functions.delete(investigation_to_delete.name)
                st.session_state.investigations.remove(investigation_to_delete)
                st.rerun()
            except Exception as e:
                st.error(f"Something went wrong while fetching resource data. Please try again. Error: {e}")

    with c2:
        if st.button("Cancel"):
            st.rerun()


class Investigation:
    def __init__(self, id: str, status: Status, name: str) -> None:
        self.id = id
        self.status = status
        self.name = name

    def __str__(self):
        return f"{self.name} {self.status} {self.id}"

    def __repr__(self):
        return self.__str__()

    def to_st(self, container):
        with container:
            with st.container(border=True):
                c1, c2, c3, c4 = st.columns([3, 3, 1, 1])
                with c1:
                    st.write(f"**{self.name}**")
                with c2:
                    st.write(self.status.name)
                with c3:
                    if st.button("Details", key=f"{self.id}_details_button", type="primary"):
                        st.session_state.current_investigation = self
                        st.switch_page("pages/InvestigationDetails.py")
                with c4:
                    if st.button("Delete", key=f"{self.id}_delete_button"):
                        # st.session_state.investigation_to_delete = self
                        # st.switch_page("pages/DeleteInvestigation.py")
                        delete_investigation(self)

    @staticmethod
    def from_clowder(clowder_investigation):
        return Investigation(
            id=clowder_investigation.id,
            name=clowder_investigation.name,
            status=Status.from_clowder_investigation(clowder_investigation),
        )  # clowder_investigation.status)
