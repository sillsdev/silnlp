from typing import List, Sequence, Tuple

from silnlp.nmt.clearml_connection import TaskIdentity


class StandInTask:
    """Plays a ClearML task's project, name and tags, and records each change asked of it."""

    def __init__(self, project: str, name: str, tags: Sequence[str]) -> None:
        self.project = project
        self.name = name
        self.tags = list(tags)
        self.changes: List[Tuple[str, object]] = []

    def get_project_name(self) -> str:
        return self.project

    def move_to_project(self, new_project_name: str) -> bool:
        self.changes.append(("project", new_project_name))
        self.project = new_project_name
        return True

    def set_name(self, name: str) -> None:
        self.changes.append(("name", name))
        self.name = name

    def get_tags(self) -> List[str]:
        return list(self.tags)

    def add_tags(self, tags: Sequence[str]) -> None:
        self.changes.append(("tags", list(tags)))
        self.tags.extend(tags)


def test_moves_renames_and_tags_a_task_created_by_another_tool() -> None:
    task = StandInTask("silnlp-web", "experiment Proj/exp --clearml-queue jobs_backlog", ["silnlp-web"])

    TaskIdentity("LangTech_Proj", "exp", ["silnlp-dev"]).apply_to(task)

    assert (task.project, task.name, task.tags) == ("LangTech_Proj", "exp", ["silnlp-web", "silnlp-dev"])


def test_leaves_a_task_silnlp_created_itself_unchanged() -> None:
    task = StandInTask("LangTech_Proj", "exp", ["silnlp-dev"])

    TaskIdentity("LangTech_Proj", "exp", ["silnlp-dev"]).apply_to(task)

    assert task.changes == []


def test_adds_only_the_tags_a_task_is_missing() -> None:
    task = StandInTask("LangTech_Proj", "exp", ["silnlp-research"])

    TaskIdentity("LangTech_Proj", "exp", ["silnlp-research", "silnlp-dev"]).apply_to(task)

    assert task.changes == [("tags", ["silnlp-dev"])]


def test_adds_no_tags_when_the_script_was_given_none() -> None:
    task = StandInTask("silnlp-web", "translate", [])

    TaskIdentity("LangTech_Proj_infer", "exp_MAT_1", []).apply_to(task)

    assert task.changes == [("project", "LangTech_Proj_infer"), ("name", "exp_MAT_1")]
