from typing import List, Set, Tuple

from silnlp.common.rename_experiment_folders import ExperimentRenames, ExperimentsRemote, Prompt, Rename, RenameSession


class FakeRemote(ExperimentsRemote):
    def __init__(self, folders: Set[str], fail_on: str = ""):
        self._folders = set(folders)
        self._fail_on = fail_on
        self.moves: List[Tuple[str, str]] = []

    def exists(self, folder: str) -> bool:
        return folder in self._folders

    def move(self, old: str, new: str) -> bool:
        if old == self._fail_on:
            return False
        self.moves.append((old, new))
        self._folders.discard(old)
        self._folders.add(new)
        return True


class ScriptedPrompt(Prompt):
    def __init__(self, *replies: str):
        self._replies = list(replies)
        self.questions: List[str] = []

    def answer(self, question: str) -> str:
        self.questions.append(question)
        return self._replies.pop(0)


def test_a_confirmed_rename_is_moved():
    remote = FakeRemote({"CookIslands"})
    done = RenameSession(remote, ScriptedPrompt("y")).run([Rename("CookIslands", "Cook_Islands")])
    assert done == 1
    assert remote.moves == [("CookIslands", "Cook_Islands")]


def test_a_declined_rename_is_not_moved():
    remote = FakeRemote({"CookIslands"})
    done = RenameSession(remote, ScriptedPrompt("n")).run([Rename("CookIslands", "Cook_Islands")])
    assert done == 0
    assert remote.moves == []


def test_a_missing_folder_is_skipped_without_asking():
    prompt = ScriptedPrompt()
    RenameSession(FakeRemote(set()), prompt).run([Rename("CookIslands", "Cook_Islands")])
    assert prompt.questions == []


def test_an_existing_destination_is_described_as_a_merge():
    prompt = ScriptedPrompt("n")
    remote = FakeRemote({"CookIslands", "Cook_Islands"})
    RenameSession(remote, prompt).run([Rename("CookIslands", "Cook_Islands")])
    assert "merged" in prompt.questions[0]


def test_quitting_stops_before_the_remaining_renames():
    remote = FakeRemote({"CookIslands", "SouthSudan"})
    renames = [Rename("CookIslands", "Cook_Islands"), Rename("SouthSudan", "South_Sudan")]
    RenameSession(remote, ScriptedPrompt("q")).run(renames)
    assert remote.moves == []


def test_a_failed_move_stops_the_session():
    remote = FakeRemote({"CookIslands", "SouthSudan"}, fail_on="CookIslands")
    renames = [Rename("CookIslands", "Cook_Islands"), Rename("SouthSudan", "South_Sudan")]
    done = RenameSession(remote, ScriptedPrompt("y", "y")).run(renames)
    assert done == 0
    assert remote.moves == []


def test_country_renames_come_before_language_renames():
    renames = ExperimentRenames().all()
    first_language = next(i for i, rename in enumerate(renames) if "/" in rename.old)
    assert all("/" not in rename.old for rename in renames[:first_language])
    assert all("/" in rename.old for rename in renames[first_language:])


def test_each_language_rename_stays_within_its_country():
    for rename in ExperimentRenames().all():
        assert rename.old.split("/")[:-1] == rename.new.split("/")[:-1]
