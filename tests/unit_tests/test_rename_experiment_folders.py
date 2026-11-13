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


def test_approving_all_moves_every_existing_folder():
    remote = FakeRemote({"CookIslands", "SouthSudan"})
    renames = [Rename("CookIslands", "Cook_Islands"), Rename("SouthSudan", "South_Sudan")]
    done = RenameSession(remote, ScriptedPrompt("y")).run(renames)
    assert done == 2
    assert remote.moves == [("CookIslands", "Cook_Islands"), ("SouthSudan", "South_Sudan")]


def test_approving_all_asks_only_once():
    prompt = ScriptedPrompt("y")
    remote = FakeRemote({"CookIslands", "SouthSudan"})
    RenameSession(remote, prompt).run([Rename("CookIslands", "Cook_Islands"), Rename("SouthSudan", "South_Sudan")])
    assert len(prompt.questions) == 1


def test_declining_moves_nothing():
    remote = FakeRemote({"CookIslands"})
    done = RenameSession(remote, ScriptedPrompt("n")).run([Rename("CookIslands", "Cook_Islands")])
    assert done == 0
    assert remote.moves == []


def test_a_missing_folder_is_left_out_and_nothing_is_asked_when_none_remain():
    prompt = ScriptedPrompt()
    RenameSession(FakeRemote(set()), prompt).run([Rename("CookIslands", "Cook_Islands")])
    assert prompt.questions == []


def test_the_plan_lists_a_merge_as_one(capsys):
    remote = FakeRemote({"CookIslands", "Cook_Islands"})
    RenameSession(remote, ScriptedPrompt("n")).run([Rename("CookIslands", "Cook_Islands")])
    assert "MERGE" in capsys.readouterr().out


def test_the_plan_lists_every_rename_before_asking(capsys):
    remote = FakeRemote({"CookIslands", "SouthSudan"})
    renames = [Rename("CookIslands", "Cook_Islands"), Rename("SouthSudan", "South_Sudan")]
    RenameSession(remote, ScriptedPrompt("n")).run(renames)
    output = capsys.readouterr().out
    assert "CookIslands -> Cook_Islands" in output
    assert "SouthSudan -> South_Sudan" in output


def test_asking_for_each_moves_only_the_approved_ones():
    remote = FakeRemote({"CookIslands", "SouthSudan"})
    renames = [Rename("CookIslands", "Cook_Islands"), Rename("SouthSudan", "South_Sudan")]
    RenameSession(remote, ScriptedPrompt("s", "n", "y")).run(renames)
    assert remote.moves == [("SouthSudan", "South_Sudan")]


def test_quitting_while_asking_for_each_stops_the_rest():
    remote = FakeRemote({"CookIslands", "SouthSudan"})
    renames = [Rename("CookIslands", "Cook_Islands"), Rename("SouthSudan", "South_Sudan")]
    RenameSession(remote, ScriptedPrompt("s", "q")).run(renames)
    assert remote.moves == []


def test_a_failed_move_does_not_stop_the_others():
    remote = FakeRemote({"CookIslands", "SouthSudan"}, fail_on="CookIslands")
    renames = [Rename("CookIslands", "Cook_Islands"), Rename("SouthSudan", "South_Sudan")]
    done = RenameSession(remote, ScriptedPrompt("y")).run(renames)
    assert done == 1
    assert remote.moves == [("SouthSudan", "South_Sudan")]


def test_a_failed_move_is_reported(capsys):
    remote = FakeRemote({"CookIslands"}, fail_on="CookIslands")
    RenameSession(remote, ScriptedPrompt("y")).run([Rename("CookIslands", "Cook_Islands")])
    assert "failed, run again to retry: CookIslands" in capsys.readouterr().out


def test_country_renames_come_before_language_renames():
    renames = ExperimentRenames().all()
    first_language = next(i for i, rename in enumerate(renames) if "/" in rename.old)
    assert all("/" not in rename.old for rename in renames[:first_language])
    assert all("/" in rename.old for rename in renames[first_language:])


def test_each_language_rename_stays_within_its_country():
    for rename in ExperimentRenames().all():
        assert rename.old.split("/")[:-1] == rename.new.split("/")[:-1]
