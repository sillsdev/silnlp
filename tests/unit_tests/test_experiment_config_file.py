import pytest
import yaml

from silnlp.nmt.config_utils import ExperimentConfigFile


def write(path, config) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(config, file)


@pytest.fixture
def project_dir(tmp_path):
    return tmp_path / "my-project"


@pytest.fixture
def exp_dir(project_dir):
    return project_dir / "my-experiment"


@pytest.fixture
def config_file(exp_dir, project_dir):
    return ExperimentConfigFile(exp_dir, project_dir)


def test_an_experiment_runs_from_its_own_config(config_file, exp_dir):
    write(exp_dir / "config.yml", {"model": "facebook/nllb-200-distilled-1.3B"})

    assert config_file.read() == {"model": "facebook/nllb-200-distilled-1.3B"}


def test_an_experiment_without_a_config_inherits_the_projects(config_file, project_dir):
    write(project_dir / "config.yml", {"model": "google/madlad400-3b-mt"})

    assert config_file.read() == {"model": "google/madlad400-3b-mt"}


def test_an_inherited_config_is_kept_with_the_experiment_that_ran_it(config_file, exp_dir, project_dir):
    write(project_dir / "config.yml", {"model": "google/madlad400-3b-mt"})

    config_file.read()

    assert (exp_dir / "config.yml").is_file()


def test_an_experiments_own_config_wins_over_the_projects(config_file, exp_dir, project_dir):
    write(project_dir / "config.yml", {"model": "google/madlad400-3b-mt"})
    write(exp_dir / "config.yml", {"model": "facebook/nllb-200-distilled-1.3B"})

    assert config_file.read() == {"model": "facebook/nllb-200-distilled-1.3B"}


def test_an_experiment_with_no_config_anywhere_says_where_it_looked(config_file, exp_dir):
    with pytest.raises(RuntimeError, match=str(exp_dir / "config.yml")):
        config_file.read()


def test_an_empty_config_is_refused(config_file, exp_dir):
    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / "config.yml").touch()

    with pytest.raises(RuntimeError, match="no contents"):
        config_file.read()
