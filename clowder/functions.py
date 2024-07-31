from typing import Optional

from filelock import FileLock

from clowder.consts import ENV, get_env
from clowder.environment import DuplicateInvestigationException, Investigation
from clowder.status import Status

if ENV is None:
    ENV = get_env()

_lock = FileLock("env.lock")


class ContextNotFoundException(Exception):
    pass


def untrack(investigation_name: str, env=None):
    with _lock:
        if env is not None:
            global ENV
            ENV = env
        ENV.get_investigation(investigation_name).delete(delete_from_clearml=False, delete_from_gdrive=False, delete_from_s3=False)  # type: ignore


def track(investigation_name: Optional[str], env=None):
    with _lock:
        if env is not None:
            global ENV
            ENV = env
        if investigation_name is not None:
            ENV.track_investigation_by_name(investigation_name)
        else:
            ENV.track_all_investigations()
    sync(investigation_name, gather_results=False, env=env)


def create_from_template(from_investigation_name: str, new_investigation_name: str, env=None):
    create(new_investigation_name, env)
    error = False
    with _lock:
        if env is not None:
            global ENV
            ENV = env
        try:
            old_investigation = ENV.get_investigation(from_investigation_name)
            new_investigation = ENV.get_investigation(new_investigation_name)
            new_investigation.import_setup_from(old_investigation)
        except:
            error = True
    if error:
        delete(new_investigation_name, env)


def delete(
    investigation_name: str,
    delete_from_clearml: bool = True,
    delete_from_google_drive: bool = True,
    delete_from_s3: bool = True,
    env=None,
):
    with _lock:
        if env is not None:
            global ENV
            ENV = env
        ENV.get_investigation(investigation_name).delete(delete_from_clearml, delete_from_google_drive, delete_from_s3)


def idfor(investigation_name: str, env=None) -> str:
    """Returns GDrive ID for investigation with name `investigation_name` in current context"""
    with _lock:
        if env is not None:
            global ENV
            ENV = env
        return ENV.get_investigation(investigation_name).id


def urlfor(investigation_name: str, env=None) -> str:
    """Returns url for investigation with name `investigation_name` in current context"""
    return f"https://drive.google.com/drive/u/0/folders/{idfor(investigation_name, env)}"


def cancel(investigation_name: str, env=None) -> bool:
    with _lock:
        if env is not None:
            global ENV
            ENV = env
        investigation = ENV.get_investigation(investigation_name)
        investigation.sync(gather_results=False, copy_all_results_to_gdrive=False)
        anything_was_canceled = investigation.cancel()
        investigation.sync(gather_results=False, copy_all_results_to_gdrive=False)
        return anything_was_canceled


def run(investigation_name: str, force_rerun: bool = False, experiments: "list[str]" = [], env=None) -> bool:
    """Runs all experiments in investigation `investigation_name` except those that are already completed or currently in progress."""
    print(f"Syncing {investigation_name} before running")
    sync(investigation_name, gather_results=False, env=env)
    with _lock:
        if env is not None:
            global ENV
            ENV = env
        investigation = ENV.get_investigation(investigation_name)
        if investigation.status.value == Status.Running.value:
            return False
        investigation.setup()
        now_running = investigation.start_investigation(force_rerun, experiments)
        if now_running:
            investigation.status = Status.Running
        print(f"Syncing {investigation_name} after running")
    sync(investigation_name, gather_results=False, env=env)
    return now_running


def setup(investigation_name: str, env=None):
    """Sets up the experiments as would happen before running - making experiment folders in GDrive and S3 as needed
    and rendering the config template into child configs"""
    with _lock:
        if env is not None:
            global ENV
            ENV = env
        investigation = ENV.get_investigation(investigation_name)
        investigation.setup()


def status(investigation_name: Optional[str], _sync: bool = True, env=None) -> dict:
    """Returns status of investigation with name `investigation_name` in the current context"""
    if _sync:
        print(f"Syncing {investigation_name if investigation_name else 'all investigations'} before gathering status")
        sync(investigation_name, gather_results=False, copy_all_results_to_gdrive=False, env=env)
    with _lock:
        if env is not None:
            global ENV
            ENV = env
        if investigation_name is not None:
            if ENV.investigation_exists(investigation_name):
                return {
                    investigation_name: {
                        "status": ENV.get_investigation(investigation_name).status,
                        "experiments": ENV.get_investigation(investigation_name).experiments,
                        "gdrive_url": urlfor(investigation_name),
                    }
                }
        return {
            inv.name: {"status": inv.status, "experiments": inv.experiments, "gdrive_url": urlfor(inv.name)}
            for inv in ENV.investigations
        }


def sync(
    investigation_name: Optional[str], gather_results: bool = True, copy_all_results_to_gdrive: bool = False, env=None
):
    with _lock:
        if env is not None:
            global ENV
            ENV = env
        if investigation_name is not None:
            ENV.get_investigation(investigation_name).sync(
                gather_results=gather_results, copy_all_results_to_gdrive=copy_all_results_to_gdrive
            )
        else:
            for investigation in ENV.investigations:
                try:
                    investigation.sync(
                        gather_results=gather_results, copy_all_results_to_gdrive=copy_all_results_to_gdrive
                    )
                except Exception as e:
                    print(f"Failed to track {investigation.name}: {e}")


def create(investigation_name: str, env=None):
    """Create an empty investigation with name `investigation_name`"""
    with _lock:
        if env is not None:
            global ENV
            ENV = env
        ENV.create_investigation(investigation_name)


def use_context(root_folder_id: str, env=None):
    """Change context to folder with id `root_folder_id` reflected in `root` field"""
    with _lock:
        if env is not None:
            global ENV
            ENV = env
        with ENV.meta.lock:
            ENV.meta.load()
            ENV.root = root_folder_id
            if root_folder_id not in ENV.meta.data:
                ENV.meta.data[root_folder_id] = {"investigations": {}}
            ENV.meta.flush()


def untrack_context(root_folder_id: str, env=None):
    """Untrack context with folder id `root_folder_id`"""
    with _lock:
        if env is not None:
            global ENV
            ENV = env
        with ENV.meta.lock:
            ENV.meta.load()
            if root_folder_id not in ENV.meta.data:
                raise ContextNotFoundException(f"Context {root_folder_id} is not tracked")
            if ENV.root == root_folder_id:
                use_context("temp")
            del ENV.meta.data[root_folder_id]
            ENV.meta.flush()


def list_contexts(env=None) -> "list[str]":
    """Lists all currently tracked contexts"""
    with _lock:
        if env is not None:
            global ENV
            ENV = env
        return ENV.meta.data.keys() - set(["current_root"])


def list_inv(env=None) -> "list[Investigation]":
    """Lists all investigations in the current context"""
    with _lock:
        if env is not None:
            global ENV
            ENV = env
        return ENV.investigations


def use_data(folder_id: str, refresh: bool = True, env=None):
    """Use scripture data from a particular gdrive folder specified by folder-id.
    (Other users will not be able to access this data). Data should be uploaded to
    the gdrive folder as complete Paratext project folders. This data folder will
    be associated only with the current context"""
    with _lock:
        if env is not None:
            global ENV
            ENV = env
    ENV.use_data(folder_id, refresh)


def current_data(env=None):
    """Return the data folder id associated with this context if any"""
    with _lock:
        if env is not None:
            global ENV
            ENV = env
    return ENV.data_folder


def unlink_data(env=None):
    """Remove the current context's association with its current private data folder"""
    with _lock:
        if env is not None:
            global ENV
            ENV = env
    ENV.unlink_data()


def list_resources(env=None) -> "list[str]":
    with _lock:
        if env is not None:
            global ENV
            ENV = env
    return ENV.list_resources()


def current_context(env=None) -> str:
    with _lock:
        if env is not None:
            global ENV
            ENV = env
    return ENV.root
