from typing import Optional

from clowder.consts import ENV, get_env
from clowder.environment import DuplicateInvestigationException, Investigation
from clowder.status import Status

if ENV is None:
    ENV = get_env()

# TODO remote logging (ignore for mvp)


def untrack(investigation_name: str):
    ENV.get_investigation(investigation_name).delete(delete_from_clearml=False, delete_from_gdrive=False, delete_from_s3=False)  # type: ignore


def track(investigation_name: Optional[str]):
    if investigation_name is not None:
        ENV.track_investigation_by_name(investigation_name)
    else:
        ENV.track_all_investigations()
    sync(investigation_name, gather_results=False)


def create_from_template(from_investigation_name: str, new_investigation_name: str):
    try:
        old_investigation = ENV.get_investigation(from_investigation_name)
        create(new_investigation_name)
        new_investigation = ENV.get_investigation(new_investigation_name)
        new_investigation.import_setup_from(old_investigation)
        return
    except DuplicateInvestigationException:
        pass
    except:
        delete(new_investigation_name)
    raise DuplicateInvestigationException(
        f"Investigation with name {from_investigation_name} already exists in the current context"
    )


def delete(
    investigation_name: str,
    delete_from_clearml: bool = True,
    delete_from_google_drive: bool = True,
    delete_from_s3: bool = True,
):
    ENV.get_investigation(investigation_name).delete(delete_from_clearml, delete_from_google_drive, delete_from_s3)


def idfor(investigation_name: str) -> str:
    """Returns GDrive ID for investigation with name `investigation_name` in current context"""
    return ENV.get_investigation(investigation_name).id


def urlfor(investigation_name: str) -> str:
    """Returns url for investigation with name `investigation_name` in current context"""
    return f"https://drive.google.com/drive/u/0/folders/{idfor(investigation_name)}"


def cancel(investigation_name: str):
    ENV.get_investigation(investigation_name).cancel()


def run(investigation_name: str, force_rerun: bool = False) -> bool:
    """Runs all experiments in investigation `investigation_name` except those that are already completed or currently in progess."""
    print(f"Syncing {investigation_name} before running")
    sync(investigation_name, gather_results=False)
    investigation = ENV.get_investigation(investigation_name)
    if investigation.status.value == Status.Running.value:
        return False
    investigation.setup()
    now_running = investigation.start_investigation(force_rerun)
    if now_running:
        investigation.status = Status.Running
    print(f"Syncing {investigation_name} after running")
    sync(investigation_name, gather_results=False)
    return now_running


def status(investigation_name: Optional[str], _sync: bool = True) -> dict:
    """Returns status of investigation with name `investigation_name` in the current context"""
    if _sync:
        print(f"Syncing {investigation_name if investigation_name else 'all investigations'} before gathering status")
        sync(investigation_name, gather_results=False, copy_all_results_to_gdrive=False)
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


def sync(investigation_name: Optional[str], gather_results: bool = True, copy_all_results_to_gdrive: bool = True):
    if investigation_name is not None:
        ENV.get_investigation(investigation_name).sync(
            gather_results=gather_results, copy_all_results_to_gdrive=copy_all_results_to_gdrive
        )
    else:
        for investigation in ENV.investigations:
            investigation.sync(gather_results=gather_results, copy_all_results_to_gdrive=copy_all_results_to_gdrive)


def create(investigation_name: str):
    """Create an empty investigation with name `investigation_name`"""
    ENV.create_investigation(investigation_name)


def use_context(root_folder_id: str):
    """Change context to folder with id `root_folder_id` reflected in `root` field"""
    ENV.root = root_folder_id
    if root_folder_id not in ENV.meta.data:
        ENV.meta.data[root_folder_id] = {"investigations": {}}
    ENV.meta.flush()


def list_inv() -> "list[Investigation]":
    """Lists all investigations in the current context"""
    return ENV.investigations


def current_context() -> str:
    return ENV.root
