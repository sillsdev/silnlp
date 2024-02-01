#!/usr/bin/env python3

import os
import sys
from pathlib import Path

sys.path.append(str(Path(os.curdir).parent.absolute().parent.absolute()))

from typing import Optional

import typer
from rich import print

from clowder import functions
from clowder.status import Status

app = typer.Typer()


@app.command("untrack")
def untrack(investigation_name: str):
    """Untrack investigation with name `investigation_name` in the current context"""
    functions.untrack(investigation_name)
    print(f"[green]Successfully untracked {investigation_name}[/green]")


@app.command("track")
def track(investigation_name: Optional[str] = None):
    """Tracks all investigations in the current context. If given an `--investigation-name`, this
    will only track a single investigation with that For your security, you've been signed out of your current session.  name in the current context"""
    functions.track(investigation_name)
    print(
        f"[green]Successfully tracked {investigation_name if investigation_name else 'all investigations in this context'}[/green]"
    )


@app.command("create-from-template")
def create_from_template(from_investigation_name: str, new_investigation_name: str):
    """Creates a new investigation with name `investigation_name` in the current context
    copying the 'investigation' spreadsheet and 'config.yml' into the new investigation
    from an existing investigation with name `from_investigation_name`"""
    functions.create_from_template(from_investigation_name, new_investigation_name)
    print(f"[green]Investigation {new_investigation_name} successfully created[/green]")


@app.command("delete")
def delete(investigation_name: str, keep_clearml: bool = False, keep_google_drive: bool = False, keep_s3: bool = False):
    """Deletes investigation with name `investigation_name` from current context. Specify `--keep-[NAMEOFSERVICE]`
    to retain data from this investigation stored by that service: clearml, google_drive or s3. If all services are
    specified to be kept, behavior is identical to that of command 'untrack'"""
    functions.delete(investigation_name, not keep_clearml, not keep_google_drive, not keep_s3)


@app.command("urlfor")
def urlfor(investigation_name: str):
    """Prints url for investigation with name `investigation_name` in current context"""
    print(functions.urlfor(investigation_name))


@app.command("idfor")
def idfor(investigation_name: str):
    """Prints GDrive ID for investigation with name `investigation_name` in current context"""
    print(functions.idfor(investigation_name))


@app.command("cancel")
def cancel(investigation_name: str):
    """Cancels a running investigation with name `investigation_name` in the current context"""
    functions.cancel(investigation_name)


@app.command("run")
def run(investigation_name: str, force_rerun: bool = False):
    print(functions.run(investigation_name, force_rerun))


@app.command("status")
def status(investigation_name: Optional[str] = None, sync: bool = True, verbose: bool = False):
    """Prints status of investigation with name `investigation_name` in the current context.
    Use `--verbose` to see more detailed information. Use `--no-sync` if you want to see
    the current status without syncing with remote services"""
    for inv_name, obj in functions.status(investigation_name, sync).items():
        color = _map_status_color(obj["status"])
        if verbose:
            print(f"[bold]{inv_name}[/bold]")
            print(f"\tStatus:[{color}]\t{obj['status'].value}[/{color}]")
            print(f"\tGDrive url:\t{obj['gdrive_url']}")
            print(f"\tExperiments:")
            for exp_name, exp_obj in obj["experiments"].items():
                print(f"\t[bold]{exp_name}[/bold]")
                print(f"\tStatus:\t{exp_obj['status']}")
                print(f"\tClearML url:\t{exp_obj.get('clearml_task_url')}")
        else:
            print(inv_name, f"[{color}]\t{obj['status'].value}[/{color}]")


@app.command("sync")
def sync(
    investigation_name: Optional[str] = None, gather_results: bool = True, copy_all_results_to_gdrive: bool = True
):
    """Sync status/data for investigation with name `investigation_name` in the current context.
    Use --no-gather-results to sync without aggregating results data. Use --no-copy-all-results-to-gdrive
    to avoid copying all results files to gdrive experiments folders (results data will still be aggregated in the spreadsheet);
    this flag is helpful if you want to conserve time or space on gdrive."""
    functions.sync(investigation_name, gather_results, copy_all_results_to_gdrive=copy_all_results_to_gdrive)
    print(
        f"[green]Successfully synced {investigation_name if investigation_name else 'all investigations in this context'}[/green]"
    )


@app.command("create")
def create(investigation_name: str):
    """Create an empty investigation with name `investigation_name`"""
    functions.create(investigation_name)
    print(f"[green]Investigation {investigation_name} successfully created")


@app.command("use-context")
def use_context(root_folder_id: str):
    """Change context to folder with id `root_folder_id`"""
    functions.use_context(root_folder_id)
    print(f"[green]Success! Now using context {root_folder_id}[/green]")


@app.command("list")
def list():
    """Lists the names of all investigations in the current context"""
    for inv in functions.list_inv():
        print(inv.name)


@app.command("current-context")
def current_context():
    """Prints the GDrive folder id for the current context"""
    print(functions.current_context())


def _map_status_color(status: Status) -> str:
    # Mysterious comparison behavior; comparing by value instead
    if status.value == Status.Created.value:
        return "blue"
    if status.value == Status.Running.value:
        return "yellow"
    if status.value == Status.Completed.value:
        return "green"
    if status.value == Status.Canceled.value:
        return "pink"
    if status.value == Status.Failed.value:
        return "red"
    return "white"


if __name__ == "__main__":
    app()
