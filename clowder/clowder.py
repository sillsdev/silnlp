from typing import Optional

import typer
from rich import print
from typing_extensions import Annotated

from clowder import functions
from clowder.status import Status

app = typer.Typer()


@app.command("untrack")
def untrack(investigation_name: str):
    """Untrack investigation with name `investigation_name` in the current context"""
    functions.untrack(investigation_name)
    print(f"[green]Successfully untracked {investigation_name}[/green]")


@app.command("track")
def track(investigation_name: Annotated[Optional[str], typer.Argument()] = None):
    """Tracks all investigations in the current context. If given an investigation name
    as an argument, this command will only track a single investigation with that name
    (if there are any) in the current context root folder"""
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
    print(f"[green]{functions.idfor(investigation_name)}[/green]")


@app.command("cancel")
def cancel(investigation_name: str):
    """Cancels a running investigation with name `investigation_name` in the current context"""
    if functions.cancel(investigation_name):
        print(f"[green]Investigation {investigation_name} successfully canceled[/green]")
    else:
        print(f"[red]Investigation {investigation_name} has no active experiments that can be canceled[/red]")


@app.command("run")
def run(investigation_name: str, force_rerun: bool = False, experiments: str = ""):
    """Runs all experiments in investigation `investigation_name` except those that are already completed or currently in progress.
    Use `--force-rerun` to forcibly rerun previously run experiments within this investigation. Pass `--experiments` as a comma-delimited
    list of experiment names  (i.e., the names of the rows in the investigation spreadsheet - e.g., "verse-counts;alignments") to
    run just those experiments."""
    experiments_list = experiments.split(";") if experiments != "" else []
    if functions.run(investigation_name, force_rerun, experiments_list):
        print(f"[green]Investigation {investigation_name} successfully started[/green]")
    else:
        print(f"[red]Investigation {investigation_name} cannot be run.[/red]")


@app.command("setup")
def setup(investigation_name: str):
    """Sets up the experiments as would happen before running - making experiment folders in GDrive and S3 as needed
    and rendering the config template into child configs. Note that setup will be run as part of the `run` command.
    This function is typically only useful for seeing generated configs before committing to running experiments."""
    functions.setup(investigation_name)
    print(f"[green]Investigation {investigation_name} successfully set up[/green]")


@app.command("status")
def status(
    investigation_name: Annotated[Optional[str], typer.Argument()] = None, sync: bool = True, verbose: bool = False
):
    """Prints status of investigation with name `investigation_name` in the current context.
    Use `--verbose` to see more detailed information. Use `--no-sync` if you want to see
    the current status without syncing with remote services"""
    for inv_name, obj in functions.status(investigation_name, sync).items():
        color = _map_status_color(obj["status"])
        if verbose:
            print(f"[bold]{inv_name}[/bold]")
            print("\tStatus:".ljust(15), f"[{color}]{obj['status'].value}[/{color}]")
            print("\tGDrive url:".ljust(15), f"{obj['gdrive_url']}")
            print("\tExperiments:".ljust(15))
            for exp_name, exp_obj in obj["experiments"].items():
                print(f"\t  [bold]{exp_name}[/bold]")
                print(f"\t  Status:".ljust(15), f"{exp_obj['status']}")
                print(f"\t  ClearML url:".ljust(15), f"{exp_obj.get('clearml_task_url')}")
        else:
            print(inv_name.ljust(25), f"[{color}]{obj['status'].value}[/{color}]")


@app.command("sync")
def sync(
    investigation_name: Annotated[Optional[str], typer.Argument()] = None,
    gather_results: bool = True,
    copy_all_results_to_gdrive: bool = False,
):
    """Sync status/data for investigation with name `investigation_name` in the current context.
    Use --no-gather-results to sync without aggregating results data. Use --copy-all-results-to-gdrive
    to copy all results files to gdrive experiments folders (results data will still be aggregated in the
    spreadsheet even when this is false)."""
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
        print(f"[green]{inv.name}[/green]")


@app.command("current-context")
def current_context():
    """Prints the GDrive folder id for the current context"""
    print(f"[green]{functions.current_context()}[/green]")


@app.command("untrack-context")
def untrack_context(root_folder_id: str):
    """Untrack context with folder id `root_folder_id`"""
    functions.untrack_context(root_folder_id)
    print(f"[green]Successfully untracked context {root_folder_id}[/green]")


@app.command("list-contexts")
def list_contexts():
    """Lists all currently tracked contexts"""
    print("[green]" + "\n".join(functions.list_contexts()) + "[/green]")


@app.command("use-data-folder")
def use_data_folder(folder_id: str):
    """Use scripture data from a particular gdrive folder specified by folder-id.
    (Other users will not be able to access this data). Data should be uploaded to
    the gdrive folder as complete Paratext project folders. This data folder will
    be associated only with the current context"""
    functions.use_data(folder_id)


@app.command("current-data-folder")
def current_data_folder():
    """Print the data folder id associated with this context if any"""
    print(f"[green]{functions.current_data()}[/green]")


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


def main():
    app()


if __name__ == "__main__":
    main()
