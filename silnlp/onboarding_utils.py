from datetime import datetime
from pathlib import Path


def append_datestamp(project_name: str) -> str:
    now = datetime.now()
    datestamp = now.strftime("%Y_%m_%d")
    return f"{project_name}_{datestamp}"


def rename_project(project_name: str, datestamp: bool, copy_from: Path | None) -> str:
    is_resource = False
    if project_name.endswith("_Resource"):
        is_resource = True
        resource_hash_path = copy_from / Path(f"{project_name}/.resource_hash") if copy_from else None
        if resource_hash_path and not resource_hash_path.exists():
            resource_hash_path.touch()
        project_name = project_name.replace("_Resource", "")
    if "-" in project_name:
        project_name = project_name.replace("-", "_")
    if datestamp and not is_resource:
        project_name = append_datestamp(project_name)
    return project_name


def copy_project(project_name: str, local_project_path: Path) -> Path:
    if local_project_path and local_project_path.exists() and local_project_path.name != project_name:
        new_local_project_path = local_project_path.parent / project_name
        if not new_local_project_path.exists():
            new_local_project_path.mkdir(parents=True, exist_ok=True)
        copy_directory(local_project_path, new_local_project_path, overwrite=True)
        local_project_path = new_local_project_path

    return local_project_path


def copy_file(source_file: Path, target_file: Path, overwrite=False) -> None:
    if target_file.exists() and not overwrite:
        return
    target_file.write_bytes(source_file.read_bytes())


def copy_directory(source_dir: Path, target_dir: Path, overwrite=False) -> None:
    if not target_dir.exists():
        target_dir.mkdir()
    for sub_item in source_dir.iterdir():
        target_item = target_dir / sub_item.name
        if sub_item.is_dir():
            copy_directory(sub_item, target_item, overwrite)
        else:
            copy_file(sub_item, target_item, overwrite)
