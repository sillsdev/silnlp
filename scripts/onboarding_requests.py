import concurrent.futures
import logging
import os
import shutil
import subprocess
import uuid
from enum import Enum
from pathlib import Path
from typing import Dict, Tuple

import requests
from clearml import Task

SF_API_TOKEN = os.getenv("SF_API_TOKEN")
UUID = str(uuid.uuid4())
ONBOARDING_PATH = os.getenv("ONBOARDING_PATH")
os.makedirs(ONBOARDING_PATH, exist_ok=True)
REPO_PATH = os.getenv("REPO_PATH")

ONBOARDING_REQUESTS_URL = "https://qa.scriptureforge.org/command-api/onboarding-requests"
PROJECTS_URL = "https://qa.scriptureforge.org/paratext-api/projects"

LOGGER = logging.getLogger(__name__)


class RequestType(Enum):
    POST = "POST"
    GET = "GET"


def send_request(type: RequestType, url: str, method: str, params: dict) -> dict:
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {SF_API_TOKEN}"}
    json_data = {
        "body": {
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
            "id": UUID,
        }
    }
    try:
        if type == RequestType.POST:
            response = requests.post(url, headers=headers, json=json_data)
        elif type == RequestType.GET:
            response = requests.get(url, headers=headers, json=json_data)
        else:
            raise ValueError("Invalid request type. Must be 'GET' or 'POST'.")
        return response
    except requests.exceptions.RequestException as e:
        LOGGER.warning(f"Error making request: {e}")
        return {}


def add_comment(request_id: str, comment: str):
    send_request(
        RequestType.POST,
        ONBOARDING_REQUESTS_URL,
        "addComment",
        {"requestId": request_id, "commentText": comment},
    )


def get_onboarding_requests() -> list[dict]:
    return send_request(RequestType.POST, ONBOARDING_REQUESTS_URL, "getAllRequests", {}).json().get("result", {})


def get_project_metadata(onboarding_request: dict) -> Tuple[str, Dict[str, str]]:
    request_metadata: Dict[str, str] = {}
    main_project_name = ""
    form_data = onboarding_request["submission"]["formData"]
    if form_data.get("partnerOrganization") != "none":
        add_comment(
            onboarding_request["id"], "This request is for a partner organization. Skipping automatic onboarding."
        )
        return {}, ""
    project_keys = [
        "projectId",
        "sourceProjectA",
        "sourceProjectB",
        "sourceProjectC",
        "draftingSourceProject",
    ]

    for key in project_keys:
        id = onboarding_request["submission"].get(key) if key == "projectId" else form_data.get(key)
        if not id:
            continue
        response = send_request(
            RequestType.GET,
            ONBOARDING_REQUESTS_URL,
            "getProjectMetadata",
            {"scriptureForgeId": id} if key == "projectId" else {"paratextId": id},
        )
        metadata = response.json().get("result", {})
        if key == "projectId":
            main_project_name = metadata.get("shortName")
        request_metadata[metadata.get("id")] = (metadata.get("paratextID"), metadata.get("shortName"))

    return request_metadata, main_project_name


def download_project(SF_id: str, main_project_name: str, project_short_name: str, paratext_id: str):
    project_url = f"{PROJECTS_URL}/{SF_id}/download"
    response = send_request(RequestType.GET, project_url, "getProjectDownloadLink", {"paratextId": SF_id})
    os.makedirs(f"{ONBOARDING_PATH}/{main_project_name}_Request", exist_ok=True)
    if len(paratext_id) == 16:
        project_short_name = f"{project_short_name}_Resource"
    with open(f"{ONBOARDING_PATH}/{main_project_name}_Request/{project_short_name}.zip", "wb") as f:
        f.write(response.content)


onboarding_requests = get_onboarding_requests()
onboarded_projects = []

if not os.path.exists(f"{ONBOARDING_PATH}/onboarded_projects.log"):
    Path(f"{ONBOARDING_PATH}/onboarded_projects.log").touch()

with open(f"{ONBOARDING_PATH}/onboarded_projects.log", "r") as f:
    onboarded_projects = f.read().splitlines()

for request in onboarding_requests:
    if request["id"] in onboarded_projects:
        onboarding_requests.remove(request)


def process_request(request):
    request_metadata, main_project_name = get_project_metadata(request)
    for SF_id, (paratext_id, project_short_name) in request_metadata.items():
        download_project(SF_id, main_project_name, project_short_name, paratext_id)
    task_name = f"Auto Onboarding - {main_project_name}"
    subprocess.run(
        [
            "python",
            f"{REPO_PATH}/scripts/automate_onboard_project.py",
            main_project_name,
            "--dir",
            f"{main_project_name}_Request",
        ]
    )
    task: Task = Task.get_task(project_name="Onboarding", task_name=task_name, tags=["silnlp-auto-onboarding"])
    with open(f"{ONBOARDING_PATH}/onboarded_projects.log", "a") as f:
        f.write(f"{request['id']}\n")
    add_comment(request["id"], f"This request is being automatically onboarded. ClearML task: {task.name}")

    try:
        task.wait_for_status()
        add_comment(request["id"], "Automatic onboarding was successful.")
    except RuntimeError as e:
        LOGGER.warningprint(e)
        add_comment(request["id"], "Automatic onboarding failed.")
    finally:
        shutil.rmtree(f"{ONBOARDING_PATH}/{main_project_name}_Request")


with concurrent.futures.ThreadPoolExecutor() as executor:
    futures = [executor.submit(process_request, request) for request in onboarding_requests]
    for future in concurrent.futures.as_completed(futures):
        try:
            future.result()
        except Exception as e:
            LOGGER.warning(f"Error processing request: {e}")
