import concurrent.futures
import logging
import os
import subprocess
import uuid
from enum import Enum
from pathlib import Path
from typing import Dict, Tuple

import requests
from clearml import Task

SF_API_TOKEN = os.getenv("SF_API_TOKEN")
SF_AUTH_USER = os.getenv("SF_AUTH_USER")
SF_AUTH_PWD = os.getenv("SF_AUTH_PWD")
SF_CLIENT_ID = os.getenv("SF_CLIENT_ID")
UUID = str(uuid.uuid4())
ONBOARDING_PATH = os.getenv("ONBOARDING_PATH")
ONBOARDING_LOG_PATH = f"{ONBOARDING_PATH}/onboarded_projects.log"
ONBOARDING_CLEANUP_PATH = f"{ONBOARDING_PATH}/paths_to_delete.txt"
os.makedirs(ONBOARDING_PATH, exist_ok=True)
REPO_PATH = os.getenv("REPO_PATH")

ONBOARDING_REQUESTS_URL = "https://scriptureforge.org/command-api/onboarding-requests"
PROJECTS_URL = "https://scriptureforge.org/paratext-api/projects"
SF_AUTH_URL = "https://login.languagetechnology.org/oauth/token"

payload = {
    "username": SF_AUTH_USER,
    "password": SF_AUTH_PWD,
    "grant_type": "http://auth0.com/oauth/grant-type/password-realm",
    "client_id": SF_CLIENT_ID,
    "realm": "Username-Password-Authentication",
    "audience": "https://scriptureforge.org/",
    "scope": "openid profile email sf_data offline_access",
}

req = requests.post(
    SF_AUTH_URL,
    json=payload,
    headers={"Content-Type": "application/json"},
)
SF_API_TOKEN = req.json().get("access_token")
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
    all_requests = send_request(RequestType.GET, ONBOARDING_REQUESTS_URL, "getAllRequests", {}).json().get("result", [])
    return [request for request in all_requests if request["status"] == "new"] if all_requests else []


def get_project_metadata(onboarding_request: dict) -> Tuple[Dict[str, str], str]:
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
        metadata = (
            send_request(
                RequestType.GET,
                ONBOARDING_REQUESTS_URL,
                "getProjectMetadata",
                {"scriptureForgeId": id} if key == "projectId" else {"paratextId": id},
            )
            .json()
            .get("result", {})
        )
        if key == "projectId":
            main_project_name = metadata.get("shortName")
        request_metadata[metadata.get("id")] = (metadata.get("paratextID"), metadata.get("shortName"))

    return request_metadata, main_project_name


def download_project(
    request_id: str, SF_id: str, main_project_name: str, project_short_name: str, paratext_id: str
) -> None:
    project_url = f"{PROJECTS_URL}/{SF_id}/download"
    response = send_request(RequestType.GET, project_url, "getProjectDownloadLink", {"paratextId": SF_id})
    if response.status_code != 200:
        add_comment(
            request_id,
            f"Failed to get download project {project_short_name}. Skipping onboarding for this project. Error: {response.text}",
        )
        return
    os.makedirs(f"{ONBOARDING_PATH}/{main_project_name}_Request", exist_ok=True)
    if paratext_id and len(paratext_id) == 16:
        project_short_name = f"{project_short_name}_Resource"
    with open(f"{ONBOARDING_PATH}/{main_project_name}_Request/{project_short_name}.zip", "wb") as f:
        f.write(response.content)


def process_request(request):
    request_metadata, main_project_name = get_project_metadata(request)
    for SF_id, (paratext_id, project_short_name) in request_metadata.items():
        download_project(request[id], SF_id, main_project_name, project_short_name, paratext_id)
    task_name = f"Auto Onboarding - {main_project_name}"
    subprocess.run(
        [
            "python",
            f"{REPO_PATH}/scripts/automate_onboard_project.py",
            f"{main_project_name}.zip",
            "--dir",
            f"{main_project_name}_Request",
            "--task-name",
            task_name,
        ]
    )
    task: Task = Task.get_task(project_name="Onboarding", task_name=task_name, tags=["silnlp-auto-onboarding"])
    with open(ONBOARDING_LOG_PATH, "a") as f:
        f.write(f"{request['id']}\n")
    with open(ONBOARDING_CLEANUP_PATH, "a") as f:
        f.write(f"{ONBOARDING_PATH}/{main_project_name}_Request\n")
    add_comment(
        request["id"],
        f"This request is being automatically onboarded. ClearML task: {task.name}. Link: {task.get_output_log_web_page()}",
    )

    try:
        task.wait_for_status()
        add_comment(request["id"], "Automatic onboarding was successful.")
    except RuntimeError as e:
        LOGGER.warning(e)
        add_comment(request["id"], "Automatic onboarding failed.")


def main():
    onboarding_requests = get_onboarding_requests()
    if not onboarding_requests:
        LOGGER.info("No new onboarding requests found.")
        return
    onboarded_projects = []

    if not os.path.exists(ONBOARDING_LOG_PATH):
        Path(ONBOARDING_LOG_PATH).touch()
    if not os.path.exists(ONBOARDING_CLEANUP_PATH):
        Path(ONBOARDING_CLEANUP_PATH).touch()

    with open(ONBOARDING_LOG_PATH, "r") as f:
        onboarded_projects = f.read().splitlines()

    for request in onboarding_requests:
        if request["id"] in onboarded_projects:
            onboarding_requests.remove(request)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_request, request) for request in onboarding_requests]
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as e:
                LOGGER.warning(f"Error processing request: {e}")


if __name__ == "__main__":
    main()
