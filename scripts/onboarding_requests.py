import concurrent.futures
import os
import subprocess
import uuid

import requests
from clearml import Task

SF_API_TOKEN = os.getenv("SF_API_TOKEN")
UUID = str(uuid.uuid4())
ONBOARDING_PATH = os.getenv("ONBOARDING_PATH")


def add_comment(request_id: str, comment: str):
    onboard_requests_url = "https://qa.scriptureforge.org/command-api/onboarding-requests"
    try:
        response = requests.post(
            onboard_requests_url,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {SF_API_TOKEN}",
            },
            json={
                "body": {
                    "jsonrpc": "2.0",
                    "method": "addComment",
                    "params": {
                        "requestId": request_id,
                        "comment": comment,
                    },
                    "id": UUID,
                },
            },
        )
    except requests.exceptions.RequestException as e:
        print(f"Error making request: {e}")


def get_onboarding_requests() -> list[dict]:
    onboard_requests_url = "https://qa.scriptureforge.org/command-api/onboarding-requests"

    try:
        response = requests.post(
            onboard_requests_url,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {SF_API_TOKEN}",
            },
            json={
                "body": {
                    "jsonrpc": "2.0",
                    "method": "getAllRequests",
                    "id": UUID,
                }
            },
        )
        return response.json().get("result")

    except requests.exceptions.RequestException as e:
        print(f"Error making request: {e}")


def get_SF_ids(onboarding_request: dict) -> list[str]:
    SF_ids = []
    form_data = onboarding_request["submission"]["formData"]
    if form_data.get("partnerOrganization") != "none":
        return []
    for key in [
        "sourceProjectA",
        "sourceProjectB",
        "draftingSourceProject",
    ]:
        pt_id = form_data.get(key)

        get_project_url = "https://qa.scriptureforge.org/command-api/projects"
        try:
            response = requests.get(
                get_project_url,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {SF_API_TOKEN}",
                },
                json={
                    "body": {
                        "jsonrpc": "2.0",
                        "method": "getProjectIDByParatextID",
                        "params": {"paratextId": pt_id},
                        "id": UUID,
                    }
                },
            )
            SF_id = response.json().get("result")
            if SF_id and SF_id not in SF_ids:
                SF_ids.append(SF_id)

        except requests.exceptions.RequestException as e:
            print(f"Error making request: {e}")
    return SF_ids


def download_project(SF_id: str, request_id: str):
    projects_url = f"https://qa.scriptureforge.org/paratext-api/projects/{SF_id}/download"

    try:
        response = requests.get(
            projects_url,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {SF_API_TOKEN}",
            },
        )
        # TODO: Save to the project's name instead of id
        os.makedirs(f"{ONBOARDING_PATH}/{request_id}", exist_ok=True)
        with open(f"{ONBOARDING_PATH}/{request_id}/{SF_id}.zip", "wb") as f:
            f.write(response.content)
    except requests.exceptions.RequestException as e:
        print(f"Error making request: {e}")


onboarding_requests = get_onboarding_requests()
# TODO: Distinguish between projects and resources

onboarded_projects = []
if os.path.exists(f"{ONBOARDING_PATH}/onboarded_projects_log.txt"):
    with open(f"{ONBOARDING_PATH}/onboarded_projects_log.txt", "r") as f:
        onboarded_projects = f.read().splitlines()

for request in onboarding_requests:
    if request["id"] in onboarded_projects:
        print(f"Request {request['id']} has already been onboarded. Skipping.")
        continue


def process_request(request):
    for SF_id in get_SF_ids(request):
        download_project(SF_id, request["id"])
    task_name = f"{request['id']}"
    subprocess.run(
        [
            "python",
            "/workspaces/silnlp/scripts/automate_onboard_project.py",
            "--task-name",
            task_name,
            "--dir",
            request["id"],
        ]
    )
    # Get a clearml task by its name
    task: Task = Task.get_task(project_name="Onboarding", task_name=task_name, tags=["silnlp-auto-onboarding"])
    with open(f"{ONBOARDING_PATH}/onboarded_projects_log.txt", "a") as f:
        f.write(f"{request['id']}\n")
    add_comment(request["id"], f"This request is being automatically onboarded. ClearML task: {task.name}")

    try:
        task.wait_for_status()
        add_comment(request["id"], "Automatic onboarding was successful.")
    except RuntimeError as e:
        print(e)
        add_comment(request["id"], "Automatic onboarding failed.")


with concurrent.futures.ThreadPoolExecutor() as executor:
    futures = [executor.submit(process_request, request) for request in onboarding_requests]
    for future in concurrent.futures.as_completed(futures):
        try:
            future.result()
        except Exception as e:
            print(f"Error processing request: {e}")
