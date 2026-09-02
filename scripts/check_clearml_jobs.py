"""Report remote ClearML jobs currently running or queued under the authenticated user.

Intended for agents to run before submitting a new remote ClearML job, to
check whether one is already in flight (across any of the user's agent
sessions). Jobs run locally (queue_name "local"/"locally", never enqueued)
are not included, since they don't contend for shared queue/GPU resources.

Usage: python scripts/check_clearml_jobs.py
Exit code: 1 if any remote job is running/queued, 0 otherwise.
"""

import base64
import json
import sys

from clearml.backend_api.services import tasks as tasks_service
from clearml.backend_api.session import Session


def _current_user_id(session: Session) -> str:
    token = session.send_request(service="auth", action="login", method="GET", json={}).json()["data"]["token"]
    payload = token.split(".")[1]
    payload += "=" * (-len(payload) % 4)
    return json.loads(base64.urlsafe_b64decode(payload))["identity"]["user"]


def main() -> int:
    session = Session()
    user_id = _current_user_id(session)

    result = session.send(
        tasks_service.GetAllRequest(
            user=[user_id],
            status=["in_progress", "queued"],
            only_fields=["name", "status", "started", "execution.queue"],
            order_by=["-started"],
        )
    )
    # tasks only get a queue when enqueued for remote execution (Task.execute_remotely);
    # locally-run tasks never have one, so this excludes them.
    active = [t for t in result.response.tasks if t.execution and t.execution.queue]

    if not active:
        print("No remote ClearML jobs currently running or queued under your account.")
        return 0

    print(f"{len(active)} remote ClearML job(s) already running/queued under your account:")
    for task in active:
        print(f"  [{task.status}] {task.name} (started: {task.started})")
    return 1


if __name__ == "__main__":
    sys.exit(main())
