from enum import Enum
from clearml import Task


class Status(Enum):
    Created = "Created"
    Running = "Running"
    Failed = "Failed"
    Canceled = "Canceled"
    Completed = "Completed"

    @staticmethod
    def from_clearml_task_statuses(statuses: "list[Task.TaskStatusEnum]", current_status: Enum) -> Enum:
        # "created", "in_progress", "stopped", "closed", "failed", "completed", "queued", "published", "publishing", "unknown"
        if len(statuses) == 0:
            return current_status
        if len(list(filter(lambda s: s == Task.TaskStatusEnum.completed, statuses))) == len(statuses):
            return Status.Completed
        if len(list(filter(lambda s: s == Task.TaskStatusEnum.stopped, statuses))) == len(statuses):
            return Status.Canceled
        if len(list(filter(lambda s: s == Task.TaskStatusEnum.failed, statuses))) > 0:
            return Status.Failed
        return current_status
