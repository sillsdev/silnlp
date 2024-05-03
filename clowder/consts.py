ENV = None
GDRIVE_SCOPE = "https://www.googleapis.com/auth/drive"
CLEARML_QUEUE = "jobs_backlog"
CLEARML_QUEUE_CPU = "cpu_only"
CLEARML_URL = "app.sil.hosted.allegro.ai"
RESULTS_CSVS_ATTRIBUTE = "results-csvs"
RESULTS_CLEARML_METRIC_ATTRIBUTE = "results-clearml-metrics"
ENTRYPOINT_ATTRIBUTE = "entrypoint"
NAME_ATTRIBUTE = "name"


def get_env(auth=None):
    global ENV
    if ENV is None:
        from clowder.environment import ClowderEnvironment

        if auth is not None:
            ENV = ClowderEnvironment(auth)
        else:
            ENV = ClowderEnvironment()
    return ENV
