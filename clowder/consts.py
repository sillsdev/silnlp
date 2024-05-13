ENV = None
GDRIVE_SCOPE = "https://www.googleapis.com/auth/drive"
CLEARML_QUEUE = "jobs_backlog"
CLEARML_QUEUE_CPU = "cpu_only"
CLEARML_URL = "app.sil.hosted.allegro.ai"
RESULTS_CSVS_ATTRIBUTE = "results-csvs"
RESULTS_CLEARML_METRIC_ATTRIBUTE = "results-clearml-metrics"
ENTRYPOINT_ATTRIBUTE = "entrypoint"
NAME_ATTRIBUTE = "name"

import os
def set_up_creds():
    if not os.path.exists('./.clowder'):
        os.mkdir('./.clowder')
        with open('./.clowder/clowder-000.json', 'w') as f:
            f.write(os.environ.get('CLOWDER_CREDENTIALS','{}'))
            os.environ['GOOGLE_CREDENTIALS_FILE'] = os.path.abspath('./.clowder/clowder-000.json')


def get_env():
    global ENV
    if ENV is None:
        from clowder.environment import ClowderEnvironment
        ENV = ClowderEnvironment()
    return ENV
