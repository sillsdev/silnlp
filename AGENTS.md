# Instructions for AI coding agents

## ClearML jobs: only one running remotely at a time, across ALL agents

Never have more than one ClearML job running **remotely** at once (submitted
to a queue and executed by a `clearml-agent` worker) — this applies across
every agent session the user has open, not just the current one. This is
about contention for shared queue/GPU resources.

This restriction does **not** apply to jobs run locally (`queue_name: local`/
`locally`, i.e. never enqueued) — run as many of those concurrently as you
like, even if they're tracked as ClearML tasks.

Before submitting any job remotely (e.g. via `silnlp/nmt/experiment.py`,
`silnlp/nmt/train.py`, `silnlp/nmt/translate.py`, alignment scripts under
`silnlp/alignment/`, or anything using `silnlp/nmt/clearml_connection.py`
with a non-local `queue_name`):

1. Run `python scripts/check_clearml_jobs.py`. It uses the ClearML API
   credentials in the environment to list any remote jobs already running
   or queued under the current user's account (a non-zero exit code means
   one exists) — this works even though you (the agent) can't browse the
   ClearML web UI.
2. If a remote job is already active, don't submit another yet — and don't
   just tell the user to wait. Keep the job in your own backlog of work to
   submit, and handle the scheduling yourself: check again (e.g. after other
   work, or by polling) and submit it as soon as the check shows no active
   remote job. Only involve the user if something looks stuck or wrong.
