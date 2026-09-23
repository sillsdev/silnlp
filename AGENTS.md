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

## Code design rules

Follow these whenever you write or review code in this repository.

### Single responsibility

- Give each class and method one clear responsibility, and name it after
  that responsibility. If you can't state the responsibility in one phrase
  without "and", find a seam and split the unit into smaller ones.
- A method or constructor with many arguments usually has more than one
  responsibility. Either split the unit, or, if the arguments travel
  together or describe one concept, group them into a class that owns that
  concept's behaviour.

### Encapsulation and ownership

- **Favor classes over bare functions.** A module-level function that
  touches an object's data belongs on that object's class. A function whose
  parameters are all primitives usually signals a missing class. Ask "whose
  data is this?" before writing a free function.
- **No getters or setters**, including Python `@property` setters. A class
  operates on its own data rather than handing it out: if a caller fetches a
  field to compute something, move that computation onto the class. Prefer a
  method that answers the caller's real question (`is_empty()` over
  `count`). Dataclasses are for behaviour-free data transfer objects only.
- **No global variables.** Put every value in the narrowest scope that
  works. When several places need a constant, encapsulate the consumers'
  actual need behind a class rather than exporting the value: callers want
  `prompt.has_examples_placeholder()`, not `EXAMPLES_PLACEHOLDER = "{examples}"`.
  `LOGGER = logging.getLogger(__name__)` is the standard exception.
- **Avoid static methods.** They behave like globals and can't be swapped
  for a mock. Prefer an instance method even when it doesn't use `self`.
  Creation methods (`from_json()`, `load()`) are the defensible exception.
- **Don't dodge circular imports** by storing a class in a variable or
  deferring a first-party import. A cycle means one file holds too much:
  move the shared piece into its own module. To vary which concrete type
  gets built, inject a factory object. Lazily importing an expensive
  optional dependency is fine.

### Testing

- **No test-only API.** Never add a method or property, or widen
  visibility, just so a test can reach it, and don't have tests reach into
  privates (`obj._thing`). Test observable behaviour. Needing to reach
  inside means the class does too much; extract the collaborator and inject
  it.
- **No monkey patching**, in tests or production. That includes assigning
  over an object's methods, patching globals, `monkeypatch.setenv` to steer a
  class that reads the environment, and mutating third-party classes at
  import time. Design a dependency-injection seam instead: pass the value or
  collaborator in, or use a real subclass. When replacing an instance patch
  with a subclass override, keep whatever the framework introspects on the
  class (for example, HuggingFace reads `inspect.signature(model_class.forward)`).

### Refactoring

- **Verify every path before moving behaviour.** Enumerate the code's
  decision points mechanically, measure real branch coverage
  (`coverage run --branch`), and don't treat a passing test suite as proof a
  path is protected. Where the old code is still callable, prefer a
  differential harness that compares old and new outputs and raised
  exceptions across a matrix of inputs. List latent bugs you find instead of
  fixing them silently inside the refactor.

### Comments and docstrings

Docstrings are comments, and every rule here applies to them equally.

- Comment only non-obvious behaviour. If a comment seems needed for
  something else, improve the naming or structure instead.
- Say *why* the code is the way it is, never how it got that way (no "added
  for X" or "used to be Y").
- Keep comments to one line, except for a genuinely subtle correctness
  point.
- Don't write the conventional summary line that just restates the
  function's name as a sentence. If no "why" survives, the function gets no
  docstring.
- Put a comment about a specific line directly above that line. A class
  docstring holds only what is true of the class as a whole, never an
  implementation detail owned by one method.
- Before each commit, re-read the comments and docstrings in
  `git diff --cached` as a separate pass and delete any that fail these rules.
