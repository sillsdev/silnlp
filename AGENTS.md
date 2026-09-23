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

Follow these whenever you write or review Python code in SILNLP (`silnlp/`,
`scripts/`, `tests/`).

Much of the existing code predates these rules. Treat the violations named
below as patterns not to copy, not as precedent. Don't clean them up inside
an unrelated change either: a refactor needs its own branch and the path
checks under "Refactoring".

### Single responsibility

- Give each class and method one clear responsibility, and name it after
  that responsibility. If you can't state the responsibility in one phrase
  without "and", find a seam and split the unit into smaller ones. The
  config classes (`Config`, `Seq2SeqConfig`, `LLMConfig`) show what happens
  otherwise: they parse YAML, build models, run training and write outputs.
  Put new behaviour in a class of its own, not on a config class.
- A method or constructor with many arguments usually has more than one
  responsibility. Either split the unit, or, if the arguments travel
  together or describe one concept (for example, a corpus pair's source and
  target settings), group them into a class that owns that concept's
  behaviour.

### Encapsulation and ownership

- **Favor classes over bare functions.** A module-level function that
  touches an object's data belongs on that object's class. For example, the
  checkpoint helpers at the top of `silnlp/nmt/seq2seq_config.py`
  (`get_best_checkpoint(model_dir)`, `delete_optimizer_state(checkpoint_path)`)
  all operate on a model directory that has no class of its own. A function
  whose parameters are all primitives usually signals a missing class like
  that.
- **No getters or setters**, including Python `@property` setters. A class
  operates on its own data rather than handing it out: if a caller fetches a
  field to compute something, move that computation onto the class. Prefer a
  method that answers the caller's real question (`is_empty()` over
  `count`). Dataclasses are for behaviour-free data transfer objects only.
- **No new module-level variables.** Put every value in the narrowest scope
  that works. When several places need a constant, encapsulate the
  consumers' actual need behind a class rather than exporting the value. In
  `silnlp/nmt/llm_config.py`, `LLMConfig` uses the module-level tuples
  `VALID_FINETUNE_METHODS` and `QUANTIZED_METHODS` only to validate a
  finetune method and ask whether it is quantized. That belongs in a small
  finetune-method class that owns the valid names privately.
  `LOGGER = logging.getLogger(__name__)` is the standard exception.
- **Avoid static methods.** They behave like globals and can't be swapped
  for a mock. Prefer an instance method even when it doesn't use `self`.
  Creation methods such as `SilNlpEnv.create_standard_environment()` are the
  defensible exception.
- **Don't dodge circular imports** by storing a class in a variable or
  importing a `silnlp` module inside a function. A cycle means one file
  holds too much: move the shared piece into its own module. To vary which
  concrete type gets built, inject a factory object, as
  `PreTrainedModelProviderFactory` does in `seq2seq_config.py`. Lazily
  importing an expensive optional dependency is fine.

### Testing

Unit tests live in `tests/unit_tests/`. End-to-end smoke tests live in
`tests/smoke_tests/`.

- **No test-only API.** Never add a method or property, or widen
  visibility, just so a test can reach it, and don't have tests reach into
  privates (`obj._thing`). Test observable behaviour. Needing to reach
  inside means the class does too much; extract the collaborator and inject
  it.
- **No monkey patching**, in tests or production. That includes assigning
  over an object's methods (such as a model's `forward` or `generate`),
  `monkeypatch.setattr`, and `monkeypatch.setenv`. Design a
  dependency-injection seam instead: pass the value or collaborator in, or
  use a real subclass.
  - `SilNlpEnv` reads `SIL_NLP_DATA_PATH` and other variables straight from
    the environment, which forces tests to set them. New code should accept
    such values as arguments and resolve the environment variable in one
    place, at the entry point.
  - `seq2seq_config.py` assigns
    `M2M100ForConditionalGeneration.prepare_decoder_input_ids_from_labels`
    at import time, changing a HuggingFace class for every process that
    imports the module. Don't add more of these.
  - When you replace a patch with a subclass override, keep whatever the
    framework inspects on the class. HuggingFace's `Trainer` (the base of
    `SilSeq2SeqTrainer`) reads `inspect.signature(model_class.forward)` to
    find the label columns, so an override with a narrower signature
    silently drops the labels and training fails far from the cause.

### Refactoring

- **Verify every path before moving behaviour.** The smoke tests pass while
  leaving large parts of the config classes unexecuted, so a green suite
  doesn't prove a path is protected. Before moving code:
  - enumerate its decision points mechanically (conditionals, ternaries,
    loops, `try` blocks, each with its `else`) rather than by reading;
  - measure real branch coverage with
    `coverage run --branch -m pytest tests/` and say which paths have no
    test;
  - where the old code is still callable, write a differential harness that
    compares old and new return values and raised exceptions across a
    matrix of inputs. For code that acts by side effect, such as writing
    experiment outputs, compare the resulting directory contents.
- List latent bugs you find on the way. Don't fix them silently inside the
  refactor.

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

### Before each commit

Run `flake8` (configured in `setup.cfg`) and the relevant tests. Then
re-read the comments and docstrings in `git diff --cached` as a separate
pass and delete any that break the rules above.
