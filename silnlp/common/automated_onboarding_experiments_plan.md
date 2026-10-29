# Plan: `auto_onboarding.py`

## Purpose

A new script, `silnlp/common/auto_onboarding.py`, takes one onboarding request
folder and creates up to four experiments with no questions asked about sources
or books. It tests whether the decisions a user makes in
`create_onboarding_experiments.py` can be automated well enough for production.
Matthew Beech will re-implement the same rules on the production side, using this
script and this plan as the reference.

`create_onboarding_experiments.py` stays as it is: interactive, with all its
options. The new script imports its helpers where they fit, and changes nothing
in it.

## Command line

```
python -m silnlp.common.auto_onboarding <folder>
```

`<folder>` is the request folder relative to `MT/experiments`, for example
`_OnboardingRequests/ABC_2026_10_01_Request`. If that folder doesn't exist,
`_Request` is added to the name and the folder is checked again, so
`_OnboardingRequests/ABC_2026_10_01` works too. There are no other options. Every
other value comes from the folder or from the fixed values in this plan.

## Inputs

All read from `<folder>`:

| File | Used for |
|---|---|
| `onboarding_config.yml` | `main_project`, `planned_books`, `draft_source`, `bt_project` |
| `corpus-stats.csv` (or `alignments/corpus-stats.csv`) | candidate sources: stems, parallel counts, alignment scores |
| `verse_counts.csv` (falling back to `MT/experiments/verse_counts/verse_counts.csv`) | how complete each book is in the target and the sources |

- `planned_books` and `draft_source` are required. If either is missing or
  empty, or a required file is missing, the script stops and says what is
  missing. `planned_books` holds book numbers as strings (`'41'`); convert
  them to book IDs.
- `bt_project` is optional. Missing, empty or null means the team has no
  written back translation.
- `completed_books` and `ref_projects` are not used. Declared completed books
  have proved unreliable, so completeness is always measured from
  `verse_counts.csv`.
- `onboarding.log` is not read.

The target is `main_project`, matched to its stem in `corpus-stats.csv`. The
experiments go in `MT/experiments/<Country>/<Language>/`, named exactly as
`create_onboarding_experiments.py` names them, including `nameOverrides.json`.

## Fixed values

| Value | Setting |
|---|---|
| Source thresholds | at least 1000 parallel verses and alignment ≥ 0.2 |
| Back translation thresholds | at least 400 parallel verses and alignment ≥ 0.25 |
| Book completeness for training | ≥ 98% of the book's verses, in the target and every source |
| Test set | the default random 250 verses |
| "Close" alignment, for choosing between sources (rule 3a) | within 0.025 |
| Model, checkpoint, seed | as in `create_onboarding_experiments.py` |

Iso codes compare equal across their 2- and 3-letter forms (`ALT_ISO` in
`silnlp/common/iso_info.py`). "Same language" means the same iso code after that
normalisation.

## Rules

### 1. Sources

A candidate is any project in `corpus-stats.csv` that passes the source thresholds
with the exceptions of the target and the back translation.
Candidates in a different script from the target are accepted as candidates.

If the `draft_source` doesn't pass the thresholds then create two experiments 
using the two best-aligned candidates.

### 2. Back translation

- If `bt_project` is set but is not in `corpus-stats.csv`, warn and continue with
  no back translation.
- If the back translation project exists print its alignment score and also:
  - the number of verses in the bt_project,
  - the number of verses in parallel with the target,
  - the number of verses in the bt_project that don't exist in the target project.
- If the alignment score or number of parallel verses are below either threshold
  write that and continue with no back translation.
- If `bt_project` and `draft_source` name the same project write that as an error
  and exit the script.
- The back translation is never a drafting source.

### 3. Experiments

At most four, in this order:

1. `draft_source` alone: in this case the `draft_source` project is the training source and the
  drafting source.
2. The best-aligned candidate alone: in this case the best-aligned project
  is the training source and the drafting source. (Back translations aren't considered here 
  since they are not to be used as the first reference or as the drafting source.)
3. `draft_source` + back translation, if they are the same language.
4. The best-aligned candidate + back translation, if they are the same language.

If `draft_source` fails the thresholds, the second-best-aligned candidate takes
its place in experiments 1 and 3, so the run uses the two best-aligned candidates.
Throughout this rule, "best-aligned" means the candidate chosen by rule 3a.
When `draft_source` is the best-aligned candidate, experiments 1 and 2 are the
same, and so are 3 and 4. Keep one of each. In a pair the source comes first and
the back translation second. The source for training is also the source for drafting
in all cases.

### 3a. Choosing between close sources

Alignment scores of the leading sources are often within a few thousandths of
each other, too close to separate them. Rule 3a lets language and edition break
the tie.

`silnlp/assets/preferredSources.json` lists preferred sources and sources to
avoid, so the lists can be edited without touching the code:

```json
{
  "preferred": {
    "eng": [["NIV11R", "NIV84", "NLT07", "BSB"]],
    "fra": [["LBS21", "LS1910"], ["PDV2017", "PDV"], ["FCR19DBL", "FRC97", "FC97"]]
  },
  "avoid": ["ESV*"]
}
```

Each inner list runs from most to least preferred, which usually means later
editions before earlier ones. Below, a "later edition" of a source is any
project before it in the same list. The language keys only group the lists; a list
compares only the projects named in it. `avoid` holds name patterns: the ESV
and its variants are avoided because of licensing.

First, `draft_source`. It is the team's choice, so it is kept even when it is
avoided (with a warning). But if a later edition of it aligns better, draft from
that instead, and say so: the team may not have seen the scores before, and can
still pick another project.

Then choose the best-aligned source:

1. Leave out the avoided candidates, unless no other candidate passes the
   thresholds.
2. Start with the highest-aligned of the remaining candidates.
3. If it isn't in `draft_source`'s language, and a candidate in that language is
   close to it and aligns better than `draft_source`, take the highest-aligned
   such candidate instead. Experiments 1 and 2 then compare two Bibles in the
   same language.
4. If a later edition is close to the one chosen, take the latest such edition
   instead. This step never chooses `draft_source` itself, so that experiments 1
   and 2 stay different.

When `draft_source` fails the thresholds, step 3 doesn't apply: steps 1, 2 and 4
choose the first source, then again, without it, the second.

In a recent request, `draft_source` was a French Bible scoring 0.2485 and BSB led
with 0.2561. LBS21 (French, 0.2507) is close to BSB and better than
`draft_source`, so step 3 takes LBS21, and it heads its own list. Had
`draft_source` been English, step 4 would have taken NIV11R (0.2504), which comes
before BSB in the English list.

### 4. Books to draft

Each experiment's training books (rule 5) are chosen first. The books to draft
are then chosen per experiment, so different experiments may draft different
books; that needs no warning.

The ordered list:

- NT: MAT, MRK, LUK, JHN, ACT, 1JN, 2JN, 3JN, 1TH, 2TH, REV, 1CO, 2TI, GAL, 1PE,
  PHP, ROM, TIT, EPH, PHM, JAS, 2PE, JUD, 1TI, COL, HEB, 2CO.
- OT: RUT, JOS, EST, JON, GEN, EXO, LEV, NUM, DEU, JDG, 1SA, 2SA, 1KI, 2KI, 1CH,
  2CH, EZR, NEH, ISA, JER, LAM, EZK, DAN, HOS, JOL, AMO, OBA, MIC, NAM, HAB, ZEP,
  HAG, ZEC, MAL, JOB, ECC, SNG, PRO, PSA.
- Deuterocanonical, the books after REV in `vref.txt`, in that order: TOB, JDT,
  ESG, WIS, SIR, BAR, LJE, S3Y, SUS, BEL, 1MA, 2MA, 3MA, 4MA, 1ES, 2ES, MAN, PS2,
  ODA, PSS, EZA, JUB, ENO. These are considered only when the target already
  contains at least one of them.

The list runs from easier to harder books. Drafting in this order lets the team
build up more translation, and so more training data, before the harder books.

If no planned book is in the NT, the NT books are left out of the list.

A book is eligible for an experiment when it is in the list, is not one of that
experiment's training books, and the drafting source's Paratext project contains
it (checked as `create_onboarding_experiments.py` checks a translate source). An
experiment drafts from its first source only.

1. Take up to two eligible planned books, in list order.
2. If that gives fewer than two, add the first eligible books from the list
   until there are two. So a single planned book is drafted with the next
   eligible book from the list.
3. If no book is eligible, warn and run the experiment without `--translate`.
   We'll add more handling if this turns out to be common.

### 5. Training books

The books that are ≥ 98% complete in the target and in every source of the
experiment. Compact full testaments to `OT`/`NT` as the existing script does.
If nothing remains, skip the experiment and say why.

### 6. Iso clash

If any source of a created experiment shares the target's iso code, the target
gets a synthetic iso code and its extract is copied, as
`create_onboarding_experiments.py` does (including reusing a copy from an
earlier run). Two changes:

- If an existing copy is older than the target's extract, rename it to
  `<stem>_1.txt` (or `_2.txt` if `_1.txt` exists, and so on) and copy again.
- The matching `MT/terms` renderings copies are overwritten with the current
  ones.

Print what happened: the shared iso code, the synthetic code, and the full names
of both extract files (and of any renamed copy).

### 7. Writing and running

- `config.yml` and `translate_config.yml` are written as the existing script
  writes them. `translate_config.yml` has one entry, for the first source.
- Existing folders are never reused. A new folder takes the next free index for
  its name: if `SRC_xyz_1` exists, the new one is `SRC_xyz_2`.
- Before writing anything, print every decision and its reason: the target,
  each candidate's figures, the back translation outcome, the books chosen,
  the experiments, and any copy into `MT/scripture`. Then ask
  `Create these experiments? [Y/n]`.
- After writing, print the run commands and ask `Run them now? [Y/n]`.
  Experiments without books to draft run without `--translate`.

In this plan "warn" and "log" mean the same thing: print to the terminal.

## Out of scope

- **Which draft reaches the team, and production logging:** Matthew's work.
  The current intent is to compare chrF3 on each model's test set and withhold
  every draft from a model scoring below 25.
- **Overriding a decision by hand:** use `create_onboarding_experiments.py`.
- **A tool for testing rule sets against past scores** (deferred). It could
  start from `find_scores_for_multi_source_experiments.py` and
  `combine_scores.py`. Errors found in `combine_scores.py` get a GitHub issue
  and are fixed on another branch.

## Design

Follow the code design rules in `AGENTS.md`:
- one class per responsibility;
- no module-level constants, so the book orders and thresholds live inside the
  classes that use them;
- values from the environment resolved only in `main()`.

A likely split:
- `OnboardingRequest` reads and validates the folder's files.
- `BackTranslationCheck` decides whether `bt_project` is usable.
- `ExperimentChooser` applies rules 1 and 3.
- `DraftBookChooser` applies rule 4.
- `TargetExtractCopy` handles rule 6.
- `run()` connects them and holds the two prompts.

Reuse the existing script's helpers for parsing `corpus-stats.csv`, book
coverage, folder naming, the translate source check and writing the configs. If
one of them needs a change to fit, copy it into the new script rather than alter
the existing one.

## Tests

`tests/unit_tests/test_auto_onboarding.py`, with small fixture folders in a temp
directory and invented project names only (project names are never committed).
Write one test per case, without `pytest.mark.parametrize`:

- Inputs: a missing required file or key stops the run; a null `bt_project`
  means no back translation; book numbers convert to IDs.
- Sources: `draft_source` failing the thresholds does not create an experiment with
  that source.
- Back translation: missing from the stats; below 400 verses; alignment below
  0.25; the same project as `draft_source`; a different language gives no pairs.
- Close sources: the same-language candidate replaces a close best-aligned
  source in another language; it doesn't when it isn't close, or aligns below
  `draft_source`; a preferred edition replaces a close older one; one that
  isn't close doesn't; an avoided source is skipped, and used when it is the
  only candidate; an avoided `draft_source` is kept; a later edition of
  `draft_source` that aligns better replaces it, and one that aligns worse
  doesn't; with `draft_source` failing, both sources follow the preferences.
- Experiments: four when everything qualifies; one when `draft_source` is the
  best-aligned candidate and there is no back translation, two when there is; the two best-aligned
  candidates when `draft_source` fails the thresholds.
- Books: two planned books drafted in ordered-list order, not planned order; a
  planned book that is a training book is skipped; the source lacks a planned
  book and the next planned book replaces it; the source lacks it and no planned
  book is left, so the ordered list supplies one; every planned book is a training
  book; no eligible book, so the experiment runs without `--translate`;
  deuterocanonical books skipped when the target has none, and considered when it
  has one; two experiments with different sources drafting different books.
- Books to draft never include a training book.
- Iso clash: a fresh copy; a reused copy; an outdated copy renamed to `_1`, then
  `_2`.

### Verification

```bash
pytest tests/unit_tests/test_auto_onboarding.py -q
pytest tests/unit_tests/test_create_onboarding_experiments.py -q   # unchanged
black silnlp/ tests/ && isort silnlp/ tests/ && flake8 silnlp/common/auto_onboarding.py
```

David runs it end to end on a few real requests: one with a usable back
translation, one with none, and one with an iso clash. He compares its choices
with what he would have picked in `create_onboarding_experiments.py`.

## TODO

- [x] David reviews this plan
- [x] Implement `auto_onboarding.py` (one PR)
- [x] Tests (40), black, isort, flake8, mypy
- [ ] David's end-to-end runs
