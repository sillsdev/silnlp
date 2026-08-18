# `create_onboarding_experiments` — usage

Turn a production onboarding request (or an analyze `corpus-stats.csv` folder)
into ready-to-run NMT experiment folders. For each selected reference the tool
writes a `config.yml` + `translate_config.yml` under
`<Country>/<Language>/<experiment>/` in the experiments tree.

> Design notes and rationale live in `create_onboarding_experiments_plan.md`.
> This file is only about *running* the tool.

## Prerequisites

- Environment configured for data access (`SIL_NLP_ENV` / bucket credentials —
  see the repo `README.md` and `bucket_setup.md`).
- The request folder present under `MT/experiments/_OnboardingRequests/`, or an
  analyze folder containing `corpus-stats.csv` (e.g. `PNG/Taupota/Align`).

## Basic usage

```bash
python -m silnlp.common.create_onboarding_experiments <request> --translate-books <books>
```

Example:

```bash
python -m silnlp.common.create_onboarding_experiments SFDH_2026_07_13 --translate-books "MAT;MRK"
```

`<request>` is the request folder name (with or without the `_Request` suffix),
or a folder relative to `MT/experiments` that holds a `corpus-stats.csv`.

The tool is **interactive**: it prints a candidate table, asks you to choose
which experiments to create, asks for `y/N` confirmation before copying any
shared extract files, and offers to run each experiment after the folders are
created. Create the folders and decline that last prompt for the preview
`--dry-run` used to give; select `none` at the experiments prompt to write nothing
at all.

## Options

| Option | Default | Purpose |
|---|---|---|
| `--translate-books <books>` | *required* | Book(s) for `translate_config.yml` (e.g. `MAT`, `"MAT;MRK"`). |
| `--training-books <books>` | `complete` | Books written to `corpus_books`, or `complete` to derive them from `verse_counts.csv` (≥98% rule; testaments compacted to `NT`/`OT`). |
| `--target <iso or project>` | auto | Force the target language/project (use the project name when two projects share an iso). |
| `--min-parallel <n>` | `2000` | Minimum parallel verse count for a reference pair. |
| `--min-alignment <a>` | `0.2` | Minimum alignment score for a reference pair. |
| `--top <n>` | `20` | Maximum number of experiments offered for selection. |
| `--translate-scripture <projects>` | auto | Override the drafting source(s) entirely (each validated like any source). |
| `--no-test` / `--test100` | 250-verse test set | Alternative test-set sizing (mutually exclusive). |
| `--run` | off | Run each created experiment without asking first. |

Book lists (`--training-books`, `--translate-books`) accept any silnlp book
selection, kept verbatim: plain lists, `OT`/`NT`, ranges (`GEN-DEU`), chapter
selections (`"MAT 1-4"`), subtractions (`"NT;-REV"`); separate with `;`, `,`, or
spaces (quote semicolons/spaces). Checkpoint is fixed at 5000 (edit the file to
change).

## Folder naming & common names

Experiments are placed under `<Country>/<Language>/`. Country and language names
come from `silnlp/assets/languageFamilies.json`, which uses official
Ethnologue/ISO forms (e.g. `Tanzania, United Republic of`). To keep folders easy
to find, `silnlp/assets/nameOverrides.json` maps those to common names
(`Tanzania`, `Russia`, `DRC`, …). Names not listed pass through unchanged.

**To add or change a common name**, edit `nameOverrides.json` — add an entry
under `"countries"` or `"languages"` whose *key* is the exact string from
`languageFamilies.json` and whose *value* is the common name you want. Do **not**
edit `languageFamilies.json` (it is a verbatim Ethnologue snapshot).

If a folder created under the *old* official name already exists, the tool prints
a warning at startup so you can merge it into the common-name folder manually; it
never moves or deletes anything itself.

## Worked example

```bash
python -m silnlp.common.create_onboarding_experiments Ndamba_2026_07_13 --translate-books "MRK"
```

For a Tanzanian target language this creates, after you pick experiments from the
table:

```
MT/experiments/
└── Tanzania/                 # common name, not "Tanzania_United_Republic_Of"
    └── Ndamba/
        ├── NIV11R_ndj_0/
        │   ├── config.yml
        │   └── translate_config.yml
        └── ...
```

## Troubleshooting

- **`Iso code '<x>' not found in languageFamilies.json`** — the target's iso is
  not in the Ethnologue snapshot; check the extract stem / `--target`.
- **`WARNING: a folder for the official country name already exists`** — a folder
  from before the common-name change exists; merge it into the new location by
  hand.
- **Source-project / translate-books warnings** — the chosen drafting source is
  missing from `MT/Paratext/projects` or lacks some `--translate-books`; the tool
  offers an alternative or you can pass `--translate-scripture`.
