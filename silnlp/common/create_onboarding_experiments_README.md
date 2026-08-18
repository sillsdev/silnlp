# How to use `create_onboarding_experiments`

The create_onboarding_experiments.py script creates NMT experiment folders.

It takes as an argument a path to a folder where it will find the details necessary
to know which experiments to create. The folders it accepts are either the
output of either the production onboarding script or the folder where analyze.py
saved a `corpus-stats.csv` file. 

From the alignment the script finds the iso code and looks up the country and 
language from the files assets/languageFamilies.json and assests/nameOverrides.json

The script finds the alignments and shows the relevant data to the user. Then 
the user can select which scriptures to use in experiments. The script then proposes
a list of possible experiments and the user chooses which to run. 

At the moment the script doesn't know about Back Translations, but I hope to 
add better handling of Back translations in the future.

For each experiment the script creates an experiment folder containing a 
`config.yml` and `translate_config.yml` under
`<Country>/<Language>/<experiment>/` in the experiments folder..

## Prerequisites

- Environment configured for data access (`SIL_NLP_ENV`.
- access to the experiments folder on Minio.
- The request folder present under `MT/experiments/_OnboardingRequests/`, or an
  analyze folder containing `corpus-stats.csv`.

## Basic usage

```bash
python -m silnlp.common.create_onboarding_experiments <request> --translate-books <books>
```

Example:

```bash
python -m silnlp.common.create_onboarding_experiments SFDH_2026_07_13 --translate-books "MAT;MRK"
```

`<request>` is the request folder (with or without the `_Request` suffix),
or a folder relative to `MT/experiments` that holds a `corpus-stats.csv`.

You can create the experiment folders then optionally edit the `config.yml` or
`translate_config.yml` files before accepting the option to run all the experiments.

## Options

| Option | Default | Purpose |
|---|---|---|
| `--translate-books <books>` | *required* | Book(s) for `translate_config.yml` (e.g. `MAT`, `"MAT;MRK"`).|
| `--training-books <books>` | `complete` | Books written to `corpus_books`, or `complete` to derive them from `verse_counts.csv` (≥98% rule; testaments compacted to `NT`/`OT`). |
| `--target <iso or project>` | auto | Force the target language/project (use the project name when two projects share an iso). |
| `--min-parallel <n>` | `2000` | Minimum parallel verse count for a reference pair. |
| `--min-alignment <a>` | `0.2` | Minimum alignment score for a reference pair. |
| `--top <n>` | `20` | Maximum number of experiments offered for selection. |
| `--translate-scripture <projects>` | auto | Override the drafting source(s) entirely (each validated like any source). |
| `--no-test` / `--test100` | 250-verse test set | Alternative test-set sizing (mutually exclusive). |


Book lists (`--training-books`, `--translate-books`) accept any silnlp book
selection, kept verbatim: plain lists, `OT`/`NT`, ranges (`GEN-DEU`), chapter
selections (`"MAT 1-4"`), subtractions (`"NT;-REV"`); separate with semicolons, commas or spaces.
Quotes are required if the list contains a semicolon or a space.

## Folder naming & common names

Experiments are placed under `experiments/<Country>/<Language>/`. Country and language names
come from `silnlp/assets/languageFamilies.json`, which uses official
Ethnologue/ISO forms (e.g. `Tanzania, United Republic of`). There is a mapping to common names
in `silnlp/assets/nameOverrides.json`. (`Tanzania`, `Russia`, `DRC`, …).

**To add or change a common name**
Edit `nameOverrides.json` by adding an entry under `"countries"` or `"languages"`
The *key* is the exact string from `languageFamilies.json` and the *value* is the common name.
Don't edit `languageFamilies.json`, it should remain a verbatim copy from the Ethnologue.

If a folder created under the official name already exists, the script gives a warning. Which is a 
good time to merge those experiments into the common-name folder manually. 

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

- **`Iso code '<x>' not found in languageFamilies.json`** the target's iso is
  not found in languageFamilies.json. Check the extract stem / `--target` and see whether 
  it is an unofficial iso code.
- **`WARNING: a folder for the official country name already exists`**.
  A folder exists that uses the official country name rather than the common-name.
  Move the existing experiments into the new location manually if you wish.
- **Source-project / translate-books warnings**
  The chosen drafting source doesn't exist in the `MT/Paratext/projects` or some `--translate-books`
  are missing or empty. You can then select an alternative drafting source.
  Or you can pass one in with the `--translate-scripture` option.
