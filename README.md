# mltoolkit

`mltoolkit` is a reusable Python framework for structured binary-classification projects.

The toolkit contains generic components for dataset metadata, exploratory data analysis, feature selection, preprocessing and modeling. Project-specific rules stay outside the toolkit, in a separate `projects/<project_name>/` folder.

## Philosophy

The main design principles are:

* keep the framework generic and reusable;
* keep business rules and hardcoded feature engineering inside each project;
* prefer simple, readable components over excessive abstraction;
* allow manual overrides for roles and variable types;
* run EDA early, before aggressively cleaning the data;
* refine preprocessing and feature engineering iteratively after inspecting the EDA report;
* keep Train/Test/OOT construction specific to the project.

A typical workflow is:

```text
Raw data
   ↓
Project preprocessing
   ↓
Project feature engineering
   ↓
Dataset + metadata
   ↓
EDA / EDA report
   ↓
Iterate preprocessing + features if needed
   ↓
Train / Validation / OOT split
   ↓
Feature selection
   ↓
Modeling preprocessing / encoding
   ↓
Baseline model
   ↓
Validation / OOT evaluation
```

## Recommended workspace structure

`mltoolkit` is intended to live alongside project-specific code rather than contain it.

```text
workspace/
│
├── mltoolkit/
│   ├── core/
│   ├── data/
│   ├── eda/
│   ├── modeling/
│   ├── preprocessing/
│   ├── project/
│   ├── selection/
│   └── ...
│
├── projects/
│   ├── __init__.py
│   └── my_project/
│       ├── __init__.py
│       ├── config.py
│       ├── preprocessing.py
│       ├── feature_engineering.py
│       ├── project.py
│       └── train.py
│
├── notebooks/
│   └── my_project/
│       ├── 01_dataset_eda.ipynb
│       ├── 02_selection.ipynb
│       └── 03_modeling.ipynb
│
├── reports/
│   └── my_project/
│
└── bootstrap.py
```

The repository includes a minimal starter project in `examples/project_template/`.

## Project configuration

The central configuration object is `mltoolkit.core.config.Config`.

Example:

```python
from mltoolkit.core.config import Config

config = Config(
    project_name="my_project",
    target="TARGET",
    date_column="DATA_REFERENCIA",
    source_column=None,

    # Optional whitelist.
    # None means that role inference determines
    # which columns are eligible features.
    feature_columns=None,

    id_columns=[
        "ID",
    ],

    ignore_columns=[
        "AUXILIARY_COLUMN",
    ],

    # Manual variable-type overrides.
    variable_types={
        "SEGMENT": "categorical",
        "AGE": "continuous",
    },

    # Manual role overrides.
    role_overrides={
        "SAMPLE": "ignored",
        "DATA_REFERENCIA": "date",
    },

    # Values isolated from regular continuous bins.
    special_values=[
        -999,
        -9999,
    ],
)
```

Useful roles currently include:

```text
feature
target
source
date
id
ignored
```

Useful variable types include:

```text
continuous
categorical
binary
ordinal
datetime
```

`feature_columns=None` is valid.

Collections such as `role_overrides`, `variable_types`, `id_columns` and `ignore_columns` should normally be omitted when unused, so their default empty dictionaries/lists are used.

## Project preprocessing

Project-specific cleaning belongs in:

```text
projects/<project_name>/preprocessing.py
```

The first version can deliberately do almost nothing:

```python
def preprocess(df):
    df = df.copy()

    return df
```

This is intentional.

Missing values, unexpected categories and outliers are useful information during the first EDA iteration.

After inspecting the report, preprocessing can progressively include rules such as:

```python
def preprocess(df):
    df = df.copy()

    df["CATEGORY"] = (
        df["CATEGORY"]
        .astype("string")
        .str.strip()
    )

    return df
```

The toolkit should not silently make business-specific cleaning decisions.

## Project feature engineering

Feature creation belongs in:

```text
projects/<project_name>/feature_engineering.py
```

Keep this file simple and explicit:

```python
def create_features(df):
    df = df.copy()

    # Example:
    # df["RATIO"] = (
    #     df["VALUE_A"]
    #     / df["VALUE_B"]
    # )

    return df
```

There is intentionally no large catalogue of generic difference/ratio helpers.

Each project has freedom to create the features it needs directly with pandas / NumPy.

A normal workflow is to add new features after one or more EDA iterations and regenerate the report.

## Project class

A project class connects project-specific preprocessing and feature engineering to the generic `BaseProject` / `Dataset`.

```python
from mltoolkit.project.project import BaseProject

from .config import config
from .feature_engineering import create_features
from .preprocessing import preprocess


class MyProject(BaseProject):

    def __init__(
        self,
        logger=None,
    ):
        super().__init__(
            config=config,
            logger=logger,
        )

    def prepare_dataset(
        self,
        dataframe,
    ):
        dataframe = preprocess(
            dataframe
        )

        dataframe = create_features(
            dataframe
        )

        return self.load_dataset(
            dataframe
        )
```

`BaseProject.load_dataset()` creates the generic `Dataset` and builds its metadata.

## Bootstrap

A small bootstrap file is useful when notebooks live outside the toolkit package.

Example:

```python
from pathlib import Path
import sys


WORKSPACE_ROOT = (
    Path(__file__)
    .resolve()
    .parent
)


if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(WORKSPACE_ROOT),
    )


from projects.my_project.project import (
    MyProject,
)


project = MyProject(
    logger=None
)
```

A notebook can then use:

```python
from bootstrap import *

import pandas as pd


raw_df = pd.read_pickle(
    "..."
)

project.prepare_dataset(
    raw_df
)

project.dataset.shape
project.dataset.features
```

For multiple projects, either use separate bootstrap files or instantiate the required project explicitly.

## Dataset and metadata

`Dataset` retains two copies when it is created:

```text
raw_df  dataframe received by Dataset
df      working dataframe stored by Dataset
```

Project preprocessing and feature engineering normally happen before `Dataset` is created, so both represent the prepared dataframe at that point.

Metadata is inferred for every column and includes:

* role;
* variable type;
* dtype;
* number of unique values;
* missing percentage;
* constant / quasi-constant flags.

Manual configuration overrides automatic inference when required.

## EDA

Create an analyzer from the prepared dataset:

```python
from mltoolkit.eda import EDAAnalyzer


eda = EDAAnalyzer(
    dataset=project.dataset,
    config=project.config,
    logger=project.logger,
)
```

Individual feature analysis:

```python
eda.feature(
    "AGE"
).summary()

eda.feature(
    "AGE"
).plot().show()

eda.feature(
    "AGE"
).target().plot().show()
```

When a source / segment column exists:

```python
eda.feature(
    "AGE"
).target(
    group="SOURCE"
).plot().show()
```

## EDA report

The report is the main EDA output.

```python
report_folder = (
    eda.export_report(
        output_folder=(
            "reports/my_project/eda"
        ),
        date_column=(
            "DATA_REFERENCIA"
        ),
        temporal_freq="Q",
        source_column=None,
        stability_by="sample",
        stability_reference="c",
        n_bins=10,
    )
)
```

The generated folder contains:

```text
index.html
strength.html
eda_summary.xlsx
features/
relationships/
```

The feature pages consolidate:

* metadata;
* summary;
* statistics;
* quality;
* distribution;
* feature vs target;
* temporal evolution;
* WoE / IV;
* lift;
* gain / KS;
* stability / PSI.

When `source_column` is supplied, target and strength analyses also compare source groups.

Without a source column, the global Feature Strength page is still generated.

## Missing values during EDA

The first EDA does not require all missing values to be imputed.

Current EDA components are designed to expose missingness rather than hide it:

* metadata reports missing percentages;
* categorical missing values appear as their own group;
* continuous target and stability binning isolate missing values;
* configured special values such as `-999` and `-9999` are isolated from regular continuous bins.

This makes it possible to decide whether missingness itself is informative before choosing an imputation strategy.

The target and structural project columns should still be validated carefully before modeling.

## Iterative EDA workflow

A practical first project iteration is:

```text
1. Read source data
2. Minimal preprocessing
3. Minimal feature engineering
4. Build Dataset
5. Generate EDA report
6. Review missing values, types, distributions,
   outliers and stability
7. Update config overrides
8. Update preprocessing
9. Add useful engineered features
10. Regenerate EDA report
11. Repeat until ready for modeling
```

This iteration is expected and is part of the toolkit design.

## Feature strength

The EDA report contains a global Feature Strength page with rankings based on:

* Information Value (IV);
* KS;
* maximum univariate lift.

When a `source_column` is provided, it also contains strength comparisons by source/group, including heatmaps.

This is useful when a feature is predictive only in certain populations.

## Stability

Stability analysis can compare populations such as construction vs OOT using a column like:

```text
sample = "c"  construction
sample = "o"  out-of-time
```

Example:

```python
eda.feature(
    "AGE"
).stability(
    by="sample",
    reference="c",
).plot_psi().show()
```

Continuous special values remain isolated in stability bins.

## Modeling workflow

The intended stage after EDA is:

```text
Development / OOT definition
   ↓
Random Train / Validation split
inside development
   ↓
FeatureFilter fit on Train
   ↓
ModelingPreprocessor fit on Train
   ↓
Transform Validation and OOT
   ↓
ForwardSelector
   ↓
Logistic baseline
   ↓
Validation / OOT evaluation
```

OOT should not be used to make feature-selection or preprocessing-fit decisions.

The modeling API is still being refined as the MVP evolves.

## Example template

See:

```text
examples/project_template/
```

The files are deliberately minimal and are intended to be copied into:

```text
projects/<project_name>/
```

and adapted to the new problem.

## Status

Current focus:

* reusable Dataset / metadata layer;
* project-level preprocessing and feature engineering;
* comprehensive EDA and HTML report;
* feature quality, relationships, strength and stability;
* binary-classification feature selection;
* logistic baseline workflow.

Further improvements can be added after the MVP workflow is stable and tested across multiple datasets.
