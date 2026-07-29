import pandas as pd


def dataset_summary(dataset):

    df = dataset.df

    return pd.DataFrame(
        {
            "metric": [
                "rows",
                "columns",
                "continuous_features",
                "categorical_features",
                "binary_features",
                "datetime_features",
            ],

            "value": [
                df.shape[0],
                df.shape[1],
                len(dataset.continuous),
                len(dataset.categorical),
                len(dataset.binary),
                len(dataset.dates),
            ]
        }
    )


def feature_summary(dataset):

    return (
        dataset.metadata
        .to_dataframe()
        .copy()
    )


def missing_summary(dataset):

    df = dataset.df

    result = pd.DataFrame(
        {
            "feature": df.columns,
            "missing_count": df.isna().sum(),
            "missing_pct": (
                df.isna()
                .mean()
                .values
            )
        }
    )

    return (
        result
        .sort_values(
            "missing_pct",
            ascending=False
        )
        .reset_index(drop=True)
    )


def statistics_summary(dataset):

    return (
        dataset.df
        .describe(
            include="all"
        )
        .T
        .reset_index()
        .rename(
            columns={
                "index": "feature"
            }
        )
    )


def single_feature_summary(
    series,
    name,
):

    return pd.DataFrame(
        {
            "feature": [name],
            "dtype": [
                str(series.dtype)
            ],
            "rows": [
                len(series)
            ],
            "unique": [
                series.nunique()
            ],
            "missing": [
                series.isna().sum()
            ],
            "missing_pct": [
                series.isna().mean()
            ],
        }
    )


def single_feature_statistics(
    series,
    name,
):

    if pd.api.types.is_numeric_dtype(series):

        return pd.DataFrame(
            {
                "feature": [name],
                "mean": [series.mean()],
                "std": [series.std()],
                "min": [series.min()],
                "median": [series.median()],
                "max": [series.max()],
            }
        )

    return pd.DataFrame(
        {
            "feature": [name],
            "unique": [
                series.nunique()
            ],
            "top": [
                series.mode().iloc[0]
                if not series.mode().empty
                else None
            ],
        }
    )
