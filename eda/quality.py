import pandas as pd
import numpy as np


def quality_analysis(
    series,
):

    result = {}

    s = series.copy()


    # basic

    result["dtype"] = str(
        s.dtype
    )

    result["observations"] = len(s)

    result["missing"] = int(
        s.isna().sum()
    )

    result["missing_pct"] = (
        s.isna().mean()
    )


    result["unique_values"] = (
        s.nunique(dropna=True)
    )


    result["unique_pct"] = (
        s.nunique(dropna=True)
        /
        len(s)
    )


    # mode

    mode = s.mode()

    result["mode"] = (
        mode.iloc[0]
        if len(mode) > 0
        else None
    )


    result["mode_pct"] = (
        s.value_counts(
            normalize=True,
            dropna=False
        )
        .iloc[0]
    )


    # numeric analysis

    if pd.api.types.is_numeric_dtype(s):

        result.update({

            "min": s.min(),

            "p01": s.quantile(0.01),

            "p05": s.quantile(0.05),

            "median": s.median(),

            "p95": s.quantile(0.95),

            "p99": s.quantile(0.99),

            "max": s.max(),

            "mean": s.mean(),

            "std": s.std(),

        })


        # outliers IQR

        q1 = s.quantile(0.25)

        q3 = s.quantile(0.75)

        iqr = q3 - q1


        lower = q1 - 1.5 * iqr

        upper = q3 + 1.5 * iqr


        outliers = (
            (s < lower)
            |
            (s > upper)
        )


        result["outliers"] = int(
            outliers.sum()
        )

        result["outliers_pct"] = (
            outliers.mean()
        )


    return pd.Series(result)
