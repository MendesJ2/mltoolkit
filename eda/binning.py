import pandas as pd


def create_bins(
    series: pd.Series,
    n_bins: int = 10,
):
    """
    Create quantile bins for continuous variables.
    """

    series = series.copy()

    if series.dropna().nunique() <= n_bins:
        return series.astype(str)

    try:

        bins = pd.qcut(
            series,
            q=n_bins,
            duplicates="drop"
        )

        return bins.astype(str)

    except ValueError:

        return series.astype(str)
