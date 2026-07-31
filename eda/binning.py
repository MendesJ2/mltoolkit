import pandas as pd


def create_bins(
    series,
    n_bins=10
):
    """
    Creates quantile bins for continuous variables.
    """

    valid = series.dropna()

    if valid.nunique() <= n_bins:
        return series

    return pd.qcut(
        series,
        q=n_bins,
        duplicates="drop"
    )
