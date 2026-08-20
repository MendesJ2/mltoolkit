import numpy as np
import pandas as pd


MISSING_LABEL = "__MISSING__"


def create_bins(
    series: pd.Series,
    n_bins: int = 10,
    special_values=None,
):
    """
    Create quantile bins while isolating special values.

    Special values, such as -999 and -9999, receive
    individual groups and are never included in qcut.
    """

    special_values = list(
        special_values or []
    )

    original = series.copy()

    result = pd.Series(
        index=original.index,
        dtype="object",
    )

    missing_mask = original.isna()

    result.loc[
        missing_mask
    ] = MISSING_LABEL

    for special_value in special_values:

        special_mask = (
            original.eq(special_value)
            & ~missing_mask
        )

        result.loc[
            special_mask
        ] = str(special_value)

    regular_mask = (
        result.isna()
        & original.notna()
    )

    regular_values = original.loc[
        regular_mask
    ]

    if regular_values.empty:
        return result

    if (
        regular_values.nunique(
            dropna=True
        )
        <= n_bins
    ):

        result.loc[
            regular_mask
        ] = regular_values.astype(str)

        return result

    try:

        regular_bins = pd.qcut(
            regular_values,
            q=n_bins,
            duplicates="drop",
        )

        result.loc[
            regular_mask
        ] = regular_bins.astype(str)

    except ValueError:

        result.loc[
            regular_mask
        ] = regular_values.astype(str)

    return result


def build_bin_order(
    binned_series,
    special_values=None,
):
    """
    Return a stable logical order for bins.

    Order:
        special values
        numeric values / intervals
        missing
    """

    special_values = list(
        special_values or []
    )

    values = (
        pd.Series(binned_series)
        .drop_duplicates()
        .tolist()
    )

    special_labels = [
        str(value)
        for value in special_values
        if str(value) in values
    ]

    missing_labels = [
        value
        for value in values
        if value == MISSING_LABEL
    ]

    regular_labels = [
        value
        for value in values
        if value not in special_labels
        and value not in missing_labels
    ]

    regular_labels = sorted(
        regular_labels,
        key=_bin_sort_key,
    )

    return (
        special_labels
        + regular_labels
        + missing_labels
    )


def _bin_sort_key(
    value,
):
    text = str(value)

    if text.startswith(
        ("(", "[")
    ):

        first_value = (
            text[1:]
            .split(",")[0]
            .strip()
        )

        try:
            return float(first_value)

        except ValueError:
            return np.inf

    try:
        return float(text)

    except ValueError:
        return np.inf


def fit_quantile_bins(
    series: pd.Series,
    n_bins: int = 10,
    special_values=None,
):
    """
    Learn quantile bin edges from a reference sample.

    Special values and missing values are excluded
    from the quantile calculation.

    Returns
    -------
    list[float] | None
        Bin edges suitable for pd.cut().
    """

    special_values = list(
        special_values or []
    )

    regular = (
        series
        .dropna()
    )

    if special_values:

        regular = regular[
            ~regular.isin(
                special_values
            )
        ]

    if regular.empty:
        return None

    if (
        regular.nunique(
            dropna=True
        )
        <= n_bins
    ):
        return None

    try:

        _, edges = pd.qcut(
            regular,
            q=n_bins,
            duplicates="drop",
            retbins=True,
        )

    except ValueError:
        return None

    edges = np.asarray(
        edges,
        dtype=float,
    )

    # Allow future/OOT values outside
    # Development min/max.
    edges[0] = -np.inf
    edges[-1] = np.inf

    return edges.tolist()


def apply_fixed_bins(
    series: pd.Series,
    bin_edges,
    special_values=None,
):
    """
    Apply previously learned continuous bin edges.

    Missing and special values remain isolated.
    """

    special_values = list(
        special_values or []
    )

    result = pd.Series(
        index=series.index,
        dtype="object",
    )

    missing_mask = (
        series.isna()
    )

    result.loc[
        missing_mask
    ] = MISSING_LABEL

    for special_value in (
        special_values
    ):

        special_mask = (
            series.eq(
                special_value
            )
            & ~missing_mask
        )

        result.loc[
            special_mask
        ] = str(
            special_value
        )

    regular_mask = (
        result.isna()
        & series.notna()
    )

    if not regular_mask.any():
        return result

    if bin_edges is None:

        result.loc[
            regular_mask
        ] = (
            series.loc[
                regular_mask
            ]
            .astype(str)
        )

        return result

    binned = pd.cut(
        series.loc[
            regular_mask
        ],
        bins=bin_edges,
        include_lowest=True,
        duplicates="drop",
    )

    result.loc[
        regular_mask
    ] = (
        binned.astype(str)
    )

    return result
