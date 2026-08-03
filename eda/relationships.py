import numpy as np
import pandas as pd

from scipy.stats import chi2_contingency


def numeric_correlation(
    df,
    columns,
    method="pearson",
):
    """
    Correlation matrix for numeric features.
    """

    valid_methods = {
        "pearson",
        "spearman",
    }

    if method not in valid_methods:
        raise ValueError(
            f"Invalid method '{method}'. "
            f"Use one of {sorted(valid_methods)}."
        )

    available_columns = [
        column
        for column in columns
        if column in df.columns
    ]

    if not available_columns:
        return pd.DataFrame()

    numeric_data = (
        df[available_columns]
        .select_dtypes(include="number")
    )

    if numeric_data.empty:
        return pd.DataFrame()

    return numeric_data.corr(
        method=method
    )


def cramers_v(
    series_x,
    series_y,
):
    """
    Bias-corrected Cramér's V.
    """

    data = pd.DataFrame(
        {
            "x": series_x,
            "y": series_y,
        }
    ).dropna()

    if data.empty:
        return np.nan

    contingency = pd.crosstab(
        data["x"],
        data["y"],
    )

    if (
        contingency.shape[0] < 2
        or contingency.shape[1] < 2
    ):
        return np.nan

    chi2 = chi2_contingency(
        contingency,
        correction=False,
    )[0]

    n = contingency.to_numpy().sum()

    if n <= 1:
        return np.nan

    phi2 = chi2 / n

    rows, columns = contingency.shape

    phi2_corrected = max(
        0,
        phi2
        - (
            (columns - 1)
            * (rows - 1)
            / (n - 1)
        ),
    )

    rows_corrected = (
        rows
        - ((rows - 1) ** 2)
        / (n - 1)
    )

    columns_corrected = (
        columns
        - ((columns - 1) ** 2)
        / (n - 1)
    )

    denominator = min(
        columns_corrected - 1,
        rows_corrected - 1,
    )

    if denominator <= 0:
        return np.nan

    return np.sqrt(
        phi2_corrected
        / denominator
    )


def categorical_association(
    df,
    columns,
):
    """
    Pairwise Cramér's V matrix.
    """

    available_columns = [
        column
        for column in columns
        if column in df.columns
    ]

    matrix = pd.DataFrame(
        index=available_columns,
        columns=available_columns,
        dtype=float,
    )

    for column_x in available_columns:

        for column_y in available_columns:

            if column_x == column_y:

                matrix.loc[
                    column_x,
                    column_y,
                ] = 1.0

                continue

            matrix.loc[
                column_x,
                column_y,
            ] = cramers_v(
                df[column_x],
                df[column_y],
            )

    return matrix


def high_relationship_pairs(
    matrix,
    threshold=0.80,
):
    """
    Return feature pairs above a relationship threshold.
    """

    if matrix.empty:
        return pd.DataFrame(
            columns=[
                "feature_1",
                "feature_2",
                "relationship",
                "absolute_relationship",
            ]
        )

    records = []

    columns = list(
        matrix.columns
    )

    for index, feature_1 in enumerate(columns):

        for feature_2 in columns[
            index + 1:
        ]:

            value = matrix.loc[
                feature_1,
                feature_2,
            ]

            if pd.isna(value):
                continue

            absolute_value = abs(value)

            if absolute_value < threshold:
                continue

            records.append(
                {
                    "feature_1": feature_1,
                    "feature_2": feature_2,
                    "relationship": value,
                    "absolute_relationship": (
                        absolute_value
                    ),
                }
            )

    if not records:
        return pd.DataFrame(
            columns=[
                "feature_1",
                "feature_2",
                "relationship",
                "absolute_relationship",
            ]
        )

    return (
        pd.DataFrame(records)
        .sort_values(
            "absolute_relationship",
            ascending=False,
        )
        .reset_index(drop=True)
    )
