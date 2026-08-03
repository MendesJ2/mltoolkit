import pandas as pd

from .quality import quality_analysis


def build_quality_report(
    dataset,
    only_features=True,
):
    """
    Build one quality row per dataset column.
    """

    if only_features:

        columns = (
            dataset.feature_columns
        )

    else:

        columns = list(
            dataset.df.columns
        )

    records = []

    for column in columns:

        quality = quality_analysis(
            dataset.df[column]
        )

        record = {
            "feature": column,
            **quality.to_dict(),
        }

        metadata = dataset.feature(
            column
        )

        record["role"] = (
            metadata.role
        )

        record["variable_type"] = (
            metadata.variable_type
        )

        record["is_constant"] = (
            metadata.is_constant
        )

        record["is_quasi_constant"] = (
            metadata.is_quasi_constant
        )

        records.append(record)

    if not records:
        return pd.DataFrame()

    report = pd.DataFrame(
        records
    )

    priority_columns = [
        "feature",
        "role",
        "variable_type",
        "dtype",
        "observations",
        "missing",
        "missing_pct",
        "unique_values",
        "unique_pct",
        "mode",
        "mode_pct",
        "is_constant",
        "is_quasi_constant",
        "outliers",
        "outliers_pct",
    ]

    existing_priority = [
        column
        for column in priority_columns
        if column in report.columns
    ]

    remaining_columns = [
        column
        for column in report.columns
        if column not in existing_priority
    ]

    return report[
        existing_priority
        + remaining_columns
    ]
