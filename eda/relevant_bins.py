from __future__ import annotations

import numpy as np
import pandas as pd

from .strength import feature_strength


def relevant_bins_analysis(
    dataset,
    *,
    columns=None,
    group=None,
    n_bins=10,
    min_lift_deviation=0.25,
    min_population_pct=0.05,
    include_global=True,
):
    """
    Identify feature bins with relevant univariate lift.

    A bin is considered relevant when:

        abs(lift - 1) >= min_lift_deviation

    and:

        population_pct >= min_population_pct

    Lift is calculated relative to the corresponding
    population:

        - global bins -> global target rate
        - grouped bins -> target rate of that group

    Parameters
    ----------
    dataset:
        mltoolkit Dataset.

    columns:
        Features to analyse. If None, use all
        dataset feature columns.

    group:
        Optional grouping column, e.g. TAREFA.

    n_bins:
        Number of bins for continuous variables.

    min_lift_deviation:
        Minimum absolute distance from neutral lift = 1.

        Example:
            0.25 accepts lift >= 1.25
            or lift <= 0.75.

    min_population_pct:
        Minimum population represented by the bin.

    include_global:
        Whether global results should also be returned
        when group is supplied.

    Returns
    -------
    pandas.DataFrame
    """

    if columns is None:
        columns = list(
            dataset.feature_columns
        )

    if group is not None:
        if group not in dataset.df.columns:
            raise ValueError(
                f"Group column '{group}' "
                "not found."
            )

    if min_lift_deviation < 0:
        raise ValueError(
            "min_lift_deviation must be >= 0."
        )

    if not (
        0 < min_population_pct <= 1
    ):
        raise ValueError(
            "min_population_pct must "
            "be between 0 and 1."
        )

    target = dataset.config.target

    special_values = getattr(
        dataset.config,
        "special_values",
        [
            -999,
            -9999,
        ],
    )

    tables = []

    for feature_name in columns:

        metadata = dataset.feature(
            feature_name
        )

        try:

            result = feature_strength(
                df=dataset.df,
                feature=feature_name,
                target=target,
                variable_type=(
                    metadata.variable_type
                ),
                n_bins=n_bins,
                group=group,
                special_values=(
                    special_values
                ),
                min_lift_population_pct=(
                    min_population_pct
                ),
            )

            table = (
                result["table"]
                .copy()
            )

            if (
                group is not None
                and not include_global
            ):

                table = table[
                    table["scope"]
                    != "global"
                ]

            table[
                "lift_deviation"
            ] = (
                table["lift"] - 1
            ).abs()

            table = table[
                (
                    table[
                        "population_pct"
                    ]
                    >= min_population_pct
                )
                &
                (
                    table[
                        "lift_deviation"
                    ]
                    >= min_lift_deviation
                )
            ].copy()

            if table.empty:
                continue

            table.insert(
                0,
                "feature",
                feature_name,
            )

            table.insert(
                1,
                "variable_type",
                metadata.variable_type,
            )

            tables.append(
                table
            )

        except (
            ValueError,
            TypeError,
            KeyError,
        ):
            continue

    if not tables:

        return pd.DataFrame(
            columns=[
                "feature",
                "variable_type",
                "scope",
                "group_value",
                "feature_group",
                "observations",
                "population_pct",
                "target_rate",
                "lift",
                "lift_deviation",
            ]
        )

    result = pd.concat(
        tables,
        ignore_index=True,
    )

    display_columns = [
        "feature",
        "variable_type",
        "scope",
        "group_value",
        "feature_group",
        "observations",
        "events",
        "population_pct",
        "target_rate",
        "lift",
        "lift_deviation",
    ]

    display_columns = [
        column
        for column in display_columns
        if column in result.columns
    ]

    return (
        result[
            display_columns
        ]
        .sort_values(
            [
                "lift_deviation",
                "population_pct",
            ],
            ascending=[
                False,
                False,
            ],
        )
        .reset_index(
            drop=True
        )
    )
