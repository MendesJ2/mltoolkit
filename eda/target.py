import numpy as np
import pandas as pd

from .binning import (
    MISSING_LABEL,
    build_bin_order,
    create_bins,
)


def target_analysis(
    df,
    feature,
    target,
    variable_type,
    n_bins=10,
    group=None,
    special_values=None,
    confidence_level=0.95,
):
    """
    Analyse feature vs binary target.

    When group is supplied, returns:
        - global analysis;
        - analysis for every group value.

    Continuous features use the same global bins for
    all groups, ensuring comparable curves.
    """

    columns = [
        feature,
        target,
    ]

    if group is not None:
        columns.append(group)

    data = df[
        columns
    ].copy()

    data = data.dropna(
        subset=[target]
    )

    if variable_type == "continuous":

        data["_feature_group"] = (
            create_bins(
                data[feature],
                n_bins=n_bins,
                special_values=(
                    special_values
                ),
            )
        )

        category_order = build_bin_order(
            data["_feature_group"],
            special_values=(
                special_values
            ),
        )

    else:

        data["_feature_group"] = (
            data[feature]
            .astype("object")
            .where(
                data[feature].notna(),
                MISSING_LABEL,
            )
            .astype(str)
        )

        category_order = (
            data["_feature_group"]
            .drop_duplicates()
            .tolist()
        )

    global_table = _aggregate_target(
        data=data,
        target=target,
        grouping=[
            "_feature_group"
        ],
        global_rate=data[target].mean(),
        confidence_level=(
            confidence_level
        ),
    )

    global_table["scope"] = "global"
    global_table["group_value"] = "Global"

    tables = [
        global_table
    ]

    if group is not None:

        group_table = _aggregate_target(
            data=data,
            target=target,
            grouping=[
                group,
                "_feature_group",
            ],
            global_rate=None,
            confidence_level=(
                confidence_level
            ),
        )
        
        group_rates = (
            data
            .groupby(
                group,
                dropna=False,
            )[target]
            .mean()
        )
        
        group_table[
            "event_rate_index"
        ] = (
            group_table["target_rate"]
            / group_table[group].map(
                group_rates
            )
        )

        group_table["scope"] = "group"
        group_table["group_value"] = (
            group_table[group].astype(str)
        )

        tables.append(
            group_table
        )

    result = pd.concat(
        tables,
        ignore_index=True,
        sort=False,
    )

    result["feature_group"] = (
        result["_feature_group"]
        .astype(str)
    )

    result["feature_order"] = (
        pd.Categorical(
            result["feature_group"],
            categories=category_order,
            ordered=True,
        )
    )

    sort_columns = [
        "scope",
        "group_value",
        "feature_order",
    ]

    result = (
        result
        .sort_values(sort_columns)
        .reset_index(drop=True)
    )

    result["population_pct"] = (
        result["observations"]
        / result.groupby(
            [
                "scope",
                "group_value",
            ]
        )["observations"]
        .transform("sum")
    )

    return result.drop(
        columns=[
            "_feature_group",
            "feature_order",
        ],
        errors="ignore",
    )


def _aggregate_target(
    data,
    target,
    grouping,
    global_rate,
    confidence_level,
):
    result = (
        data
        .groupby(
            grouping,
            dropna=False,
            observed=False,
        )
        .agg(
            observations=(
                target,
                "count",
            ),
            events=(
                target,
                "sum",
            ),
            target_rate=(
                target,
                "mean",
            ),
        )
        .reset_index()
    )

    result["non_events"] = (
        result["observations"]
        - result["events"]
    )

    if global_rate is not None:
    
        result["event_rate_index"] = (
            result["target_rate"]
            / global_rate
        )
    
    else:
    
        result["event_rate_index"] = np.nan

    lower, upper = _wilson_interval(
        events=result["events"],
        observations=(
            result["observations"]
        ),
        confidence_level=(
            confidence_level
        ),
    )

    result["target_rate_ci_lower"] = lower
    result["target_rate_ci_upper"] = upper

    return result


def _wilson_interval(
    events,
    observations,
    confidence_level=0.95,
):
    # z=1.96 for the default 95% confidence interval.
    # The MVP keeps this dependency-free.
    z = (
        1.96
        if confidence_level == 0.95
        else 1.96
    )

    observations = observations.astype(float)
    events = events.astype(float)

    proportion = (
        events
        / observations.replace(0, np.nan)
    )

    denominator = (
        1
        + (z ** 2 / observations)
    )

    centre = (
        proportion
        + z ** 2
        / (2 * observations)
    ) / denominator

    distance = (
        z
        * np.sqrt(
            (
                proportion
                * (1 - proportion)
                / observations
            )
            + (
                z ** 2
                / (
                    4
                    * observations ** 2
                )
            )
        )
        / denominator
    )

    return (
        (centre - distance).clip(0, 1),
        (centre + distance).clip(0, 1),
    )
