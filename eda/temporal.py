import pandas as pd

from .binning import MISSING_LABEL


def temporal_analysis(
    df,
    feature,
    target,
    date,
    variable_type,
    freq="M",
    group=None,
    special_values=None,
):
    """
    Analyse feature evolution through time.

    Continuous:
        mean and median by target.

    Categorical:
        category share by target.
    
    Binary:
        proportion of feature == 1 by target.

    Special values are excluded from continuous averages.
    """

    special_values = list(
        special_values or []
    )

    columns = [
        feature,
        target,
        date,
    ]

    if group is not None:
        columns.append(group)

    data = df[
        columns
    ].copy()

    data[date] = pd.to_datetime(
        data[date],
        errors="coerce",
    )

    data = data.dropna(
        subset=[
            date,
            target,
        ]
    )

    data["period"] = (
        data[date]
        .dt
        .to_period(freq)
        .astype(str)
    )

    target_grouping = [
        "period"
    ]

    if group is not None:
        target_grouping.append(group)

    target_summary = (
        data
        .groupby(
            target_grouping,
            dropna=False,
        )
        .agg(
            observations=(
                target,
                "count",
            ),
            target_rate=(
                target,
                "mean",
            ),
        )
        .reset_index()
    )

    if variable_type == "continuous":
    
        feature_data = data[
            ~data[feature].isin(
                special_values
            )
        ].dropna(
            subset=[
                feature
            ]
        )
    
        feature_grouping = [
            "period"
        ]
    
        if group is not None:
            feature_grouping.append(
                group
            )
    
        feature_grouping.append(
            target
        )
    
        feature_summary = (
            feature_data
            .groupby(
                feature_grouping,
                dropna=False,
            )
            .agg(
                mean_feature=(
                    feature,
                    "mean",
                ),
                median_feature=(
                    feature,
                    "median",
                ),
                observations=(
                    feature,
                    "count",
                ),
            )
            .reset_index()
        )
    
        analysis_type = "continuous"
    
    
    elif variable_type == "binary":
    
        feature_grouping = [
            "period"
        ]
    
        if group is not None:
            feature_grouping.append(
                group
            )
    
        feature_grouping.append(
            target
        )
    
        feature_summary = (
            data
            .groupby(
                feature_grouping,
                dropna=False,
            )
            .agg(
                binary_rate=(
                    feature,
                    "mean",
                ),
                observations=(
                    feature,
                    "count",
                ),
            )
            .reset_index()
        )
    
        analysis_type = "binary"
    
    
    else:
    
        data["_category"] = (
            data[feature]
            .astype("object")
            .where(
                data[feature].notna(),
                MISSING_LABEL,
            )
            .astype(str)
        )
    
        feature_grouping = [
            "period"
        ]
    
        if group is not None:
            feature_grouping.append(
                group
            )
    
        feature_grouping.extend(
            [
                target,
                "_category",
            ]
        )
    
        feature_summary = (
            data
            .groupby(
                feature_grouping,
                dropna=False,
            )
            .size()
            .reset_index(
                name="observations"
            )
        )
    
        share_grouping = [
            "period"
        ]
    
        if group is not None:
            share_grouping.append(
                group
            )
    
        share_grouping.append(
            target
        )
    
        feature_summary[
            "category_share"
        ] = (
            feature_summary[
                "observations"
            ]
            / feature_summary.groupby(
                share_grouping
            )["observations"]
            .transform("sum")
        )
    
        feature_summary = (
            feature_summary.rename(
                columns={
                    "_category": "category"
                }
            )
        )
    
        analysis_type = "categorical"

    return {
        "target": target_summary,
        "feature": feature_summary,
        "analysis_type": analysis_type,
    }
