import numpy as np
import pandas as pd

from .target import target_analysis


def feature_strength(
    df,
    feature,
    target,
    variable_type,
    n_bins=10,
    smoothing=0.5,
    group=None,
    special_values=None,
    min_lift_population_pct=0.05,
):
    """
    Calculate WoE, IV, lift, gain and KS globally
    and, optionally, by group.

    The same continuous bins are used globally and
    across all group values.
    """

    table = target_analysis(
        df=df,
        feature=feature,
        target=target,
        variable_type=variable_type,
        n_bins=n_bins,
        group=group,
        special_values=special_values,
    ).copy()

    table = _calculate_strength_columns(
        table=table,
        smoothing=smoothing,
    )

    metrics = _build_strength_metrics(
        table=table,
        feature=feature,
        min_lift_population_pct=(
            min_lift_population_pct
        ),
    )

    global_metrics = (
        metrics[
            metrics["scope"] == "global"
        ]
        .iloc[0]
    )

    return {
        "table": table,
        "metrics": global_metrics,
        "group_metrics": metrics,
    }


def _calculate_strength_columns(
    table,
    smoothing,
):
    """
    Calculate metrics independently within each scope/group.
    """

    result_tables = []

    for (
        scope,
        group_value,
    ), segment in table.groupby(
        [
            "scope",
            "group_value",
        ],
        dropna=False,
        sort=False,
    ):
        segment = segment.copy()

        total_events = segment[
            "events"
        ].sum()

        total_non_events = segment[
            "non_events"
        ].sum()

        total_observations = segment[
            "observations"
        ].sum()

        if (
            total_events == 0
            or total_non_events == 0
            or total_observations == 0
        ):
            segment["event_distribution"] = np.nan
            segment["non_event_distribution"] = np.nan
            segment["woe"] = np.nan
            segment["iv_component"] = np.nan
            segment["lift"] = np.nan
            segment["cumulative_population_pct"] = np.nan
            segment["cumulative_event_pct"] = np.nan
            segment["cumulative_non_event_pct"] = np.nan
            segment["gain"] = np.nan
            segment["ks"] = np.nan
            segment["cumulative_lift"] = np.nan

            result_tables.append(segment)
            continue

        n_groups = len(segment)

        segment["event_distribution"] = (
            segment["events"] + smoothing
        ) / (
            total_events
            + smoothing * n_groups
        )

        segment["non_event_distribution"] = (
            segment["non_events"] + smoothing
        ) / (
            total_non_events
            + smoothing * n_groups
        )

        segment["woe"] = np.log(
            segment["event_distribution"]
            / segment["non_event_distribution"]
        )

        segment["iv_component"] = (
            segment["event_distribution"]
            - segment["non_event_distribution"]
        ) * segment["woe"]

        segment_target_rate = (
            total_events
            / total_observations
        )

        segment["lift"] = (
            segment["target_rate"]
            / segment_target_rate
        )

        # Gain e KS exigem ordenação por propensão.
        segment = (
            segment
            .sort_values(
                "target_rate",
                ascending=False,
            )
            .reset_index(drop=True)
        )

        segment["population_pct"] = (
            segment["observations"]
            / total_observations
        )

        segment[
            "cumulative_population_pct"
        ] = segment[
            "population_pct"
        ].cumsum()

        segment[
            "cumulative_event_pct"
        ] = (
            segment["events"].cumsum()
            / total_events
        )

        segment[
            "cumulative_non_event_pct"
        ] = (
            segment["non_events"].cumsum()
            / total_non_events
        )

        segment["gain"] = (
            segment[
                "cumulative_event_pct"
            ]
        )

        segment["ks"] = (
            segment[
                "cumulative_event_pct"
            ]
            - segment[
                "cumulative_non_event_pct"
            ]
        ).abs()

        segment["cumulative_lift"] = (
            segment[
                "cumulative_event_pct"
            ]
            / segment[
                "cumulative_population_pct"
            ].replace(0, np.nan)
        )

        result_tables.append(segment)

    return pd.concat(
        result_tables,
        ignore_index=True,
    )


def _build_strength_metrics(
    table,
    feature,
    min_lift_population_pct=0.05,
):
    records = []

    for (
        scope,
        group_value,
    ), segment in table.groupby(
        [
            "scope",
            "group_value",
        ],
        dropna=False,
        sort=False,
    ):

        observations = segment[
            "observations"
        ].sum()

        events = segment[
            "events"
        ].sum()

        # ---------------------------------------------
        # Relevant lift
        # ---------------------------------------------
        
        relevant_bins = segment[
            segment["population_pct"]
            >= min_lift_population_pct
        ].copy()
        
        if relevant_bins.empty:
        
            relevant_lift = np.nan
            relevant_lift_deviation = np.nan
            relevant_lift_population_pct = np.nan
            relevant_lift_group = None
        
        else:
        
            # Distance from neutral lift = 1.
            #
            # This captures both:
            #   lift > 1  -> positive signal
            #   lift < 1  -> negative signal
        
            relevant_bins[
                "_lift_deviation"
            ] = (
                relevant_bins["lift"] - 1
            ).abs()
        
            best_lift_index = (
                relevant_bins[
                    "_lift_deviation"
                ]
                .idxmax()
            )
        
            best_lift_row = (
                relevant_bins.loc[
                    best_lift_index
                ]
            )
        
            relevant_lift = (
                best_lift_row["lift"]
            )
        
            relevant_lift_deviation = (
                best_lift_row[
                    "_lift_deviation"
                ]
            )
        
            relevant_lift_population_pct = (
                best_lift_row[
                    "population_pct"
                ]
            )
        
            relevant_lift_group = (
                best_lift_row[
                    "feature_group"
                ]
            )

        records.append(
            {
                "feature": feature,
                "scope": scope,
                "group_value": group_value,

                "iv": segment[
                    "iv_component"
                ].sum(
                    min_count=1
                ),

                "max_ks": segment[
                    "ks"
                ].max(),

                "max_lift": segment[
                    "lift"
                ].max(),

                "relevant_lift": (
                    relevant_lift
                ),
                
                "relevant_lift_deviation": (
                    relevant_lift_deviation
                ),
                
                "relevant_lift_population_pct": (
                    relevant_lift_population_pct
                ),
                
                "relevant_lift_group": (
                    relevant_lift_group
                ),

                "global_target_rate": (
                    events / observations
                    if observations > 0
                    else np.nan
                ),

                "n_groups": len(segment),
                "observations": observations,
                "events": events,
            }
        )

    return pd.DataFrame(
        records
    )


def feature_strength_by_group(
    df,
    feature,
    target,
    variable_type,
    group,
    n_bins=10,
    smoothing=0.5,
    special_values=None,
    min_lift_population_pct=0.05,
):
    """
    Return one metrics row globally and per group.
    """

    result = feature_strength(
        df=df,
        feature=feature,
        target=target,
        variable_type=variable_type,
        n_bins=n_bins,
        smoothing=smoothing,
        group=group,
        special_values=special_values,
        min_lift_population_pct=(
            min_lift_population_pct
        ),
    )

    return result["group_metrics"]
