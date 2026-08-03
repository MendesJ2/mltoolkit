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
):
    """
    Calculate univariate predictive-strength metrics.

    Metrics:
        - WoE
        - IV
        - Lift
        - Cumulative gain
        - KS
    """

    table = target_analysis(
        df=df,
        feature=feature,
        target=target,
        variable_type=variable_type,
        n_bins=n_bins,
    ).copy()

    total_events = table["events"].sum()
    total_non_events = table["non_events"].sum()
    total_observations = table["observations"].sum()

    if total_events == 0:
        raise ValueError(
            "Target contains no events."
        )

    if total_non_events == 0:
        raise ValueError(
            "Target contains no non-events."
        )

    n_groups = len(table)

    table["event_distribution"] = (
        table["events"] + smoothing
    ) / (
        total_events
        + smoothing * n_groups
    )

    table["non_event_distribution"] = (
        table["non_events"] + smoothing
    ) / (
        total_non_events
        + smoothing * n_groups
    )

    table["woe"] = np.log(
        table["event_distribution"]
        / table["non_event_distribution"]
    )

    table["iv_component"] = (
        table["event_distribution"]
        - table["non_event_distribution"]
    ) * table["woe"]

    global_target_rate = (
        total_events
        / total_observations
    )

    table["lift"] = (
        table["target_rate"]
        / global_target_rate
    )

    # Para gain e KS, ordenar do maior para o menor target rate.
    table = (
        table
        .sort_values(
            "target_rate",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    table["population_pct"] = (
        table["observations"]
        / total_observations
    )

    table["cumulative_population_pct"] = (
        table["population_pct"]
        .cumsum()
    )

    table["cumulative_event_pct"] = (
        table["events"]
        .cumsum()
        / total_events
    )

    table["cumulative_non_event_pct"] = (
        table["non_events"]
        .cumsum()
        / total_non_events
    )

    table["gain"] = (
        table["cumulative_event_pct"]
    )

    table["ks"] = (
        table["cumulative_event_pct"]
        - table["cumulative_non_event_pct"]
    ).abs()

    metrics = pd.Series(
        {
            "feature": feature,
            "iv": table[
                "iv_component"
            ].sum(),
            "max_ks": table["ks"].max(),
            "max_lift": table["lift"].max(),
            "global_target_rate": (
                global_target_rate
            ),
            "n_groups": len(table),
            "observations": (
                total_observations
            ),
            "events": total_events,
        }
    )

    return {
        "table": table,
        "metrics": metrics,
    }
