from __future__ import annotations

import numpy as np
import pandas as pd

def temporal_model_performance(
    *,
    dates,
    y,
    probabilities,
    frequency="Q",
    min_observations=100,
):
    """
    Evaluate model discrimination through time.

    Parameters
    ----------
    dates:
        Observation dates aligned with y.

    y:
        Binary target.

    probabilities:
        Model probabilities.

    frequency:
        Pandas period frequency.
        Examples:
            "M" -> month
            "Q" -> quarter
            "Y" -> year

    min_observations:
        Minimum observations required to calculate
        period Gini.
    """

    from sklearn.metrics import (
        roc_auc_score,
    )

    data = pd.DataFrame(
        {
            "date": pd.to_datetime(
                dates
            ),
            "target": np.asarray(
                y
            ),
            "probability": np.asarray(
                probabilities
            ),
        }
    )

    data = data.dropna(
        subset=[
            "date",
            "target",
            "probability",
        ]
    )

    data["period"] = (
        data["date"]
        .dt.to_period(
            frequency
        )
        .astype(str)
    )

    records = []

    for period, sample in (
        data.groupby(
            "period",
            sort=True,
        )
    ):

        observations = len(
            sample
        )

        events = (
            sample["target"].sum()
        )

        target_rate = (
            sample["target"].mean()
        )

        if (
            observations
            < min_observations
            or sample[
                "target"
            ].nunique() < 2
        ):

            auc = np.nan
            gini = np.nan

        else:

            auc = roc_auc_score(
                sample["target"],
                sample["probability"],
            )

            gini = (
                2 * auc - 1
            )

        records.append(
            {
                "period": period,
                "observations": (
                    observations
                ),
                "events": events,
                "target_rate": (
                    target_rate
                ),
                "auc": auc,
                "gini": gini,
            }
        )

    return pd.DataFrame(
        records
    )
