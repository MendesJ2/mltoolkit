import pandas as pd

from scipy.stats import (
    chi2_contingency,
    kruskal,
)


def comparison_test(
    data,
    feature_name,
    group,
    variable_type,
):
    """
    Compare feature distributions across groups.

    Continuous:
        Kruskal-Wallis test.

    Categorical or binary:
        Chi-square independence test.
    """

    required_columns = {
        feature_name,
        group,
    }

    missing_columns = (
        required_columns
        - set(data.columns)
    )

    if missing_columns:
        raise ValueError(
            "Missing columns for comparison test: "
            f"{sorted(missing_columns)}"
        )

    test_data = data[
        [
            feature_name,
            group,
        ]
    ].dropna()

    if test_data.empty:
        raise ValueError(
            "No valid observations available for comparison."
        )

    if variable_type == "continuous":

        samples = [
            group_data[feature_name].to_numpy()
            for _, group_data in test_data.groupby(group)
            if len(group_data) > 0
        ]

        if len(samples) < 2:
            raise ValueError(
                "At least two groups are required "
                "for the Kruskal-Wallis test."
            )

        statistic, p_value = kruskal(
            *samples,
            nan_policy="omit",
        )

        return pd.Series(
            {
                "test": "Kruskal-Wallis",
                "statistic": statistic,
                "p_value": p_value,
                "significant_5pct": p_value < 0.05,
                "n_groups": len(samples),
                "n_observations": len(test_data),
            }
        )

    contingency = pd.crosstab(
        test_data[group],
        test_data[feature_name],
    )

    if contingency.shape[0] < 2:
        raise ValueError(
            "At least two groups are required "
            "for the Chi-square test."
        )

    if contingency.shape[1] < 2:
        raise ValueError(
            "The feature must have at least two categories "
            "for the Chi-square test."
        )

    statistic, p_value, degrees_freedom, expected = (
        chi2_contingency(contingency)
    )

    expected_below_5 = int(
        (expected < 5).sum()
    )

    return pd.Series(
        {
            "test": "Chi-square",
            "statistic": statistic,
            "p_value": p_value,
            "significant_5pct": p_value < 0.05,
            "degrees_freedom": degrees_freedom,
            "n_groups": contingency.shape[0],
            "n_categories": contingency.shape[1],
            "n_observations": int(
                contingency.to_numpy().sum()
            ),
            "expected_cells_below_5": expected_below_5,
        }
    )
