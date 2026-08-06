import numpy as np
import pandas as pd

from .binning import create_bins


MISSING_LABEL = "__MISSING__"


def stability_analysis(
    df,
    feature,
    by,
    variable_type,
    reference=None,
    n_bins=10,
    epsilon=1e-6,
    special_values=None,
):
    """
    Calculate PSI between a reference population
    and all other populations defined by `by`.
    """

    required_columns = {
        feature,
        by,
    }

    missing_columns = (
        required_columns
        - set(df.columns)
    )

    if missing_columns:
        raise ValueError(
            "Missing columns: "
            f"{sorted(missing_columns)}"
        )

    data = df[
        [
            feature,
            by,
        ]
    ].copy()

    data = data.dropna(
        subset=[by]
    )

    if data.empty:
        raise ValueError(
            "No observations available "
            "for stability analysis."
        )

    groups = list(
        data[by]
        .drop_duplicates()
    )

    if len(groups) < 2:
        raise ValueError(
            "Stability analysis requires "
            "at least two populations."
        )

    if reference is None:
        reference = groups[0]

    if reference not in groups:
        raise ValueError(
            f"Reference '{reference}' "
            f"not found in column '{by}'."
        )

    if variable_type == "continuous":

        data["_stability_group"] = (
            create_bins(
                data[feature],
                n_bins=n_bins,
                special_values=(
                    special_values
                ),
            )
            .astype(str)
        )

    else:

        data["_stability_group"] = (
            data[feature]
            .astype("object")
            .where(
                data[feature].notna(),
                MISSING_LABEL,
            )
            .astype(str)
        )

    distribution = (
        data
        .groupby(
            [
                by,
                "_stability_group",
            ],
            dropna=False,
        )
        .size()
        .reset_index(
            name="observations"
        )
    )

    distribution["distribution"] = (
        distribution["observations"]
        / distribution.groupby(by)[
            "observations"
        ].transform("sum")
    )

    all_feature_groups = (
        distribution[
            "_stability_group"
        ]
        .drop_duplicates()
        .tolist()
    )

    reference_distribution = (
        distribution[
            distribution[by] == reference
        ][
            [
                "_stability_group",
                "distribution",
            ]
        ]
        .rename(
            columns={
                "distribution": (
                    "reference_distribution"
                )
            }
        )
    )

    detail_records = []
    summary_records = []

    for comparison_group in groups:

        if comparison_group == reference:
            continue

        comparison_distribution = (
            distribution[
                distribution[by]
                == comparison_group
            ][
                [
                    "_stability_group",
                    "observations",
                    "distribution",
                ]
            ]
            .rename(
                columns={
                    "observations": (
                        "comparison_observations"
                    ),
                    "distribution": (
                        "comparison_distribution"
                    ),
                }
            )
        )

        detail = pd.DataFrame(
            {
                "_stability_group": (
                    all_feature_groups
                )
            }
        )

        detail = detail.merge(
            reference_distribution,
            on="_stability_group",
            how="left",
        )

        detail = detail.merge(
            comparison_distribution,
            on="_stability_group",
            how="left",
        )

        detail[
            "reference_distribution"
        ] = detail[
            "reference_distribution"
        ].fillna(0)

        detail[
            "comparison_distribution"
        ] = detail[
            "comparison_distribution"
        ].fillna(0)

        detail[
            "reference_distribution_adjusted"
        ] = (
            detail["reference_distribution"]
            .clip(lower=epsilon)
        )

        detail[
            "comparison_distribution_adjusted"
        ] = (
            detail["comparison_distribution"]
            .clip(lower=epsilon)
        )

        detail["psi_component"] = (
            detail[
                "comparison_distribution_adjusted"
            ]
            - detail[
                "reference_distribution_adjusted"
            ]
        ) * np.log(
            detail[
                "comparison_distribution_adjusted"
            ]
            / detail[
                "reference_distribution_adjusted"
            ]
        )

        detail["reference"] = reference
        detail["comparison"] = (
            comparison_group
        )

        psi = detail[
            "psi_component"
        ].sum()

        summary_records.append(
            {
                "reference": reference,
                "comparison": (
                    comparison_group
                ),
                "psi": psi,
                "stability": (
                    _psi_interpretation(psi)
                ),
            }
        )

        detail_records.append(detail)

    return {
        "summary": pd.DataFrame(
            summary_records
        ),
        "detail": pd.concat(
            detail_records,
            ignore_index=True,
        ),
        "distribution": distribution,
        "reference": reference,
    }


def _psi_interpretation(
    psi,
):

    if psi < 0.10:
        return "stable"

    if psi < 0.25:
        return "moderate_shift"

    return "significant_shift"
