from __future__ import annotations

import numpy as np
import pandas as pd

from mltoolkit.eda.binning import (
    MISSING_LABEL,
    apply_fixed_bins,
    fit_quantile_bins,
)


class FeatureRelationshipStability:
    """
    Compare feature -> target relationships between
    a Development reference sample and OOT.

    Continuous bins are learned only on Development
    and frozen when applied to OOT.
    """

    def __init__(
        self,
        *,
        target,
        variable_types,
        n_bins=10,
        special_values=None,
    ):
        self.target = target
        self.variable_types = dict(
            variable_types
        )
        self.n_bins = n_bins
        self.special_values = (
            list(
                special_values
                or []
            )
        )

        self.table = None
        self.summary = None
        self.bin_definitions = {}

    def fit(
        self,
        development,
        oot,
        *,
        features=None,
    ):
        """
        Analyse relationship stability.
        """

        self._validate_sample(
            development,
            "development",
        )

        self._validate_sample(
            oot,
            "oot",
        )

        if features is None:

            features = [
                feature
                for feature
                in self.variable_types
                if (
                    feature
                    in development.columns
                    and feature
                    in oot.columns
                )
            ]

        tables = []

        for feature in features:

            variable_type = (
                self.variable_types[
                    feature
                ]
            )

            dev_group, oot_group = (
                self._prepare_feature_groups(
                    development=development,
                    oot=oot,
                    feature=feature,
                    variable_type=(
                        variable_type
                    ),
                )
            )

            dev_table = (
                self._aggregate(
                    feature_group=dev_group,
                    target=development[
                        self.target
                    ],
                    sample="development",
                )
            )

            oot_table = (
                self._aggregate(
                    feature_group=oot_group,
                    target=oot[
                        self.target
                    ],
                    sample="oot",
                )
            )

            merged = (
                dev_table
                .merge(
                    oot_table,
                    on="feature_group",
                    how="outer",
                    suffixes=(
                        "_development",
                        "_oot",
                    ),
                )
            )

            merged.insert(
                0,
                "feature",
                feature,
            )

            merged.insert(
                1,
                "variable_type",
                variable_type,
            )

            merged[
                "population_pct_delta"
            ] = (
                merged[
                    "population_pct_oot"
                ]
                - merged[
                    "population_pct_development"
                ]
            )

            merged[
                "target_rate_delta"
            ] = (
                merged[
                    "target_rate_oot"
                ]
                - merged[
                    "target_rate_development"
                ]
            )

            merged[
                "lift_delta"
            ] = (
                merged["lift_oot"]
                - merged[
                    "lift_development"
                ]
            )

            merged[
                "abs_lift_delta"
            ] = (
                merged[
                    "lift_delta"
                ]
                .abs()
            )

            merged[
                "abs_target_rate_delta"
            ] = (
                merged[
                    "target_rate_delta"
                ]
                .abs()
            )

            tables.append(
                merged
            )

        if not tables:

            self.table = (
                pd.DataFrame()
            )

            self.summary = (
                pd.DataFrame()
            )

            return self

        self.table = pd.concat(
            tables,
            ignore_index=True,
        )

        self.summary = (
            self._build_summary(
                self.table
            )
        )

        return self

    def _prepare_feature_groups(
        self,
        *,
        development,
        oot,
        feature,
        variable_type,
    ):

        if variable_type == "continuous":

            edges = fit_quantile_bins(
                development[
                    feature
                ],
                n_bins=self.n_bins,
                special_values=(
                    self.special_values
                ),
            )

            self.bin_definitions[
                feature
            ] = edges

            dev_group = (
                apply_fixed_bins(
                    development[
                        feature
                    ],
                    bin_edges=edges,
                    special_values=(
                        self.special_values
                    ),
                )
            )

            oot_group = (
                apply_fixed_bins(
                    oot[
                        feature
                    ],
                    bin_edges=edges,
                    special_values=(
                        self.special_values
                    ),
                )
            )

        else:

            dev_group = (
                self._categorical_group(
                    development[
                        feature
                    ]
                )
            )

            oot_group = (
                self._categorical_group(
                    oot[
                        feature
                    ]
                )
            )

        return (
            dev_group,
            oot_group,
        )

    @staticmethod
    def _categorical_group(
        series,
    ):

        return (
            series
            .astype("object")
            .where(
                series.notna(),
                MISSING_LABEL,
            )
            .astype(str)
        )

    @staticmethod
    def _aggregate(
        *,
        feature_group,
        target,
        sample,
    ):

        data = pd.DataFrame(
            {
                "feature_group": (
                    feature_group
                ),
                "target": target,
            }
        )

        data = data.dropna(
            subset=["target"]
        )

        global_rate = (
            data[
                "target"
            ].mean()
        )

        table = (
            data
            .groupby(
                "feature_group",
                dropna=False,
                observed=False,
            )
            .agg(
                observations=(
                    "target",
                    "size",
                ),
                events=(
                    "target",
                    "sum",
                ),
                target_rate=(
                    "target",
                    "mean",
                ),
            )
            .reset_index()
        )

        table[
            "population_pct"
        ] = (
            table[
                "observations"
            ]
            / table[
                "observations"
            ].sum()
        )

        table["lift"] = (
            table[
                "target_rate"
            ]
            / global_rate
        )

        rename = {
            column: (
                f"{column}_{sample}"
            )
            for column
            in table.columns
            if column
            != "feature_group"
        }

        return table.rename(
            columns=rename
        )

    @staticmethod
    def _build_summary(
        table,
    ):

        records = []

        for feature, data in (
            table.groupby(
                "feature",
                sort=False,
            )
        ):

            valid = data[
                data[
                    "abs_lift_delta"
                ].notna()
            ]

            if valid.empty:

                records.append(
                    {
                        "feature": feature,
                        "max_abs_lift_delta": (
                            np.nan
                        ),
                        "weighted_abs_lift_delta": (
                            np.nan
                        ),
                        "max_abs_target_rate_delta": (
                            np.nan
                        ),
                        "worst_bin": None,
                    }
                )

                continue

            worst_index = (
                valid[
                    "abs_lift_delta"
                ]
                .idxmax()
            )

            worst_row = (
                valid.loc[
                    worst_index
                ]
            )

            weights = (
                valid[
                    "population_pct_development"
                ]
                .fillna(0)
            )

            if weights.sum() > 0:

                weighted_delta = (
                    (
                        valid[
                            "abs_lift_delta"
                        ]
                        * weights
                    ).sum()
                    / weights.sum()
                )

            else:

                weighted_delta = (
                    np.nan
                )

            records.append(
                {
                    "feature": feature,

                    "max_abs_lift_delta": (
                        valid[
                            "abs_lift_delta"
                        ].max()
                    ),

                    "weighted_abs_lift_delta": (
                        weighted_delta
                    ),

                    "max_abs_target_rate_delta": (
                        valid[
                            "abs_target_rate_delta"
                        ].max()
                    ),

                    "worst_bin": (
                        worst_row[
                            "feature_group"
                        ]
                    ),

                    "development_lift_worst_bin": (
                        worst_row[
                            "lift_development"
                        ]
                    ),

                    "oot_lift_worst_bin": (
                        worst_row[
                            "lift_oot"
                        ]
                    ),

                    "development_population_worst_bin": (
                        worst_row[
                            "population_pct_development"
                        ]
                    ),

                    "oot_population_worst_bin": (
                        worst_row[
                            "population_pct_oot"
                        ]
                    ),
                }
            )

        return (
            pd.DataFrame(
                records
            )
            .sort_values(
                "weighted_abs_lift_delta",
                ascending=False,
                na_position="last",
            )
            .reset_index(
                drop=True
            )
        )

    def _validate_sample(
        self,
        df,
        name,
    ):

        if self.target not in df.columns:
            raise ValueError(
                f"Target '{self.target}' "
                f"not found in {name}."
            )

        y = df[
            self.target
        ].dropna()

        if not set(
            y.unique()
        ).issubset(
            {0, 1}
        ):
            raise ValueError(
                "Target must contain "
                "only 0 and 1."
            )
