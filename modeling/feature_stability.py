from __future__ import annotations

import numpy as np
import pandas as pd


class FeatureRelationshipStability:
    """
    Compare feature -> target relationships across:

        - train
        - validation
        - oot

    The input matrices are expected to be already
    transformed by the ModelingPreprocessor.

    Continuous feature bins are learned only on Train
    and then frozen for Validation and OOT.
    """

    def __init__(
        self,
        *,
        n_bins=10,
    ):
        self.n_bins = n_bins

        self.table = None
        self.summary = None

        self.bin_definitions = {}
        self.feature_types = {}

    # =====================================================
    # Public API
    # =====================================================

    def fit(
        self,
        *,
        X_train,
        y_train,
        X_valid,
        y_valid,
        X_oot,
        y_oot,
        features=None,
    ):
        """
        Fit relationship stability analysis.
        """

        (
            X_train,
            y_train,
        ) = self._prepare_sample(
            X_train,
            y_train,
            "train",
        )

        (
            X_valid,
            y_valid,
        ) = self._prepare_sample(
            X_valid,
            y_valid,
            "validation",
        )

        (
            X_oot,
            y_oot,
        ) = self._prepare_sample(
            X_oot,
            y_oot,
            "oot",
        )

        self._validate_columns(
            X_train,
            X_valid,
            X_oot,
        )

        if features is None:

            features = list(
                X_train.columns
            )

        else:

            features = list(
                features
            )

        missing_features = (
            set(features)
            - set(X_train.columns)
        )

        if missing_features:

            raise ValueError(
                "Features not found in X_train: "
                f"{sorted(missing_features)}"
            )

        tables = []

        for feature in features:

            feature_type = (
                self._infer_feature_type(
                    X_train[feature]
                )
            )

            self.feature_types[
                feature
            ] = feature_type

            train_group = (
                self._fit_feature_bins(
                    X_train[feature],
                    feature=feature,
                    feature_type=(
                        feature_type
                    ),
                )
            )

            valid_group = (
                self._apply_feature_bins(
                    X_valid[feature],
                    feature=feature,
                    feature_type=(
                        feature_type
                    ),
                )
            )

            oot_group = (
                self._apply_feature_bins(
                    X_oot[feature],
                    feature=feature,
                    feature_type=(
                        feature_type
                    ),
                )
            )

            train_table = (
                self._aggregate(
                    feature_group=(
                        train_group
                    ),
                    target=y_train,
                    sample="train",
                )
            )

            valid_table = (
                self._aggregate(
                    feature_group=(
                        valid_group
                    ),
                    target=y_valid,
                    sample="validation",
                )
            )

            oot_table = (
                self._aggregate(
                    feature_group=(
                        oot_group
                    ),
                    target=y_oot,
                    sample="oot",
                )
            )

            feature_table = (
                train_table
                .merge(
                    valid_table,
                    on="feature_group",
                    how="outer",
                )
                .merge(
                    oot_table,
                    on="feature_group",
                    how="outer",
                )
            )

            feature_table.insert(
                0,
                "feature",
                feature,
            )

            feature_table.insert(
                1,
                "feature_type",
                feature_type,
            )

            feature_table = (
                self._add_deltas(
                    feature_table
                )
            )

            tables.append(
                feature_table
            )

        if not tables:

            self.table = pd.DataFrame()
            self.summary = pd.DataFrame()

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

    def feature_table(
        self,
        feature,
    ):
        """
        Return detailed stability table
        for one feature.
        """

        if self.table is None:
            raise RuntimeError(
                "The analysis has not been fitted."
            )

        return (
            self.table[
                self.table[
                    "feature"
                ] == feature
            ]
            .reset_index(
                drop=True
            )
        )

    # =====================================================
    # Feature binning
    # =====================================================

    @staticmethod
    def _infer_feature_type(
        series,
    ):
        unique_values = set(
            series
            .dropna()
            .unique()
        )

        if unique_values.issubset(
            {0, 1}
        ):
            return "binary"

        return "continuous"

    def _fit_feature_bins(
        self,
        series,
        *,
        feature,
        feature_type,
    ):
        if feature_type == "binary":

            self.bin_definitions[
                feature
            ] = None

            return (
                series
                .astype("object")
                .where(
                    series.notna(),
                    "__MISSING__",
                )
                .astype(str)
            )

        clean = (
            series
            .dropna()
        )

        if clean.empty:

            self.bin_definitions[
                feature
            ] = None

            return pd.Series(
                "__MISSING__",
                index=series.index,
            )

        try:

            _, edges = pd.qcut(
                clean,
                q=self.n_bins,
                duplicates="drop",
                retbins=True,
            )

            edges = np.asarray(
                edges,
                dtype=float,
            )

            edges[0] = -np.inf
            edges[-1] = np.inf

            self.bin_definitions[
                feature
            ] = edges

            return self._apply_continuous_bins(
                series,
                edges,
            )

        except ValueError:

            self.bin_definitions[
                feature
            ] = None

            return (
                series
                .astype("object")
                .where(
                    series.notna(),
                    "__MISSING__",
                )
                .astype(str)
            )

    def _apply_feature_bins(
        self,
        series,
        *,
        feature,
        feature_type,
    ):
        if feature_type == "binary":

            return (
                series
                .astype("object")
                .where(
                    series.notna(),
                    "__MISSING__",
                )
                .astype(str)
            )

        edges = (
            self.bin_definitions[
                feature
            ]
        )

        if edges is None:

            return (
                series
                .astype("object")
                .where(
                    series.notna(),
                    "__MISSING__",
                )
                .astype(str)
            )

        return self._apply_continuous_bins(
            series,
            edges,
        )

    @staticmethod
    def _apply_continuous_bins(
        series,
        edges,
    ):
        result = pd.Series(
            index=series.index,
            dtype="object",
        )

        missing_mask = (
            series.isna()
        )

        result.loc[
            missing_mask
        ] = "__MISSING__"

        regular_mask = (
            ~missing_mask
        )

        result.loc[
            regular_mask
        ] = (
            pd.cut(
                series.loc[
                    regular_mask
                ],
                bins=edges,
                include_lowest=True,
                duplicates="drop",
            )
            .astype(str)
        )

        return result

    # =====================================================
    # Aggregation
    # =====================================================

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
            data["target"].mean()
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

    # =====================================================
    # Deltas
    # =====================================================

    @staticmethod
    def _add_deltas(
        table,
    ):
        table[
            "lift_delta_validation"
        ] = (
            table[
                "lift_validation"
            ]
            - table[
                "lift_train"
            ]
        )

        table[
            "lift_delta_oot"
        ] = (
            table[
                "lift_oot"
            ]
            - table[
                "lift_train"
            ]
        )

        table[
            "abs_lift_delta_validation"
        ] = (
            table[
                "lift_delta_validation"
            ]
            .abs()
        )

        table[
            "abs_lift_delta_oot"
        ] = (
            table[
                "lift_delta_oot"
            ]
            .abs()
        )

        table[
            "target_rate_delta_validation"
        ] = (
            table[
                "target_rate_validation"
            ]
            - table[
                "target_rate_train"
            ]
        )

        table[
            "target_rate_delta_oot"
        ] = (
            table[
                "target_rate_oot"
            ]
            - table[
                "target_rate_train"
            ]
        )

        table[
            "population_delta_validation"
        ] = (
            table[
                "population_pct_validation"
            ]
            - table[
                "population_pct_train"
            ]
        )

        table[
            "population_delta_oot"
        ] = (
            table[
                "population_pct_oot"
            ]
            - table[
                "population_pct_train"
            ]
        )

        return table

    # =====================================================
    # Summary
    # =====================================================

    @staticmethod
    def _build_summary(
        table,
    ):
        records = []

        for (
            feature,
            data,
        ) in table.groupby(
            "feature",
            sort=False,
        ):

            valid = data[
                data[
                    "abs_lift_delta_oot"
                ].notna()
            ].copy()

            if valid.empty:
                continue

            worst_index = (
                valid[
                    "abs_lift_delta_oot"
                ]
                .idxmax()
            )

            worst_row = (
                valid.loc[
                    worst_index
                ]
            )

            train_weights = (
                valid[
                    "population_pct_train"
                ]
                .fillna(0)
            )

            if train_weights.sum() > 0:

                weighted_oot = (
                    (
                        valid[
                            "abs_lift_delta_oot"
                        ]
                        * train_weights
                    ).sum()
                    / train_weights.sum()
                )

                weighted_validation = (
                    (
                        valid[
                            "abs_lift_delta_validation"
                        ]
                        * train_weights
                    ).sum()
                    / train_weights.sum()
                )

            else:

                weighted_oot = np.nan
                weighted_validation = (
                    np.nan
                )

            records.append(
                {
                    "feature": feature,

                    "max_abs_lift_delta_validation": (
                        valid[
                            "abs_lift_delta_validation"
                        ].max()
                    ),

                    "weighted_abs_lift_delta_validation": (
                        weighted_validation
                    ),

                    "max_abs_lift_delta_oot": (
                        valid[
                            "abs_lift_delta_oot"
                        ].max()
                    ),

                    "weighted_abs_lift_delta_oot": (
                        weighted_oot
                    ),

                    "worst_bin_oot": (
                        worst_row[
                            "feature_group"
                        ]
                    ),

                    "lift_train_worst_bin": (
                        worst_row[
                            "lift_train"
                        ]
                    ),

                    "lift_validation_worst_bin": (
                        worst_row[
                            "lift_validation"
                        ]
                    ),

                    "lift_oot_worst_bin": (
                        worst_row[
                            "lift_oot"
                        ]
                    ),

                    "population_train_worst_bin": (
                        worst_row[
                            "population_pct_train"
                        ]
                    ),

                    "population_oot_worst_bin": (
                        worst_row[
                            "population_pct_oot"
                        ]
                    ),
                }
            )

        if not records:
            return pd.DataFrame()

        return (
            pd.DataFrame(
                records
            )
            .sort_values(
                [
                    "weighted_abs_lift_delta_oot",
                    "max_abs_lift_delta_oot",
                ],
                ascending=False,
                na_position="last",
            )
            .reset_index(
                drop=True
            )
        )

    # =====================================================
    # Validation
    # =====================================================

    @staticmethod
    def _prepare_sample(
        X,
        y,
        name,
    ):
        if not isinstance(
            X,
            pd.DataFrame,
        ):
            X = pd.DataFrame(X)

        else:
            X = X.copy()

        if isinstance(
            y,
            pd.Series,
        ):
            y = y.copy()

        else:
            y = pd.Series(
                y,
                index=X.index,
            )

        if not y.index.equals(
            X.index
        ):
            y = y.reindex(
                X.index
            )

        if y.isna().any():

            raise ValueError(
                f"y_{name} contains "
                "missing values."
            )

        unique_values = set(
            y.unique()
        )

        if not unique_values.issubset(
            {0, 1}
        ):

            raise ValueError(
                f"y_{name} must "
                "contain only 0 and 1."
            )

        non_numeric = [
            column
            for column in X.columns
            if not pd.api.types
            .is_numeric_dtype(
                X[column]
            )
        ]

        if non_numeric:

            raise TypeError(
                f"X_{name} contains "
                "non-numeric columns: "
                f"{non_numeric}"
            )

        return (
            X,
            y.astype(int),
        )

    @staticmethod
    def _validate_columns(
        X_train,
        X_valid,
        X_oot,
    ):
        train_columns = set(
            X_train.columns
        )

        for name, X in [
            (
                "validation",
                X_valid,
            ),
            (
                "oot",
                X_oot,
            ),
        ]:

            if set(
                X.columns
            ) != train_columns:

                raise ValueError(
                    f"X_{name} columns "
                    "do not match X_train."
                )
