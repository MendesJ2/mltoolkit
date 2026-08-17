from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from mltoolkit.eda.stability import stability_analysis
from mltoolkit.eda.strength import feature_strength


class FeatureFilter:
    """
    Apply auditable filters before feature selection.

    Supported filters:
        - constant
        - quasi-constant
        - missing percentage
        - categorical cardinality
        - Information Value
        - PSI
        - numeric correlation

    Notes
    -----
    This class does not transform the dataframe and does not
    perform missing-value imputation.

    It only produces:
        - selected_features
        - removed_features
        - report
    """

    def __init__(
        self,
        dataset,
        *,
        max_missing_pct: float | None = None,
        max_mode_pct: float | None = 0.99,
        max_categories: int | None = None,
        min_iv: float | None = None,
        iv_n_bins: int = 10,
        min_lift_deviation: float | None = None,
        lift_min_population_pct: float = 0.05,
        lift_group: str | None = None,
        max_psi: float | None = None,
        stability_by: str | None = None,
        stability_reference=None,
        stability_n_bins: int = 10,
        max_correlation: float | None = 0.80,
        correlation_method: str = "spearman",
        feature_priority: (
            Mapping[str, float]
            | Sequence[str]
            | None
        ) = None,
    ):
        self.dataset = dataset

        self.max_missing_pct = max_missing_pct
        self.max_mode_pct = max_mode_pct
        self.max_categories = max_categories

        self.min_iv = min_iv
        self.iv_n_bins = iv_n_bins

        self.min_lift_deviation = (
            min_lift_deviation
        )
        
        self.lift_min_population_pct = (
            lift_min_population_pct
        )
        
        self.lift_group = lift_group
        self.max_psi = max_psi
        self.stability_by = stability_by
        self.stability_reference = stability_reference
        self.stability_n_bins = stability_n_bins

        self.max_correlation = max_correlation
        self.correlation_method = correlation_method

        self.feature_priority = feature_priority

        self.report = None
        self.selected_features = []
        self.removed_features = []

        self._validate_parameters()

    # =====================================================
    # Public API
    # =====================================================

    def fit(self):
        """
        Apply filters and generate the audit report.
        """

        features = list(
            self.dataset.feature_columns
        )
        
        if not features:
            raise ValueError(
                "No eligible features found. "
                "Check config.feature_columns and feature roles."
            )

        report = self._build_base_report(
            features
        )

        report = self._apply_basic_filters(
            report
        )

        report = self._apply_strength_filter(
            report
        )

        report = self._apply_psi_filter(
            report
        )

        report = self._apply_correlation_filter(
            report
        )

        report["decision"] = np.where(
            report["reasons"].str.len() == 0,
            "keep",
            "remove",
        )

        report["reason"] = (
            report["reasons"]
            .apply(
                lambda reasons: "; ".join(reasons)
            )
        )

        report = report.drop(
            columns=["reasons"]
        )

        self.report = (
            report
            .sort_values(
                [
                    "decision",
                    "feature",
                ],
                ascending=[
                    True,
                    True,
                ],
            )
            .reset_index(drop=True)
        )

        self.selected_features = (
            self.report.loc[
                self.report["decision"] == "keep",
                "feature",
            ]
            .tolist()
        )

        self.removed_features = (
            self.report.loc[
                self.report["decision"] == "remove",
                "feature",
            ]
            .tolist()
        )

        return self

    def fit_transform(self):
        """
        Fit filters and return dataframe with selected features.
        """

        self.fit()

        return self.dataset.df[
            self.selected_features
        ].copy()

    def summary(self):
        """
        Compact filter summary.
        """

        self._check_is_fitted()

        return pd.Series(
            {
                "input_features": len(
                    self.report
                ),
                "selected_features": len(
                    self.selected_features
                ),
                "removed_features": len(
                    self.removed_features
                ),
            }
        )

    # =====================================================
    # Base report
    # =====================================================

    def _build_base_report(
        self,
        features,
    ):
        records = []

        for feature_name in features:

            metadata = self.dataset.feature(
                feature_name
            )

            series = self.dataset.df[
                feature_name
            ]

            value_counts = (
                series.value_counts(
                    normalize=True,
                    dropna=False,
                )
            )

            mode_pct = (
                value_counts.iloc[0]
                if len(value_counts) > 0
                else np.nan
            )

            records.append(
                {
                    "feature": feature_name,
                    "variable_type": (
                        metadata.variable_type
                    ),
                    "dtype": metadata.dtype,
                    "observations": len(series),
                    "missing_pct": (
                        series.isna().mean()
                    ),
                    "n_unique": (
                        series.nunique(
                            dropna=True
                        )
                    ),
                    "mode_pct": mode_pct,
                    "is_constant": (
                        series.nunique(
                            dropna=False
                        )
                        <= 1
                    ),
                    "is_quasi_constant": (
                        mode_pct
                        >= self.max_mode_pct
                        if (
                            self.max_mode_pct
                            is not None
                            and not pd.isna(
                                mode_pct
                            )
                        )
                        else False
                    ),
                    "iv": np.nan,
                    
                    "relevant_lift": np.nan,
                    "relevant_lift_deviation": np.nan,
                    "relevant_lift_population_pct": np.nan,
                    "relevant_lift_group": None,
                    "relevant_lift_scope": None,
                    
                    "max_psi": np.nan,
                    
                    "correlated_with": None,
                    "correlation": np.nan,
                    
                    "reasons": [],
                }
            )

        return pd.DataFrame(records)

    # =====================================================
    # Basic filters
    # =====================================================

    def _apply_basic_filters(
        self,
        report,
    ):
        for index, row in report.iterrows():

            reasons = list(
                row["reasons"]
            )

            if row["is_constant"]:
                reasons.append(
                    "constant"
                )

            elif row["is_quasi_constant"]:
                reasons.append(
                    "quasi_constant"
                )

            if (
                self.max_missing_pct
                is not None
                and row["missing_pct"]
                > self.max_missing_pct
            ):
                reasons.append(
                    "missing_above_threshold"
                )

            if (
                self.max_categories
                is not None
                and row["variable_type"]
                in {
                    "categorical",
                    "ordinal",
                }
                and row["n_unique"]
                > self.max_categories
            ):
                reasons.append(
                    "cardinality_above_threshold"
                )

            report.at[
                index,
                "reasons",
            ] = reasons

        return report

    # =====================================================
    # IV filter
    # =====================================================

    def _apply_strength_filter(
        self,
        report,
    ):
        """
        Filter features using predictive strength.
    
        A feature is kept when:
            - IV is above min_iv;
            OR
            - a sufficiently large bin has lift sufficiently
              far from the neutral value of 1.
    
        Lift deviation is defined as:
    
            abs(lift - 1)
    
        Therefore both positive and negative signals
        are considered.
        """
    
        if (
            self.min_iv is None
            and self.min_lift_deviation is None
        ):
            return report
    
        target = (
            self.dataset.config.target
        )
    
        for index, row in (
            report.iterrows()
        ):
    
            if row["reasons"]:
                continue
    
            feature_name = (
                row["feature"]
            )
    
            try:
    
                result = feature_strength(
                    df=self.dataset.df,
                    feature=feature_name,
                    target=target,
                    variable_type=(
                        row["variable_type"]
                    ),
                    n_bins=self.iv_n_bins,
                    group=self.lift_group,
                    special_values=getattr(
                        self.dataset.config,
                        "special_values",
                        None,
                    ),
                    min_lift_population_pct=(
                        self.lift_min_population_pct
                    ),
                )
    
                # =========================================
                # Global IV
                # =========================================
    
                global_metrics = (
                    result["metrics"]
                )
    
                iv = global_metrics["iv"]
    
                report.at[
                    index,
                    "iv",
                ] = iv
    
                # =========================================
                # Best relevant lift
                # =========================================
    
                strength_metrics = (
                    result["group_metrics"]
                )
    
                valid_lift_rows = (
                    strength_metrics[
                        strength_metrics[
                            "relevant_lift_deviation"
                        ].notna()
                    ]
                )
    
                if valid_lift_rows.empty:
    
                    best_deviation = np.nan
    
                else:
    
                    best_index = (
                        valid_lift_rows[
                            "relevant_lift_deviation"
                        ]
                        .idxmax()
                    )
    
                    best_row = (
                        valid_lift_rows.loc[
                            best_index
                        ]
                    )
    
                    best_deviation = (
                        best_row[
                            "relevant_lift_deviation"
                        ]
                    )
    
                    report.at[
                        index,
                        "relevant_lift",
                    ] = (
                        best_row[
                            "relevant_lift"
                        ]
                    )
    
                    report.at[
                        index,
                        "relevant_lift_deviation",
                    ] = best_deviation
    
                    report.at[
                        index,
                        "relevant_lift_population_pct",
                    ] = (
                        best_row[
                            "relevant_lift_population_pct"
                        ]
                    )
    
                    report.at[
                        index,
                        "relevant_lift_group",
                    ] = (
                        best_row[
                            "relevant_lift_group"
                        ]
                    )
    
                    report.at[
                        index,
                        "relevant_lift_scope",
                    ] = (
                        best_row[
                            "group_value"
                        ]
                    )
    
                # =========================================
                # Decision
                # =========================================
    
                iv_ok = (
                    self.min_iv is not None
                    and pd.notna(iv)
                    and iv >= self.min_iv
                )
    
                lift_ok = (
                    self.min_lift_deviation
                    is not None
                    and pd.notna(
                        best_deviation
                    )
                    and best_deviation
                    >= self.min_lift_deviation
                )
    
                # Both configured:
                # either criterion is sufficient.
                if (
                    self.min_iv is not None
                    and self.min_lift_deviation
                    is not None
                ):
    
                    keep_strength = (
                        iv_ok
                        or lift_ok
                    )
    
                elif self.min_iv is not None:
    
                    keep_strength = iv_ok
    
                else:
    
                    keep_strength = lift_ok
    
                if not keep_strength:
    
                    reasons = list(
                        report.at[
                            index,
                            "reasons",
                        ]
                    )
    
                    reasons.append(
                        "predictive_strength_below_threshold"
                    )
    
                    report.at[
                        index,
                        "reasons",
                    ] = reasons
    
            except (
                ValueError,
                TypeError,
                KeyError,
            ) as error:
    
                reasons = list(
                    report.at[
                        index,
                        "reasons",
                    ]
                )
    
                reasons.append(
                    "strength_calculation_failed"
                )
    
                report.at[
                    index,
                    "reasons",
                ] = reasons
    
                report.at[
                    index,
                    "strength_error",
                ] = str(error)
    
        return report

    # =====================================================
    # PSI filter
    # =====================================================

    def _apply_psi_filter(
        self,
        report,
    ):
        if self.max_psi is None:
            return report

        if self.stability_by is None:
            raise ValueError(
                "stability_by is required "
                "when max_psi is defined."
            )

        for index, row in report.iterrows():

            if row["reasons"]:
                continue

            feature_name = row[
                "feature"
            ]

            try:

                result = stability_analysis(
                    df=self.dataset.df,
                    feature=feature_name,
                    by=self.stability_by,
                    variable_type=row[
                        "variable_type"
                    ],
                    reference=(
                        self.stability_reference
                    ),
                    n_bins=(
                        self.stability_n_bins
                    ),
                )

                max_psi = (
                    result["summary"]["psi"]
                    .max()
                )

                report.at[
                    index,
                    "max_psi",
                ] = max_psi

                if max_psi > self.max_psi:

                    reasons = list(
                        report.at[
                            index,
                            "reasons",
                        ]
                    )

                    reasons.append(
                        "psi_above_threshold"
                    )

                    report.at[
                        index,
                        "reasons",
                    ] = reasons

            except (
                ValueError,
                TypeError,
                KeyError,
            ) as error:

                reasons = list(
                    report.at[
                        index,
                        "reasons",
                    ]
                )

                reasons.append(
                    "psi_calculation_failed"
                )

                report.at[
                    index,
                    "reasons",
                ] = reasons

                report.at[
                    index,
                    "psi_error",
                ] = str(error)

        return report

    # =====================================================
    # Correlation filter
    # =====================================================

    def _apply_correlation_filter(
        self,
        report,
    ):
        if self.max_correlation is None:
            return report

        candidate_features = (
            report.loc[
                report["reasons"].apply(
                    len
                )
                == 0
            ]
        )

        numeric_features = (
            candidate_features.loc[
                candidate_features[
                    "variable_type"
                ].isin(
                    {
                        "continuous",
                        "binary",
                        "ordinal",
                    }
                ),
                "feature",
            ]
            .tolist()
        )

        if len(numeric_features) < 2:
            return report

        correlation_matrix = (
            self.dataset.df[
                numeric_features
            ]
            .corr(
                method=self.correlation_method
            )
            .abs()
        )

        ordered_features = (
            self._order_features(
                numeric_features,
                report,
            )
        )

        kept_features = []

        for feature_name in ordered_features:

            if not kept_features:
                kept_features.append(
                    feature_name
                )
                continue

            correlations = (
                correlation_matrix.loc[
                    feature_name,
                    kept_features,
                ]
            )

            if correlations.empty:
                kept_features.append(
                    feature_name
                )
                continue

            max_corr = correlations.max()

            if (
                pd.isna(max_corr)
                or max_corr
                <= self.max_correlation
            ):
                kept_features.append(
                    feature_name
                )
                continue

            correlated_with = (
                correlations.idxmax()
            )

            index = report.index[
                report["feature"]
                == feature_name
            ][0]

            reasons = list(
                report.at[
                    index,
                    "reasons",
                ]
            )

            reasons.append(
                "correlation_above_threshold"
            )

            report.at[
                index,
                "reasons",
            ] = reasons

            report.at[
                index,
                "correlated_with",
            ] = correlated_with

            report.at[
                index,
                "correlation",
            ] = max_corr

        return report

    # =====================================================
    # Feature priority
    # =====================================================

    def _order_features(
        self,
        features,
        report,
    ):
        """
        Define which feature is retained in correlated pairs.

        Priority options:
            1. Mapping feature -> numeric score.
            2. Explicit ordered sequence.
            3. IV descending, if IV was calculated.
            4. Original dataset order.
        """

        if isinstance(
            self.feature_priority,
            Mapping,
        ):

            return sorted(
                features,
                key=lambda feature: (
                    self.feature_priority.get(
                        feature,
                        -np.inf,
                    )
                ),
                reverse=True,
            )

        if (
            self.feature_priority
            is not None
            and not isinstance(
                self.feature_priority,
                str,
            )
        ):

            explicit_order = list(
                self.feature_priority
            )

            explicit_position = {
                feature: position
                for position, feature
                in enumerate(
                    explicit_order
                )
            }

            return sorted(
                features,
                key=lambda feature: (
                    explicit_position.get(
                        feature,
                        len(explicit_order),
                    )
                ),
            )

        iv_values = (
            report
            .set_index("feature")["iv"]
        )

        if iv_values.notna().any():

            return sorted(
                features,
                key=lambda feature: (
                    iv_values.get(
                        feature,
                        -np.inf,
                    )
                ),
                reverse=True,
            )

        dataset_order = {
            feature: position
            for position, feature
            in enumerate(
                self.dataset.feature_columns
            )
        }

        return sorted(
            features,
            key=lambda feature: (
                dataset_order[feature]
            ),
        )

    # =====================================================
    # Validation
    # =====================================================

    def _validate_parameters(self):

        if (
            self.max_missing_pct
            is not None
            and not 0
            <= self.max_missing_pct
            <= 1
        ):
            raise ValueError(
                "max_missing_pct must be "
                "between 0 and 1."
            )

        if (
            self.max_mode_pct
            is not None
            and not 0
            <= self.max_mode_pct
            <= 1
        ):
            raise ValueError(
                "max_mode_pct must be "
                "between 0 and 1."
            )

        if (
            self.max_correlation
            is not None
            and not 0
            <= self.max_correlation
            <= 1
        ):
            raise ValueError(
                "max_correlation must be "
                "between 0 and 1."
            )

        if self.correlation_method not in {
            "pearson",
            "spearman",
        }:
            raise ValueError(
                "correlation_method must be "
                "'pearson' or 'spearman'."
            )
        
        if (
            self.min_lift_deviation
            is not None
            and self.min_lift_deviation < 0
        ):
            raise ValueError(
                "min_lift_deviation must be >= 0."
            )
        
        if not (
            0
            < self.lift_min_population_pct
            <= 1
        ):
            raise ValueError(
                "lift_min_population_pct must "
                "be between 0 and 1."
            )

    def _check_is_fitted(self):

        if self.report is None:
            raise RuntimeError(
                "FeatureFilter is not fitted. "
                "Call fit() first."
            )

    def __repr__(self):

        if self.report is None:

            return (
                "FeatureFilter("
                "status='not_fitted'"
                ")"
            )

        return (
            "FeatureFilter("
            f"selected={len(self.selected_features)}, "
            f"removed={len(self.removed_features)}"
            ")"
        )
