from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
import statsmodels.api as sm

from numpy.linalg import LinAlgError
from sklearn.metrics import roc_auc_score
from statsmodels.tools.sm_exceptions import (
    PerfectSeparationError,
)


class ForwardSelector:
    """
    Forward feature selection for binary classification.

    Selection criteria:
        - Candidate statistical significance on Train.
        - Maximum allowed correlation with selected features.
        - Marginal validation Gini improvement.

    Candidate order:
        - Univariate validation Gini.
        - Explicit priority mapping.
        - Explicit ordered feature list.

    Notes
    -----
    X_train and X_valid must already be numeric and processed.

    Do not include an intercept column. It is added internally.
    """

    def __init__(
        self,
        *,
        significance_level: float | None = 0.05,
        max_correlation: float | None = 0.80,
        max_features: int | None = 15,
        min_gini_increase: float = 0.0001,
        correlation_method: str = "spearman",
        candidate_priority: (
            Mapping[str, float]
            | Sequence[str]
            | None
        ) = None,
        candidate_order: str = "univariate_gini",
        max_candidates_per_step: int | None = None,
        verbose: bool = True,
    ):
        self.significance_level = significance_level
        self.max_correlation = max_correlation
        self.max_features = max_features
        self.min_gini_increase = min_gini_increase
        self.correlation_method = correlation_method
        self.candidate_priority = candidate_priority
        self.candidate_order = candidate_order
        self.max_candidates_per_step = (
            max_candidates_per_step
        )
        self.verbose = verbose

        self.selected_features = []
        self.removed_features = []

        self.candidate_scores = None
        self.history = None
        self.final_model = None

        self.train_gini = None
        self.validation_gini = None

        self._correlation_matrix = None
        self._is_fitted = False

        self._validate_parameters()

    # =====================================================
    # Public API
    # =====================================================

    def fit(
        self,
        X_train,
        y_train,
        X_valid,
        y_valid,
    ):
        """
        Run forward selection.
        """

        X_train = self._prepare_X(
            X_train,
            name="X_train",
        )

        X_valid = self._prepare_X(
            X_valid,
            name="X_valid",
        )

        y_train = self._prepare_y(
            y_train,
            X_train.index,
            name="y_train",
        )

        y_valid = self._prepare_y(
            y_valid,
            X_valid.index,
            name="y_valid",
        )

        self._validate_datasets(
            X_train=X_train,
            X_valid=X_valid,
            y_train=y_train,
            y_valid=y_valid,
        )

        self._correlation_matrix = (
            X_train.corr(
                method=self.correlation_method
            ).abs()
        )

        self.candidate_scores = (
            self._calculate_candidate_scores(
                X_train=X_train,
                y_train=y_train,
                X_valid=X_valid,
                y_valid=y_valid,
            )
        )

        self.selected_features = []

        current_validation_gini = 0.0
        history_records = []

        step = 1

        while True:

            if (
                self.max_features is not None
                and len(self.selected_features)
                >= self.max_features
            ):
                break

            candidates = [
                feature
                for feature
                in self._ordered_candidates()
                if feature
                not in self.selected_features
            ]

            if self.max_candidates_per_step is not None:

                candidates = candidates[
                    :self.max_candidates_per_step
                ]

            if not candidates:
                break

            best_result = None

            for feature in candidates:

                correlation_result = (
                    self._check_correlation(
                        feature=feature,
                    )
                )

                if not correlation_result[
                    "accepted"
                ]:
                    continue

                current_features = (
                    self.selected_features
                    + [feature]
                )

                result = self._evaluate_candidate(
                    feature=feature,
                    features=current_features,
                    X_train=X_train,
                    y_train=y_train,
                    X_valid=X_valid,
                    y_valid=y_valid,
                )

                if result is None:
                    continue

                if (
                    self.significance_level
                    is not None
                    and result["p_value"]
                    > self.significance_level
                ):
                    continue

                result.update(
                    correlation_result
                )

                if (
                    best_result is None
                    or result["validation_gini"]
                    > best_result[
                        "validation_gini"
                    ]
                ):
                    best_result = result

            if best_result is None:
                break

            marginal_increase = (
                best_result["validation_gini"]
                - current_validation_gini
            )

            if (
                marginal_increase
                < self.min_gini_increase
            ):
                break

            selected_feature = (
                best_result["feature"]
            )

            self.selected_features.append(
                selected_feature
            )

            current_validation_gini = (
                best_result["validation_gini"]
            )

            history_records.append(
                {
                    "step": step,
                    "feature": selected_feature,
                    "n_features": len(
                        self.selected_features
                    ),
                    "candidate_score": (
                        self.candidate_scores.loc[
                            selected_feature,
                            "priority_score",
                        ]
                    ),
                    "p_value": best_result[
                        "p_value"
                    ],
                    "max_correlation": (
                        best_result[
                            "max_correlation"
                        ]
                    ),
                    "correlated_with": (
                        best_result[
                            "correlated_with"
                        ]
                    ),
                    "train_gini": (
                        best_result[
                            "train_gini"
                        ]
                    ),
                    "validation_gini": (
                        best_result[
                            "validation_gini"
                        ]
                    ),
                    "marginal_gini_increase": (
                        marginal_increase
                    ),
                    "aic": best_result["aic"],
                    "bic": best_result["bic"],
                }
            )

            if self.verbose:

                print(
                    f"Step {step}: "
                    f"added '{selected_feature}' | "
                    f"Validation Gini: "
                    f"{current_validation_gini:.4f} | "
                    f"Increase: "
                    f"{marginal_increase:.4f} | "
                    f"p-value: "
                    f"{best_result['p_value']:.4g}"
                )

            step += 1

        self.history = pd.DataFrame(
            history_records
        )

        self.removed_features = [
            feature
            for feature in X_train.columns
            if feature
            not in self.selected_features
        ]

        self._fit_final_model(
            X_train=X_train,
            y_train=y_train,
            X_valid=X_valid,
            y_valid=y_valid,
        )

        self._is_fitted = True

        return self

    def transform(
        self,
        X,
    ):
        """
        Return dataframe with selected features.
        """

        self._check_is_fitted()

        missing_features = (
            set(self.selected_features)
            - set(X.columns)
        )

        if missing_features:
            raise ValueError(
                "Missing selected features: "
                f"{sorted(missing_features)}"
            )

        return X[
            self.selected_features
        ].copy()

    def fit_transform(
        self,
        X_train,
        y_train,
        X_valid,
        y_valid,
    ):
        """
        Fit selector and transform Train and Validation.
        """

        self.fit(
            X_train=X_train,
            y_train=y_train,
            X_valid=X_valid,
            y_valid=y_valid,
        )

        return (
            self.transform(X_train),
            self.transform(X_valid),
        )

    def summary(
        self,
    ):
        """
        Compact selection summary.
        """

        self._check_is_fitted()

        return pd.Series(
            {
                "selected_features": len(
                    self.selected_features
                ),
                "train_gini": self.train_gini,
                "validation_gini": (
                    self.validation_gini
                ),
                "aic": self.final_model.aic,
                "bic": self.final_model.bic,
            }
        )

    def coefficients(
        self,
    ):
        """
        Final model coefficients and odds ratios.
        """

        self._check_is_fitted()

        result = pd.DataFrame(
            {
                "coefficient": (
                    self.final_model.params
                ),
                "std_error": (
                    self.final_model.bse
                ),
                "p_value": (
                    self.final_model.pvalues
                ),
            }
        )

        result["odds_ratio"] = np.exp(
            result["coefficient"]
        )

        result.index.name = "feature"

        return result.reset_index()

    # =====================================================
    # Candidate priority
    # =====================================================

    def _calculate_candidate_scores(
        self,
        X_train,
        y_train,
        X_valid,
        y_valid,
    ):
        records = []

        for feature in X_train.columns:

            result = self._evaluate_candidate(
                feature=feature,
                features=[feature],
                X_train=X_train,
                y_train=y_train,
                X_valid=X_valid,
                y_valid=y_valid,
            )

            if result is None:

                train_gini = np.nan
                validation_gini = np.nan
                p_value = np.nan

            else:

                train_gini = result[
                    "train_gini"
                ]

                validation_gini = result[
                    "validation_gini"
                ]

                p_value = result[
                    "p_value"
                ]

            records.append(
                {
                    "feature": feature,
                    "univariate_train_gini": (
                        train_gini
                    ),
                    "univariate_validation_gini": (
                        validation_gini
                    ),
                    "univariate_p_value": (
                        p_value
                    ),
                }
            )

        scores = (
            pd.DataFrame(records)
            .set_index("feature")
        )

        scores["priority_score"] = (
            self._build_priority_scores(
                scores
            )
        )

        return scores.sort_values(
            [
                "priority_score",
                "univariate_validation_gini",
            ],
            ascending=False,
        )

    def _build_priority_scores(
        self,
        scores,
    ):
        if isinstance(
            self.candidate_priority,
            Mapping,
        ):

            return pd.Series(
                {
                    feature: (
                        self.candidate_priority.get(
                            feature,
                            -np.inf,
                        )
                    )
                    for feature
                    in scores.index
                }
            )

        if (
            self.candidate_priority
            is not None
            and not isinstance(
                self.candidate_priority,
                str,
            )
        ):

            explicit_order = list(
                self.candidate_priority
            )

            positions = {
                feature: position
                for position, feature
                in enumerate(
                    explicit_order
                )
            }

            return pd.Series(
                {
                    feature: -positions.get(
                        feature,
                        len(explicit_order),
                    )
                    for feature
                    in scores.index
                }
            )

        if (
            self.candidate_order
            == "univariate_gini"
        ):

            return scores[
                "univariate_validation_gini"
            ].fillna(-np.inf)

        if (
            self.candidate_order
            == "univariate_pvalue"
        ):

            return (
                -scores[
                    "univariate_p_value"
                ].fillna(np.inf)
            )

        if self.candidate_order == "original":

            return pd.Series(
                {
                    feature: -position
                    for position, feature
                    in enumerate(
                        scores.index
                    )
                }
            )

        raise ValueError(
            "Invalid candidate_order."
        )

    def _ordered_candidates(
        self,
    ):
        return self.candidate_scores.index.tolist()

    # =====================================================
    # Candidate evaluation
    # =====================================================

    def _evaluate_candidate(
        self,
        feature,
        features,
        X_train,
        y_train,
        X_valid,
        y_valid,
    ):
        try:

            train_matrix = sm.add_constant(
                X_train[features],
                has_constant="add",
            )

            valid_matrix = sm.add_constant(
                X_valid[features],
                has_constant="add",
            )

            valid_matrix = valid_matrix[
                train_matrix.columns
            ]

            model = sm.Logit(
                y_train,
                train_matrix,
            ).fit(
                disp=0,
                maxiter=200,
            )

            train_probability = model.predict(
                train_matrix
            )

            valid_probability = model.predict(
                valid_matrix
            )

            train_gini = self._gini(
                y_train,
                train_probability,
            )

            validation_gini = self._gini(
                y_valid,
                valid_probability,
            )

            return {
                "feature": feature,
                "p_value": float(
                    model.pvalues[feature]
                ),
                "train_gini": train_gini,
                "validation_gini": (
                    validation_gini
                ),
                "aic": model.aic,
                "bic": model.bic,
            }

        except (
            PerfectSeparationError,
            LinAlgError,
            ValueError,
            KeyError,
        ):
            return None

    def _check_correlation(
        self,
        feature,
    ):
        if (
            self.max_correlation is None
            or not self.selected_features
        ):
            return {
                "accepted": True,
                "max_correlation": np.nan,
                "correlated_with": None,
            }

        correlations = (
            self._correlation_matrix.loc[
                feature,
                self.selected_features,
            ]
        )

        max_correlation = (
            correlations.max()
        )

        correlated_with = (
            correlations.idxmax()
        )

        return {
            "accepted": (
                pd.isna(max_correlation)
                or max_correlation
                <= self.max_correlation
            ),
            "max_correlation": (
                max_correlation
            ),
            "correlated_with": (
                correlated_with
            ),
        }

    # =====================================================
    # Final model
    # =====================================================

    def _fit_final_model(
        self,
        X_train,
        y_train,
        X_valid,
        y_valid,
    ):
        if not self.selected_features:
            raise RuntimeError(
                "Forward selection did not select "
                "any feature."
            )

        train_matrix = sm.add_constant(
            X_train[
                self.selected_features
            ],
            has_constant="add",
        )

        valid_matrix = sm.add_constant(
            X_valid[
                self.selected_features
            ],
            has_constant="add",
        )

        valid_matrix = valid_matrix[
            train_matrix.columns
        ]

        self.final_model = sm.Logit(
            y_train,
            train_matrix,
        ).fit(
            disp=0,
            maxiter=200,
        )

        train_probability = (
            self.final_model.predict(
                train_matrix
            )
        )

        validation_probability = (
            self.final_model.predict(
                valid_matrix
            )
        )

        self.train_gini = self._gini(
            y_train,
            train_probability,
        )

        self.validation_gini = self._gini(
            y_valid,
            validation_probability,
        )

    # =====================================================
    # Validation
    # =====================================================

    @staticmethod
    def _gini(
        y_true,
        probability,
    ):
        return (
            2
            * roc_auc_score(
                y_true,
                probability,
            )
            - 1
        )

    @staticmethod
    def _prepare_X(
        X,
        name,
    ):
        if not isinstance(
            X,
            pd.DataFrame,
        ):
            X = pd.DataFrame(X)

        X = X.copy()

        if "Intercept" in X.columns:
            raise ValueError(
                f"{name} must not contain "
                "'Intercept'."
            )

        if "const" in X.columns:
            raise ValueError(
                f"{name} must not contain "
                "'const'."
            )

        non_numeric = X.select_dtypes(
            exclude="number"
        ).columns.tolist()

        if non_numeric:
            raise TypeError(
                f"{name} contains non-numeric "
                f"columns: {non_numeric}"
            )

        if X.isna().any().any():
            missing_columns = (
                X.columns[
                    X.isna().any()
                ].tolist()
            )

            raise ValueError(
                f"{name} contains missing values "
                f"in: {missing_columns}"
            )

        return X.astype(float)

    @staticmethod
    def _prepare_y(
        y,
        index,
        name,
    ):
        y = pd.Series(
            y,
            index=index,
            name=name,
        ).astype(int)

        unique_values = set(
            y.dropna().unique()
        )

        if not unique_values.issubset(
            {0, 1}
        ):
            raise ValueError(
                f"{name} must be binary 0/1."
            )

        if len(unique_values) < 2:
            raise ValueError(
                f"{name} must contain both classes."
            )

        return y

    @staticmethod
    def _validate_datasets(
        X_train,
        X_valid,
        y_train,
        y_valid,
    ):
        if list(X_train.columns) != list(
            X_valid.columns
        ):
            raise ValueError(
                "X_train and X_valid must have "
                "the same columns in the same order."
            )

        if len(X_train) != len(y_train):
            raise ValueError(
                "X_train and y_train have "
                "different lengths."
            )

        if len(X_valid) != len(y_valid):
            raise ValueError(
                "X_valid and y_valid have "
                "different lengths."
            )

    def _validate_parameters(
        self,
    ):
        if (
            self.significance_level
            is not None
            and not 0
            < self.significance_level
            <= 1
        ):
            raise ValueError(
                "significance_level must be "
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

        if (
            self.max_features is not None
            and self.max_features < 1
        ):
            raise ValueError(
                "max_features must be positive."
            )

        if self.min_gini_increase < 0:
            raise ValueError(
                "min_gini_increase cannot "
                "be negative."
            )

        if self.correlation_method not in {
            "pearson",
            "spearman",
        }:
            raise ValueError(
                "correlation_method must be "
                "'pearson' or 'spearman'."
            )

        valid_orders = {
            "univariate_gini",
            "univariate_pvalue",
            "original",
        }

        if (
            self.candidate_priority is None
            and self.candidate_order
            not in valid_orders
        ):
            raise ValueError(
                "candidate_order must be one of "
                f"{sorted(valid_orders)}."
            )

    def _check_is_fitted(
        self,
    ):
        if not self._is_fitted:
            raise RuntimeError(
                "ForwardSelector is not fitted."
            )

    def __repr__(
        self,
    ):
        if not self._is_fitted:

            return (
                "ForwardSelector("
                "status='not_fitted'"
                ")"
            )

        return (
            "ForwardSelector("
            f"selected={len(self.selected_features)}, "
            f"validation_gini="
            f"{self.validation_gini:.4f}"
            ")"
        )
