from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.ensemble import (
    GradientBoostingClassifier,
)

from sklearn.metrics import (
    roc_auc_score,
)


class TreeModel:
    """
    Gradient-boosting baseline for binary classification.

    Intended to be directly comparable with LogisticModel.

    X must already be:
        - numeric
        - without missing values
        - transformed consistently across samples

    Scaling is not required by the model, but scaled
    features can still be used for direct comparison
    with an existing Logistic pipeline.
    """

    def __init__(
        self,
        *,
        n_estimators=150,
        learning_rate=0.05,
        max_depth=2,
        min_samples_leaf=30,
        subsample=0.8,
        random_state=42,
    ):
        self.n_estimators = (
            n_estimators
        )

        self.learning_rate = (
            learning_rate
        )

        self.max_depth = (
            max_depth
        )

        self.min_samples_leaf = (
            min_samples_leaf
        )

        self.subsample = (
            subsample
        )

        self.random_state = (
            random_state
        )

        self.model = None
        self.feature_names = None

        self.train_probabilities = None
        self.train_auc = None
        self.train_gini = None

        self._is_fitted = False

    # =====================================================
    # Public API
    # =====================================================

    def fit(
        self,
        X,
        y,
    ):
        X = self._prepare_X(
            X,
            fitting=True,
        )
    
        y = self._prepare_y(
            y,
            index=X.index,
        )
    
        self.model = (
            GradientBoostingClassifier(
                n_estimators=(
                    self.n_estimators
                ),
                learning_rate=(
                    self.learning_rate
                ),
                max_depth=(
                    self.max_depth
                ),
                min_samples_leaf=(
                    self.min_samples_leaf
                ),
                subsample=(
                    self.subsample
                ),
                random_state=(
                    self.random_state
                ),
            )
        )
    
        self.model.fit(
            X,
            y,
        )
    
        # Model is now fitted.
        self._is_fitted = True
    
        self.train_probabilities = (
            self.predict_proba(
                X
            )
        )
    
        self.train_auc = (
            roc_auc_score(
                y,
                self.train_probabilities,
            )
        )
    
        self.train_gini = (
            2 * self.train_auc - 1
        )
    
        return self

    def predict_proba(
        self,
        X,
    ):
        self._check_is_fitted()

        X = self._prepare_X(
            X,
            fitting=False,
        )

        probabilities = (
            self.model
            .predict_proba(X)[
                :, 1
            ]
        )

        return pd.Series(
            probabilities,
            index=X.index,
            name="probability",
        )

    def predict(
        self,
        X,
        threshold=0.50,
    ):
        if not 0 <= threshold <= 1:
            raise ValueError(
                "threshold must be "
                "between 0 and 1."
            )

        probabilities = (
            self.predict_proba(X)
        )

        return (
            probabilities
            >= threshold
        ).astype(int)

    def feature_importance(
        self,
    ):
        """
        Native impurity-based tree importance.

        Prefer SHAP for model interpretation.
        """

        self._check_is_fitted()

        result = pd.DataFrame(
            {
                "feature": (
                    self.feature_names
                ),
                "importance": (
                    self.model
                    .feature_importances_
                ),
            }
        )

        return (
            result
            .sort_values(
                "importance",
                ascending=False,
            )
            .reset_index(
                drop=True
            )
        )

    # =====================================================
    # Validation
    # =====================================================

    def _prepare_X(
        self,
        X,
        *,
        fitting,
    ):
        if not isinstance(
            X,
            pd.DataFrame,
        ):
            X = pd.DataFrame(X)

        else:
            X = X.copy()

        non_numeric = [
            column
            for column in X.columns
            if not (
                pd.api.types
                .is_numeric_dtype(
                    X[column]
                )
            )
        ]

        if non_numeric:
            raise TypeError(
                "X contains non-numeric "
                f"columns: {non_numeric}"
            )

        if X.isna().any().any():
            raise ValueError(
                "X contains missing values."
            )

        if fitting:

            self.feature_names = (
                X.columns.tolist()
            )

        else:

            missing = (
                set(
                    self.feature_names
                )
                - set(X.columns)
            )

            if missing:
                raise ValueError(
                    "Missing model features: "
                    f"{sorted(missing)}"
                )

            X = X[
                self.feature_names
            ]

        return X

    @staticmethod
    def _prepare_y(
        y,
        *,
        index,
    ):
        if not isinstance(
            y,
            pd.Series,
        ):
            y = pd.Series(
                y,
                index=index,
            )

        else:
            y = y.copy()

        if not y.index.equals(index):
            y = y.reindex(index)

        if y.isna().any():
            raise ValueError(
                "y contains missing values."
            )

        unique_values = set(
            y.unique()
        )

        if not unique_values.issubset(
            {0, 1}
        ):
            raise ValueError(
                "y must contain "
                "only 0 and 1."
            )

        return y.astype(int)

    def _check_is_fitted(
        self,
    ):
        if not self._is_fitted:
            raise RuntimeError(
                "TreeModel is not fitted."
            )

    def __repr__(
        self,
    ):
        return (
            "TreeModel("
            f"n_estimators={self.n_estimators}, "
            f"learning_rate={self.learning_rate}, "
            f"max_depth={self.max_depth}, "
            f"min_samples_leaf={self.min_samples_leaf}"
            ")"
        )
