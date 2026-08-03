from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm

from numpy.linalg import LinAlgError
from sklearn.metrics import roc_auc_score
from statsmodels.tools.sm_exceptions import (
    ConvergenceWarning,
    PerfectSeparationError,
)


class LogisticModel:
    """
    Interpretable logistic-regression baseline using Statsmodels.

    Parameters
    ----------
    max_iter:
        Maximum number of optimization iterations.

    significance_level:
        Confidence level used in coefficient diagnostics.

    add_intercept:
        Whether to add an intercept internally.

    verbose:
        Whether Statsmodels prints optimization information.

    Notes
    -----
    X must already be:

        - numeric;
        - without missing values;
        - transformed consistently across samples;
        - without an intercept/constant column.
    """

    def __init__(
        self,
        *,
        max_iter: int = 200,
        significance_level: float = 0.05,
        add_intercept: bool = True,
        verbose: bool = False,
    ):
        self.max_iter = max_iter
        self.significance_level = (
            significance_level
        )
        self.add_intercept = add_intercept
        self.verbose = verbose

        self.model = None
        self.feature_names = None

        self.train_probabilities = None
        self.train_gini = None
        self.train_auc = None

        self.converged = None

        self._is_fitted = False

        self._validate_parameters()

    # =====================================================
    # Public API
    # =====================================================

    def fit(
        self,
        X,
        y,
    ):
        """
        Fit logistic-regression model.
        """

        X = self._prepare_X(
            X,
            fitting=True,
        )

        y = self._prepare_y(
            y,
            index=X.index,
        )

        design_matrix = (
            self._build_design_matrix(X)
        )

        try:

            with warnings.catch_warnings(
                record=True
            ) as captured_warnings:

                warnings.simplefilter(
                    "always",
                    ConvergenceWarning,
                )

                self.model = sm.Logit(
                    y,
                    design_matrix,
                ).fit(
                    disp=self.verbose,
                    maxiter=self.max_iter,
                )

                convergence_warnings = [
                    warning
                    for warning
                    in captured_warnings
                    if issubclass(
                        warning.category,
                        ConvergenceWarning,
                    )
                ]

            self.converged = bool(
                self.model.mle_retvals.get(
                    "converged",
                    len(
                        convergence_warnings
                    )
                    == 0,
                )
            )

        except PerfectSeparationError as error:

            raise RuntimeError(
                "Perfect separation detected. "
                "One or more features perfectly predict "
                "the target."
            ) from error

        except LinAlgError as error:

            raise RuntimeError(
                "The logistic model could not be fitted "
                "because the design matrix is singular. "
                "Check multicollinearity and duplicate "
                "features."
            ) from error

        self.train_probabilities = (
            pd.Series(
                self.model.predict(
                    design_matrix
                ),
                index=X.index,
                name="probability",
            )
        )

        self.train_auc = roc_auc_score(
            y,
            self.train_probabilities,
        )

        self.train_gini = (
            2 * self.train_auc - 1
        )

        self._is_fitted = True

        return self

    def predict_proba(
        self,
        X,
    ):
        """
        Predict event probability.
        """

        self._check_is_fitted()

        X = self._prepare_X(
            X,
            fitting=False,
        )

        design_matrix = (
            self._build_design_matrix(X)
        )

        probabilities = self.model.predict(
            design_matrix
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
        """
        Predict binary class.
        """

        if not 0 <= threshold <= 1:
            raise ValueError(
                "threshold must be between 0 and 1."
            )

        probabilities = self.predict_proba(
            X
        )

        return (
            probabilities
            >= threshold
        ).astype(int).rename(
            "prediction"
        )

    def coefficients(
        self,
    ):
        """
        Return coefficients, odds ratios, p-values
        and confidence intervals.
        """

        self._check_is_fitted()

        confidence_interval = (
            self.model.conf_int(
                alpha=self.significance_level
            )
        )

        confidence_interval.columns = [
            "coefficient_ci_lower",
            "coefficient_ci_upper",
        ]

        result = pd.DataFrame(
            {
                "coefficient": (
                    self.model.params
                ),
                "std_error": (
                    self.model.bse
                ),
                "z_value": (
                    self.model.tvalues
                ),
                "p_value": (
                    self.model.pvalues
                ),
            }
        )

        result = result.join(
            confidence_interval
        )

        result["odds_ratio"] = np.exp(
            result["coefficient"]
        )

        result["odds_ratio_ci_lower"] = (
            np.exp(
                result[
                    "coefficient_ci_lower"
                ]
            )
        )

        result["odds_ratio_ci_upper"] = (
            np.exp(
                result[
                    "coefficient_ci_upper"
                ]
            )
        )

        result["significant"] = (
            result["p_value"]
            < self.significance_level
        )

        result.index.name = "feature"

        return result.reset_index()

    def summary(
        self,
    ):
        """
        Compact model summary.
        """

        self._check_is_fitted()

        return pd.Series(
            {
                "observations": int(
                    self.model.nobs
                ),
                "features": len(
                    self.feature_names
                ),
                "converged": self.converged,
                "log_likelihood": (
                    self.model.llf
                ),
                "aic": self.model.aic,
                "bic": self.model.bic,
                "pseudo_r2": (
                    self.model.prsquared
                ),
                "train_auc": (
                    self.train_auc
                ),
                "train_gini": (
                    self.train_gini
                ),
            }
        )

    def statsmodels_summary(
        self,
    ):
        """
        Return native Statsmodels summary.
        """

        self._check_is_fitted()

        return self.model.summary()

    def feature_effects(
        self,
        exclude_intercept=True,
    ):
        """
        Return features ordered by absolute coefficient.
        """

        result = self.coefficients()

        if exclude_intercept:

            result = result[
                ~result["feature"].isin(
                    {
                        "const",
                        "Intercept",
                    }
                )
            ]

        result["absolute_coefficient"] = (
            result["coefficient"].abs()
        )

        return (
            result
            .sort_values(
                "absolute_coefficient",
                ascending=False,
            )
            .reset_index(drop=True)
        )

    # =====================================================
    # Internal preparation
    # =====================================================

    def _prepare_X(
        self,
        X,
        fitting,
    ):
        if not isinstance(
            X,
            pd.DataFrame,
        ):
            raise TypeError(
                "X must be a pandas DataFrame."
            )

        X = X.copy()

        forbidden_columns = {
            "const",
            "Intercept",
        }

        present_forbidden = (
            forbidden_columns
            & set(X.columns)
        )

        if present_forbidden:

            raise ValueError(
                "X must not contain an intercept "
                "column. Found: "
                f"{sorted(present_forbidden)}"
            )

        if X.columns.duplicated().any():

            duplicated_columns = (
                X.columns[
                    X.columns.duplicated()
                ].tolist()
            )

            raise ValueError(
                "X contains duplicated columns: "
                f"{duplicated_columns}"
            )

        non_numeric = (
            X.select_dtypes(
                exclude="number"
            )
            .columns
            .tolist()
        )

        if non_numeric:

            raise TypeError(
                "X contains non-numeric columns: "
                f"{non_numeric}"
            )

        if X.isna().any().any():

            missing_columns = (
                X.columns[
                    X.isna().any()
                ].tolist()
            )

            raise ValueError(
                "X contains missing values in: "
                f"{missing_columns}"
            )

        if fitting:

            if X.shape[1] == 0:
                raise ValueError(
                    "X contains no features."
                )

            self.feature_names = list(
                X.columns
            )

        else:

            missing_features = (
                set(self.feature_names)
                - set(X.columns)
            )

            extra_features = (
                set(X.columns)
                - set(self.feature_names)
            )

            if missing_features:

                raise ValueError(
                    "Prediction data is missing "
                    "model features: "
                    f"{sorted(missing_features)}"
                )

            if extra_features:

                raise ValueError(
                    "Prediction data contains "
                    "unexpected features: "
                    f"{sorted(extra_features)}"
                )

            X = X[
                self.feature_names
            ]

        return X.astype(float)

    @staticmethod
    def _prepare_y(
        y,
        index,
    ):
        if isinstance(
            y,
            pd.Series,
        ):

            y = y.copy()

            if not y.index.equals(index):

                y = y.reindex(index)

        else:

            y = pd.Series(
                y,
                index=index,
            )

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
                "y must contain only 0 and 1. "
                f"Found: {sorted(unique_values)}"
            )

        if len(unique_values) < 2:

            raise ValueError(
                "y must contain both classes."
            )

        return y.astype(int)

    def _build_design_matrix(
        self,
        X,
    ):
        if not self.add_intercept:
            return X

        return sm.add_constant(
            X,
            has_constant="add",
        )

    # =====================================================
    # Validation
    # =====================================================

    def _validate_parameters(
        self,
    ):
        if self.max_iter < 1:

            raise ValueError(
                "max_iter must be positive."
            )

        if not (
            0
            < self.significance_level
            < 1
        ):

            raise ValueError(
                "significance_level must be "
                "between 0 and 1."
            )

    def _check_is_fitted(
        self,
    ):
        if not self._is_fitted:

            raise RuntimeError(
                "LogisticModel is not fitted."
            )

    def __repr__(
        self,
    ):
        if not self._is_fitted:

            return (
                "LogisticModel("
                "status='not_fitted'"
                ")"
            )

        return (
            "LogisticModel("
            f"features={len(self.feature_names)}, "
            f"train_gini={self.train_gini:.4f}, "
            f"converged={self.converged}"
            ")"
        )
