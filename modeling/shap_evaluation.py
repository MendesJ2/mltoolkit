from __future__ import annotations

import numpy as np
import pandas as pd


class SHAPEvaluation:
    """
    SHAP analysis for a fitted tree model.
    """

    def __init__(
        self,
        model,
    ):
        if not getattr(
            model,
            "_is_fitted",
            False,
        ):
            raise RuntimeError(
                "Model must be fitted "
                "before SHAP analysis."
            )

        self.tree_model = model
        self.explainer = None

        self.X = None
        self.values = None

    def fit(
        self,
        X,
    ):
        try:
            import shap

        except ImportError as error:

            raise ImportError(
                "SHAP is required for "
                "SHAPEvaluation. "
                "Install package 'shap'."
            ) from error

        X = (
            self.tree_model
            ._prepare_X(
                X,
                fitting=False,
            )
        )

        self.X = X

        self.explainer = (
            shap.TreeExplainer(
                self.tree_model.model
            )
        )

        explanation = (
            self.explainer(
                X
            )
        )

        values = (
            explanation.values
        )

        # Defensive handling for
        # multi-output SHAP formats.
        if values.ndim == 3:

            values = values[
                :, :, -1
            ]

        self.values = pd.DataFrame(
            values,
            index=X.index,
            columns=X.columns,
        )

        return self

    def importance(
        self,
    ):
        """
        Mean absolute SHAP value per feature.
        """

        self._check_is_fitted()

        result = pd.DataFrame(
            {
                "feature": (
                    self.values.columns
                ),

                "mean_abs_shap": (
                    self.values
                    .abs()
                    .mean()
                    .values
                ),

                "mean_shap": (
                    self.values
                    .mean()
                    .values
                ),
            }
        )

        return (
            result
            .sort_values(
                "mean_abs_shap",
                ascending=False,
            )
            .reset_index(
                drop=True
            )
        )

    def values_table(
        self,
    ):
        self._check_is_fitted()

        return self.values.copy()

    def dependence_table(
        self,
        feature,
    ):
        """
        Return raw feature value + SHAP contribution.

        Useful for plotting / inspecting nonlinear
        relationships.
        """

        self._check_is_fitted()

        if feature not in (
            self.values.columns
        ):
            raise ValueError(
                f"Feature '{feature}' "
                "not found."
            )

        return pd.DataFrame(
            {
                "feature_value": (
                    self.X[feature]
                ),

                "shap_value": (
                    self.values[
                        feature
                    ]
                ),
            },
            index=self.X.index,
        )

    def plot_summary(
        self,
        max_display=20,
    ):
        self._check_is_fitted()

        import shap

        shap.summary_plot(
            self.values.values,
            self.X,
            feature_names=(
                self.X.columns
            ),
            max_display=max_display,
        )

    def plot_dependence(
        self,
        feature,
        interaction_feature="auto",
    ):
        """
        SHAP dependence plot.

        interaction_feature='auto' lets SHAP identify
        the strongest interaction candidate.
        """

        self._check_is_fitted()

        import shap

        shap.dependence_plot(
            feature,
            self.values.values,
            self.X,
            interaction_index=(
                interaction_feature
            ),
        )

    def _check_is_fitted(
        self,
    ):
        if self.values is None:
            raise RuntimeError(
                "SHAPEvaluation has not "
                "been fitted."
            )
