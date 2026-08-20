from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
)


class ModelEvaluation:
    """
    Evaluate binary-classification probabilities
    across one or more samples.

    Typical samples:
        - train
        - validation
        - oot
    """

    def __init__(
        self,
        *,
        n_bins=10,
    ):
        self.n_bins = n_bins

        self.metrics = None
        self.deciles = None

        self._samples = {}

    # =====================================================
    # Public API
    # =====================================================

    def add_sample(
        self,
        name,
        y_true,
        probabilities,
    ):
        """
        Add one evaluation sample.
        """

        y_true = self._prepare_y(
            y_true
        )

        probabilities = (
            self._prepare_probabilities(
                probabilities,
                index=y_true.index,
            )
        )

        self._samples[name] = {
            "y_true": y_true,
            "probabilities": probabilities,
        }

        self._calculate()

        return self

    def summary(
        self,
    ):
        return self.metrics.copy()

    def lift_table(
        self,
        sample=None,
    ):
        """
        Return decile/lift table.

        If sample is None, return all samples.
        """

        if sample is None:
            return self.deciles.copy()

        return (
            self.deciles[
                self.deciles[
                    "sample"
                ] == sample
            ]
            .reset_index(drop=True)
        )

    # =====================================================
    # Plots
    # =====================================================

    def plot_lift(
        self,
    ):
        """
        Compare lift by decile across samples.
        """

        fig = go.Figure()

        for (
            sample_name,
            data,
        ) in self.deciles.groupby(
            "sample",
            sort=False,
        ):

            fig.add_trace(
                go.Scatter(
                    x=data["decile"],
                    y=data["lift"],
                    mode="lines+markers",
                    name=str(sample_name),
                )
            )

        fig.add_hline(
            y=1,
            line_dash="dash",
        )

        fig.update_layout(
            title="Lift by decile",
            template="plotly_white",
            height=500,
            xaxis={
                "title": "Decile",
                "dtick": 1,                
            },
            yaxis={
                "title": "Lift",
                "rangemode": "tozero",
            },
        )

        return fig

    def plot_target_rate(
        self,
    ):
        """
        Compare observed target rate by decile.
        """

        fig = go.Figure()

        for (
            sample_name,
            data,
        ) in self.deciles.groupby(
            "sample",
            sort=False,
        ):

            fig.add_trace(
                go.Scatter(
                    x=data["decile"],
                    y=data[
                        "target_rate"
                    ],
                    mode="lines+markers",
                    name=str(sample_name),
                )
            )

        fig.update_layout(
            title=(
                "Observed target rate "
                "by decile"
            ),
            template="plotly_white",
            height=500,
            xaxis={
                "title": "Decile",
                "dtick": 1,
            },
            yaxis={
                "title": "Target rate",
                "tickformat": ".2%",
                "rangemode": "tozero",
            },
        )

        return fig

    def plot_cumulative_lift(
        self,
    ):
        """
        Compare cumulative lift across samples.
        """

        fig = go.Figure()

        for (
            sample_name,
            data,
        ) in self.deciles.groupby(
            "sample",
            sort=False,
        ):

            fig.add_trace(
                go.Scatter(
                    x=data[
                        "cumulative_population_pct"
                    ],
                    y=data[
                        "cumulative_lift"
                    ],
                    mode="lines+markers",
                    name=str(sample_name),
                )
            )

        fig.add_hline(
            y=1,
            line_dash="dash",
        )

        fig.update_layout(
            title="Cumulative lift",
            template="plotly_white",
            height=500,
            xaxis={
                "title": (
                    "Cumulative population"
                ),
                "tickformat": ".0%",
            },
            yaxis={
                "title": "Cumulative lift",
            },
        )

        return fig

    # =====================================================
    # Calculation
    # =====================================================

    def _calculate(
        self,
    ):
        metric_records = []
        decile_tables = []

        for (
            sample_name,
            sample,
        ) in self._samples.items():

            y = sample["y_true"]
            prob = sample[
                "probabilities"
            ]

            auc = roc_auc_score(
                y,
                prob,
            )

            gini = 2 * auc - 1

            ks = self._ks(
                y,
                prob,
            )

            metric_records.append(
                {
                    "sample": sample_name,
                    "observations": len(y),
                    "events": int(
                        y.sum()
                    ),
                    "target_rate": (
                        y.mean()
                    ),
                    "auc": auc,
                    "gini": gini,
                    "ks": ks,
                }
            )

            decile_table = (
                self._build_deciles(
                    sample_name=sample_name,
                    y=y,
                    probabilities=prob,
                )
            )

            decile_tables.append(
                decile_table
            )

        self.metrics = (
            pd.DataFrame(
                metric_records
            )
        )

        self.deciles = pd.concat(
            decile_tables,
            ignore_index=True,
        )

    # =====================================================
    # Deciles
    # =====================================================

    def _build_deciles(
        self,
        *,
        sample_name,
        y,
        probabilities,
    ):
        data = pd.DataFrame(
            {
                "target": y,
                "probability": probabilities,
            }
        )

        ranks = (
            data["probability"]
            .rank(
                method="first",
                ascending=True,
            )
        )

        data["decile"] = (
            pd.qcut(
                ranks,
                q=self.n_bins,
                labels=False,
            )
            + 1
        )

        global_rate = (
            data["target"].mean()
        )

        table = (
            data
            .groupby(
                "decile",
                observed=True,
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
                avg_probability=(
                    "probability",
                    "mean",
                ),
                min_probability=(
                    "probability",
                    "min",
                ),
                max_probability=(
                    "probability",
                    "max",
                ),
            )
            .reset_index()
        )

        table["population_pct"] = (
            table["observations"]
            / table[
                "observations"
            ].sum()
        )

        table["event_pct"] = (
            table["events"]
            / table["events"].sum()
        )

        table["lift"] = (
            table["target_rate"]
            / global_rate
        )

        cumulative = (
            table
            .sort_values(
                "decile",
                ascending=False,
            )
            .copy()
        )
        
        cumulative[
            "cumulative_population_pct"
        ] = (
            cumulative[
                "population_pct"
            ]
            .cumsum()
        )
        
        cumulative[
            "cumulative_event_pct"
        ] = (
            cumulative[
                "event_pct"
            ]
            .cumsum()
        )
        
        cumulative[
            "cumulative_lift"
        ] = (
            cumulative[
                "cumulative_event_pct"
            ]
            / cumulative[
                "cumulative_population_pct"
            ]
        )
        
        table = (
            table
            .drop(
                columns=[
                    "cumulative_population_pct",
                    "cumulative_event_pct",
                    "cumulative_lift",
                ],
                errors="ignore",
            )
            .merge(
                cumulative[
                    [
                        "decile",
                        "cumulative_population_pct",
                        "cumulative_event_pct",
                        "cumulative_lift",
                    ]
                ],
                on="decile",
                how="left",
            )
            .sort_values(
                "decile"
            )
            .reset_index(
                drop=True
            )
        )

        table.insert(
            0,
            "sample",
            sample_name,
        )

        return table

    # =====================================================
    # Metrics
    # =====================================================

    @staticmethod
    def _ks(
        y,
        probabilities,
    ):
        fpr, tpr, _ = roc_curve(
            y,
            probabilities,
        )

        return float(
            np.max(
                np.abs(
                    tpr - fpr
                )
            )
        )

    # =====================================================
    # Validation
    # =====================================================

    @staticmethod
    def _prepare_y(
        y,
    ):
        if isinstance(
            y,
            pd.Series,
        ):
            y = y.copy()

        else:
            y = pd.Series(y)

        if y.isna().any():
            raise ValueError(
                "y_true contains missing values."
            )

        unique_values = set(
            y.unique()
        )

        if not unique_values.issubset(
            {0, 1}
        ):
            raise ValueError(
                "y_true must contain only 0 and 1."
            )

        if len(unique_values) < 2:
            raise ValueError(
                "y_true must contain both classes."
            )

        return y.astype(int)

    @staticmethod
    def _prepare_probabilities(
        probabilities,
        *,
        index,
    ):
        if isinstance(
            probabilities,
            pd.Series,
        ):
            probabilities = (
                probabilities.copy()
            )

            if not (
                probabilities.index
                .equals(index)
            ):
                probabilities = (
                    probabilities
                    .reindex(index)
                )

        else:
            probabilities = pd.Series(
                probabilities,
                index=index,
            )

        if probabilities.isna().any():
            raise ValueError(
                "Probabilities contain "
                "missing values."
            )

        if (
            (probabilities < 0).any()
            or (probabilities > 1).any()
        ):
            raise ValueError(
                "Probabilities must be "
                "between 0 and 1."
            )

        return probabilities.astype(
            float
        )

    def __repr__(
        self,
    ):
        samples = list(
            self._samples.keys()
        )

        return (
            "ModelEvaluation("
            f"samples={samples}, "
            f"n_bins={self.n_bins}"
            ")"
        )
