from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go


class ModelComparison:
    """
    Compare multiple ModelEvaluation objects.
    """

    def __init__(
        self,
    ):
        self._models = {}

    def add(
        self,
        name,
        evaluation,
    ):
        if (
            evaluation.metrics
            is None
        ):
            raise ValueError(
                "Evaluation contains "
                "no results."
            )

        self._models[name] = (
            evaluation
        )

        return self

    @property
    def table(
        self,
    ):
        records = []

        for (
            model_name,
            evaluation,
        ) in self._models.items():

            table = (
                evaluation.metrics
            )

            for _, row in (
                table.iterrows()
            ):

                records.append(
                    {
                        "model": (
                            model_name
                        ),
                        "sample": (
                            row["sample"]
                        ),
                        "observations": (
                            row[
                                "observations"
                            ]
                        ),
                        "target_rate": (
                            row[
                                "target_rate"
                            ]
                        ),
                        "auc": row["auc"],
                        "gini": row["gini"],
                        "ks": row["ks"],
                    }
                )

        return pd.DataFrame(
            records
        )

    def pivot(
        self,
        metric="gini",
    ):
        table = self.table

        if metric not in {
            "auc",
            "gini",
            "ks",
            "target_rate",
        }:
            raise ValueError(
                "Unsupported metric."
            )

        return (
            table
            .pivot(
                index="model",
                columns="sample",
                values=metric,
            )
        )

    def plot(
        self,
        metric="gini",
    ):
        pivot = self.pivot(
            metric=metric
        )

        fig = go.Figure()

        for sample in (
            pivot.columns
        ):

            fig.add_trace(
                go.Bar(
                    x=pivot.index,
                    y=pivot[sample],
                    name=str(sample),
                )
            )

        fig.update_layout(
            title=(
                f"Model comparison "
                f"— {metric.upper()}"
            ),
            template="plotly_white",
            height=500,
            barmode="group",
            xaxis_title="Model",
            yaxis_title=(
                metric.upper()
            ),
        )

        return fig

    def generalization_gap(
        self,
        *,
        train_sample="train",
        validation_sample=(
            "validation"
        ),
        metric="gini",
    ):
        pivot = self.pivot(
            metric=metric
        )

        required = {
            train_sample,
            validation_sample,
        }

        missing = (
            required
            - set(pivot.columns)
        )

        if missing:
            raise ValueError(
                "Missing samples: "
                f"{sorted(missing)}"
            )

        result = (
            pivot[
                [
                    train_sample,
                    validation_sample,
                ]
            ]
            .copy()
        )

        result[
            "generalization_gap"
        ] = (
            result[
                train_sample
            ]
            - result[
                validation_sample
            ]
        )

        return (
            result
            .sort_values(
                "generalization_gap",
                ascending=False,
            )
        )
