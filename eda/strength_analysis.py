from mltoolkit.eda.plots.strength import (
    plot_gain_ks,
    plot_lift,
    plot_woe,
)


class StrengthAnalysis:
    """
    Container for univariate feature-strength results.
    """

    def __init__(
        self,
        feature_name,
        table,
        metrics,
    ):
        self.feature_name = feature_name
        self.table = table
        self.metrics = metrics

    @property
    def iv(self):
        return self.metrics["iv"]

    @property
    def ks(self):
        return self.metrics["max_ks"]

    def plot_woe(self):

        return plot_woe(
            table=self.table,
            feature_name=self.feature_name,
        )

    def plot_lift(self):

        return plot_lift(
            table=self.table,
            feature_name=self.feature_name,
        )

    def plot_gain(self):

        return plot_gain_ks(
            table=self.table,
            feature_name=self.feature_name,
        )

    def summary(self):

        return self.metrics

    def __repr__(self):

        return (
            "StrengthAnalysis("
            f"feature_name='{self.feature_name}', "
            f"iv={self.iv:.4f}, "
            f"ks={self.ks:.4f}"
            ")"
        )
