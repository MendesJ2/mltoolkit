from mltoolkit.eda.plots.stability import (
    plot_psi,
    plot_stability_distribution,
)


class StabilityAnalysis:
    """
    Container for PSI and distribution stability.
    """

    def __init__(
        self,
        feature_name,
        by,
        summary,
        detail,
        distribution,
        reference,
    ):
        self.feature_name = feature_name
        self.by = by
        self.summary = summary
        self.table = summary
        self.detail = detail
        self.distribution = distribution
        self.reference = reference

    def plot_distribution(self):

        return plot_stability_distribution(
            distribution=self.distribution,
            feature_name=self.feature_name,
            by=self.by,
        )

    def plot_psi(self):

        return plot_psi(
            summary=self.summary,
            feature_name=self.feature_name,
        )

    def __repr__(self):

        return (
            "StabilityAnalysis("
            f"feature_name='{self.feature_name}', "
            f"by='{self.by}', "
            f"reference='{self.reference}'"
            ")"
        )
