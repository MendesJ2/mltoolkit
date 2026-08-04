from mltoolkit.eda.plots.temporal import (
    plot_feature_temporal,
    plot_target_temporal,
    plot_volume_temporal,
)


class TemporalAnalysis:

    def __init__(
        self,
        feature_name,
        target_table,
        feature_table,
        variable_type,
        target_name,
        group=None,
    ):
        self.feature_name = feature_name
        self.target_table = target_table
        self.feature_table = feature_table
        self.variable_type = variable_type
        self.target_name = target_name
        self.group = group

    def summary(self):

        return {
            "target": self.target_table,
            "feature": self.feature_table,
        }

    def plot_target(self):

        return plot_target_temporal(
            table=self.target_table,
            feature_name=self.feature_name,
            group=self.group,
        )

    def plot_feature(self):

        if self.variable_type != "continuous":
            raise ValueError(
                "plot_feature() is only available "
                "for continuous variables in MVP1."
            )

        return plot_feature_temporal(
            table=self.feature_table,
            feature_name=self.feature_name,
            group=self.group,
        )

    def plot_volume(self):

        return plot_volume_temporal(
            table=self.target_table,
            group=self.group,
        )

    def __repr__(self):

        return (
            "TemporalAnalysis("
            f"feature_name='{self.feature_name}', "
            f"variable_type='{self.variable_type}', "
            f"group='{self.group}'"
            ")"
        )
