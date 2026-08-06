from mltoolkit.eda.plots.target import (
    plot_target_analysis,
)


class FeatureAnalysis:
    """
    Container for feature-vs-target analysis.
    """

    def __init__(
        self,
        feature_name,
        table,
        variable_type,
        global_rate,
        group=None,
    ):
        self.feature_name = feature_name
        self.table = table
        self.variable_type = variable_type
        self.global_rate = global_rate
        self.group = group

    def plot(self):

        return plot_target_analysis(
            table=self.table,
            feature_name=self.feature_name,
            variable_type=(
                self.variable_type
            ),
            group=self.group,
        )

    def get_table(self):

        return self.table

    def __repr__(self):

        return (
            "FeatureAnalysis("
            f"feature_name='{self.feature_name}', "
            f"variable_type='{self.variable_type}', "
            f"group='{self.group}'"
            ")"
        )
