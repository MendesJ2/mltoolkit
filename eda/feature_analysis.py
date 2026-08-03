from mltoolkit.eda.plots.target import (
    plot_target_categorical,
    plot_target_continuous,
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
    ):
        self.feature_name = feature_name
        self.table = table
        self.variable_type = variable_type
        self.global_rate = global_rate

    def plot(self):

        if self.variable_type == "continuous":

            return plot_target_continuous(
                table=self.table,
                feature_name=self.feature_name,
                global_rate=self.global_rate,
            )

        if self.variable_type in {
            "binary",
            "categorical",
            "ordinal",
        }:

            return plot_target_categorical(
                table=self.table,
                feature_name=self.feature_name,
                global_rate=self.global_rate,
            )

        raise ValueError(
            "Target plot not implemented for variable type "
            f"'{self.variable_type}'."
        )

    def get_table(self):
        return self.table

    def __repr__(self):
        return (
            "FeatureAnalysis("
            f"feature_name='{self.feature_name}', "
            f"variable_type='{self.variable_type}'"
            ")"
        )
