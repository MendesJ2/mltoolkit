from mltoolkit.eda.plots.temporal import (
    plot_target_temporal,
    plot_feature_temporal,
    plot_volume_temporal,
)


class TemporalAnalysis:


    def __init__(
        self,
        feature_name,
        target_table,
        feature_table,
        variable_type,
    ):

        self.feature_name = feature_name

        self.target_table = target_table

        self.feature_table = feature_table

        self.variable_type = variable_type



    def summary(self):

        return {

            "target": self.target_table,

            "feature": self.feature_table,

        }



    def plot_target(self):

        return plot_target_temporal(
            self.target_table,
            self.feature_name,
        )



    def plot_feature(self):

        return plot_feature_temporal(
            self.feature_table,
            self.feature_name,
        )



    def plot_volume(self):

        return plot_volume_temporal(
            self.target_table,
        )
