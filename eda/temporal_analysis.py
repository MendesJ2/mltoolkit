from mltoolkit.eda.plots.temporal import (
    plot_binary_feature_temporal,
    plot_categorical_feature_temporal,
    plot_continuous_feature_temporal,
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
        analysis_type,
        group=None,
    ):
        self.feature_name = feature_name
        self.target_table = target_table
        self.feature_table = feature_table
        self.variable_type = variable_type
        self.target_name = target_name
        self.analysis_type = analysis_type
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

    def plot_feature(
        self,
        statistic="mean",
    ):

        if (
            self.analysis_type
            == "continuous"
        ):

            return (
                plot_continuous_feature_temporal(
                    table=self.feature_table,
                    feature_name=(
                        self.feature_name
                    ),
                    target_name=(
                        self.target_name
                    ),
                    group=self.group,
                    statistic=statistic,
                )
            )

        if (
            self.analysis_type
            == "binary"
        ):
        
            return (
                plot_binary_feature_temporal(
                    table=self.feature_table,
                    feature_name=(
                        self.feature_name
                    ),
                    target_name=(
                        self.target_name
                    ),
                    group=self.group,
                )
            )
        
        return (
            plot_categorical_feature_temporal(
                table=self.feature_table,
                feature_name=(
                    self.feature_name
                ),
                target_name=(
                    self.target_name
                ),
                group=self.group,
            )
        )

    def plot_volume(self):

        return plot_volume_temporal(
            table=self.target_table,
            group=self.group,
        )
