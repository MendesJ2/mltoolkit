class ComparisonAnalysis:


    def __init__(
        self,
        feature_name,
        table,
        variable_type,
        group,
    ):

        self.feature_name = feature_name

        self.table = table

        self.variable_type = variable_type

        self.group = group


    def plot(self):

        from mltoolkit.eda.plots.comparison import (
            plot_comparison
        )

        return plot_comparison(
            self.table,
            self.feature_name,
            self.variable_type,
            self.group,
        )


    def test(self):

        from mltoolkit.eda.statistics.comparison import (
            comparison_test
        )

        return comparison_test(
            self.table,
            self.variable_type,
        )
