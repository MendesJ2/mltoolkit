from mltoolkit.eda.plots.comparison import plot_comparison
from mltoolkit.eda.statistics.comparison import comparison_test


class ComparisonAnalysis:
    """
    Results of comparing one feature across population groups.
    """

    def __init__(
        self,
        feature_name,
        group,
        variable_type,
        table,
        data,
    ):
        self.feature_name = feature_name
        self.group = group
        self.variable_type = variable_type
        self.table = table
        self.data = data

    def plot(self):
        return plot_comparison(
            data=self.data,
            table=self.table,
            feature_name=self.feature_name,
            variable_type=self.variable_type,
            group=self.group,
        )

    def test(self):
        return comparison_test(
            data=self.data,
            feature_name=self.feature_name,
            group=self.group,
            variable_type=self.variable_type,
        )
