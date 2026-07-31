class FeatureAnalysis:
    """
    Container for feature analysis results.
    """


    def __init__(
        self,
        feature_name,
        table=None,
    ):

        self.feature_name = feature_name

        self.table = table



    def get_table(self):

        return self.table



    def summary(self):

        return self.table



    def __repr__(self):

        return (
            f"FeatureAnalysis("
            f"{self.feature_name})"
        )
