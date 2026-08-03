class QualityAnalysis:


    def __init__(
        self,
        feature_name,
        table,
    ):

        self.feature_name = feature_name

        self.table = table



    def summary(self):

        return self.table
