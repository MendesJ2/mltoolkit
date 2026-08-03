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
