import pandas as pd


class FeatureTransformer:
    """
    Executes project-defined feature functions.
    """


    def __init__(
        self,
        feature_set,
    ):

        self.feature_set = feature_set


    def transform(
        self,
        df: pd.DataFrame,
    ):

        result = df.copy()


        for name, function in self.feature_set.items():

            result[name] = function(
                result
            )


        return result
