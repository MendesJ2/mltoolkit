import pandas as pd


class Metadata:

    def __init__(self):

        self.features = []

    def add(self, feature):

        self.features.append(feature)

    def to_dataframe(self):

        return pd.DataFrame(
            [vars(feature) for feature in self.features]
        )

    @property
    def continuous(self):

        return [
            feature.name
            for feature in self.features
            if feature.variable_type == "continuous"
        ]

    @property
    def categorical(self):

        return [
            feature.name
            for feature in self.features
            if feature.variable_type == "categorical"
        ]

    @property
    def binary(self):

        return [
            feature.name
            for feature in self.features
            if feature.variable_type == "binary"
        ]

    @property
    def dates(self):

        return [
            feature.name
            for feature in self.features
            if feature.variable_type == "datetime"
        ]
