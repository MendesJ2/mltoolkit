import pandas as pd


class Metadata:

    def __init__(self):

        self.features = []

    def add(self, feature):

        self.features.append(feature)

    def to_dataframe(self):

        return pd.DataFrame([vars(f) for f in self.features])

    @property
    def continuous(self):

        return [
            f.name
            for f in self.features
            if f.variable_type == "continuous"
        ]

    @property
    def categorical(self):

        return [
            f.name
            for f in self.features
            if f.variable_type == "categorical"
        ]

    @property
    def binary(self):

        return [
            f.name
            for f in self.features
            if f.variable_type == "binary"
        ]

    @property
    def dates(self):

        return [
            f.name
            for f in self.features
            if f.variable_type == "datetime"
        ]

    @property
    def ids(self):

        return [
            f.name
            for f in self.features
            if f.role == "id"
        ]
