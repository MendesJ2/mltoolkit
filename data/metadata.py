import pandas as pd

from .feature import Feature


class Metadata:
    """
    Holds metadata for every feature in the dataset.
    """

    def __init__(self):

        self._features = {}

    # =====================================================
    # Basic Operations
    # =====================================================

    def add(self, feature: Feature):

        self._features[feature.name] = feature

    def get(self, name):

        return self._features[name]

    def exists(self, name):

        return name in self._features

    # =====================================================
    # Properties
    # =====================================================

    @property
    def features(self):

        return list(self._features.values())

    @property
    def names(self):

        return list(self._features.keys())

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
    def ordinal(self):

        return [
            feature.name
            for feature in self.features
            if feature.variable_type == "ordinal"
        ]

    @property
    def datetime(self):

        return [
            feature.name
            for feature in self.features
            if feature.variable_type == "datetime"
        ]

    @property
    def eligible(self):
    
        return [
            f.name
            for f in self.features
            if f.role == "feature"
        ]

    # =====================================================
    # Export
    # =====================================================

    def to_dataframe(self):

        return pd.DataFrame(
            [vars(feature) for feature in self.features]
        )
