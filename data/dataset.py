import pandas as pd

from mltoolkit.core.base import BaseComponent

from .feature import Feature
from .metadata import Metadata


class Dataset(BaseComponent):

    def __init__(self,
                 dataframe,
                 config=None,
                 logger=None):

        super().__init__(config, logger)

        self.df = dataframe.copy()

        self.metadata = Metadata()

        self._build_metadata()

    @property
    def shape(self):

        return self.df.shape

    @property
    def features(self):

        return self.metadata.to_dataframe()

    @property
    def continuous(self):

        return self.metadata.continuous

    @property
    def categorical(self):

        return self.metadata.categorical

    @property
    def binary(self):

        return self.metadata.binary

    @property
    def dates(self):

        return self.metadata.dates

    def summary(self):

        print(f"Rows: {self.df.shape[0]:,}")

        print(f"Columns: {self.df.shape[1]}")

        print(f"Continuous: {len(self.continuous)}")

        print(f"Categorical: {len(self.categorical)}")

        print(f"Binary: {len(self.binary)}")

        print(f"Datetime: {len(self.dates)}")
