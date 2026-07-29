import pandas as pd

from mltoolkit.core.base import BaseComponent


class Dataset(BaseComponent):

    def __init__(self, dataframe: pd.DataFrame, config=None, logger=None):

        super().__init__(config, logger)

        self.df = dataframe.copy()

    @property
    def shape(self):

        return self.df.shape

    @property
    def columns(self):

        return list(self.df.columns)

    def head(self, n=5):

        return self.df.head(n)

    def summary(self):

        self.info("Dataset summary")

        print(f"Rows: {self.df.shape[0]}")

        print(f"Columns: {self.df.shape[1]}")
