import pandas as pd

from mltoolkit.core.base import BaseComponent

from .feature import Feature
from .metadata import Metadata


class Dataset(BaseComponent):

    def __init__(
        self,
        dataframe: pd.DataFrame,
        config=None,
        logger=None,
    ):

        super().__init__(config=config, logger=logger)

        self.df = dataframe.copy()

        self.metadata = Metadata()

        self._build_metadata()

    ####################################################################
    # Public Properties
    ####################################################################

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

    ####################################################################
    # Public Methods
    ####################################################################

    def summary(self):

        print(f"Rows: {self.df.shape[0]:,}")

        print(f"Columns: {self.df.shape[1]}")

        print(f"Continuous: {len(self.continuous)}")

        print(f"Categorical: {len(self.categorical)}")

        print(f"Binary: {len(self.binary)}")

        print(f"Datetime: {len(self.dates)}")

    ####################################################################
    # Private Methods
    ####################################################################

    def _build_metadata(self):

        for column in self.df.columns:

            series = self.df[column]

            feature = Feature(

                name=column,

                dtype=str(series.dtype),

                role=self._infer_role(column),

                variable_type=self._infer_variable_type(series),

                n_unique=series.nunique(dropna=False),

                missing_pct=series.isna().mean(),

                is_constant=series.nunique(dropna=False) == 1,

                is_quasi_constant=(
                    series.value_counts(
                        normalize=True,
                        dropna=False
                    ).max()
                    > 0.99
                ),
            )

            self.metadata.add(feature)

    def _infer_role(self, column):

        if (
            self.config is not None
            and column == self.config.target
        ):
            return "target"

        if (
            self.config is not None
            and column == self.config.date_column
        ):
            return "date"

        if (
            self.config is not None
            and column in self.config.id_columns
        ):
            return "id"

        return "feature"

    def _infer_variable_type(self, series):

        if pd.api.types.is_datetime64_any_dtype(series):

            return "datetime"

        unique = series.nunique()

        if unique == 2:

            return "binary"

        if pd.api.types.is_numeric_dtype(series):

            if unique < 10:

                return "categorical"

            return "continuous"

        return "categorical"
