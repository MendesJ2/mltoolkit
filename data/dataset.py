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

    ###################################
    # PRIVATE METHODS
    ###################################

    def _build_metadata(self):

        for col in self.df.columns:

            s = self.df[col]

            n_unique = s.nunique(dropna=False)

            missing = s.isna().mean()

            is_constant = n_unique == 1

            is_quasi_constant = (
                s.value_counts(normalize=True,
                               dropna=False)
                .max()
                > 0.99
            )

            role = self._infer_role(col)

            variable_type = self._infer_variable_type(s)

            feature = Feature(

                name=col,

                dtype=str(s.dtype),

                role=role,

                variable_type=variable_type,

                n_unique=n_unique,

                missing_pct=missing,

                is_constant=is_constant,

                is_quasi_constant=is_quasi_constant,

            )

            self.metadata.add(feature)

    def _infer_role(self, column):

        if column == self.config.target:
    
            return "target"
    
        if column == self.config.date_column:
    
            return "date"
    
        if column.lower().endswith("id"):
    
            return "id"
    
        return "feature"


    def _infer_variable_type(self, series):
    
        if pd.api.types.is_datetime64_any_dtype(series):
    
            return "datetime"
    
        nunique = series.nunique()
    
        if nunique == 2:
    
            return "binary"
    
        if pd.api.types.is_numeric_dtype(series):
    
            if nunique < 10:
    
                return "categorical"
    
            return "continuous"
    
        return "categorical"
