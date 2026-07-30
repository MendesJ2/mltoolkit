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

        super().__init__(
            config=config,
            logger=logger,
        )

        self.df = dataframe.copy()

        self.metadata = Metadata()

        self._build_metadata()

    # =====================================================
    # Public Properties
    # =====================================================

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
    def ordinal(self):

        return self.metadata.ordinal

    @property
    def dates(self):

        return self.metadata.datetime

    # =====================================================
    # Public Methods
    # =====================================================

    def summary(self):

        return pd.DataFrame(
            {
                "Metric": [
                    "Rows",
                    "Columns",
                    "Continuous",
                    "Categorical",
                    "Binary",
                    "Ordinal",
                    "Datetime",
                ],
                "Value": [
                    self.df.shape[0],
                    self.df.shape[1],
                    len(self.continuous),
                    len(self.categorical),
                    len(self.binary),
                    len(self.ordinal),
                    len(self.dates),
                ],
            }
        )

    def feature(self, name):

        return self.metadata.get(name)

    # =====================================================
    # Metadata
    # =====================================================

    def _build_metadata(self):

        for column in self.df.columns:

            series = self.df[column]

            feature = Feature(

                name=column,

                dtype=str(series.dtype),

                role=self._infer_role(column),

                variable_type=self._infer_variable_type(
                    column,
                    series,
                ),

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

    # =====================================================
    # Infer role
    # =====================================================

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
            and column == self.config.source_column
        ):
            return "source"

        if (
            self.config is not None
            and column in self.config.id_columns
        ):
            return "id"

        return "feature"

    # =====================================================
    # Infer variable type
    # =====================================================

    def _infer_variable_type(
        self,
        column,
        series,
    ):

        # ----------------------------------------------
        # Manual override
        # ----------------------------------------------

        if (
            self.config is not None
            and column in self.config.variable_types
        ):
            return self.config.variable_types[column]

        # ----------------------------------------------
        # Automatic inference
        # ----------------------------------------------

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
