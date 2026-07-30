from mltoolkit.core.base import BaseComponent

from .summarizers import (
    single_feature_summary,
    single_feature_statistics,
)

from .plots import (
    plot_continuous,
    plot_categorical,
    plot_binary,
)


class EDAFeature(BaseComponent):
    """
    Object representing a single feature analysis.
    """

    def __init__(
        self,
        dataset,
        feature_name,
        config=None,
        logger=None,
    ):

        super().__init__(
            config=config,
            logger=logger,
        )

        self.dataset = dataset

        self.feature_name = feature_name

        self.series = dataset.df[feature_name]

    @property
    def metadata(self):

        return (
            self.dataset.metadata
            .to_dataframe()
            .query(
                "name == @self.feature_name"
            )
        )

    def summary(self):

        return single_feature_summary(
            self.series,
            self.feature_name,
        )

    def statistics(self):

        return single_feature_statistics(
            self.series,
            self.feature_name,
        )

    def plot(self):
    
        variable_type = (
            self.dataset
            .feature(self.feature_name)
            .variable_type
        )
    
    
        if variable_type == "continuous":
    
            return plot_continuous(
                self.series,
                self.feature_name
            )
    
    
        if variable_type == "binary":
    
            return plot_binary(
                self.series,
                self.feature_name
            )
    
    
        if variable_type == "categorical":
    
            return plot_categorical(
                self.series,
                self.feature_name
            )
    
    
        raise ValueError(
            f"No plot implemented for {variable_type}"
        )
