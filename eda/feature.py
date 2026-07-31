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

from .target import target_summary

from .target import target_analysis

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

    def target(self):
        
        target = (
            self.dataset
            .config
            .target
        )
    
    
        return target_summary(
    
            self.dataset.df,
    
            self.feature_name,
    
            target
    
        )

    def target_analysis(
        self,
        n_bins=10
    ):
    
        metadata = (
            self.dataset
            .feature(
                self.feature_name
            )
        )
    
        return target_analysis(
    
            df=self.dataset.df,
    
            feature=self.feature_name,
    
            target=self.dataset.config.target,
    
            variable_type=metadata.variable_type,
    
            n_bins=n_bins
    
        )
