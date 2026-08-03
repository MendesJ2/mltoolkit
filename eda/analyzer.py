from mltoolkit.core.base import BaseComponent

from .feature import EDAFeature
from .summarizers import (
    dataset_summary,
    feature_summary,
    missing_summary,
    statistics_summary,
)

class EDAAnalyzer(BaseComponent):
    """
    Main entry point for exploratory data analysis.
    """

    def __init__(
        self,
        dataset,
        config=None,
        logger=None,
    ):

        super().__init__(
            config=config,
            logger=logger,
        )

        self.dataset = dataset

    def summary(self):

        return dataset_summary(
            self.dataset
        )

    def feature_summary(self):

        return feature_summary(
            self.dataset
        )

    def missing_summary(self):

        return missing_summary(
            self.dataset
        )

    def statistics(self):

        return statistics_summary(
            self.dataset
        )

    def feature(self, name):

        if name not in self.dataset.df.columns:

            raise ValueError(
                f"Feature '{name}' not found in dataset."
            )

        return EDAFeature(
            dataset=self.dataset,
            feature_name=name,
            config=self.config,
            logger=self.logger,
        )

    
        )
