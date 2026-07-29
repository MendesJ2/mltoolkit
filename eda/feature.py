from mltoolkit.core.base import BaseComponent

from .summarizers import (
    single_feature_summary,
    single_feature_statistics,
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
