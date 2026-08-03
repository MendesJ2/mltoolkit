from mltoolkit.core.base import BaseComponent

from .feature import EDAFeature
from .summarizers import (
    dataset_summary,
    feature_summary,
    missing_summary,
    statistics_summary,
)

from .relationship_analysis import (
    RelationshipAnalysis,
)

from .relationships import (
    categorical_association,
    numeric_correlation,
)

from .quality_report import (
    build_quality_report,
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

    
    def correlation(
        self,
        method="pearson",
        columns=None,
    ):
    
        if columns is None:
    
            columns = [
                feature
                for feature in (
                    self.dataset.continuous
                    + self.dataset.ordinal
                    + self.dataset.binary
                )
                if (
                    self.dataset
                    .feature(feature)
                    .role
                    == "feature"
                )
            ]
    
        matrix = numeric_correlation(
            df=self.dataset.df,
            columns=columns,
            method=method,
        )
    
        return RelationshipAnalysis(
            matrix=matrix,
            method=method,
        )
    
    
    def categorical_relationships(
        self,
        columns=None,
    ):
    
        if columns is None:
    
            columns = [
                feature
                for feature in (
                    self.dataset.categorical
                    + self.dataset.binary
                    + self.dataset.ordinal
                )
                if (
                    self.dataset
                    .feature(feature)
                    .role
                    == "feature"
                )
            ]
    
        matrix = categorical_association(
            df=self.dataset.df,
            columns=columns,
        )
    
        return RelationshipAnalysis(
            matrix=matrix,
            method="cramers_v",
        )

    def quality_report(
        self,
        only_features=True,
    ):
    
        return build_quality_report(
            dataset=self.dataset,
            only_features=only_features,
        )
