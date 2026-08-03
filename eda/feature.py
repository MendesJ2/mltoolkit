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

from .target import target_analysis

from .feature_analysis import FeatureAnalysis

from .comparison import compare_feature
from .comparison_analysis import ComparisonAnalysis

from .temporal import temporal_analysis
from .temporal_analysis import TemporalAnalysis

from .quality import quality_analysis
from .quality_analysis import QualityAnalysis

from .strength import feature_strength
from .strength_analysis import StrengthAnalysis

from .stability import stability_analysis
from .stability_analysis import StabilityAnalysis

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

    def target(
        self,
        n_bins=10
    ):
    
        metadata = (
            self.dataset.metadata
            .to_dataframe()
            .query(
                "name == @self.feature_name"
            )
            .iloc[0]
        )
    
    
        table = target_analysis(
    
            df=self.dataset.df,
    
            feature=self.feature_name,
    
            target=self.dataset.config.target,
    
            variable_type=metadata["variable_type"],
    
            n_bins=n_bins
        )
    
    
        return FeatureAnalysis(
            feature_name=self.feature_name,
            table=table,
            variable_type=metadata[
                "variable_type"
            ],
            global_rate=self.dataset.df[
                self.dataset.config.target
            ].mean(),
        )

    def compare(
        self,
        by,
    ):
    
        if by not in self.dataset.df.columns:
            raise ValueError(
                f"Group column '{by}' not found in dataset."
            )
    
        metadata = (
            self.dataset.metadata
            .to_dataframe()
            .query(
                "name == @self.feature_name"
            )
            .iloc[0]
        )
    
        variable_type = metadata[
            "variable_type"
        ]
    
        table = compare_feature(
            df=self.dataset.df,
            feature=self.feature_name,
            group=by,
            variable_type=variable_type,
        )
    
        data = self.dataset.df[
            [
                self.feature_name,
                by,
            ]
        ].copy()
    
        return ComparisonAnalysis(
            feature_name=self.feature_name,
            group=by,
            variable_type=variable_type,
            table=table,
            data=data,
        )


    def temporal(
        self,
        date,
        freq="M",
        group=None,
    ):
    
        if date not in self.dataset.df.columns:
            raise ValueError(
                f"Date column '{date}' not found."
            )
    
        if (
            group is not None
            and group not in self.dataset.df.columns
        ):
            raise ValueError(
                f"Group column '{group}' not found."
            )
    
        metadata = (
            self.dataset.metadata
            .to_dataframe()
            .query(
                "name == @self.feature_name"
            )
            .iloc[0]
        )
    
        result = temporal_analysis(
            df=self.dataset.df,
            feature=self.feature_name,
            target=self.dataset.config.target,
            date=date,
            variable_type=metadata[
                "variable_type"
            ],
            freq=freq,
            group=group,
        )
    
        return TemporalAnalysis(
            feature_name=self.feature_name,
            target_table=result["target"],
            feature_table=result["feature"],
            variable_type=metadata[
                "variable_type"
            ],
            group=group,
        )


    def quality(self):
    
    
        table = quality_analysis(
            self.series
        )
    
    
        return QualityAnalysis(
    
            feature_name=self.feature_name,
    
            table=table
    
        )


    def strength(
        self,
        n_bins=10,
        smoothing=0.5,
    ):
    
        metadata = (
            self.dataset.metadata
            .to_dataframe()
            .query(
                "name == @self.feature_name"
            )
            .iloc[0]
        )
    
        result = feature_strength(
            df=self.dataset.df,
            feature=self.feature_name,
            target=self.dataset.config.target,
            variable_type=metadata[
                "variable_type"
            ],
            n_bins=n_bins,
            smoothing=smoothing,
        )
    
        return StrengthAnalysis(
            feature_name=self.feature_name,
            table=result["table"],
            metrics=result["metrics"],
        )

    def stability(
        self,
        by,
        reference=None,
        n_bins=10,
    ):
    
        if by not in self.dataset.df.columns:
            raise ValueError(
                f"Column '{by}' not found."
            )
    
        metadata = (
            self.dataset.metadata
            .to_dataframe()
            .query(
                "name == @self.feature_name"
            )
            .iloc[0]
        )
    
        result = stability_analysis(
            df=self.dataset.df,
            feature=self.feature_name,
            by=by,
            variable_type=metadata[
                "variable_type"
            ],
            reference=reference,
            n_bins=n_bins,
        )
    
        return StabilityAnalysis(
            feature_name=self.feature_name,
            by=by,
            summary=result["summary"],
            detail=result["detail"],
            distribution=result[
                "distribution"
            ],
            reference=result["reference"],
        )
