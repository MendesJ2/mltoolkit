from mltoolkit.project.project import BaseProject

from .config import config
from .feature_engineering import create_features
from .preprocessing import preprocess


class ProjectTemplate(BaseProject):
    """
    Example project implementation.

    Project-specific transformations happen
    before creating the generic mltoolkit Dataset.
    """

    def __init__(
        self,
        logger=None,
    ):
        super().__init__(
            config=config,
            logger=logger,
        )

    def prepare_dataset(
        self,
        dataframe,
    ):
        """
        Prepare project data and create
        the mltoolkit Dataset.

        Order:
            1. preprocessing
            2. feature engineering
            3. Dataset / metadata
        """

        dataframe = preprocess(
            dataframe
        )

        dataframe = create_features(
            dataframe
        )

        return self.load_dataset(
            dataframe
        )
