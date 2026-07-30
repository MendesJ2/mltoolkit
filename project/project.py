from mltoolkit.core.base import BaseComponent
from mltoolkit.data.dataset import Dataset
from projects.insurance.features import build_features
from mltoolkit.features import FeatureTransformer

class BaseProject(BaseComponent):

    def __init__(
        self,
        config,
        logger,
    ):

        super().__init__(
            config=config,
            logger=logger,
        )

        self.dataset = None

    def load_dataset(self, dataframe):

        self.dataset = Dataset(
            dataframe=dataframe,
            config=self.config,
            logger=self.logger,
        )

        self.info("Dataset loaded.")

        features = build_features()
    
        transformer = FeatureTransformer(
            features
        )
        
        
        df_features = transformer.transform(
            dataset.df
        )
        
        
        dataset.add_features(
            df_features,
            features.names()
        )

    def summary(self):

        print("=" * 50)

        print(self.config.project_name)

        print("=" * 50)

        if self.dataset is not None:

            self.dataset.summary()

        else:

            print("No dataset loaded.")
