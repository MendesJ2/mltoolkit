from mltoolkit.core.base import BaseComponent
from mltoolkit.data.dataset import Dataset


class BaseProject(BaseComponent):
    """
    Base class for ML projects.
    """

    def __init__(self, config, logger):
        super().__init__(config=config, logger=logger)

        self.dataset = None

    def load_dataset(self, dataframe):
        """
        Load a pandas DataFrame into the project.
        """

        self.dataset = Dataset(
            dataframe=dataframe,
            config=self.config,
            logger=self.logger
        )

        self.info("Dataset loaded successfully.")

    def summary(self):
        """
        Display a summary of the project.
        """

        print("=" * 50)
        print(self.config.project_name)
        print("=" * 50)

        if self.dataset is not None:
            self.dataset.summary()
        else:
            print("No dataset loaded.")
