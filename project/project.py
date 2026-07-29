from mltoolkit.core.base import BaseComponent

from mltoolkit.data.dataset import Dataset


class BaseProject(BaseComponent):

    def __init__(self, config, logger):

        super().__init__(config, logger)

        self.dataset = None

    def load_dataset(self, dataframe):

        self.dataset = Dataset(
            dataframe,
            config=self.config,
            logger=self.logger,
        )

        self.info("Dataset loaded.")

    def summary(self):

        print()

        print("=" * 40)

        print(self.config.project_name)

        print("=" * 40)

        if self.dataset:

            self.dataset.summary()
