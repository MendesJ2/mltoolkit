from mltoolkit.core.base import BaseComponent
from mltoolkit.data.dataset import Dataset


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


    def load_dataset(
        self,
        dataframe,
    ):

        self.dataset = Dataset(
            dataframe=dataframe,
            config=self.config,
            logger=self.logger,
        )

        self.info(
            "Dataset loaded."
        )

        return self.dataset


    def summary(self):

        print("=" * 50)

        print(
            self.config.project_name
        )

        print("=" * 50)


        if self.dataset is not None:

            self.dataset.summary()

        else:

            print(
                "No dataset loaded."
            )
