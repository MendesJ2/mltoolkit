from mltoolkit.project.project import BaseProject


class InsuranceProject(BaseProject):

    def __init__(self,
                 config,
                 logger):

        super().__init__(
            config=config,
            logger=logger
        )
