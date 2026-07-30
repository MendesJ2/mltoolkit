class FeatureSet:
    """
    Container for project-specific feature definitions.

    The toolkit does not know how features are created.
    Each project defines its own logic.
    """

    def __init__(self):

        self.features = {}


    def add(
        self,
        name,
        function,
    ):

        if name in self.features:
            raise ValueError(
                f"Feature '{name}' already exists."
            )

        self.features[name] = function


    def items(self):

        return self.features.items()


    def names(self):

        return list(
            self.features.keys()
        )
