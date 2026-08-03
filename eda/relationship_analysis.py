from mltoolkit.eda.plots.relationships import (
    plot_relationship_matrix,
)

from mltoolkit.eda.relationships import (
    high_relationship_pairs,
)


class RelationshipAnalysis:
    """
    Container for a relationship matrix.
    """

    def __init__(
        self,
        matrix,
        method,
    ):
        self.matrix = matrix
        self.table = matrix
        self.method = method

    def plot(self):

        if self.method == "cramers_v":

            return plot_relationship_matrix(
                matrix=self.matrix,
                title="Cramér's V",
                minimum=0,
                maximum=1,
            )

        return plot_relationship_matrix(
            matrix=self.matrix,
            title=(
                f"Correlação "
                f"{self.method.capitalize()}"
            ),
            minimum=-1,
            maximum=1,
        )

    def high_pairs(
        self,
        threshold=0.80,
    ):

        return high_relationship_pairs(
            matrix=self.matrix,
            threshold=threshold,
        )

    def __repr__(self):

        return (
            "RelationshipAnalysis("
            f"method='{self.method}', "
            f"n_features={len(self.matrix)}"
            ")"
        )
