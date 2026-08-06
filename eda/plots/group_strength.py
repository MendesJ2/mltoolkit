import plotly.express as px


def plot_group_strength_heatmap(
    table,
    metric="iv",
):
    """
    Plot feature strength by group.

    Expected columns:
        feature
        group_value
        metric
    """

    matrix = (
        table
        .pivot_table(
            index="feature",
            columns="group_value",
            values=metric,
            aggfunc="first",
        )
    )

    fig = px.imshow(
        matrix,
        text_auto=".3f",
        aspect="auto",
        title=(
            f"{metric.upper()} por grupo"
        ),
    )

    fig.update_layout(
        template="plotly_white",
        height=max(
            500,
            30 * len(matrix),
        ),
        xaxis_title="Grupo",
        yaxis_title="Feature",
    )

    return fig
