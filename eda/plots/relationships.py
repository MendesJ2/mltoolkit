import plotly.express as px


def plot_relationship_matrix(
    matrix,
    title,
    minimum=-1,
    maximum=1,
):
    """
    Plot relationship matrix as a Plotly heatmap.
    """

    if matrix.empty:
        raise ValueError(
            "Relationship matrix is empty."
        )

    fig = px.imshow(
        matrix,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="RdBu_r",
        zmin=minimum,
        zmax=maximum,
        title=title,
    )

    fig.update_layout(
        template="plotly_white",
        height=max(
            500,
            len(matrix) * 28,
        ),
        width=max(
            700,
            len(matrix.columns) * 35,
        ),
        xaxis_title="Feature",
        yaxis_title="Feature",
    )

    fig.update_xaxes(
        tickangle=45
    )

    return fig
