import plotly.graph_objects as go


def plot_continuous(
    series,
    title
):

    fig = go.Figure()


    fig.add_trace(
        go.Histogram(
            x=series,
            nbinsx=40
        )
    )


    fig.update_layout(
        title=title,
        template="plotly_white",
        height=500
    )


    return fig
