import plotly.graph_objects as go


def plot_categorical(
    series,
    title
):

    counts = (
        series
        .value_counts(dropna=False)
        .head(20)
    )


    fig = go.Figure()


    fig.add_trace(
        go.Bar(
            x=counts.index.astype(str),
            y=counts.values
        )
    )


    fig.update_layout(
        title=title,
        template="plotly_white",
        height=500
    )


    return fig
