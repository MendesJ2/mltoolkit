import plotly.graph_objects as go


def plot_target_temporal(
    table,
    feature_name=None,
):
    """
    Target rate evolution over time.
    """

    fig = go.Figure()


    fig.add_trace(
        go.Scatter(
            x=table["period"],
            y=table["target_rate"],
            mode="lines+markers",
            name="Target Rate",
        )
    )


    fig.update_layout(

        title=(
            f"Target Rate temporal"
            if feature_name is None
            else f"{feature_name} - Target Rate temporal"
        ),

        xaxis_title="Period",

        yaxis_title="Target Rate",

        height=500,

    )


    return fig



def plot_feature_temporal(
    table,
    feature_name,
):
    """
    Feature mean evolution by target.
    """

    fig = go.Figure()


    for target_value in sorted(
        table["target"].unique()
    ):

        temp = table[
            table["target"] == target_value
        ]


        fig.add_trace(

            go.Scatter(

                x=temp["period"],

                y=temp["mean_feature"],

                mode="lines+markers",

                name=f"Target={target_value}"

            )

        )


    fig.update_layout(

        title=(
            f"{feature_name} temporal "
            "by target"
        ),

        xaxis_title="Period",

        yaxis_title=(
            f"Mean {feature_name}"
        ),

        height=500,

    )


    return fig



def plot_volume_temporal(
    table,
):

    fig = go.Figure()


    fig.add_trace(

        go.Bar(

            x=table["period"],

            y=table["observations"],

            name="Observations"

        )

    )


    fig.update_layout(

        title="Observations temporal",

        xaxis_title="Period",

        yaxis_title="Observations",

        height=400,

    )


    return fig
