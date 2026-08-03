import plotly.express as px


def plot_comparison(
    table,
    feature_name,
    variable_type,
    group,
):


    if variable_type == "continuous":

        fig = px.box(
            table,
            x=group,
            y="value",
            points=False,
            title=f"{feature_name} vs {group}"
        )


    else:

        fig = px.bar(
            table,
            x=group,
            y="percentage",
            color=feature_name,
            barmode="group",
            title=f"{feature_name} vs {group}"
        )


    return fig
