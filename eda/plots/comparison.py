import plotly.express as px


def plot_comparison(
    data,
    table,
    feature_name,
    variable_type,
    group,
):
    """
    Plot feature distribution across groups.
    """

    if variable_type == "continuous":

        plot_data = data[
            [
                feature_name,
                group,
            ]
        ].dropna()

        fig = px.box(
            plot_data,
            x=group,
            y=feature_name,
            points=False,
            title=f"{feature_name} por {group}",
        )

        fig.update_layout(
            template="plotly_white",
            height=500,
            xaxis_title=group,
            yaxis_title=feature_name,
        )

        return fig

    fig = px.bar(
        table,
        x=group,
        y="percentage",
        color=feature_name,
        barmode="group",
        title=f"Distribuição de {feature_name} por {group}",
        custom_data=["observations"],
    )

    fig.update_traces(
        hovertemplate=(
            f"{group}: %{{x}}"
            f"<br>{feature_name}: %{{fullData.name}}"
            "<br>Percentagem: %{y:.2%}"
            "<br>Observações: %{customdata[0]:,.0f}"
            "<extra></extra>"
        )
    )

    fig.update_layout(
        template="plotly_white",
        height=500,
        xaxis_title=group,
        yaxis_title="Percentagem",
        yaxis_tickformat=".1%",
        legend_title=feature_name,
    )

    return fig
