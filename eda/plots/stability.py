import plotly.express as px


def plot_stability_distribution(
    distribution,
    feature_name,
    by,
):

    fig = px.line(
        distribution,
        x="_stability_group",
        y="distribution",
        color=by,
        markers=True,
        title=(
            f"Estabilidade de {feature_name} "
            f"por {by}"
        ),
    )

    fig.update_layout(
        template="plotly_white",
        height=500,
        xaxis_title=feature_name,
        yaxis_title="Distribuição",
        yaxis_tickformat=".1%",
        legend_title=by,
    )

    fig.update_xaxes(
        tickangle=45
    )

    return fig


def plot_psi(
    summary,
    feature_name,
):

    fig = px.bar(
        summary,
        x="comparison",
        y="psi",
        title=f"PSI — {feature_name}",
        custom_data=[
            "reference",
            "stability",
        ],
    )

    fig.update_traces(
        hovertemplate=(
            "Comparação: %{x}"
            "<br>PSI: %{y:.4f}"
            "<br>Referência: %{customdata[0]}"
            "<br>Classificação: %{customdata[1]}"
            "<extra></extra>"
        )
    )

    fig.add_hline(
        y=0.10,
        line_dash="dash",
        annotation_text="0.10",
    )

    fig.add_hline(
        y=0.25,
        line_dash="dash",
        annotation_text="0.25",
    )

    fig.update_layout(
        template="plotly_white",
        height=450,
        xaxis_title="População",
        yaxis_title="PSI",
    )

    return fig
