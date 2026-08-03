import plotly.graph_objects as go


def plot_target_continuous(
    table,
    feature_name,
    global_rate,
):
    """
    Volume and target rate by continuous-feature bin.
    """

    category_column = table.columns[0]

    x_values = (
        table[category_column]
        .astype(str)
    )

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=x_values,
            y=table["observations"],
            name="Observações",
            opacity=0.4,
            yaxis="y",
            customdata=table[
                [
                    "events",
                    "non_events",
                ]
            ],
            hovertemplate=(
                "Bin: %{x}"
                "<br>Observações: %{y:,.0f}"
                "<br>Adesões: %{customdata[0]:,.0f}"
                "<br>Não adesões: %{customdata[1]:,.0f}"
                "<extra></extra>"
            ),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=x_values,
            y=table["target_rate"],
            name="Taxa de adesão",
            mode="lines+markers",
            yaxis="y2",
            customdata=table[
                [
                    "event_rate_index",
                ]
            ],
            hovertemplate=(
                "Bin: %{x}"
                "<br>Taxa de adesão: %{y:.2%}"
                "<br>Índice vs média: %{customdata[0]:.2f}"
                "<extra></extra>"
            ),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=x_values,
            y=[global_rate] * len(table),
            name="Taxa global",
            mode="lines",
            line={
                "dash": "dash",
            },
            yaxis="y2",
            hovertemplate=(
                "Taxa global: %{y:.2%}"
                "<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        title=f"{feature_name} vs target",
        template="plotly_white",
        height=500,
        xaxis={
            "title": feature_name,
            "tickangle": 45,
        },
        yaxis={
            "title": "Observações",
            "rangemode": "tozero",
        },
        yaxis2={
            "title": "Taxa de adesão",
            "overlaying": "y",
            "side": "right",
            "tickformat": ".1%",
            "rangemode": "tozero",
        },
        legend={
            "orientation": "h",
            "y": 1.12,
        },
        hovermode="x unified",
    )

    return fig


def plot_target_categorical(
    table,
    feature_name,
    global_rate,
):
    """
    Target rate by categorical or binary feature.
    """

    category_column = table.columns[0]

    x_values = (
        table[category_column]
        .astype(str)
    )

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=x_values,
            y=table["target_rate"],
            name="Taxa de adesão",
            text=[
                f"{value:,.0f} obs."
                for value in table[
                    "observations"
                ]
            ],
            textposition="inside",
            customdata=table[
                [
                    "observations",
                    "events",
                    "event_rate_index",
                ]
            ],
            hovertemplate=(
                f"{feature_name}: %{{x}}"
                "<br>Taxa de adesão: %{y:.2%}"
                "<br>Observações: %{customdata[0]:,.0f}"
                "<br>Adesões: %{customdata[1]:,.0f}"
                "<br>Índice vs média: %{customdata[2]:.2f}"
                "<extra></extra>"
            ),
        )
    )

    fig.add_hline(
        y=global_rate,
        line_dash="dash",
        annotation_text=(
            f"Média global: {global_rate:.2%}"
        ),
        annotation_position="top right",
    )

    fig.update_layout(
        title=f"{feature_name} vs target",
        template="plotly_white",
        height=500,
        xaxis_title=feature_name,
        yaxis_title="Taxa de adesão",
        yaxis_tickformat=".1%",
        yaxis_rangemode="tozero",
        showlegend=False,
    )

    return fig
