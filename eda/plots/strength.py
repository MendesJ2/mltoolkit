import plotly.graph_objects as go


def _global_table(
    table,
):
    """
    Strength plots display the global analysis by default.
    """

    if "scope" in table.columns:
        table = table[
            table["scope"] == "global"
        ].copy()

    return table


def plot_woe(
    table,
    feature_name,
):
    table = _global_table(table)

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=table["feature_group"],
            y=table["woe"],
            name="WoE",
            customdata=table[
                [
                    "observations",
                    "target_rate",
                    "iv_component",
                ]
            ],
            hovertemplate=(
                "Grupo: %{x}"
                "<br>WoE: %{y:.4f}"
                "<br>Observações: %{customdata[0]:,.0f}"
                "<br>Target rate: %{customdata[1]:.2%}"
                "<br>IV componente: %{customdata[2]:.4f}"
                "<extra></extra>"
            ),
        )
    )

    fig.add_hline(
        y=0,
        line_dash="dash",
    )

    fig.update_layout(
        title=f"WoE — {feature_name}",
        template="plotly_white",
        height=500,
        xaxis={
            "title": feature_name,
            "categoryorder": "array",
            "categoryarray": (
                table["feature_group"]
                .tolist()
            ),
            "tickangle": 45,
        },
        yaxis_title="Weight of Evidence",
    )

    return fig


def plot_lift(
    table,
    feature_name,
):
    table = _global_table(table)

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=table["feature_group"],
            y=table["observations"],
            name="Observações",
            opacity=0.30,
            yaxis="y",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=table["feature_group"],
            y=table["lift"],
            name="Lift",
            mode="lines+markers",
            yaxis="y2",
            customdata=table[
                [
                    "target_rate",
                    "events",
                    "population_pct",
                ]
            ],
            hovertemplate=(
                "Grupo: %{x}"
                "<br>Lift: %{y:.3f}"
                "<br>Target rate: %{customdata[0]:.2%}"
                "<br>Positivos: %{customdata[1]:,.0f}"
                "<br>População: %{customdata[2]:.1%}"
                "<extra></extra>"
            ),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=table["feature_group"],
            y=[1] * len(table),
            name="Lift = 1",
            mode="lines",
            yaxis="y2",
            line={
                "dash": "dash",
            },
        )
    )

    fig.update_layout(
        title=f"Lift univariado — {feature_name}",
        template="plotly_white",
        height=500,
        xaxis={
            "title": feature_name,
            "categoryorder": "array",
            "categoryarray": (
                table["feature_group"]
                .tolist()
            ),
            "tickangle": 45,
        },
        yaxis={
            "title": "Observações",
        },
        yaxis2={
            "title": "Lift",
            "overlaying": "y",
            "side": "right",
            "rangemode": "tozero",
        },
        legend={
            "orientation": "h",
            "y": 1.13,
        },
    )

    return fig


def plot_gain_ks(
    table,
    feature_name,
):
    table = _global_table(table)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=table[
                "cumulative_population_pct"
            ],
            y=table[
                "cumulative_event_pct"
            ],
            mode="lines+markers",
            name="Gain acumulado",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=table[
                "cumulative_population_pct"
            ],
            y=table[
                "cumulative_non_event_pct"
            ],
            mode="lines+markers",
            name="Não eventos acumulados",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Aleatório",
            line={
                "dash": "dash",
            },
        )
    )

    fig.update_layout(
        title=f"Gain e KS — {feature_name}",
        template="plotly_white",
        height=500,
        xaxis_title="População acumulada",
        yaxis_title="Percentagem acumulada",
        xaxis_tickformat=".0%",
        yaxis_tickformat=".0%",
    )

    return fig
