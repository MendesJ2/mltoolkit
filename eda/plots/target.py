import plotly.graph_objects as go


def plot_target_analysis(
    table,
    feature_name,
    variable_type,
    group=None,
):
    """
    Plot target rate globally and, optionally, by group.
    """

    if variable_type == "continuous":

        return _plot_continuous_target(
            table=table,
            feature_name=feature_name,
            group=group,
        )

    return _plot_categorical_target(
        table=table,
        feature_name=feature_name,
        group=group,
    )


def _plot_continuous_target(
    table,
    feature_name,
    group,
):
    fig = go.Figure()

    global_table = (
        table[
            table["scope"] == "global"
        ]
        .copy()
    )

    global_rate = (
        global_table["events"].sum()
        / global_table[
            "observations"
        ].sum()
    )

    # Volume global
    fig.add_trace(
        go.Bar(
            x=global_table[
                "feature_group"
            ],
            y=global_table[
                "observations"
            ],
            name="Observações",
            opacity=0.25,
            yaxis="y",
            customdata=global_table[
                [
                    "population_pct",
                    "events",
                ]
            ],
            hovertemplate=(
                "Bin: %{x}"
                "<br>Observações: %{y:,.0f}"
                "<br>População: "
                "%{customdata[0]:.1%}"
                "<br>Positivos: "
                "%{customdata[1]:,.0f}"
                "<extra></extra>"
            ),
        )
    )

    # Target rate global por bin
    fig.add_trace(
        go.Scatter(
            x=global_table[
                "feature_group"
            ],
            y=global_table[
                "target_rate"
            ],
            name="Global",
            mode="lines+markers",
            yaxis="y2",
            line={
                "width": 4,
            },
            customdata=global_table[
                [
                    "observations",
                    "events",
                    "population_pct",
                    "event_rate_index",
                ]
            ],
            hovertemplate=(
                "Bin: %{x}"
                "<br>Target rate: %{y:.2%}"
                "<br>Observações: "
                "%{customdata[0]:,.0f}"
                "<br>Positivos: "
                "%{customdata[1]:,.0f}"
                "<br>População: "
                "%{customdata[2]:.1%}"
                "<br>Lift: "
                "%{customdata[3]:.2f}"
                "<extra></extra>"
            ),
        )
    )

    # Target rate por grupo/TAREFA
    if group is not None:

        group_table = table[
            table["scope"] == "group"
        ]

        for group_value, data in (
            group_table.groupby(
                "group_value",
                sort=False,
            )
        ):

            fig.add_trace(
                go.Scatter(
                    x=data[
                        "feature_group"
                    ],
                    y=data[
                        "target_rate"
                    ],
                    name=str(group_value),
                    mode="lines+markers",
                    yaxis="y2",
                    line={
                        "width": 2,
                    },
                    customdata=data[
                        [
                            "observations",
                            "events",
                            "population_pct",
                            "event_rate_index",
                        ]
                    ],
                    hovertemplate=(
                        f"{group}: {group_value}"
                        "<br>Bin: %{x}"
                        "<br>Target rate: "
                        "%{y:.2%}"
                        "<br>Observações: "
                        "%{customdata[0]:,.0f}"
                        "<br>Positivos: "
                        "%{customdata[1]:,.0f}"
                        "<br>População do grupo: "
                        "%{customdata[2]:.1%}"
                        "<br>Lift: "
                        "%{customdata[3]:.2f}"
                        "<extra></extra>"
                    ),
                )
            )

    # Média global horizontal
    fig.add_trace(
        go.Scatter(
            x=global_table[
                "feature_group"
            ],
            y=[
                global_rate
            ] * len(global_table),
            name=(
                f"Média global "
                f"({global_rate:.2%})"
            ),
            mode="lines",
            yaxis="y2",
            line={
                "dash": "dash",
                "width": 2,
            },
            hovertemplate=(
                "Média global: %{y:.2%}"
                "<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        title=f"{feature_name} vs target",
        template="plotly_white",
        height=550,
        hovermode="x unified",
        xaxis={
            "title": feature_name,
            "categoryorder": "array",
            "categoryarray": (
                global_table[
                    "feature_group"
                ].tolist()
            ),
            "tickangle": 45,
        },
        yaxis={
            "title": "Observações",
            "rangemode": "tozero",
        },
        yaxis2={
            "title": "Target rate",
            "overlaying": "y",
            "side": "right",
            "tickformat": ".1%",
            "rangemode": "tozero",
        },
        legend={
            "orientation": "h",
            "y": 1.15,
        },
    )

    return fig


def _plot_categorical_target(
    table,
    feature_name,
    group,
):
    fig = go.Figure()

    global_table = (
        table[
            table["scope"] == "global"
        ]
        .copy()
    )

    global_rate = (
        global_table["events"].sum()
        / global_table[
            "observations"
        ].sum()
    )

    tables = [
        (
            "Global",
            global_table,
        )
    ]

    if group is not None:

        for group_value, data in (
            table[
                table["scope"] == "group"
            ]
            .groupby(
                "group_value",
                sort=False,
            )
        ):

            tables.append(
                (
                    str(group_value),
                    data,
                )
            )

    # Barras global + TAREFAS
    for label, data in tables:

        fig.add_trace(
            go.Bar(
                x=data[
                    "feature_group"
                ],
                y=data[
                    "target_rate"
                ],
                name=label,
                customdata=data[
                    [
                        "observations",
                        "events",
                        "population_pct",
                        "event_rate_index",
                    ]
                ],
                hovertemplate=(
                    f"{label}"
                    "<br>Categoria: %{x}"
                    "<br>Target rate: %{y:.2%}"
                    "<br>Observações: "
                    "%{customdata[0]:,.0f}"
                    "<br>Positivos: "
                    "%{customdata[1]:,.0f}"
                    "<br>População: "
                    "%{customdata[2]:.1%}"
                    "<br>Lift: "
                    "%{customdata[3]:.2f}"
                    "<extra></extra>"
                ),
            )
        )

    # Média global horizontal
    fig.add_hline(
        y=global_rate,
        line_dash="dash",
        line_width=2,
        annotation_text=(
            f"Média global: "
            f"{global_rate:.2%}"
        ),
        annotation_position="top right",
    )

    fig.update_layout(
        title=f"{feature_name} vs target",
        template="plotly_white",
        height=550,
        barmode="group",
        xaxis={
            "title": feature_name,
            "categoryorder": "array",
            "categoryarray": (
                global_table[
                    "feature_group"
                ].tolist()
            ),
            "tickangle": 45,
        },
        yaxis={
            "title": "Target rate",
            "tickformat": ".1%",
            "rangemode": "tozero",
        },
        legend_title=(
            group
            if group is not None
            else None
        ),
    )

    return fig
