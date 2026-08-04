import plotly.graph_objects as go


def plot_target_temporal(
    table,
    feature_name=None,
    group=None,
):
    """
    Target-rate evolution over time.

    When group is provided, creates one line per group value.
    """

    fig = go.Figure()

    if group is None:

        fig.add_trace(
            go.Scatter(
                x=table["period"],
                y=table["target_rate"],
                mode="lines+markers",
                name="Target rate",
                customdata=table[
                    ["observations"]
                ],
                hovertemplate=(
                    "Período: %{x}"
                    "<br>Taxa de adesão: %{y:.2%}"
                    "<br>Observações: %{customdata[0]:,.0f}"
                    "<extra></extra>"
                ),
            )
        )

    else:

        if group not in table.columns:
            raise ValueError(
                f"Group column '{group}' "
                "not found in target table."
            )

        for group_value, group_data in (
            table.groupby(
                group,
                dropna=False,
                sort=False,
            )
        ):

            group_data = group_data.sort_values(
                "period"
            )

            fig.add_trace(
                go.Scatter(
                    x=group_data["period"],
                    y=group_data["target_rate"],
                    mode="lines+markers",
                    name=str(group_value),
                    customdata=group_data[
                        ["observations"]
                    ],
                    hovertemplate=(
                        f"{group}: {group_value}"
                        "<br>Período: %{x}"
                        "<br>Taxa de adesão: %{y:.2%}"
                        "<br>Observações: %{customdata[0]:,.0f}"
                        "<extra></extra>"
                    ),
                )
            )

    title = "Evolução temporal da taxa de adesão"

    if feature_name is not None:
        title = (
            f"{feature_name} — "
            "evolução temporal da taxa de adesão"
        )

    fig.update_layout(
        title=title,
        template="plotly_white",
        height=500,
        xaxis_title="Período",
        yaxis_title="Taxa de adesão",
        yaxis_tickformat=".1%",
        yaxis_rangemode="tozero",
        legend_title=group,
        hovermode="x unified",
    )

    return fig


def plot_feature_temporal(
    table,
    feature_name,
    target_name,
    group=None,
):
    """
    Evolution of the mean continuous feature by target.

    When group is provided, creates one line for each
    combination of group and target.
    """

    required_columns = {
        "period",
        "target",
        "mean_feature",
    }

    missing_columns = (
        required_columns
        - set(table.columns)
    )

    if missing_columns:
        raise ValueError(
            "Feature temporal plot is only available "
            "for continuous variables. Missing columns: "
            f"{sorted(missing_columns)}"
        )

    fig = go.Figure()

    if group is None:

        for target_value, target_data in (
            table.groupby(
                "target",
                dropna=False,
                sort=False,
            )
        ):

            target_data = target_data.sort_values(
                "period"
            )

            fig.add_trace(
                go.Scatter(
                    x=target_data["period"],
                    y=target_data["mean_feature"],
                    mode="lines+markers",
                    name=f"Target={target_value}",
                    customdata=target_data[
                        ["observations"]
                    ],
                    hovertemplate=(
                        f"Target: {target_value}"
                        "<br>Período: %{x}"
                        f"<br>Média {feature_name}: "
                        "%{y:,.2f}"
                        "<br>Observações: "
                        "%{customdata[0]:,.0f}"
                        "<extra></extra>"
                    ),
                )
            )

    else:

        if group not in table.columns:
            raise ValueError(
                f"Group column '{group}' "
                "not found in feature table."
            )

        grouping_columns = [
            group,
            "target",
        ]

        for keys, line_data in (
            table.groupby(
                grouping_columns,
                dropna=False,
                sort=False,
            )
        ):

            group_value, target_value = keys

            line_data = line_data.sort_values(
                "period"
            )

            fig.add_trace(
                go.Scatter(
                    x=line_data["period"],
                    y=line_data["mean_feature"],
                    mode="lines+markers",
                    name=(
                        f"{group_value} | "
                        f"Target={target_value}"
                    ),
                    customdata=line_data[
                        ["observations"]
                    ],
                    hovertemplate=(
                        f"{group}: {group_value}"
                        f"<br>Target: {target_value}"
                        "<br>Período: %{x}"
                        f"<br>Média {feature_name}: "
                        "%{y:,.2f}"
                        "<br>Observações: "
                        "%{customdata[0]:,.0f}"
                        "<extra></extra>"
                    ),
                )
            )

    fig.update_layout(
        title=(
            f"Evolução temporal de {feature_name} "
            "por target"
        ),
        template="plotly_white",
        height=500,
        xaxis_title="Período",
        yaxis_title=f"Média de {feature_name}",
        legend_title=(
            group
            if group is not None
            else "Target"
        ),
        hovermode="x unified",
    )

    return fig


def plot_volume_temporal(
    table,
    group=None,
):
    """
    Observation volume over time.

    When group is provided, creates stacked bars by group.
    """

    fig = go.Figure()

    if group is None:

        fig.add_trace(
            go.Bar(
                x=table["period"],
                y=table["observations"],
                name="Observações",
                hovertemplate=(
                    "Período: %{x}"
                    "<br>Observações: %{y:,.0f}"
                    "<extra></extra>"
                ),
            )
        )

    else:

        if group not in table.columns:
            raise ValueError(
                f"Group column '{group}' "
                "not found in volume table."
            )

        for group_value, group_data in (
            table.groupby(
                group,
                dropna=False,
                sort=False,
            )
        ):

            group_data = group_data.sort_values(
                "period"
            )

            fig.add_trace(
                go.Bar(
                    x=group_data["period"],
                    y=group_data["observations"],
                    name=str(group_value),
                    hovertemplate=(
                        f"{group}: {group_value}"
                        "<br>Período: %{x}"
                        "<br>Observações: %{y:,.0f}"
                        "<extra></extra>"
                    ),
                )
            )

    fig.update_layout(
        title="Evolução temporal do volume",
        template="plotly_white",
        height=450,
        xaxis_title="Período",
        yaxis_title="Observações",
        legend_title=group,
        barmode=(
            "stack"
            if group is not None
            else "group"
        ),
        hovermode="x unified",
    )

    return fig
