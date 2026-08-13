import plotly.graph_objects as go


def plot_target_temporal(
    table,
    feature_name=None,
    group=None,
):
    fig = go.Figure()

    grouping = (
        group
        if group is not None
        else None
    )

    if grouping is None:

        groups = [
            (
                "Global",
                table,
            )
        ]

    else:

        groups = list(
            table.groupby(
                grouping,
                dropna=False,
                sort=False,
            )
        )

    for label, data in groups:

        data = data.sort_values(
            "period"
        )

        fig.add_trace(
            go.Scatter(
                x=data["period"],
                y=data["target_rate"],
                mode="lines+markers",
                name=str(label),
                customdata=data[
                    ["observations"]
                ],
                hovertemplate=(
                    "Período: %{x}"
                    "<br>Target rate: %{y:.2%}"
                    "<br>Observações: %{customdata[0]:,.0f}"
                    "<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title="Target rate temporal",
        template="plotly_white",
        height=500,
        xaxis_title="Período",
        yaxis_title="Target rate",
        yaxis_tickformat=".1%",
        hovermode="x unified",
        legend_title=group,
    )

    return fig


def plot_continuous_feature_temporal(
    table,
    feature_name,
    target_name,
    group=None,
    statistic="mean",
):
    """
    Continuous temporal evolution by target.

    statistic:
        mean
        median
    """

    valid_statistics = {
        "mean",
        "median",
    }

    if statistic not in valid_statistics:
        raise ValueError(
            "statistic must be 'mean' or 'median'."
        )

    value_column = (
        f"{statistic}_feature"
    )

    fig = go.Figure()

    grouping_columns = [
        target_name
    ]

    if group is not None:
        grouping_columns.insert(
            0,
            group,
        )

    for keys, data in table.groupby(
        grouping_columns,
        dropna=False,
        sort=False,
    ):

        if not isinstance(
            keys,
            tuple,
        ):
            keys = (
                keys,
            )

        if group is None:

            target_value = keys[0]

            label = (
                f"Target={target_value}"
            )

        else:

            group_value, target_value = keys

            label = (
                f"{group_value} | "
                f"Target={target_value}"
            )

        data = data.sort_values(
            "period"
        )

        fig.add_trace(
            go.Scatter(
                x=data["period"],
                y=data[value_column],
                mode="lines+markers",
                name=label,
                customdata=data[
                    ["observations"]
                ],
                hovertemplate=(
                    "Período: %{x}"
                    f"<br>{statistic.capitalize()}: "
                    "%{y:,.2f}"
                    "<br>Observações: %{customdata[0]:,.0f}"
                    "<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title=(
            f"{feature_name} temporal "
            f"({statistic}) por target"
        ),
        template="plotly_white",
        height=550,
        xaxis_title="Período",
        yaxis_title=(
            f"{statistic.capitalize()} "
            f"de {feature_name}"
        ),
        hovermode="x unified",
    )

    return fig


def plot_categorical_feature_temporal(
    table,
    feature_name,
    target_name,
    group=None,
):
    """
    Category-share evolution over time by target.

    Each line represents category x target and,
    optionally, group.
    """

    fig = go.Figure()

    grouping_columns = [
        "category",
        target_name,
    ]

    if group is not None:
        grouping_columns.insert(
            0,
            group,
        )

    for keys, data in table.groupby(
        grouping_columns,
        dropna=False,
        sort=False,
    ):

        if not isinstance(
            keys,
            tuple,
        ):
            keys = (
                keys,
            )

        if group is None:

            category, target_value = keys

            label = (
                f"{category} | "
                f"Target={target_value}"
            )

        else:

            (
                group_value,
                category,
                target_value,
            ) = keys

            label = (
                f"{group_value} | "
                f"{category} | "
                f"Target={target_value}"
            )

        data = data.sort_values(
            "period"
        )

        fig.add_trace(
            go.Scatter(
                x=data["period"],
                y=data["category_share"],
                mode="lines+markers",
                name=label,
                customdata=data[
                    ["observations"]
                ],
                hovertemplate=(
                    "Período: %{x}"
                    "<br>Share: %{y:.2%}"
                    "<br>Observações: %{customdata[0]:,.0f}"
                    "<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title=(
            f"Distribuição temporal de "
            f"{feature_name} por target"
        ),
        template="plotly_white",
        height=600,
        xaxis_title="Período",
        yaxis_title="Share da categoria",
        yaxis_tickformat=".1%",
        hovermode="x unified",
    )

    return fig


def plot_volume_temporal(
    table,
    group=None,
):
    fig = go.Figure()

    if group is None:

        groups = [
            (
                "Observações",
                table,
            )
        ]

    else:

        groups = list(
            table.groupby(
                group,
                dropna=False,
                sort=False,
            )
        )

    for label, data in groups:

        data = data.sort_values(
            "period"
        )

        fig.add_trace(
            go.Bar(
                x=data["period"],
                y=data["observations"],
                name=str(label),
            )
        )

    fig.update_layout(
        title="Volume temporal",
        template="plotly_white",
        height=450,
        xaxis_title="Período",
        yaxis_title="Observações",
        barmode=(
            "stack"
            if group is not None
            else "group"
        ),
        legend_title=group,
    )

    return fig

def plot_binary_feature_temporal(
    table,
    feature_name,
    target_name,
    group=None,
):
    """
    Temporal evolution of the binary rate
    by target and optionally group.

    binary_rate represents the proportion
    of observations where feature == 1.
    """

    fig = go.Figure()

    grouping_columns = [
        target_name
    ]

    if group is not None:
        grouping_columns.insert(
            0,
            group,
        )

    for keys, data in table.groupby(
        grouping_columns,
        dropna=False,
        sort=False,
    ):

        if not isinstance(
            keys,
            tuple,
        ):
            keys = (
                keys,
            )

        if group is None:

            target_value = keys[0]

            label = (
                f"Target={target_value}"
            )

        else:

            (
                group_value,
                target_value,
            ) = keys

            label = (
                f"{group_value} | "
                f"Target={target_value}"
            )

        data = data.sort_values(
            "period"
        )

        fig.add_trace(
            go.Scatter(
                x=data["period"],
                y=data["binary_rate"],
                mode="lines+markers",
                name=label,
                customdata=data[
                    ["observations"]
                ],
                hovertemplate=(
                    "Período: %{x}"
                    "<br>Rate: %{y:.2%}"
                    "<br>Observações: "
                    "%{customdata[0]:,.0f}"
                    "<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title=(
            f"{feature_name} temporal "
            "por target"
        ),
        template="plotly_white",
        height=550,
        xaxis_title="Período",
        yaxis_title=(
            f"% {feature_name}=1"
        ),
        yaxis_tickformat=".1%",
        hovermode="x unified",
    )

    return fig
