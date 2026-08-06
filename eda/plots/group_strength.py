import pandas as pd
import plotly.express as px


def plot_group_strength_heatmap(
    table,
    metric="iv",
):
    """
    Plot feature strength globally and by group.

    Features are ordered by their global metric,
    from highest to lowest.
    """

    required_columns = {
        "feature",
        "group_value",
        metric,
    }

    missing_columns = (
        required_columns
        - set(table.columns)
    )

    if missing_columns:
        raise ValueError(
            "Missing columns for strength heatmap: "
            f"{sorted(missing_columns)}"
        )

    plot_table = table[
        [
            "feature",
            "group_value",
            metric,
        ]
    ].copy()

    global_order = (
        plot_table[
            plot_table["group_value"]
            == "Global"
        ]
        .sort_values(
            metric,
            ascending=False,
            na_position="last",
        )["feature"]
        .tolist()
    )

    remaining_features = [
        feature
        for feature in (
            plot_table["feature"]
            .drop_duplicates()
            .tolist()
        )
        if feature not in global_order
    ]

    feature_order = (
        global_order
        + remaining_features
    )

    matrix = (
        plot_table
        .pivot_table(
            index="feature",
            columns="group_value",
            values=metric,
            aggfunc="first",
        )
        .reindex(feature_order)
    )

    column_order = [
        column
        for column in [
            "Global"
        ]
        if column in matrix.columns
    ]

    column_order.extend(
        [
            column
            for column in matrix.columns
            if column != "Global"
        ]
    )

    matrix = matrix[
        column_order
    ]

    figure = px.imshow(
        matrix,
        text_auto=".3f",
        aspect="auto",
        labels={
            "x": "Grupo",
            "y": "Feature",
            "color": metric.upper(),
        },
        title=(
            f"{metric.upper()} por grupo"
        ),
    )

    figure.update_layout(
        template="plotly_white",
        height=max(
            550,
            29 * len(matrix),
        ),
        xaxis_title="Grupo",
        yaxis_title="Feature",
    )

    return figure
