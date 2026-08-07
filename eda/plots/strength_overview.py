import plotly.express as px


def plot_global_strength(
    table,
    metric="iv",
):
    """
    Horizontal ranking of features by a strength metric.
    """

    required = {
        "feature",
        metric,
    }

    missing = required - set(table.columns)

    if missing:
        raise ValueError(
            f"Missing columns: {sorted(missing)}"
        )

    plot_table = (
        table[
            ["feature", metric]
        ]
        .dropna()
        .sort_values(
            metric,
            ascending=True,
        )
    )

    fig = px.bar(
        plot_table,
        x=metric,
        y="feature",
        orientation="h",
        text_auto=".3f",
        title=f"{metric.upper()} global",
    )

    fig.update_layout(
        template="plotly_white",
        height=max(
            500,
            28 * len(plot_table),
        ),
        xaxis_title=metric.upper(),
        yaxis_title="Feature",
    )

    return fig
