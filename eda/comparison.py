def compare_source(
    df,
    source,
    target
):

    return (

        df
        .groupby(source)[target]
        .agg(
            [
                "count",
                "mean"
            ]
        )

        .rename(
            columns={
                "mean":"target_rate"
            }
        )

        .reset_index()

    )
