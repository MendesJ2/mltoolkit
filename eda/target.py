import pandas as pd


def target_summary(
    df,
    feature,
    target
):

    result = (

        df
        .groupby(feature)[target]
        .agg(
            [
                "count",
                "mean"
            ]
        )

        .reset_index()

    )


    result.rename(
        columns={
            "mean": "target_rate"
        },
        inplace=True
    )


    return result
