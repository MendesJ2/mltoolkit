import pandas as pd


def temporal_analysis(
    df,
    feature,
    target,
    date,
    variable_type,
    freq="M",
):
    """
    Analyse feature evolution through time.
    """


    data = df[
        [
            feature,
            target,
            date,
        ]
    ].copy()


    data[date] = pd.to_datetime(
        data[date]
    )


    data["period"] = (
        data[date]
        .dt
        .to_period(freq)
        .astype(str)
    )


    # volume e target por período

    target_summary = (

        data
        .groupby("period")

        .agg(

            observations=(
                target,
                "count"
            ),

            target_rate=(
                target,
                "mean"
            )

        )

        .reset_index()

    )


    if variable_type == "continuous":

        feature_summary = (

            data
            .groupby(
                [
                    "period",
                    target
                ]
            )

            .agg(

                mean_feature=(
                    feature,
                    "mean"
                ),

                observations=(
                    feature,
                    "count"
                )

            )

            .reset_index()

        )


    else:

        feature_summary = (

            data
            .groupby(
                [
                    "period",
                    target
                ]
            )[feature]

            .count()

            .reset_index(
                name="observations"
            )

        )


    return {

        "target": target_summary,

        "feature": feature_summary,

    }
