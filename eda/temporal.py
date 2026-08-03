import pandas as pd


def temporal_analysis(
    df,
    feature,
    target,
    date,
    variable_type,
    freq="M",
    group=None,
):

    cols = [
        feature,
        target,
        date,
    ]

    if group is not None:
        cols.append(group)


    data = df[cols].copy()


    data[date] = pd.to_datetime(
        data[date]
    )


    data["period"] = (
        data[date]
        .dt
        .to_period(freq)
        .astype(str)
    )


    grouping = ["period"]

    if group is not None:
        grouping.append(group)


    # target evolution

    target_summary = (

        data
        .groupby(grouping)

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


    # feature evolution

    feature_grouping = grouping + [target]


    if variable_type == "continuous":

        feature_summary = (

            data
            .groupby(feature_grouping)

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
            .groupby(feature_grouping)

            .agg(

                observations=(
                    feature,
                    "count"
                )

            )

            .reset_index()

        )


    return {

        "target": target_summary,

        "feature": feature_summary,

    }
