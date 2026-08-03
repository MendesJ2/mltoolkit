import pandas as pd


def compare_feature(
    df,
    feature,
    group,
    variable_type,
    n_bins=10,
):

    data = df[
        [
            feature,
            group,
        ]
    ].copy()


    if variable_type == "continuous":

        summary = (

            data
            .groupby(group)

            .agg(

                observations=(
                    feature,
                    "count"
                ),

                mean=(
                    feature,
                    "mean"
                ),

                median=(
                    feature,
                    "median"
                ),

                std=(
                    feature,
                    "std"
                ),

                min=(
                    feature,
                    "min"
                ),

                max=(
                    feature,
                    "max"
                ),

            )

            .reset_index()

        )

    else:

        summary = (

            data
            .groupby(
                [
                    group,
                    feature,
                ],
                dropna=False,
            )

            .size()

            .reset_index(
                name="observations"
            )

        )

        summary["percentage"] = (

            summary["observations"]

            /

            summary.groupby(group)[
                "observations"
            ].transform("sum")

        )


    return summary
