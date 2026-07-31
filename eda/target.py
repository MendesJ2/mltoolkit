import pandas as pd

from .binning import create_bins


def target_analysis(
    df,
    feature,
    target,
    variable_type,
    n_bins=10
):

    data = df[[feature, target]].copy()


    if variable_type == "continuous":

        data["_bin"] = create_bins(
            data[feature],
            n_bins=n_bins
        )

        group = "_bin"

    else:

        group = feature


    result = (

        data
        .groupby(group, dropna=False)

        .agg(

            observations=(target, "size"),

            events=(target, "sum"),

            target_rate=(target, "mean")

        )

        .reset_index()

    )


    result["non_events"] = (
        result["observations"]
        - result["events"]
    )


    return result
