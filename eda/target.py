import pandas as pd

from .binning import create_bins



def target_analysis(
    df: pd.DataFrame,
    feature: str,
    target: str,
    variable_type: str,
    n_bins: int = 10,
):
    """
    Analyse feature vs binary target.

    Returns:
        DataFrame with volume and target rate.
    """


    data = df[
        [
            feature,
            target
        ]
    ].copy()


    if variable_type == "continuous":

        data["_bin"] = create_bins(
            data[feature],
            n_bins=n_bins
        )

        group_column = "_bin"

    else:

        group_column = feature



    result = (

        data
        .groupby(
            group_column,
            dropna=False
        )

        .agg(

            observations=(
                target,
                "count"
            ),

            events=(
                target,
                "sum"
            ),

            target_rate=(
                target,
                "mean"
            )

        )

        .reset_index()

    )


    result["non_events"] = (
        result["observations"]
        -
        result["events"]
    )


    result["event_rate_index"] = (
        result["target_rate"]
        /
        df[target].mean()
    )


    return result
