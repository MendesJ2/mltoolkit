def preprocess(df):
    """
    Project-specific preprocessing.

    Keep the first version minimal.

    The initial EDA should expose:
        - missing values;
        - unexpected categories;
        - outliers;
        - data-quality problems.

    Add cleaning rules iteratively after
    inspecting the EDA report.
    """

    df = df.copy()

    # =====================================================
    # Examples
    # =====================================================

    # Clean categorical values:
    #
    # df["CATEGORY"] = (
    #     df["CATEGORY"]
    #     .astype("string")
    #     .str.strip()
    # )

    # Filter project population:
    #
    # df = df[
    #     df["ELIGIBLE"] == 1
    # ]

    # Replace values if required:
    #
    # df["VARIABLE"] = (
    #     df["VARIABLE"]
    #     .replace(...)
    # )

    return df
