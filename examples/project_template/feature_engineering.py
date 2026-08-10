import pandas as pd
import numpy as np

def create_features(df):
    """
    Project-specific feature engineering.

    Keep feature creation explicit and simple.

    New features can be added after each
    EDA iteration.
    """

    df = df.copy()

    # =====================================================
    # Examples
    # =====================================================

    # Ratio:
    #
    # df["RATIO"] = (
    #     df["VALUE_A"]
    #     / df["VALUE_B"]
    # )

    # Difference:
    #
    # df["VALUE_DIFFERENCE"] = (
    #     df["VALUE_A"]
    #     - df["VALUE_B"]
    # )

    # Boolean feature:
    #
    # df["HIGH_VALUE"] = (
    #     df["VALUE_A"] > 10000
    # ).astype(int)

    # np.select can also be used for
    # project-specific business rules.
    #
    # Example:
    #
    # df["PRODUCT_GROUP"] = np.select(
    #     [
    #         df["PRODUCT"].isin(["A", "B"]),
    #         df["PRODUCT"].eq("C"),
    #     ],
    #     [
    #         "GROUP_1",
    #         "GROUP_2",
    #     ],
    #     default="OTHER",
    # )

    # Filtering after feature creation is allowed:
    #
    # df = df[
    #     df["PRODUCT_GROUP"] != "EXCLUDE"
    # ]

    return df
