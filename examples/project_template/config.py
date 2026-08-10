from mltoolkit.core.config import Config


config = Config(
    # =====================================================
    # Project
    # =====================================================

    project_name="project_template",

    target="TARGET",

    # =====================================================
    # Structural columns
    # =====================================================

    date_column="REFERENCE_DATE",

    source_column=None, # this column serves to use a variable to split the target variable across some group or segment (for instance: product)

    id_columns=[
        "ID",
    ],

    # =====================================================
    # Features
    # =====================================================

    # None:
    #   features are determined through role inference.
    #
    # Alternatively, provide an explicit whitelist:
    #
    # feature_columns=[
    #     "AGE",
    #     "INCOME",
    #     "SEGMENT",
    # ],
    feature_columns=None,

    ignore_columns=[
        "AUXILIARY_COLUMN",
    ],

    # =====================================================
    # Variable type overrides
    # =====================================================

    variable_types={
        # "AGE": "continuous",
        # "SEGMENT": "categorical",
        # "FLAG": "binary",
        # "RISK_LEVEL": "ordinal",
        # "REFERENCE_DATE": "datetime",
    },

    # =====================================================
    # Role overrides
    # =====================================================

    role_overrides={
        # "SAMPLE": "ignored",
        # "ANOTHER_DATE": "date",
    },

    # =====================================================
    # Special values
    # =====================================================

    # Values such as -999 / -9999 can represent
    # missing values with business meaning.
    #
    # They are kept separate from regular
    # continuous bins during EDA.
    special_values=[
        -999,
        -9999,
    ],

    # =====================================================
    # Modeling defaults
    # =====================================================

    random_state=42,

    test_size=0.20,
)
