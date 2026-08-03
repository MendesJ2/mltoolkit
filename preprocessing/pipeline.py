from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
)


class ModelingPreprocessor:
    """
    Preprocessing pipeline for classification models.

    Transformations
    ---------------
    continuous:
        StandardScaler.

    categorical:
        OneHotEncoder with unknown-category handling.

    binary:
        Passthrough.

    ordinal:
        StandardScaler or passthrough.

    Notes
    -----
    Missing-value treatment must be performed before
    using this pipeline.

    The pipeline must be fitted only on the training data.
    """

    def __init__(
        self,
        *,
        continuous_features=None,
        categorical_features=None,
        binary_features=None,
        ordinal_features=None,
        scale_continuous=True,
        scale_ordinal=True,
        drop_first=True,
        handle_unknown="ignore",
    ):
        self.continuous_features = list(
            continuous_features or []
        )

        self.categorical_features = list(
            categorical_features or []
        )

        self.binary_features = list(
            binary_features or []
        )

        self.ordinal_features = list(
            ordinal_features or []
        )

        self.scale_continuous = (
            scale_continuous
        )

        self.scale_ordinal = scale_ordinal
        self.drop_first = drop_first
        self.handle_unknown = handle_unknown

        self.transformer = None
        self.feature_names_out = None
        self.input_features = None

        self._is_fitted = False

        self._validate_feature_groups()

    # =====================================================
    # Alternative constructor
    # =====================================================

    @classmethod
    def from_dataset(
        cls,
        dataset,
        *,
        selected_features=None,
        scale_continuous=True,
        scale_ordinal=True,
        drop_first=True,
        handle_unknown="ignore",
    ):
        """
        Create preprocessor using Dataset metadata.
        """

        if selected_features is None:

            selected_features = list(
                dataset.feature_columns
            )

        else:

            selected_features = list(
                selected_features
            )

        selected_set = set(
            selected_features
        )

        continuous_features = [
            feature
            for feature in dataset.continuous
            if feature in selected_set
            and dataset.feature(feature).role
            == "feature"
        ]

        categorical_features = [
            feature
            for feature in dataset.categorical
            if feature in selected_set
            and dataset.feature(feature).role
            == "feature"
        ]

        binary_features = [
            feature
            for feature in dataset.binary
            if feature in selected_set
            and dataset.feature(feature).role
            == "feature"
        ]

        ordinal_features = [
            feature
            for feature in dataset.ordinal
            if feature in selected_set
            and dataset.feature(feature).role
            == "feature"
        ]

        identified_features = set(
            continuous_features
            + categorical_features
            + binary_features
            + ordinal_features
        )

        unidentified_features = (
            selected_set
            - identified_features
        )

        if unidentified_features:

            raise ValueError(
                "Selected features without a supported "
                "variable type: "
                f"{sorted(unidentified_features)}"
            )

        return cls(
            continuous_features=(
                continuous_features
            ),
            categorical_features=(
                categorical_features
            ),
            binary_features=binary_features,
            ordinal_features=ordinal_features,
            scale_continuous=scale_continuous,
            scale_ordinal=scale_ordinal,
            drop_first=drop_first,
            handle_unknown=handle_unknown,
        )

    # =====================================================
    # Public API
    # =====================================================

    def fit(
        self,
        X,
        y=None,
    ):
        """
        Fit preprocessing transformations.
        """

        X = self._prepare_input(
            X,
            fitting=True,
        )

        self.transformer = (
            self._build_transformer()
        )

        self.transformer.fit(
            X,
            y,
        )

        self.feature_names_out = (
            self._extract_feature_names()
        )

        self._is_fitted = True

        return self

    def transform(
        self,
        X,
    ):
        """
        Transform data and return a pandas DataFrame.
        """

        self._check_is_fitted()

        X = self._prepare_input(
            X,
            fitting=False,
        )

        transformed = (
            self.transformer.transform(X)
        )

        if hasattr(
            transformed,
            "toarray",
        ):
            transformed = (
                transformed.toarray()
            )

        return pd.DataFrame(
            transformed,
            columns=self.feature_names_out,
            index=X.index,
        ).astype(float)

    def fit_transform(
        self,
        X,
        y=None,
    ):
        """
        Fit and transform training data.
        """

        self.fit(
            X,
            y,
        )

        return self.transform(X)

    def get_feature_names_out(
        self,
    ):
        """
        Return transformed feature names.
        """

        self._check_is_fitted()

        return list(
            self.feature_names_out
        )

    def summary(
        self,
    ):
        """
        Return preprocessing summary.
        """

        summary = {
            "continuous_features": len(
                self.continuous_features
            ),
            "categorical_features": len(
                self.categorical_features
            ),
            "binary_features": len(
                self.binary_features
            ),
            "ordinal_features": len(
                self.ordinal_features
            ),
            "input_features": len(
                self.all_features
            ),
            "fitted": self._is_fitted,
        }

        if self._is_fitted:

            summary[
                "output_features"
            ] = len(
                self.feature_names_out
            )

        return pd.Series(summary)

    # =====================================================
    # Properties
    # =====================================================

    @property
    def all_features(
        self,
    ):
        return (
            self.continuous_features
            + self.categorical_features
            + self.binary_features
            + self.ordinal_features
        )

    # =====================================================
    # Transformer
    # =====================================================

    def _build_transformer(
        self,
    ):
        transformers = []

        if self.continuous_features:

            continuous_transformer = (
                StandardScaler()
                if self.scale_continuous
                else "passthrough"
            )

            transformers.append(
                (
                    "continuous",
                    continuous_transformer,
                    self.continuous_features,
                )
            )

        if self.categorical_features:

            categorical_transformer = (
                self._create_one_hot_encoder()
            )

            transformers.append(
                (
                    "categorical",
                    categorical_transformer,
                    self.categorical_features,
                )
            )

        if self.binary_features:

            transformers.append(
                (
                    "binary",
                    "passthrough",
                    self.binary_features,
                )
            )

        if self.ordinal_features:

            ordinal_transformer = (
                StandardScaler()
                if self.scale_ordinal
                else "passthrough"
            )

            transformers.append(
                (
                    "ordinal",
                    ordinal_transformer,
                    self.ordinal_features,
                )
            )

        if not transformers:
            raise ValueError(
                "No features were supplied to "
                "the preprocessor."
            )

        return ColumnTransformer(
            transformers=transformers,
            remainder="drop",
            verbose_feature_names_out=False,
        )

    def _create_one_hot_encoder(
        self,
    ):
        drop = (
            "first"
            if self.drop_first
            else None
        )

        parameters = {
            "drop": drop,
            "handle_unknown": (
                self.handle_unknown
            ),
            "dtype": float,
        }

        try:

            return OneHotEncoder(
                sparse_output=False,
                **parameters,
            )

        except TypeError:

            # Compatibility with older
            # scikit-learn versions.
            return OneHotEncoder(
                sparse=False,
                **parameters,
            )

    # =====================================================
    # Feature names
    # =====================================================

    def _extract_feature_names(
        self,
    ):
        try:

            names = (
                self.transformer
                .get_feature_names_out()
            )

            return [
                str(name)
                for name in names
            ]

        except AttributeError:

            return (
                self._extract_feature_names_legacy()
            )

    def _extract_feature_names_legacy(
        self,
    ):
        names = []

        names.extend(
            self.continuous_features
        )

        if self.categorical_features:

            encoder = (
                self.transformer
                .named_transformers_[
                    "categorical"
                ]
            )

            try:

                categorical_names = (
                    encoder
                    .get_feature_names_out(
                        self.categorical_features
                    )
                )

            except AttributeError:

                categorical_names = (
                    encoder.get_feature_names(
                        self.categorical_features
                    )
                )

            names.extend(
                categorical_names
            )

        names.extend(
            self.binary_features
        )

        names.extend(
            self.ordinal_features
        )

        return [
            str(name)
            for name in names
        ]

    # =====================================================
    # Input validation
    # =====================================================

    def _prepare_input(
        self,
        X,
        fitting,
    ):
        if not isinstance(
            X,
            pd.DataFrame,
        ):
            raise TypeError(
                "X must be a pandas DataFrame."
            )

        missing_columns = (
            set(self.all_features)
            - set(X.columns)
        )

        if missing_columns:

            raise ValueError(
                "Missing preprocessing columns: "
                f"{sorted(missing_columns)}"
            )

        X = X[
            self.all_features
        ].copy()

        if X.isna().any().any():

            columns_with_missing = (
                X.columns[
                    X.isna().any()
                ].tolist()
            )

            raise ValueError(
                "Missing values remain after project "
                "preprocessing in columns: "
                f"{columns_with_missing}"
            )

        self._validate_numeric_groups(
            X
        )

        if fitting:

            self.input_features = list(
                X.columns
            )

        elif (
            list(X.columns)
            != self.input_features
        ):

            raise ValueError(
                "Input columns differ from the columns "
                "used during fit."
            )

        return X

    def _validate_numeric_groups(
        self,
        X,
    ):
        numeric_features = (
            self.continuous_features
            + self.binary_features
            + self.ordinal_features
        )

        invalid_numeric = [
            feature
            for feature in numeric_features
            if not pd.api.types.is_numeric_dtype(
                X[feature]
            )
        ]

        if invalid_numeric:

            raise TypeError(
                "The following continuous, binary or "
                "ordinal features are not numeric: "
                f"{invalid_numeric}"
            )

        for feature in self.binary_features:

            unique_values = set(
                X[feature]
                .dropna()
                .unique()
            )

            if not unique_values.issubset(
                {0, 1}
            ):

                raise ValueError(
                    f"Binary feature '{feature}' "
                    "must contain only 0 and 1. "
                    f"Found: {sorted(unique_values)}"
                )

    def _validate_feature_groups(
        self,
    ):
        feature_groups = {
            "continuous": (
                self.continuous_features
            ),
            "categorical": (
                self.categorical_features
            ),
            "binary": (
                self.binary_features
            ),
            "ordinal": (
                self.ordinal_features
            ),
        }

        seen = {}

        for group_name, features in (
            feature_groups.items()
        ):

            duplicates = (
                len(features)
                != len(set(features))
            )

            if duplicates:

                raise ValueError(
                    f"Duplicate features inside "
                    f"'{group_name}'."
                )

            for feature in features:

                if feature in seen:

                    raise ValueError(
                        f"Feature '{feature}' appears "
                        f"in both '{seen[feature]}' "
                        f"and '{group_name}'."
                    )

                seen[feature] = group_name

    def _check_is_fitted(
        self,
    ):
        if not self._is_fitted:

            raise RuntimeError(
                "ModelingPreprocessor is not fitted."
            )

    def __repr__(
        self,
    ):
        if not self._is_fitted:

            return (
                "ModelingPreprocessor("
                f"input_features="
                f"{len(self.all_features)}, "
                "status='not_fitted'"
                ")"
            )

        return (
            "ModelingPreprocessor("
            f"input_features="
            f"{len(self.all_features)}, "
            f"output_features="
            f"{len(self.feature_names_out)}, "
            "status='fitted'"
            ")"
        )
