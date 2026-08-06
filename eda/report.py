from __future__ import annotations

import html
import re
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.io as pio


class EDAReport:
    """
    Export a complete EDA report.

    Output
    ------
    index.html:
        Searchable overview with links to feature pages.

    eda_summary.xlsx:
        Global tables for filtering and prioritisation.

    features/<feature>.html:
        One consolidated page per feature.

    relationships/*.html:
        Pearson, Spearman and Cramér's V matrices.
    """

    def __init__(
        self,
        eda,
        output_folder,
        *,
        date_column=None,
        temporal_freq="Q",
        source_column=None,
        stability_by=None,
        stability_reference=None,
        n_bins=10,
        include_ignored=False,
    ):
        self.eda = eda
        self.dataset = eda.dataset

        self.output_folder = Path(
            output_folder
        )

        self.features_folder = (
            self.output_folder
            / "features"
        )

        self.relationships_folder = (
            self.output_folder
            / "relationships"
        )

        self.date_column = date_column
        self.temporal_freq = temporal_freq
        self.source_column = source_column

        self.stability_by = stability_by
        self.stability_reference = (
            stability_reference
        )

        self.n_bins = n_bins
        self.include_ignored = (
            include_ignored
        )

        self.overview = None
        self.quality = None
        self.strength = None
        self.stability = None
        self.group_strength = None

        self.errors = []

    # =====================================================
    # Public API
    # =====================================================

    def export(self):
        """
        Generate the complete EDA report.
        """

        self._create_folders()

        features = self._get_features()

        self.quality = (
            self.eda.quality_report(
                only_features=(
                    not self.include_ignored
                )
            )
        )

        strength_records = []
        stability_records = []
        overview_records = []

        for position, feature_name in enumerate(
            features
        ):

            previous_feature = (
                features[position - 1]
                if position > 0
                else None
            )

            next_feature = (
                features[position + 1]
                if position < len(features) - 1
                else None
            )

            result = self._export_feature(
                feature_name=feature_name,
                previous_feature=previous_feature,
                next_feature=next_feature,
            )

            overview_records.append(
                result["overview"]
            )

            if result["strength"] is not None:
                strength_records.append(
                    result["strength"]
                )

            if result["stability"] is not None:
                stability_records.extend(
                    result["stability"]
                )

        self.overview = pd.DataFrame(
            overview_records
        )

        self.strength = pd.DataFrame(
            strength_records
        )

        self.stability = pd.DataFrame(
            stability_records
        )

        relationship_tables = (
            self._export_relationships()
        )
        
        self.group_strength = (
            self._export_group_strength()
        )
        
        self._export_excel(
            relationship_tables
        )
        
        self._export_index()
        
        return self.output_folder

    # =====================================================
    # Feature report
    # =====================================================

    def _export_feature(
        self,
        feature_name,
        previous_feature=None,
        next_feature=None,
    ):
        feature = self.eda.feature(
            feature_name
        )

        metadata = (
            self.dataset.feature(
                feature_name
            )
        )

        filename = self._feature_filename(
            feature_name
        )

        sections = []

        # -----------------------------------------------
        # Metadata
        # -----------------------------------------------

        metadata_table = pd.DataFrame(
            {
                "Metric": [
                    "Feature",
                    "Role",
                    "Variable type",
                    "Dtype",
                    "Unique values",
                    "Missing percentage",
                    "Constant",
                    "Quasi-constant",
                ],
                "Value": [
                    feature_name,
                    metadata.role,
                    metadata.variable_type,
                    metadata.dtype,
                    metadata.n_unique,
                    metadata.missing_pct,
                    metadata.is_constant,
                    metadata.is_quasi_constant,
                ],
            }
        )

        sections.append(
            self._table_section(
                "Metadata",
                metadata_table,
            )
        )

        # -----------------------------------------------
        # Summary and statistics
        # -----------------------------------------------

        sections.append(
            self._safe_table_section(
                title="Summary",
                function=feature.summary,
                feature_name=feature_name,
            )
        )

        sections.append(
            self._safe_table_section(
                title="Statistics",
                function=feature.statistics,
                feature_name=feature_name,
            )
        )

        sections.append(
            self._safe_table_section(
                title="Quality",
                function=lambda: (
                    feature.quality().table
                ),
                feature_name=feature_name,
            )
        )

        # -----------------------------------------------
        # Univariate plot
        # -----------------------------------------------

        sections.append(
            self._safe_plot_section(
                title="Distribution",
                function=feature.plot,
                feature_name=feature_name,
            )
        )

        # -----------------------------------------------
        # Target
        # -----------------------------------------------

        target_analysis = None

        try:
            target_analysis = feature.target(
                n_bins=self.n_bins,
                group=self.source_column,
            )

            target_display_table = (
                target_analysis.table[
                    [
                        "group_value",
                        "feature_group",
                        "observations",
                        "target_rate",
                        "population_pct",
                    ]
                ]
                .rename(
                    columns={
                        "group_value": (
                            self.source_column
                            or "Population"
                        )
                    }
                )
                .reset_index(drop=True)
            )
            
            sections.append(
                self._table_section(
                    "Feature vs target — table",
                    target_display_table,
                )
            )

            sections.append(
                self._plot_section(
                    "Feature vs target",
                    target_analysis.plot(),
                )
            )

        except Exception as error:
            sections.append(
                self._error_section(
                    "Feature vs target",
                    error,
                )
            )

            self._register_error(
                feature_name,
                "target",
                error,
            )

        # -----------------------------------------------
        # Comparison by source
        # -----------------------------------------------

        if (
            self.source_column is not None
            and self.source_column
            in self.dataset.df.columns
            and feature_name
            != self.source_column
        ):

            try:
                comparison = feature.compare(
                    by=self.source_column
                )

                sections.append(
                    self._table_section(
                        (
                            "Comparison by "
                            f"{self.source_column}"
                        ),
                        comparison.table,
                    )
                )

                sections.append(
                    self._plot_section(
                        (
                            "Distribution by "
                            f"{self.source_column}"
                        ),
                        comparison.plot(),
                    )
                )

                comparison_test = (
                    comparison.test()
                )

                sections.append(
                    self._table_section(
                        "Comparison statistical test",
                        comparison_test,
                    )
                )

            except Exception as error:
                sections.append(
                    self._error_section(
                        "Comparison by source",
                        error,
                    )
                )

                self._register_error(
                    feature_name,
                    "comparison",
                    error,
                )

        # -----------------------------------------------
        # Temporal
        # -----------------------------------------------

        if (
            self.date_column is not None
            and self.date_column
            in self.dataset.df.columns
            and feature_name
            != self.date_column
        ):

            try:
                temporal = feature.temporal(
                    date=self.date_column,
                    freq=self.temporal_freq,
                    group=self.source_column,
                )

                sections.append(
                    self._plot_section(
                        "Temporal target rate",
                        temporal.plot_target(),
                    )
                )

                sections.append(
                    self._plot_section(
                        "Temporal volume",
                        temporal.plot_volume(),
                    )
                )

                sections.append(
                    self._plot_section(
                        (
                            "Temporal feature "
                            "evolution by target"
                        ),
                        temporal.plot_feature(),
                    )
                )
                if (
                    metadata.variable_type
                    == "continuous"
                ):
                    sections.append(
                        self._plot_section(
                            (
                                "Temporal feature "
                                "median by target"
                            ),
                            temporal.plot_feature(
                                statistic="median"
                            ),
                        )
                    )

            except Exception as error:
                sections.append(
                    self._error_section(
                        "Temporal analysis",
                        error,
                    )
                )

                self._register_error(
                    feature_name,
                    "temporal",
                    error,
                )

        # -----------------------------------------------
        # Strength
        # -----------------------------------------------

        strength_record = None

        try:
            strength = feature.strength(
                n_bins=self.n_bins,
                group=self.source_column,
            )

            strength_record = (
                strength.metrics.to_dict()
            )

            strength_record[
                "feature"
            ] = feature_name

            sections.append(
                self._table_section(
                    "Feature strength",
                    strength.group_metrics[
                        [
                            "group_value",
                            "iv",
                            "max_ks",
                            "max_lift",
                            "observations",
                            "events",
                        ]
                    ],
                )
            )

            strength_display_table = (
                strength.table[
                    [
                        "group_value",
                        "feature_group",
                        "observations",
                        "target_rate",
                        "woe",
                        "iv_component",
                        "lift",
                    ]
                ]
                .rename(
                    columns={
                        "group_value": (
                            self.source_column
                            or "Population"
                        )
                    }
                )
            )
            
            sections.append(
                self._table_section(
                    "WoE / IV table",
                    strength_display_table,
                )
            )

            sections.append(
                self._plot_section(
                    "Weight of Evidence",
                    strength.plot_woe(),
                )
            )

            sections.append(
                self._plot_section(
                    "Univariate lift",
                    strength.plot_lift(),
                )
            )

            sections.append(
                self._plot_section(
                    "Gain and KS",
                    strength.plot_gain(),
                )
            )

        except Exception as error:
            sections.append(
                self._error_section(
                    "Feature strength",
                    error,
                )
            )

            self._register_error(
                feature_name,
                "strength",
                error,
            )

        # -----------------------------------------------
        # Stability
        # -----------------------------------------------

        stability_records = None
        max_psi = np.nan

        if (
            self.stability_by is not None
            and self.stability_by
            in self.dataset.df.columns
            and feature_name
            != self.stability_by
        ):

            try:
                stability = feature.stability(
                    by=self.stability_by,
                    reference=(
                        self.stability_reference
                    ),
                    n_bins=self.n_bins,
                )

                stability_records = (
                    stability.summary
                    .assign(
                        feature=feature_name
                    )
                    .to_dict(
                        orient="records"
                    )
                )

                if not stability.summary.empty:
                    max_psi = (
                        stability.summary[
                            "psi"
                        ].max()
                    )

                sections.append(
                    self._table_section(
                        "Stability summary",
                        stability.summary,
                    )
                )

                sections.append(
                    self._plot_section(
                        "PSI",
                        stability.plot_psi(),
                    )
                )

                sections.append(
                    self._plot_section(
                        "Distribution stability",
                        (
                            stability
                            .plot_distribution()
                        ),
                    )
                )

            except Exception as error:
                sections.append(
                    self._error_section(
                        "Stability",
                        error,
                    )
                )

                self._register_error(
                    feature_name,
                    "stability",
                    error,
                )

        # -----------------------------------------------
        # Final HTML
        # -----------------------------------------------

        page_html = self._feature_page(
            feature_name=feature_name,
            sections=sections,
            previous_feature=previous_feature,
            next_feature=next_feature,
        )

        output_path = (
            self.features_folder
            / filename
        )

        output_path.write_text(
            page_html,
            encoding="utf-8",
        )

        quality_row = self._quality_row(
            feature_name
        )

        overview = {
            "feature": feature_name,
            "role": metadata.role,
            "variable_type": (
                metadata.variable_type
            ),
            "missing_pct": (
                quality_row.get(
                    "missing_pct",
                    metadata.missing_pct,
                )
            ),
            "n_unique": (
                quality_row.get(
                    "unique_values",
                    metadata.n_unique,
                )
            ),
            "mode_pct": quality_row.get(
                "mode_pct",
                np.nan,
            ),
            "outliers_pct": (
                quality_row.get(
                    "outliers_pct",
                    np.nan,
                )
            ),
            "iv": (
                strength_record.get(
                    "iv",
                    np.nan,
                )
                if strength_record
                is not None
                else np.nan
            ),
            "ks": (
                strength_record.get(
                    "max_ks",
                    np.nan,
                )
                if strength_record
                is not None
                else np.nan
            ),
            "max_psi": max_psi,
            "is_constant": (
                metadata.is_constant
            ),
            "is_quasi_constant": (
                metadata.is_quasi_constant
            ),
            "page": (
                f"features/{filename}"
            ),
        }

        overview["status"] = (
            self._feature_status(
                overview
            )
        )

        return {
            "overview": overview,
            "strength": strength_record,
            "stability": stability_records,
        }

    # =====================================================
    # Relationships
    # =====================================================

    def _export_relationships(
        self,
    ):
        tables = {}

        relationship_methods = [
            (
                "pearson",
                lambda: self.eda.correlation(
                    method="pearson"
                ),
            ),
            (
                "spearman",
                lambda: self.eda.correlation(
                    method="spearman"
                ),
            ),
            (
                "cramers_v",
                (
                    self.eda
                    .categorical_relationships
                ),
            ),
        ]

        for name, function in (
            relationship_methods
        ):

            try:
                analysis = function()

                tables[name] = (
                    analysis.table
                )

                output_path = (
                    self.relationships_folder
                    / f"{name}.html"
                )

                page = self._simple_page(
                    title=name,
                    body=self._plot_html(
                        analysis.plot()
                    ),
                    back_link="../index.html",
                )

                output_path.write_text(
                    page,
                    encoding="utf-8",
                )

            except Exception as error:
                self._register_error(
                    "__global__",
                    f"relationship_{name}",
                    error,
                )

        return tables
    # =====================================================
    # Group Strength
    # =====================================================
    
    def _export_group_strength(
        self,
    ):
        """
        Export global feature-strength heatmaps by source group.
    
        Creates:
            strength_by_group.html
    
        Contains:
            - IV heatmap
            - KS heatmap
            - maximum lift heatmap
            - detailed metrics table
        """
    
        if self.source_column is None:
            return pd.DataFrame()
    
        if (
            self.source_column
            not in self.dataset.df.columns
        ):
            return pd.DataFrame()
    
        try:
    
            table = (
                self.eda.strength_by_group(
                    group=self.source_column,
                    n_bins=self.n_bins,
                )
            )
    
            if table.empty:
                return table
    
            metric_configurations = [
                (
                    "iv",
                    "Information Value por grupo",
                ),
                (
                    "max_ks",
                    "KS máximo por grupo",
                ),
                (
                    "max_lift",
                    "Lift máximo por grupo",
                ),
            ]
    
            sections = []
    
            for metric, title in (
                metric_configurations
            ):
    
                if metric not in table.columns:
                    continue
    
                figure = (
                    self.eda
                    .plot_strength_by_group(
                        group=self.source_column,
                        metric=metric,
                        n_bins=self.n_bins,
                    )
                )
    
                figure.update_layout(
                    title=title
                )
    
                sections.append(
                    self._plot_section(
                        title,
                        figure,
                    )
                )
    
            display_columns = [
                "feature",
                "group_value",
                "iv",
                "max_ks",
                "max_lift",
                "global_target_rate",
                "observations",
                "events",
            ]
    
            display_columns = [
                column
                for column in display_columns
                if column in table.columns
            ]
    
            display_table = (
                table[
                    display_columns
                ]
                .sort_values(
                    [
                        "feature",
                        "group_value",
                    ],
                    kind="stable",
                )
                .reset_index(drop=True)
            )
    
            sections.append(
                self._table_section(
                    "Métricas por feature e grupo",
                    display_table,
                )
            )
    
            body = f"""
            <div class="navigation">
                <a href="index.html">
                    ← Voltar ao índice
                </a>
            </div>
    
            <h1>
                Feature strength por
                {html.escape(self.source_column)}
            </h1>
    
            <p>
                Comparação do poder preditivo global e por
                {html.escape(self.source_column)}.
                Uma feature pode ter baixo IV global e ainda
                ser relevante para uma tarefa específica.
            </p>
    
            {
                ''.join(
                    self._wrap_section(
                        section,
                        index,
                    )
                    for index, section
                    in enumerate(sections)
                )
            }
            """
    
            page = self._simple_page(
                title=(
                    "Feature strength por grupo"
                ),
                body=body,
            )
    
            output_path = (
                self.output_folder
                / "strength_by_group.html"
            )
    
            output_path.write_text(
                page,
                encoding="utf-8",
            )
    
            return table
    
        except Exception as error:
    
            self._register_error(
                "__global__",
                "strength_by_group",
                error,
            )
    
            return pd.DataFrame()
    
    
    # =====================================================
    # Excel
    # =====================================================

    def _export_excel(
        self,
        relationship_tables,
    ):
        output_path = (
            self.output_folder
            / "eda_summary.xlsx"
        )

        with pd.ExcelWriter(
            output_path,
            engine="openpyxl",
        ) as writer:

            if self.overview is not None:
                self.overview.to_excel(
                    writer,
                    sheet_name="Overview",
                    index=False,
                )

            if (
                self.quality is not None
                and not self.quality.empty
            ):
                self.quality.to_excel(
                    writer,
                    sheet_name="Quality",
                    index=False,
                )

            if (
                self.strength is not None
                and not self.strength.empty
            ):
                self.strength.to_excel(
                    writer,
                    sheet_name="Strength",
                    index=False,
                )

            if (
                self.group_strength is not None
                and not self.group_strength.empty
            ):
                self.group_strength.to_excel(
                    writer,
                    sheet_name="Strength by group",
                    index=False,
                )
            
            if (
                self.stability is not None
                and not self.stability.empty
            ):
                self.stability.to_excel(
                    writer,
                    sheet_name="Stability",
                    index=False,
                )

            for name, table in (
                relationship_tables.items()
            ):

                if table.empty:
                    continue

                table.to_excel(
                    writer,
                    sheet_name=name[:31],
                )

            if self.errors:

                pd.DataFrame(
                    self.errors
                ).to_excel(
                    writer,
                    sheet_name="Errors",
                    index=False,
                )

    # =====================================================
    # Index
    # =====================================================

    def _export_index(
        self,
    ):
        table_rows = []
    
        for _, row in self.overview.iterrows():
    
            table_rows.append(
                f"""
                <tr>
                    <td>
                        <a href="{html.escape(row['page'])}">
                            {html.escape(str(row['feature']))}
                        </a>
                    </td>
    
                    <td>
                        {html.escape(
                            str(row['variable_type'])
                        )}
                    </td>
    
                    <td>
                        {
                            self._format_percent(
                                row["missing_pct"]
                            )
                        }
                    </td>
    
                    <td>
                        {
                            self._format_number(
                                row["n_unique"]
                            )
                        }
                    </td>
    
                    <td>
                        {
                            self._format_number(
                                row["iv"],
                                4,
                            )
                        }
                    </td>
    
                    <td>
                        {
                            self._format_number(
                                row["ks"],
                                4,
                            )
                        }
                    </td>
    
                    <td>
                        {
                            self._format_number(
                                row["max_psi"],
                                4,
                            )
                        }
                    </td>
    
                    <td>
                        <span
                            class="
                                status
                                status-{row['status']}
                            "
                        >
                            {row["status"]}
                        </span>
                    </td>
                </tr>
                """
            )
    
        strength_link = ""
    
        if (
            self.group_strength is not None
            and not self.group_strength.empty
        ):
            group_name = (
                self.source_column
                or "grupo"
            )
    
            strength_link = (
                '<a href="strength_by_group.html">'
                f"Strength por "
                f"{html.escape(str(group_name))}"
                "</a>"
            )
    
        body = f"""
        <h1>EDA Report</h1>
    
        <div class="summary-links">
    
            <a href="eda_summary.xlsx">
                Download Excel summary
            </a>
    
            {strength_link}
    
            <a href="relationships/pearson.html">
                Pearson
            </a>
    
            <a href="relationships/spearman.html">
                Spearman
            </a>
    
            <a href="relationships/cramers_v.html">
                Cramér's V
            </a>
    
        </div>
    
        <input
            type="text"
            id="featureSearch"
            placeholder="Search feature..."
            onkeyup="filterFeatures()"
        >
    
        <table id="featureTable">
    
            <thead>
                <tr>
                    <th onclick="sortTable(0)">
                        Feature
                    </th>
    
                    <th onclick="sortTable(1)">
                        Type
                    </th>
    
                    <th onclick="sortTable(2)">
                        Missing
                    </th>
    
                    <th onclick="sortTable(3)">
                        Unique
                    </th>
    
                    <th onclick="sortTable(4)">
                        IV
                    </th>
    
                    <th onclick="sortTable(5)">
                        KS
                    </th>
    
                    <th onclick="sortTable(6)">
                        PSI
                    </th>
    
                    <th onclick="sortTable(7)">
                        Status
                    </th>
                </tr>
            </thead>
    
            <tbody>
                {''.join(table_rows)}
            </tbody>
    
        </table>
        """
    
        page = self._simple_page(
            title="EDA Report",
            body=body,
            include_index_javascript=True,
        )
    
        (
            self.output_folder
            / "index.html"
        ).write_text(
            page,
            encoding="utf-8",
        )

    # =====================================================
    # HTML helpers
    # =====================================================

    def _feature_page(
        self,
        feature_name,
        sections,
        previous_feature,
        next_feature,
    ):
        navigation = [
            '<a href="../index.html">← Index</a>'
        ]

        if previous_feature is not None:
            navigation.append(
                (
                    '<a href="'
                    f'{self._feature_filename(previous_feature)}'
                    '">← Previous</a>'
                )
            )

        if next_feature is not None:
            navigation.append(
                (
                    '<a href="'
                    f'{self._feature_filename(next_feature)}'
                    '">Next →</a>'
                )
            )

        body = f"""
        <div class="navigation">
            {' '.join(navigation)}
        </div>

        <h1>{html.escape(feature_name)}</h1>

        <div class="toc">
            <strong>Sections</strong>
            <ul>
                {
                    ''.join(
                        f'<li><a href="#section-{index}">'
                        f'{self._section_title(section)}'
                        '</a></li>'
                        for index, section
                        in enumerate(sections)
                    )
                }
            </ul>
        </div>

        {
            ''.join(
                self._wrap_section(
                    section,
                    index,
                )
                for index, section
                in enumerate(sections)
            )
        }

        <div class="navigation bottom">
            {' '.join(navigation)}
        </div>
        """

        return self._simple_page(
            title=feature_name,
            body=body,
            back_link="../index.html",
        )

    def _table_section(
        self,
        title,
        table,
    ):
        if isinstance(
            table,
            pd.Series,
        ):
            table = (
                table
                .rename("Value")
                .to_frame()
                .reset_index()
                .rename(
                    columns={
                        "index": "Metric"
                    }
                )
            )

        elif not isinstance(
            table,
            pd.DataFrame,
        ):
            table = pd.DataFrame(
                table
            )

        return {
            "title": title,
            "content": table.to_html(
                index=False,
                border=0,
                classes="dataframe",
                na_rep="",
            ),
        }

    def _plot_section(
        self,
        title,
        figure,
    ):
        return {
            "title": title,
            "content": self._plot_html(
                figure
            ),
        }

    def _safe_table_section(
        self,
        title,
        function,
        feature_name,
    ):
        try:
            return self._table_section(
                title,
                function(),
            )

        except Exception as error:
            self._register_error(
                feature_name,
                title,
                error,
            )

            return self._error_section(
                title,
                error,
            )

    def _safe_plot_section(
        self,
        title,
        function,
        feature_name,
    ):
        try:
            return self._plot_section(
                title,
                function(),
            )

        except Exception as error:
            self._register_error(
                feature_name,
                title,
                error,
            )

            return self._error_section(
                title,
                error,
            )

    @staticmethod
    def _error_section(
        title,
        error,
    ):
        return {
            "title": title,
            "content": (
                '<div class="error">'
                f"{html.escape(str(error))}"
                "</div>"
            ),
        }

    @staticmethod
    def _plot_html(
        figure,
    ):
        return pio.to_html(
            figure,
            full_html=False,
            include_plotlyjs="cdn",
            config={
                "responsive": True,
                "displaylogo": False,
            },
        )

    @staticmethod
    def _wrap_section(
        section,
        index,
    ):
        return f"""
        <section id="section-{index}">
            <h2>{html.escape(section['title'])}</h2>
            {section['content']}
        </section>
        """

    @staticmethod
    def _section_title(
        section,
    ):
        return html.escape(
            section["title"]
        )

    def _simple_page(
        self,
        title,
        body,
        back_link=None,
        include_index_javascript=False,
    ):
        back = ""

        if back_link is not None:
            back = (
                f'<a href="{back_link}">'
                "← Back"
                "</a>"
            )

        javascript = (
            self._index_javascript()
            if include_index_javascript
            else ""
        )

        return f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="utf-8">
            <meta
                name="viewport"
                content="width=device-width, initial-scale=1"
            >
            <title>{html.escape(title)}</title>
            <style>
                {self._css()}
            </style>
        </head>
        <body>
            {back}
            {body}
            {javascript}
        </body>
        </html>
        """

    # =====================================================
    # Utilities
    # =====================================================

    def _get_features(
        self,
    ):
        if self.include_ignored:
            return list(
                self.dataset.df.columns
            )

        return list(
            self.dataset.feature_columns
        )

    def _quality_row(
        self,
        feature_name,
    ):
        if (
            self.quality is None
            or self.quality.empty
        ):
            return {}

        result = self.quality[
            self.quality["feature"]
            == feature_name
        ]

        if result.empty:
            return {}

        return result.iloc[0].to_dict()

    @staticmethod
    def _feature_status(
        row,
    ):
        if row["is_constant"]:
            return "remove"

        if row["is_quasi_constant"]:
            return "review"

        if (
            pd.notna(row["missing_pct"])
            and row["missing_pct"] > 0.50
        ):
            return "review"

        if (
            pd.notna(row["max_psi"])
            and row["max_psi"] >= 0.25
        ):
            return "review"

        return "ok"

    def _feature_filename(
        self,
        feature_name,
    ):
        safe_name = re.sub(
            r"[^a-zA-Z0-9_.-]+",
            "_",
            str(feature_name),
        )

        return f"{safe_name}.html"

    def _register_error(
        self,
        feature,
        section,
        error,
    ):
        self.errors.append(
            {
                "feature": feature,
                "section": section,
                "error_type": (
                    type(error).__name__
                ),
                "error": str(error),
            }
        )

    def _create_folders(
        self,
    ):
        self.output_folder.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.features_folder.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.relationships_folder.mkdir(
            parents=True,
            exist_ok=True,
        )

    @staticmethod
    def _format_percent(
        value,
    ):
        if pd.isna(value):
            return ""

        return f"{value:.1%}"

    @staticmethod
    def _format_number(
        value,
        decimals=0,
    ):
        if pd.isna(value):
            return ""

        return f"{value:.{decimals}f}"

    @staticmethod
    def _css():
        return """
        body {
            max-width: 1500px;
            margin: 0 auto;
            padding: 30px;
            font-family: Arial, sans-serif;
            color: #222;
            background: #fafafa;
        }

        h1, h2 {
            color: #1f2937;
        }

        h2 {
            border-bottom: 1px solid #ddd;
            padding-bottom: 8px;
        }

        section {
            background: white;
            margin: 24px 0;
            padding: 22px;
            border-radius: 8px;
            box-shadow: 0 1px 4px rgba(0,0,0,.08);
        }

        table {
            width: 100%;
            border-collapse: collapse;
            background: white;
        }

        th, td {
            padding: 9px 12px;
            border-bottom: 1px solid #ddd;
            text-align: left;
        }

        th {
            background: #f1f5f9;
            cursor: pointer;
            position: sticky;
            top: 0;
        }

        tr:hover {
            background: #f8fafc;
        }

        a {
            color: #2563eb;
            text-decoration: none;
            margin-right: 14px;
        }

        a:hover {
            text-decoration: underline;
        }

        #featureSearch {
            width: 100%;
            box-sizing: border-box;
            padding: 12px;
            margin: 20px 0;
            border: 1px solid #bbb;
            border-radius: 6px;
        }

        .navigation {
            display: flex;
            gap: 12px;
            margin-bottom: 20px;
        }

        .navigation.bottom {
            margin-top: 30px;
        }

        .summary-links {
            display: flex;
            flex-wrap: wrap;
            gap: 12px;
            margin: 20px 0;
        }

        .toc {
            background: white;
            padding: 15px 20px;
            border-radius: 8px;
        }

        .error {
            background: #fee2e2;
            color: #991b1b;
            padding: 12px;
            border-radius: 6px;
        }

        .status {
            padding: 4px 9px;
            border-radius: 10px;
            font-weight: bold;
            font-size: 12px;
        }

        .status-ok {
            background: #dcfce7;
            color: #166534;
        }

        .status-review {
            background: #fef3c7;
            color: #92400e;
        }

        .status-remove {
            background: #fee2e2;
            color: #991b1b;
        }
        """

    @staticmethod
    def _index_javascript():
        return """
        <script>
        function filterFeatures() {
            const input =
                document.getElementById("featureSearch");

            const filter =
                input.value.toLowerCase();

            const table =
                document.getElementById("featureTable");

            const rows =
                table.getElementsByTagName("tr");

            for (let i = 1; i < rows.length; i++) {
                const text =
                    rows[i].innerText.toLowerCase();

                rows[i].style.display =
                    text.includes(filter)
                    ? ""
                    : "none";
            }
        }

        function sortTable(columnIndex) {
            const table =
                document.getElementById("featureTable");

            const body =
                table.tBodies[0];

            const rows =
                Array.from(body.rows);

            const ascending =
                table.dataset.sortColumn != columnIndex
                || table.dataset.sortOrder != "asc";

            rows.sort((rowA, rowB) => {
                const valueA =
                    rowA.cells[columnIndex]
                    .innerText.trim();

                const valueB =
                    rowB.cells[columnIndex]
                    .innerText.trim();

                const numberA =
                    parseFloat(
                        valueA.replace("%", "")
                    );

                const numberB =
                    parseFloat(
                        valueB.replace("%", "")
                    );

                let comparison;

                if (
                    !Number.isNaN(numberA)
                    && !Number.isNaN(numberB)
                ) {
                    comparison = numberA - numberB;
                } else {
                    comparison =
                        valueA.localeCompare(valueB);
                }

                return ascending
                    ? comparison
                    : -comparison;
            });

            rows.forEach(
                row => body.appendChild(row)
            );

            table.dataset.sortColumn =
                columnIndex;

            table.dataset.sortOrder =
                ascending ? "asc" : "desc";
        }
        </script>
        """
