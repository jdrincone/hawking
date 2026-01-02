"""
Orchestration logic for the Fazenda automation report.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, List, Optional, Tuple

from core.s3 import S3AssetManager
from core.viz import plot_bar
from core.render_svg import generate_sackoff_svg

from automation.fazenda.config import FazendaConfig
from automation.fazenda.data import load_and_clean_data
from automation.fazenda.metrics import (
    compute_sackoff,
    build_summary_table,
    build_monthly_summary_table,
    extract_svg_params
)

logger = logging.getLogger(__name__)

class FazendaReportOrchestrator:
    """Manages the lifecycle of a Fazenda report generation run."""
    
    def __init__(self, s3: S3AssetManager, config: FazendaConfig):
        self.s3 = s3
        self.config = config

    def run(self):
        """Executes the full automated orchestration flow."""
        logger.info("Step 1: Loading data...")
        df_raw = load_and_clean_data(self.s3, self.config)

        logger.info("Step 2: Identifying outliers and selecting relevant diets...")
        df_cut, df_bad, df_both_cut, df_both_diet = self._prepare_data(df_raw)

        logger.info("Step 3: Building reports and visualizations...")
        self._build_bad_records_report(df_bad)
        self._build_global_performance_report(df_cut)
        self._build_monthly_evolution_report(df_cut)
        self._build_diet_performance_report(df_both_cut, df_both_diet)

        logger.info("Fazenda automation completed.")

    def _prepare_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str]]:
        """Separates valid data from outliers and filters by diet coverage."""
        q_min = df["sackoff_op"].quantile(self.config.q_low)
        q_max = df["sackoff_op"].quantile(self.config.q_high)

        cond_range = (df["sackoff_op"] >= q_min) & (df["sackoff_op"] <= q_max)
        cond_exclude = df["op"].isin(self.config.exclude_ops)
        
        cond_good = cond_range & ~cond_exclude
        df_cut = df[cond_good].copy()
        df_bad = df[~cond_good].copy()

        # Selection of diets that have both 'Con Adiflow' and 'Sin Adiflow'
        df_group_diet = compute_sackoff(df_cut, cols=["Dieta"])
        mask = (df_group_diet
                .groupby('Dieta')['Tiene Adiflow']
                .transform('nunique')
                .eq(2))

        df_both_diet = (df_group_diet[mask]
                .sort_values(['Dieta', 'Tiene Adiflow']))["Dieta"].unique().tolist()
        df_both_cut = df_cut[df_cut["Dieta"].isin(df_both_diet)]

        return df_cut, df_bad, df_both_cut, df_both_diet

    def _finalize_report_dataframe(self, df: pd.DataFrame, rename_map: Dict[str, str], index_cols: List[str]) -> pd.DataFrame:
        """Helper to format report dataframes (numeric conversion, selection, renaming)."""
        res = df.copy()
        for col in rename_map.keys():
            if col in res.columns:
                res[col] = pd.to_numeric(res[col], errors='coerce')
        
        cols_to_keep = index_cols + [c for c in rename_map.keys() if c in res.columns]
        return res[cols_to_keep].rename(columns=rename_map).round(2)

    def _build_bad_records_report(self, df_bad: pd.DataFrame):
        """Saves records that were excluded due to anomalies."""
        cols = [
            'date', 'op', 'Dieta', 'panificadas', 'entregadas', 
            'Producción (Ton)', 'Anulación (Ton)', 'diff', 'sackoff_op', 
            'Tiene Adiflow', 'pdi_agro', 'dureza_agro', 'finos_agro'
        ]
        
        bad_rename_map = {
            'diff': self.config.METRIC_RENAME_MAP.get('diferencia', 'Diferencia (Ton)'),
            'panificadas': self.config.METRIC_RENAME_MAP.get('production', 'Planificadas (Ton)'),
            'entregadas': 'Entregadas (Ton)',
            'sackoff_op': self.config.METRIC_RENAME_MAP.get('sackoff', 'Sackoff (%)'),
            'pdi_agro': self.config.METRIC_RENAME_MAP.get('pdi', 'Pdi (%)'),
            'dureza_agro': self.config.METRIC_RENAME_MAP.get('dureza', 'Dureza (kg/cm2)'),
            'finos_agro': self.config.METRIC_RENAME_MAP.get('finos', 'Finos (%)')
        }

        df_bad_dep = df_bad[cols].round(2).rename(columns=bad_rename_map).sort_values(["date"], ascending=False)
        self.s3.save_dataframe(df_bad_dep, self.config.artifact_bad_records_csv)

    def _build_global_performance_report(self, df_cut: pd.DataFrame) -> pd.DataFrame:
        """Processes global performance metrics and generates related artifacts."""
        summary = compute_sackoff(df_cut, cols=[])
        
        # Mapping specialized for global summary
        rename_map = self.config.METRIC_RENAME_MAP.copy()
        if 'sackoff' in rename_map:
            rename_map['sackoff'] = rename_map.get('sackoff_prom', 'Sackoff Prom (%)')

        summary_dep = self._finalize_report_dataframe(summary, rename_map, index_cols=['Tiene Adiflow'])
        self.s3.save_dataframe(summary_dep, self.config.artifact_global_performance_csv)

        # Economic/Savings Estimation
        savings = build_summary_table(summary_dep)
        self.s3.save_dataframe(savings, self.config.artifact_economic_savings_csv)

        # SVG Infographic
        svg_params = extract_svg_params(
            df=summary_dep,
            fecha_ini=df_cut["date"].min().strftime("%d-%b"),  
            fecha_fin=df_cut["date"].max().strftime("%d-%b"),
            pct_datos=98
        )
        svg_content = generate_sackoff_svg(
            template_path=self.config.svg_template_path, 
            **svg_params
        )
        self.s3.save_svg_content(svg_content, self.config.artifact_performance_infographic_svg)
        
        return summary_dep

    def _build_monthly_evolution_report(self, df_cut: pd.DataFrame):
        """Processes monthly evolution data and charts."""
        df_monthly = compute_sackoff(df_cut, cols=["month"])
        
        summary_month_dep = self._finalize_report_dataframe(df_monthly, self.config.METRIC_RENAME_MAP, index_cols=['month', 'Tiene Adiflow'])
        self.s3.save_dataframe(summary_month_dep, self.config.artifact_monthly_evolution_csv)

        # Comparison table
        comp_table = build_monthly_summary_table(
            summary_month_dep,
            category_order=sorted(summary_month_dep['month'].unique()),
        )
        comp_table.rename(columns={"Category": "Mes"}, inplace=True)
        self.s3.save_dataframe(comp_table, self.config.artifact_monthly_comparison_table_csv)

        # Monthly Evolution Chart
        m = df_monthly["month"].dt.to_timestamp()
        df_monthly["month_lbl"] = m.dt.strftime("%b-%Y")
        cats = pd.date_range(m.min(), m.max(), freq="MS").strftime("%b-%Y")
        df_monthly["month_lbl"] = pd.Categorical(df_monthly["month_lbl"], categories=cats, ordered=True)

        fig = plot_bar(
            df_monthly.round(2),
            x_col="month_lbl",
            y_col="sackoff",
            group_col="Tiene Adiflow",
            order_x=sorted(df_monthly["month_lbl"].unique(), reverse=True),
            title="Evolución Mensual del Sackoff",
            cat_base="Sin Adiflow",
            show_delta=True,
            x_title="Mes",
            y_title="Sackoff (%)",
            text_format=".2f",
            delta_unit="%",
            hover_data_cols=['reales', 'production'],
            height=500,
            width=1000,
        )
        self.s3.save_plotly_html(fig, self.config.artifact_monthly_evolution_chart_html)

    def _build_diet_performance_report(self, df_both_cut: pd.DataFrame, df_both_diet: List[str]):
        """Processes performance metrics specifically for different diet types."""
        df_diet = compute_sackoff(df_both_cut, cols=["Dieta"])
        
        # Identify top results (where 'Con Adiflow' > 'Sin Adiflow')
        pivot = df_diet.pivot_table(index="Dieta", columns="Tiene Adiflow", values="sackoff", aggfunc="mean")
        best_results = (
            pivot[pivot["Con Adiflow"] > pivot["Sin Adiflow"]]
            .assign(delta=lambda x: x["Con Adiflow"] - x["Sin Adiflow"])
            .sort_values("delta", ascending=False)
            .reset_index().round(2)
        )
        self.s3.save_dataframe(best_results, self.config.artifact_top_diet_results_csv)

        # Standard Diet Summary
        summary_diet_dep = self._finalize_report_dataframe(df_diet, self.config.METRIC_RENAME_MAP, index_cols=['Dieta', 'Tiene Adiflow'])
        # Filter only diet that shown improvement
        summary_diet_dep = summary_diet_dep[summary_diet_dep["Dieta"].isin(best_results["Dieta"])]
        self.s3.save_dataframe(summary_diet_dep, self.config.artifact_diet_performance_csv)

        # Diet comparison table based on recovered tons
        diet_comp_table = build_monthly_summary_table(
            summary_diet_dep,
            month_col="Dieta",
            category_order=best_results["Dieta"].tolist(),
        )
        diet_comp_table.rename(columns={"Category": "Dieta"}, inplace=True)
        diet_comp_table = diet_comp_table.sort_values("Toneladas Recuperadas", ascending=False)
        self.s3.save_dataframe(diet_comp_table, self.config.artifact_diet_comparison_table_csv)

        # Diet Comparison Chart
        fig = plot_bar(
            df_diet.round(2),
            x_col="Dieta",
            y_col="sackoff",
            group_col="Tiene Adiflow",
            order_x=df_both_diet,
            title="Comparativo de Sackoff por Dieta",
            cat_base="Sin Adiflow",
            show_delta=True,
            x_title="Dieta",
            y_title="Sackoff (%)",
            text_format=".2f",
            delta_unit="%",
            hover_data_cols=['reales', 'production'],
            height=500,
            width=1000,
        )
        self.s3.save_plotly_html(fig, self.config.artifact_diet_comparison_chart_html)
