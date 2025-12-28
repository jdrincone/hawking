"""
Orchestration logic for the El Dorado automation report.
"""

import os
import logging
import pandas as pd
from jinja2 import Template
from typing import Dict, Any, Optional

from core.s3 import S3AssetManager
from core.viz import plot_line, plot_cdf, plot_heatmap, plot_corr_triangle

from automation.el_dorado.config import ElDoradoConfig
from automation.el_dorado.data import load_all_data
from automation.el_dorado.metrics import (
    compute_micro_prevalence, compute_eviscerado_summary, 
    prepare_combined_metrics, filter_last_month_data
)

logger = logging.getLogger(__name__)

class ElDoradoReportOrchestrator:
    def __init__(self, s3: S3AssetManager, config: ElDoradoConfig):
        self.s3 = s3
        self.config = config
        self.df_micro: Optional[pd.DataFrame] = None
        self.df_evi: Optional[pd.DataFrame] = None
        self.cats: list = []

    def run(self) -> str:
        logger.info("Step 1: Loading El Dorado data...")
        self.df_micro, self.df_evi, self.cats = load_all_data(self.s3, self.config)
        
        logger.info("Step 2: Calculating monthly aggregations...")
        df_micro_month = compute_micro_prevalence(self.df_micro)
        df_evi_month = compute_eviscerado_summary(self.df_evi)
        
        self.s3.save_dataframe(df_micro_month, "summary_micro_month.csv")
        self.s3.save_dataframe(df_evi_month, "summary_evi_month.csv")
        
        logger.info("Step 3: Generating line plots (Micro vs Kilos)...")
        self._generate_prevalence_plots(df_micro_month, df_evi_month)

        logger.info("Step 4: Generating Detailed Heatmaps...")
        self._generate_detail_heatmaps(df_micro_month)
        
        logger.info("Step 5: Generating CDF plots (Chiller Distribution)...")
        self._generate_cdf_plots()

        logger.info("Step 6: Generating Correlation Analysis...")
        self._generate_correlation_triangle()
        
        logger.info("Step 7: Rendering YAML configuration...")
        yaml_path = self._render_yaml(df_micro_month, df_evi_month)
        
        return yaml_path

    def _generate_prevalence_plots(self, df_micro_month: pd.DataFrame, df_evi_month: pd.DataFrame):
        for micro in ["Salmonella", "Campylobacter"]:
            df_plot = prepare_combined_metrics(df_micro_month, df_evi_month, micro)
            
            fig = plot_line(
                df=df_plot,
                x_col="month",
                y_col="prev",
                group_col="etapa",
                secondary_y_col="Descuento (Kg)",
                secondary_y_title="Kilogramos",
                x_title="Mes",
                y_title="Prevalencia (%)",
                title=f"Prevalencia {micro} vs Kilos de Descuento",
                line_colors=self.config.line_colors,
                height=500,
                width=900,
            )
            self.s3.save_plotly_html(fig, f"{micro.lower()}_line.html")

    def _generate_cdf_plots(self):
        # Chiller data for the last month
        df_chiller_last = filter_last_month_data(self.df_micro)
        df_chiller_last = df_chiller_last[df_chiller_last["etapa"] == "salida chiller"]
        
        for col in self.config.cdf_columns:
            if col in df_chiller_last.columns:
                fig = plot_cdf(
                    df_chiller_last, 
                    value_col=col, 
                    group_col='month', 
                    width=800, 
                    height=500,
                    title=f"Distribución Acumulada de {col.upper()} (Último Mes)"
                )
                self.s3.save_plotly_html(fig, f"cdf_{col}.html")

    def _generate_detail_heatmaps(self, df_micro_month: pd.DataFrame):
        """Generates detailed heatmaps for prevalence."""
        last_month = df_micro_month["month"].max()
        df_last = df_micro_month[df_micro_month["month"] == last_month]
        
        # 1. Overall Heatmap (Stage vs Microorganism)
        fig_gen = plot_heatmap(
            df=df_last,
            x_col="etapa",
            y_col="microorganismo",
            value_col="prev",
            secondary_col="n_analysis",
            secondary_aggfunc="sum",
            aggfunc="mean",
            x_order=["ciegos", "Poll Ext Visceras", "prechiller", "salida chiller"],
            value_unit="%",
            title=f"<b>Prevalencias y muestras en {last_month}</b>",
            show_secondary_labels=True,
            secondary_position="bottom",
            width=800, height=300,
        )
        self.s3.save_plotly_html(fig_gen, "prevalencia_heatmap.html")

        # 2. Farm-level Heatmaps
        from automation.el_dorado.metrics import compute_farm_prevalence
        df_farm = compute_farm_prevalence(self.df_micro)
        df_farm_last = df_farm[df_farm["month"] == last_month]
        
        for micro in df_micro_month["microorganismo"].unique():
            df_sub = df_farm_last[df_farm_last["microorganismo"] == micro]
            fig_farm = plot_heatmap(
                df=df_sub,
                x_col="etapa",
                y_col="granja",
                value_col="prev",
                secondary_col="n_analysis",
                secondary_aggfunc="sum",
                aggfunc="mean",
                x_order=["ciegos", "Poll Ext Visceras", "prechiller", "salida chiller"],
                value_unit="%",
                title=f"<b>Prevalencia de {micro} en {last_month} por granja</b>",
                show_secondary_labels=True,
                secondary_position="bottom",
                width=800, height=500,
            )
            self.s3.save_plotly_html(fig_farm, f"prev_last_month_{micro}.html")

    def _generate_correlation_triangle(self):
        """Generates the correlation triangle for chiller data."""
        # Use chiller data from last month
        df_chiller = self.df_micro[self.df_micro["etapa"] == "salida chiller"]
        df_chiller_last = filter_last_month_data(df_chiller)
        
        if df_chiller_last.empty:
            logger.warning("No data for correlation triangle.")
            return

        # Prepare pivot for correlation as in the notebook
        df_pivot = df_chiller_last.pivot_table(
            index=['ph', 'orp', 'cloro'],
            columns='microorganismo',
            values=['have_micro'],
            aggfunc='mean'
        )
        df_pivot.columns = [f'have_micro_{micro.lower()}' for _, micro in df_pivot.columns]
        df_final = df_pivot.reset_index()

        corr_cols = ['ph', 'have_micro_campylobacter', 'cloro', 'orp', 'have_micro_salmonella']
        # Filter available columns
        available_cols = [c for c in corr_cols if c in df_final.columns]
        
        if len(available_cols) < 2:
            logger.warning(f"Not enough columns for correlation. Available: {available_cols}")
            return

        fig = plot_corr_triangle(
            df=df_final,
            value_cols=available_cols,
            title="Correlación Parámetros Control vs Micro",
            width=750, height=650,
        )
        self.s3.save_plotly_html(fig, "correlacion_chiller.html")

    def _render_yaml(self, df_micro: pd.DataFrame, df_evi: pd.DataFrame) -> str:
        template_path = os.path.join("yamls", "templates", "el_dorado.yaml.j2")
        if not os.path.exists(template_path):
            logger.error(f"Template not found at {template_path}")
            return ""

        with open(template_path, "r") as f:
            template = Template(f.read())
            
        # Basic dynamic summaries for YAML
        last_month = self.cats[-1] if self.cats else "N/A"
        
        render_params = {
            "descripcion": self.config.descripcion,
            "periodo_analisis": f"{self.cats[0]} a {self.cats[-1]}" if self.cats else "N/A",
            "ultimo_mes": last_month,
            "total_muestras": len(self.df_micro),
        }
        
        rendered_yaml = template.render(**render_params)
        output_yaml_path = os.path.join("yamls", f"{self.config.notebook_name}.yaml")
        
        with open(output_yaml_path, "w") as f:
            f.write(rendered_yaml)
            
        return output_yaml_path

def run_report_logic(s3: S3AssetManager, config_dict: Dict[str, Any]) -> str:
    config = ElDoradoConfig(**config_dict)
    orchestrator = ElDoradoReportOrchestrator(s3, config)
    return orchestrator.run()
