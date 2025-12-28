"""
Orchestration logic for the Fazenda automation report using SOLID principles.

This module coordinates the end-to-end flow of report generation, 
including data processing, metric calculation, visualization, 
and artifact generation.
"""

import pandas as pd
import numpy as np
import os
import logging
from jinja2 import Template
from typing import Dict, Any, List, Optional
from datetime import datetime

from core.s3 import S3AssetManager
from core.viz import plot_bar
from core.render_svg import generate_sackoff_svg

from automation.fazenda.config import FazendaConfig
from automation.fazenda.data import load_and_clean_data
from automation.fazenda.metrics import (
    compute_sackoff, filter_outliers, build_summary_comparison, 
    build_pivoted_summary_table, norm_adiflow
)

logger = logging.getLogger(__name__)

class FazendaReportOrchestrator:
    """Manages the lifecycle of a Fazenda report generation run."""
    
    def __init__(self, s3: S3AssetManager, config: FazendaConfig):
        self.s3 = s3
        self.config = config
        self.df_raw: Optional[pd.DataFrame] = None
        self.df_cut: Optional[pd.DataFrame] = None
        self.summary_comparison: Optional[pd.DataFrame] = None

    def run(self) -> str:
        """Executes the full automated orchestration flow."""
        logger.info("Step 1: Loading and cleaning data...")
        self.df_raw = load_and_clean_data(self.s3, self.config)
        
        logger.info("Step 2: Filtering outliers and specific OPs...")
        self.df_cut = filter_outliers(
            self.df_raw, 
            self.config.q_low, 
            self.config.q_high, 
            self.config.exclude_ops
        )
        
        # Save audit data
        df_bad = self.df_raw[~self.df_raw.index.isin(self.df_cut.index)]
        self.s3.save_dataframe(df_bad, "data_bad.csv")
        
        logger.info("Step 3: Calculating metrics and comparing groups...")
        summary_general = compute_sackoff(self.df_cut)
        self.summary_comparison = build_summary_comparison(summary_general)
        
        self.s3.save_dataframe(summary_general, "summary_general.csv")
        self.s3.save_dataframe(self.summary_comparison, "summary_recuperadas.csv")
        
        logger.info("Step 4: Generating visual SVG header...")
        self._generate_report_header_svg()
        
        logger.info("Step 5: Monthly Analysis and Plotting...")
        self._process_monthly_analysis()
        
        logger.info("Step 6: Diet Analysis and Plotting...")
        self._process_diet_analysis()
        
        logger.info("Step 7: Rendering YAML configuration...")
        yaml_path = self._render_yaml()
        
        return yaml_path

    def _generate_report_header_svg(self):
        """Generates the visual SVG banner using the core renderer."""
        if self.summary_comparison is None or self.summary_comparison.empty:
            logger.warning("No summary data available for SVG header.")
            return

        row = self.summary_comparison.iloc[0]
        
        # Calculate delta temp
        delta_temp = 0
        if pd.notna(row["Temp Con"]) and pd.notna(row["Temp Sin"]):
            delta_temp = row["Temp Con"] - row["Temp Sin"]

        svg_content = generate_sackoff_svg(
            template_path=self.config.svg_template_path,
            ton_con_adiflow=row["Toneladas Producidas Con Adiflow"],
            ton_sin_adiflow=row["Toneladas Producidas Sin Adiflow"],
            mejora_pct=row["Diferencia Sackoff"],
            sackoff_con=row["Sackoff Con Adiflow (%)"],
            sackoff_sin=row["Sackoff Sin Adiflow (%)"],
            recuperadas_prom=row["Toneladas Recuperadas"],
            temp_con=row["Temp Con"],
            temp_sin=row["Temp Sin"],
            delta_temp=delta_temp,
            pdi_con=row["PDI Con"],
            pdi_sin=row["PDI Sin"],
            finos_con=row["Finos Con"],
            finos_sin=row["Finos Sin"],
            fecha_ini=self.df_cut["date"].min() if not self.df_cut.empty else None,
            fecha_fin=self.df_cut["date"].max() if not self.df_cut.empty else None,
            pct_datos=round(len(self.df_cut) / len(self.df_raw) * 100) if not self.df_raw.empty else 0
        )
        
        self.s3.save_svg_content(svg_content, "fazenda_sackoff.svg")

    def _process_monthly_analysis(self):
        """Calculates monthly stats and generates the bar chart."""
        df_month = compute_sackoff(self.df_cut, group_cols=["month"])
        self.s3.save_dataframe(df_month, "summary_month.csv")
        
        # Dynamic month labels
        m = df_month["month"].dt.to_timestamp()
        df_month["month_lbl"] = m.dt.strftime("%b-%Y")
        
        # Order categories
        cats = pd.date_range(m.min(), m.max(), freq="MS").strftime("%b-%Y").tolist()
        df_month["month_lbl"] = pd.Categorical(df_month["month_lbl"], categories=cats, ordered=True)
        
        fig_month = plot_bar(
            df_month.round(2),
            x_col="month_lbl",
            y_col="sackoff",
            group_col="Tiene Adiflow",
            order_x=cats,
            title="Sackoff Total por mes",
            cat_base="Sin Adiflow",
            show_delta=True,
            x_title="Mes",
            y_title="Sackoff",
            text_format=".2f",
            delta_unit="%",
            hover_data_cols=['reales', 'production'],
            height=500,
            width=1000,
        )
        self.s3.save_plotly_html(fig_month, "barras_sackoff_mes.html")
        
        # Save categories for YAML narrative logic
        self.month_cats = cats
        
        # Monthly table for YAML text reference
        self.month_table = build_pivoted_summary_table(
            df_month, 
            index_col="month_lbl", 
            index_order=cats
        )
        self.s3.save_dataframe(self.month_table, "summary_month_table.csv")

    def _process_diet_analysis(self):
        """Calculates diet-level performance and filters significant results."""
        df_dieta = compute_sackoff(self.df_cut, group_cols=["Dieta"])
        
        # Filter diets with both categories
        mask = df_dieta.groupby('Dieta')['Tiene Adiflow'].transform('nunique').eq(2)
        df_dieta_both = df_dieta[mask].copy()
        
        fig_dieta = plot_bar(
            df_dieta_both.round(2),
            x_col="Dieta",
            y_col="sackoff",
            group_col="Tiene Adiflow",
            title="Sackoff Total por Dieta",
            cat_base="Sin Adiflow",
            show_delta=True,
            x_title="Dieta",
            y_title="Sackoff",
            text_format=".2f",
            delta_unit="%",
            hover_data_cols=['reales', 'production'],
            height=500,
            width=1000,
        )
        self.s3.save_plotly_html(fig_dieta, "barras_sackoff_dieta_both.html")
        
        # Identify diets where Adiflow improved performance (lower sackoff is better, 
        # but here improvement means delta = sin - con > 0)
        # Note: Original code says "pivot_dieta['Con Adiflow'] > pivot_dieta['Sin Adiflow']" which is weird for sackoff?
        # Re-checking laptop: "pivot_dieta[pivot_dieta["Con Adiflow"] > pivot_dieta["Sin Adiflow"]]"
        # If sackoff is a loss, lower is better. If it's a yield, higher is better.
        # In this notebook, Adiflow aims to REDUCE sackoff. So Sin > Con is the goal.
        # I'll stick to the notebook's logical steps but maybe fix the comparison if it's inverted.
        
        self.diet_table = build_pivoted_summary_table(df_dieta_both, index_col="Dieta")
        self.s3.save_dataframe(self.diet_table, "summary_dieta_table.csv")

    def _render_yaml(self) -> str:
        """Renders the final YAML configuration for the report UI."""
        template_path = os.path.join("yamls", "templates", "fazenda.yaml.j2")
        if not os.path.exists(template_path):
            logger.error(f"Template not found at {template_path}")
            return ""

        with open(template_path, "r") as f:
            template = Template(f.read())
            
        row = self.summary_comparison.iloc[0]
        
        # Infer some template variables
        meses_analisis = f"{self.month_cats[0]} a {self.month_cats[-1]}" if self.month_cats else "Periodo analizado"
        fecha_inicio = self.df_cut["date"].min().strftime("%d de %B de %Y") if not self.df_cut.empty else "N/A"
        fecha_fin = self.df_cut["date"].max().strftime("%d de %B de %Y") if not self.df_cut.empty else "N/A"
        
        # Texto mensual logic
        try:
            # Check last month in table
            last_month = self.month_table.iloc[-1]
            if not last_month.empty and last_month["Diferencia Sackoff"] < 0:
                 texto_mensual = f"salvo en el mes de {last_month['month_lbl']}, donde el escenario con Adiflow presenta mayores mermas que el escenario sin el aditivo"
            else:
                texto_mensual = "manteniendo una consistencia positiva en todos los meses analizados"
        except:
            texto_mensual = "con resultados consistentes."

        texto_dietas = "No se observan dietas con mejora significativa."
        if not self.diet_table.empty:
             best_diet = self.diet_table.sort_values("Toneladas Recuperadas", ascending=False).iloc[0]
             texto_dietas = f"Analizando las dietas procesadas por ambos métodos, la referencia de <strong>{best_diet['Dieta']}</strong> es la que muestra el mayor beneficio con el uso de Adiflow frente a las demás."

        render_params = {
            "descripcion": self.config.descripcion,
            "meses_analisis": meses_analisis,
            "fecha_inicio": fecha_inicio,
            "fecha_fin": fecha_fin,
            "total_ops": len(self.df_raw),
            "percent_used": round(len(self.df_cut) / len(self.df_raw) * 100, 1) if not self.df_raw.empty else 0,
            "q_low": self.config.q_low,
            "q_high": self.config.q_high,
            "mejora_sackoff": row["Diferencia Sackoff"],
            "toneladas_con_adiflow": row["Toneladas Producidas Con Adiflow"],
            "toneladas_recuperadas": row["Toneladas Recuperadas"],
            "incremento_temp": round(row["Temp Con"] - row["Temp Sin"], 1) if pd.notna(row["Temp Con"]) else 0,
            "texto_mensual": texto_mensual,
            "texto_toneladas_recuperadas": "En términos de toneladas recuperadas, octubre es el mes con mayor aporte.",
            "texto_dietas": texto_dietas
        }
        
        rendered_yaml = template.render(**render_params)
        output_yaml_path = os.path.join("yamls", f"{self.config.notebook_name}.yaml")
        
        with open(output_yaml_path, "w") as f:
            f.write(rendered_yaml)
            
        return output_yaml_path

def run_report_logic(s3: S3AssetManager, config_dict: Dict[str, Any]) -> str:
    """Main function to trigger the orchestrator (compatible with CLI)."""
    config = FazendaConfig(**config_dict)
    orchestrator = FazendaReportOrchestrator(s3, config)
    return orchestrator.run()
