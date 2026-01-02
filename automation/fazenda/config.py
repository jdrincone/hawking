from pydantic import BaseModel, Field
from typing import List, Dict

class FazendaConfig(BaseModel):
    notebook_name: str = "fazenda_efecto_adiflow_en_sackoff"
    q_low: float = 0.01
    q_high: float = 0.99
    exclude_ops: List[str] = ["21944", "21864"]
    descripcion: str = "Automatización del reporte de Fazenda para el análisis del efecto de Adiflow en el sackoff."
    cut_date_ensayo: str = "2025-09-01"

    # S3 Input Paths
    path_base: str = "raw/fazenda/"
    sap_file: str = "sackoff_fazenda_n8n.xlsx"
    sap_sheet: str = "SAP"
    cap_sheet: str = "CAP"
    quality_file: str = "bd_fazenda_n8n.xlsx"
    quality_agro_sheet: str = "CALIDAD AGROINDUSTRIA"
    quality_control_sheet: str = "3_control_prod_peletizado"
    
    # SVG Template Path (S3)
    svg_template_path: str = f"s3://galileo-c4e9a2f1/svg_template/fazenda_sackoff.svg"

    # S3 Output Filenames (Descriptive)
    artifact_bad_records_csv: str = "registros_excluidos_rango_anomalos.csv"
    artifact_global_performance_csv: str = "desempeno_global_adiflow.csv"
    artifact_economic_savings_csv: str = "ahorro_economico_estimado_recuperacion.csv"
    artifact_monthly_evolution_csv: str = "evolucion_mensual_sackoff.csv"
    artifact_monthly_comparison_table_csv: str = "tabla_comparativa_mensual_adiflow.csv"
    artifact_diet_performance_csv: str = "desempeno_por_dieta_adiflow.csv"
    artifact_top_diet_results_csv: str = "mejores_resultados_por_dieta.csv"
    artifact_diet_comparison_table_csv: str = "tabla_comparativa_dietas_recuperacion.csv"
    
    # Visualization Artifacts
    artifact_performance_infographic_svg: str = "infografia_desempeno_fazenda.svg"
    artifact_monthly_evolution_chart_html: str = "grafico_evolucion_mensual_sackoff.html"
    artifact_diet_comparison_chart_html: str = "grafico_comparativo_dietas_both.html"

    # Column Mapping for Reports
    METRIC_RENAME_MAP: Dict[str, str] = {
        'ops': 'OPs',
        'production': 'Planificadas (Ton)',
        'reales': 'Producidas (Ton)',
        'anulation': 'Anuladas (Ton)',
        'sackoff': 'Sackoff (%)',
        'sackoff_prom': 'Sackoff Prom (%)', # Specialized for summary
        'diferencia': "Diferencia (Ton)",
        'temp1_acond_c': 'Temp (°C)',
        'pdi': 'Pdi (%)',
        'finos': 'Finos (%)',
        'dureza': 'Dureza (Kg/cm2)',
    }