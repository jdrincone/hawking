"""
Visualization module for Okuo reports.

This module provides a suite of high-level Plotly-based functions for generating
various types of charts (gauges, bars, heatmaps, lines, etc.) with a consistent
corporate aesthetic. All functions are designed to be data-agnostic, accepting
pandas DataFrames and column names as primary inputs.
"""

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
from typing import Optional, List, Dict, Union, Tuple, Any, Sequence
import math

# Default color palette for consistent branding across visualizations
DEFAULT_COLORS = ["#1C8074", "#666666", "#E4572E", "#29B6F6", "#FFA726"]


def plot_gauge_grid(
    df: pd.DataFrame,
    group_col: str,
    value_col: str,
    target_ranges: Union[Tuple[float, float], Dict[str, Tuple[float, float]]],
    *,
    title_prefix: str = "",
    unit: str = "",
    label_metric: str = "Compliance",
    label_count: str = "N",
    count_col: Optional[str] = None,
    order_groups: Optional[List[str]] = None,
    gauge_colors: List[str] = ["#EF5350", "#66BB6A", "#FFA726"],
    pct_thresholds: Tuple[float, float] = (70.0, 90.0),
    decimals: int = 1,
    height: int = 400,
    width: int = 1100,
    horizontal_spacing: float = 0.05,
    output_path: Optional[str] = None
) -> go.Figure:
    """
    Generates a horizontal grid of Gauge (speedometer) charts for group comparison.

    Each gauge displays:
    1. A central numerical value (mean of value_col).
    2. A colored arc representing the percentage of values within target_ranges.
    3. Contextual labels for total counts (N) and the reference range.

    Args:
        df (pd.DataFrame): The source data.
        group_col (str): Column to group by (creates one gauge per group).
        value_col (str): Numeric column to calculate the mean and compliance.
        target_ranges (Union[Tuple[float, float], Dict[str, Tuple[float, float]]]): 
            The 'good' range. Can be a single (min, max) for all groups or a 
            mapping of group names to their specific (min, max) tuples.
        title_prefix (str, optional): Text to prepend to the group name in gauge titles.
        unit (str, optional): Unit suffix for the central value (e.g., "%", "kg").
        label_metric (str, optional): Descriptive name for the compliance metric (default: "Compliance").
        label_count (str, optional): Label for the sample size count (default: "N").
        count_col (Optional[str], optional): Column to use for unique counts. If None, uses row count.
        order_groups (Optional[List[str]], optional): Explicit order for gauges from left to right.
        gauge_colors (List[str], optional): Colors for [Low, Mid, High] compliance zones.
        pct_thresholds (Tuple[float, float], optional): Percentage cutoffs for the compliance arc colors.
        decimals (int, optional): Precision for the central displayed value.
        height (int, optional): Total figure height in pixels.
        width (int, optional): Total figure width in pixels.
        horizontal_spacing (float, optional): Spacing between subplots (0 to 1).
        output_path (Optional[str], optional): If provided, saves the result as an HTML file.

    Returns:
        go.Figure: A Plotly Figure object containing the gauge grid.
    """

    # 1. Validation and Sorting
    unique_groups = df[group_col].dropna().unique().astype(str).tolist()
    groups = order_groups if order_groups is not None else unique_groups
    n_groups = len(groups)

    if n_groups == 0:
        return go.Figure()

    # 2. Canvas Configuration
    fig = make_subplots(
        rows=1, cols=n_groups,
        specs=[[{'type': 'indicator'}] * n_groups],
        horizontal_spacing=horizontal_spacing
    )

    # 3. Geometry Calculation for Perfect Alignment
    total_spacing_width = horizontal_spacing * (n_groups - 1)
    subplot_width = (1 - total_spacing_width) / n_groups

    # 4. Group Iteration
    for i, group_name in enumerate(groups):
        idx_col = i + 1  # Plotly uses 1-based indexing for columns

        # A. Get Target Range (Dynamic or Static)
        if isinstance(target_ranges, dict):
            low, high = target_ranges.get(group_name, (0, 0))
        else:
            low, high = target_ranges

        # B. Filtering and Math Calculation
        mask = df[group_col].astype(str) == str(group_name)
        subset = pd.to_numeric(df.loc[mask, value_col], errors="coerce").dropna()

        if subset.empty:
            mean_val = 0.0
            compliance_pct = 0.0
            n_count = 0
        else:
            mean_val = subset.mean()
            # Generic calculation: % of data falling within the range (inclusive)
            compliance_pct = (subset.between(low, high).mean() * 100)
            n_count = df.loc[mask, count_col].nunique() if count_col else len(subset)

        # C. Gauge Construction (Visual)
        fig.add_trace(
            go.Indicator(
                mode="gauge",
                value=compliance_pct,
                title={
                    "text": f"<b>{title_prefix} {group_name}</b>".strip(),
                    "font": {"size": 16, "color": "#333333"},
                },
                gauge={
                    "axis": {"range": [0, 100], "tickformat": ".0f"},
                    "bar": {"color": "#2D3748", "thickness": 0.6},
                    "bgcolor": "white",
                    "steps": [
                        {"range": [0, pct_thresholds[0]], "color": gauge_colors[0]},
                        {"range": [pct_thresholds[0], pct_thresholds[1]], "color": gauge_colors[1]},
                        {"range": [pct_thresholds[1], 100], "color": gauge_colors[2]},
                    ],
                    "threshold": {
                        "line": {"color": "red", "width": 2},
                        "thickness": 0.75,
                        "value": compliance_pct
                    }
                }
            ),
            row=1, col=idx_col
        )

        # D. Geometric Center Calculation (Paper Reference)
        x_center = (i * (subplot_width + horizontal_spacing)) + (subplot_width / 2)

        # E. Text Annotations

        # 1. Main Value (Large number in center)
        fig.add_annotation(
            x=x_center, y=0.45, xref="paper", yref="paper",
            text=f"<b>{mean_val:.{decimals}f}</b><span style='font-size:0.6em'>{unit}</span>",
            showarrow=False,
            font=dict(size=28, color="#1F2937", family="Arial"),
            xanchor="center"
        )

        # 2. Secondary Text (Compliance and Count)
        fig.add_annotation(
            x=x_center, y=0.35, xref="paper", yref="paper",
            text=f"{label_metric}: <b>{compliance_pct:.0f}%</b><br>"
                 f"<span style='font-size:0.9em; color:gray'>{label_count}: {n_count}</span>",
            showarrow=False,
            font=dict(size=13, color="#4B5563"),
            align="center",
            xanchor="center", yanchor="top"
        )

        # 3. Bottom Text (Target Range)
        fig.add_annotation(
            x=x_center, y=0.15, xref="paper", yref="paper",
            text=f"Ref: {low} - {high} {unit}",
            showarrow=False,
            font=dict(size=11, color="#9CA3AF"),
            xanchor="center"
        )

    # 5. Final Adjustments
    fig.update_layout(
        template="plotly_white",
        height=height,
        width=width,
        margin=dict(l=40, r=40, t=80, b=40),
        font=dict(family="Inter, sans-serif"),
    )

    if output_path:
        fig.write_html(output_path)

    return fig


def plot_bar(
        df: pd.DataFrame,
        x_col: str,
        y_col: str,
        # --- Agrupación y Ordenamiento ---
        group_col: Optional[str] = None,
        order_x: Optional[List[str]] = None,  # Orden forzado eje X
        order_groups: Optional[List[str]] = None,  # Orden forzado leyenda
        cat_base:  Optional[str] = None, #Categoria base (Sin Adiflow/Con Paro ...)
        filter_empty: bool = True,  # Ocultar categorías X sin datos

        # --- Modo Compacto (La nueva lógica) ---
        compact_mode: bool = False,  # True = usa cálculo manual de coordenadas (sin huecos)
        cluster_width: float = 0.8,  # Ancho total del grupo (0 a 1)
        bar_width_scale: float = 0.9,  # Ancho relativo de las barras (0 a 1)

        # --- Configuración General ---
        barmode: str = "group",  # 'group' o 'stack' (compact_mode fuerza 'overlay' visualmente)

        # --- Lógica de Deltas ---
        show_delta: bool = False,
        delta_reference_group: Optional[str] = None,
        delta_unit: str = "",

        # --- Datos Extra ---
        hover_data_cols: Optional[List[str]] = None,

        # --- Estética ---
        title: str = "",
        x_title: Optional[str] = None,
        y_title: Optional[str] = None,
        text_format: str = ".1f",
        bar_colors: Union[List[str], Dict[str, str], None] = None,

        height: int = 500,
        width: int = 1000,
        font_family: str = "Inter, Arial, sans-serif",
        output_path: Optional[str] = None
) -> go.Figure:
    """
    Advanced bar chart function supporting clustering, stacking, and custom 'compact' layout.

    Features:
    - Delta calculations between two groups (shown as indicators above bars).
    - Compact mode: Eliminates empty spaces in grouped bars by calculating coordinates manually.
    - Custom ordering for both X-axis and groups.
    - Automatic color mapping for groups.

    Args:
        df (pd.DataFrame): Data source.
        x_col (str): Column for the X-axis categories.
        y_col (str): Numeric column for the bar heights.
        group_col (Optional[str], optional): Column to cluster/stack bars by.
        order_x (Optional[List[str]], optional): Explicit order for X categories.
        order_groups (Optional[List[str]], optional): Explicit order for legend groups.
        cat_base (Optional[str], optional): Reference category for delta comparison.
        filter_empty (bool, optional): If True, hides X categories with no data.
        compact_mode (bool, optional): If True, uses manual coordinate calculation for tight grouping.
        cluster_width (float, optional): Total width utilized by a category cluster (0 to 1).
        bar_width_scale (float, optional): Scaling factor for individual bar widths.
        barmode (str, optional): Plotly bar mode ('group' or 'stack').
        show_delta (bool, optional): If True, shows numerical differences between clusters.
        delta_reference_group (Optional[str], optional): Group to use as '0' for deltas.
        delta_unit (str, optional): Suffix for delta labels.
        hover_data_cols (Optional[List[str]], optional): Additional columns for the hover tooltip.
        title (str, optional): Chart title.
        x_title (Optional[str], optional): X-axis label.
        y_title (Optional[str], optional): Y-axis label.
        text_format (str, optional): D3-format string for bar labels (e.g., ".2f").
        bar_colors (Union[List[str], Dict[str, str], None], optional): List or map of colors for groups.
        height (int, optional): Figure height.
        width (int, optional): Figure width.
        font_family (str, optional): Font for all text elements.
        output_path (Optional[str], optional): Path to save the HTML file.

    Returns:
        go.Figure: The rendered Plotly Figure.
    """

    # Eliminamos filas donde falten datos clave para evitar errores de cálculo
    d = df.dropna(subset=[x_col, y_col]).copy()
    if group_col:
        d = d.dropna(subset=[group_col])
    else:
        group_col = "_dummy_group_"
        d[group_col] = "Total"

    # A. Eje X
    actual_x = d[x_col].unique().astype(str)
    final_x_order = None

    if order_x:
        if filter_empty:
            final_x_order = [x for x in order_x if x in actual_x]
            # Agregar remanentes no listados al final
            others = [x for x in actual_x if x not in final_x_order]
            final_x_order.extend(others)
        else:
            final_x_order = order_x
    else:
        # Si no hay orden forzado, ordenamos alfabéticamente
        final_x_order = sorted(actual_x)

    # B. Grupos
    actual_groups = d[group_col].unique().astype(str)
    final_group_order = None

    if order_groups:
        final_group_order = [g for g in order_groups if g in actual_groups]
        others_g = [g for g in actual_groups if g not in final_group_order]
        final_group_order.extend(others_g)
    else:
        final_group_order = sorted(actual_groups)

    # Ordenar el DataFrame Físicamente
    # Esto es crucial para que el cálculo de "compactación" asigne los offsets en orden correcto
    d['_sort_x'] = pd.Categorical(d[x_col].astype(str), categories=final_x_order, ordered=True)
    d['_sort_g'] = pd.Categorical(d[group_col].astype(str), categories=final_group_order, ordered=True)
    d = d.sort_values(by=['_sort_x', '_sort_g'])

    # Asignación de Colores
    default_colors = DEFAULT_COLORS
    if isinstance(bar_colors, dict):
        color_map = bar_colors
    elif isinstance(bar_colors, list):
        color_map = {g: bar_colors[i % len(bar_colors)] for i, g in enumerate(final_group_order)}
    else:
        color_map = {g: default_colors[i % len(default_colors)] for i, g in enumerate(final_group_order)}

    # -------------------------------------------------------------------------
    #  CÁLCULO DE COORDENADAS (LÓGICA COMPACTA vs ESTÁNDAR)
    # -------------------------------------------------------------------------

    # Mapeo de Categoría X -> Índice Numérico (0, 1, 2...)
    # Esto sirve tanto para el modo compacto como para posicionar los Deltas
    x_to_idx = {cat: i for i, cat in enumerate(final_x_order)}
    d["_px"] = d[x_col].astype(str).map(x_to_idx).astype(float)

    # Filtramos solo lo que vamos a graficar (dentro del orden X)
    d = d.dropna(subset=["_px"])

    if compact_mode and barmode == "group":
        # --- Lógica Compacta ---
        # 1. Rankear grupos dentro de cada X (0, 1, 2...) basándose en el orden ya establecido
        d["_rank"] = d.groupby("_px")[group_col].transform(lambda s: pd.factorize(s, sort=False)[0])

        # 2. Contar cuántos grupos reales hay en cada X
        k_per_x = d.groupby("_px")["_rank"].transform('max') + 1

        # 3. Calcular Ancho y Offset
        # _w = Ancho de la barra individual
        # _off = Desplazamiento desde el centro (0, 1, 2)
        d["_w"] = (cluster_width / k_per_x) * bar_width_scale
        d["_off"] = (d["_rank"] - (k_per_x - 1) / 2) * (cluster_width / k_per_x)
        d["_final_x"] = d["_px"] + d["_off"]

        plot_barmode = "overlay"
    else:
        # --- Lógica Estándar ---
        d["_w"] = None  # Plotly automático
        d["_final_x"] = d[x_col]  # Usamos la categoría directa
        plot_barmode = barmode

    # -------------------------------------------------------------------------
    # CONSTRUCCIÓN DEL GRÁFICO
    # -------------------------------------------------------------------------
    fig = go.Figure()

    for g in final_group_order:
        subset = d[d[group_col].astype(str) == str(g)]
        if subset.empty: continue

        # Hover Data
        customdata = None
        hovertemplate_suffix = ""

        # Preparamos datos para hover: [NombreX, Grupo, ...Extras]
        hover_base_cols = [x_col, group_col]
        extra_cols = hover_data_cols if hover_data_cols else []
        all_hover_cols = hover_base_cols + extra_cols

        # Extraemos valores de manera segura
        try:
            c_values = [subset[c].astype(str).values for c in all_hover_cols]
            customdata = np.stack(c_values, axis=-1)

            # Construir template
            # Index 0 is X, Index 1 is Group
            hovertemplate_suffix = f"<br><b>{group_col}:</b> %{{customdata[1]}}"
            for i, c_name in enumerate(extra_cols):
                # Offset +2 porque 0 y 1 son base
                hovertemplate_suffix += f"<br><b>{c_name}:</b> %{{customdata[{i + 2}]}}"
        except Exception:
            pass

            # Añadir Traza
        fig.add_trace(go.Bar(
            name=str(g),
            x=subset["_final_x"],  # Puede ser numérico (compacto) o categórico (estándar)
            y=subset[y_col],
            width=subset["_w"] if compact_mode else None,
            marker_color=color_map.get(g, "#333333"),
            text=subset[y_col],
            texttemplate=f"%{{text:{text_format}}}",
            textposition="auto" if barmode == "stack" else "outside",
            marker_line_width=0,
            customdata=customdata,
            hovertemplate=(
                f"<b>{x_col}:</b> %{{customdata[0]}}<br>"  # Usamos customdata para el nombre X real
                f"<b>{y_col}:</b> %{{y:{text_format}}}"
                f"{hovertemplate_suffix}<extra></extra>"
            )
        ))

    # -------------------------------------------------------------------------
    # LÓGICA DE DELTAS
    # -------------------------------------------------------------------------
    if show_delta and group_col != "_dummy_group_" and barmode == "group":
        pivot = d.pivot_table(index="_px", columns=group_col, values=y_col, aggfunc='sum')

        cols = pivot.columns.tolist()
        if len(cols) >= 2:
            if cat_base is not None:
                base = cat_base
            else:
                base = delta_reference_group if delta_reference_group in cols else cols[0]
            target = next((c for c in cols if c != base), None)

            if target:
                delta_vals = pivot[target] - pivot[base]
                max_vals = pivot[[base, target]].max(axis=1)

                offset_y = d[y_col].max() * 0.10 if not d.empty else 1

                # Símbolos y Colores
                symbols = np.where(delta_vals > 0, "triangle-up",
                                   np.where(delta_vals < 0, "triangle-down", "circle"))
                colors = np.where(delta_vals > 0, "#1C8074",
                                  np.where(delta_vals < 0, "#E4572E", "gray"))
                texts = [f"{'+' if v > 0 else ''}{v:{text_format}}{delta_unit}"
                         if pd.notna(v) and v != 0 else "" for v in delta_vals]

                # Coordenada X para los Deltas:
                # En modo compacto o estándar, queremos centrar el delta en la categoría.
                # Como pivotamos por "_px" (índice numérico), la X es numérica.
                # Si estamos en modo estándar, Plotly espera categórico en X si las barras son categóricas.

                delta_x = pivot.index  # Esto es 0.0, 1.0, 2.0

                # Si NO es modo compacto, debemos mapear los índices numéricos de vuelta a nombres
                # para que el Scatter coincida con el eje categórico de Plotly
                if not compact_mode:
                    # Invertir el mapa: índice -> nombre
                    idx_to_x = {v: k for k, v in x_to_idx.items()}
                    delta_x = delta_x.map(idx_to_x)

                fig.add_trace(go.Scatter(
                    x=delta_x,
                    y=max_vals + offset_y,
                    mode='markers+text',
                    marker=dict(symbol=symbols, color=colors, size=12, line=dict(width=1, color="black")),
                    text=texts,
                    textposition="top center",
                    textfont=dict(size=11, color="black"),
                    hoverinfo="skip",
                    showlegend=False
                ))

    # Configuración de Eje X para modo Compacto
    xaxis_config = dict(
        title_text=x_title or x_col,
        showline=True, linecolor="black", linewidth=1,
        ticks="outside", tickcolor="black",
        tickfont=dict(color="black"),
        title_font=dict(color="black")
    )

    if compact_mode and barmode == "group":
        # En modo compacto, el eje X es numérico (0, 1, 2).
        # Debemos "disfrazarlo" poniendo los textos manualmente en esas posiciones.
        xaxis_config.update(dict(
            tickmode="array",
            tickvals=list(range(len(final_x_order))),
            ticktext=final_x_order,
            range=[-0.5, len(final_x_order) - 0.5]  # Márgenes limpios
        ))
    else:
        # Modo estándar: forzamos el orden de categorías
        xaxis_config.update(dict(
            categoryorder='array',
            categoryarray=final_x_order
        ))

    fig.update_layout(
        title={'text': f"<b>{title}<b>", 'y': 0.95, 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top'},
        title_font=dict(color="black"),
        barmode=plot_barmode,
        bargap=0.4,
        font=dict(family=font_family, color="black"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=40, r=40, t=80, b=40),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
            title_text="", font=dict(color="black"),
            bgcolor="rgba(255,255,255,0.8)", bordercolor="black", borderwidth=1
        ),
        hoverlabel=dict(font_color="black", bgcolor="white", bordercolor="black"),
        height=height, width=width,
        xaxis=xaxis_config,
        yaxis=dict(
            title_text=y_title or y_col,
            showline=True, linecolor="black", linewidth=1,
            showgrid=True, gridcolor="rgba(0,0,0,0.1)", zeroline=True, zerolinecolor="rgba(0,0,0,0.2)",
            tickfont=dict(color="black"), title_font=dict(color="black")
        )
    )

    if output_path:
        fig.write_html(output_path, include_plotlyjs="cdn")

    return fig




def plot_heatmap(
        df: pd.DataFrame,
        x_col: str,
        y_col: str,
        value_col: str,
        # --- Métrica Secundaria (Opcional) ---
        secondary_col: Optional[str] = None,
        secondary_aggfunc: str = "sum",

        # --- Agregación y Orden ---
        aggfunc: str = "mean",
        x_order: Optional[List[str]] = None,
        y_order: Optional[List[str]] = None,

        # --- Formato de Valores ---
        value_unit: str = "",  # Ej: "%", " $", " Ton"
        unit_position: str = "suffix",  # "suffix" (10 %) o "prefix" ($ 10)
        decimals_value: int = 2,
        decimals_secondary: int = 1,
        secondary_prefix: str = "",  # Ej: "N=" para que se vea "N=150"

        # --- Colores ---
        colorscale: Union[str, List[Tuple[float, str]]] = DEFAULT_COLORS,  # Nombre (str) o Lista Personalizada
        zmin: Optional[float] = None,
        zmax: Optional[float] = None,
        zmid: Optional[float] = None,  # Para centrar la escala (ej: en 0 para desviaciones)

        # --- Diseño y Textos ---
        title: str = "",
        show_secondary_labels: bool = True,
        secondary_position: str = "bottom",  # 'top', 'bottom', 'left', 'right'
        font_color: str = "black",  # Color base de los textos
        center_font_size: int = 14,
        secondary_font_size: int = 10,
        width: int = 800,
        height: int = 500,
        transparent_bg: bool = True,
        output_path: Optional[str] = None
) -> go.Figure:
    """
    Generates a high-detail Heatmap with optional double labeling per cell.

    Ideal for matrix comparisons (e.g., Performance vs Category). Supports showing
    a primary value (centered) and a secondary helper value (e.g., sample size)
    in a corner of each cell.

    Args:
        df (pd.DataFrame): Input data in tidy format.
        x_col (str): Column for horizontal axis categories.
        y_col (str): Column for vertical axis categories.
        value_col (str): Column for the primary metric (color intensity and large label).
        secondary_col (Optional[str], optional): Metric for the minor label (e.g., 'N').
        secondary_aggfunc (str, optional): Aggregation for the secondary metric.
        aggfunc (str, optional): Aggregation for the primary metric ('mean', 'sum', etc.).
        x_order (Optional[List[str]], optional): Sequence for X categories.
        y_order (Optional[List[str]], optional): Sequence for Y categories.
        value_unit (str, optional): Unit suffix/prefix for the primary label.
        unit_position (str, optional): Where to place the unit ('suffix' or 'prefix').
        decimals_value (int, optional): Decimal precision for primary label.
        decimals_secondary (int, optional): Decimal precision for secondary label.
        secondary_prefix (str, optional): Prefix for secondary label (e.g., "n=").
        colorscale (Union[str, list], optional): Plotly colorscale name or custom list.
        zmin/zmax/zmid (Optional[float]): Manual overrides for the color intensity range.
        title (str, optional): Heatmap title.
        show_secondary_labels (bool, optional): Toggle visibility of minor labels.
        secondary_position (str, optional): Placement of minor labels ('top', 'bottom', etc.).
        font_color (str, optional): Color for all labels.
        center_font_size (int, optional): Size of the main cell label.
        secondary_font_size (int, optional): Size of the minor cell label.
        width (int, optional): Figure width.
        height (int, optional): Figure height.
        transparent_bg (bool, optional): Use a clear background.
        output_path (Optional[str], optional): HTML save destination.

    Returns:
        go.Figure: The rendered Heatmap.
    """

    # --- 1. Helper interno para formateo de números ---
    def _fmt_value(v: float, decimals: int, unit: str, pos: str) -> str:
        if pd.isna(v): return ""
        formatted_num = f"{v:.{decimals}f}"
        if not unit: return formatted_num
        # Espacio fino (\u202F) para sufijos, nada para prefijos
        sep = "\u202F" if pos == "suffix" else ""
        if pos == "prefix": return f"{unit}{formatted_num}"
        return f"{formatted_num}{sep}{unit}"

    # --- 2. Creación de Tablas Dinámicas (Pivot) ---
    # Tabla Principal (Color y Texto Central)
    pt_val = pd.pivot_table(df, index=y_col, columns=x_col, values=value_col, aggfunc=aggfunc)

    # Tabla Secundaria (Texto pequeño), si aplica
    if secondary_col:
        pt_sec = pd.pivot_table(df, index=y_col, columns=x_col, values=secondary_col, aggfunc=secondary_aggfunc)
    else:
        # Crear estructura vacía idéntica para evitar errores
        #pt_sec = pd.DataFrame(index=pt_val.index, columns=pt_val.columns)
        pt_sec = pd.pivot_table(df, index=y_col, columns=x_col, values=secondary_col, aggfunc=secondary_aggfunc)

    # --- 3. Ordenamiento de Ejes ---
    # Si no se da orden, se usa el alfanumérico o el existente
    if x_order is None: x_order = sorted(pt_val.columns.astype(str).tolist())
    if y_order is None: y_order = sorted(pt_val.index.astype(str).tolist())

    # Reindexar para asegurar que la matriz de datos coincida con el orden visual
    pt_val = pt_val.reindex(index=y_order, columns=x_order)
    pt_sec = pt_sec.reindex(index=y_order, columns=x_order)

    # Convertir a matrices Numpy para Plotly
    z_main = pt_val.values.astype(float)
    z_sec = pt_sec.values.astype(float)

    # --- 4. Generación de Matrices de Texto ---
    # Texto Central
    text_center = np.empty(z_main.shape, dtype=object)
    for i in range(z_main.shape[0]):
        for j in range(z_main.shape[1]):
            text_center[i, j] = _fmt_value(z_main[i, j], decimals_value, value_unit, unit_position)

    # Texto Secundario
    text_secondary = np.empty(z_sec.shape, dtype=object)
    if secondary_col and show_secondary_labels:
        for i in range(z_sec.shape[0]):
            for j in range(z_sec.shape[1]):
                val = z_sec[i, j]
                if pd.notna(val):
                    text_secondary[i, j] = f"{secondary_prefix}{val:.{decimals_secondary}f}"
                else:
                    text_secondary[i, j] = ""
    else:
        text_secondary[:] = ""

    # --- 5. Configuración de Escala de Color ---
    # Calcular min/max automáticos si no se proveen, ignorando NaNs
    if zmin is None and np.nanmin(z_main) is not np.nan: zmin = np.nanmin(z_main)
    if zmax is None and np.nanmax(z_main) is not np.nan: zmax = np.nanmax(z_main)

    # --- 6. Construcción de la Figura Base (Heatmap) ---
    # Template para el tooltip (hover)
    hover_lines = [
        f"<b>{x_col}:</b> %{{x}}",
        f"<b>{y_col}:</b> %{{y}}",
        f"<b>{value_col}:</b> %{{text}}",  # Usa el texto formateado con unidad
    ]
    if secondary_col:
        hover_lines.append(f"<b>{secondary_col}:</b> %{{customdata:.{decimals_secondary}f}}")

    fig = go.Figure(data=go.Heatmap(
        z=z_main,
        x=x_order,
        y=y_order,
        colorscale=colorscale,
        zmin=zmin, zmax=zmax, zmid=zmid,
        text=text_center,  # Se usa para display en hover
        customdata=z_sec,  # Datos extra para hover
        hovertemplate="<br>".join(hover_lines) + "<extra></extra>",
        colorbar=dict(
            title=value_unit if value_unit else value_col,
            title_font=dict(color=font_color),  # Título de la barra de color
            tickfont=dict(color=font_color),  # Números de la barra de color
            thickness=15,
            outlinecolor=font_color,
            outlinewidth=1
        ),
        xgap=1, ygap=1  # Pequeño espacio blanco entre celdas para limpieza visual
    ))

    # --- 7. Añadir Anotaciones de Texto (Layers) ---
    annotations = []

    # Mapa de desplazamientos para el texto secundario
    pos_map = {
        "top": (0, 15, "center", "bottom"),
        "bottom": (0, -15, "center", "top"),
        "left": (-20, 0, "right", "middle"),
        "right": (20, 0, "left", "middle")
    }
    sec_x_s, sec_y_s, sec_x_anc, sec_y_anc = pos_map.get(secondary_position, (0, -15, "center", "top"))

    # Iterar sobre cada celda para poner los textos
    for i, yy in enumerate(y_order):
        for j, xx in enumerate(x_order):
            # A. Texto Central
            if text_center[i, j]:
                annotations.append(dict(
                    x=xx, y=yy,
                    text=str(text_center[i, j]),
                    showarrow=False,
                    font=dict(color=font_color, size=center_font_size, family="Arial")
                ))

            # B. Texto Secundario
            if text_secondary[i, j]:
                annotations.append(dict(
                    x=xx, y=yy,
                    text=str(text_secondary[i, j]),
                    showarrow=False,
                    xshift=sec_x_s, yshift=sec_y_s,  
                    xanchor=sec_x_anc, yanchor=sec_y_anc,
                    font=dict(color=font_color, size=secondary_font_size)
                ))

    fig.update_layout(annotations=annotations)

    # --- 8. Configuración Final del Layout y Ejes (La corrección clave) ---
    axis_style = dict(
        showline=True,
        linecolor=font_color,  
        linewidth=1.5,
        mirror=True,
        tickfont=dict(color=font_color, size=11), 
        title_font=dict(color=font_color), 
        ticks="outside",
        ticklen=5,
        gridcolor="rgba(0,0,0,0)" 
    )

    fig.update_xaxes(**axis_style, title=x_col, side="bottom")
    fig.update_yaxes(**axis_style, title=y_col, autorange="reversed")

    layout_updates = dict(
        title=dict(
            text=f"<b>{title}<b>",
            x=0.5,
            font=dict(color=font_color, size=18)
        ),
        margin=dict(l=60, r=60, t=80, b=60),
        width=width,
        height=height,
        font=dict(family="Inter, Arial, sans-serif", color=font_color),
    )

    if transparent_bg:
        layout_updates.update(dict(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)"))

    fig.update_layout(**layout_updates)
    
    if output_path:
        fig.write_html(output_path, include_plotlyjs="cdn")

    return fig



def plot_line(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    group_col: Optional[str] = None,
    order_x: Optional[List[str]] = None,
    order_groups: Optional[List[str]] = None,
    
    # --- Dual Axis Logic ---
    secondary_y_col: Optional[str] = None,
    secondary_y_title: Optional[str] = None,

    # --- Line Specifics ---
    line_width: int = 3,
    marker_size: int = 8,
    show_markers: bool = True,
    interpolation: str = "linear",

    # --- Extra Data ---
    hover_data_cols: Optional[List[str]] = None,

    # --- Esthetics ---
    title: str = "",
    x_title: Optional[str] = None,
    y_title: Optional[str] = None,
    text_format: str = ".1f",
    y_as_pct: bool = False,
    line_colors: Union[List[str], Dict[str, str], None] = None,
    font_family: str = "Inter, Arial, sans-serif",
    height: int = 500,
    width: int = 1000,
    show_legend: bool = True,
    output_path: Optional[str] = None
) -> go.Figure:
    """
    Renders a multi-series line chart with support for dual Y-axes.

    Args:
        df (pd.DataFrame): Data source.
        x_col (str): Column for the X-axis (usually time or categories).
        y_col (str): Column for the primary numeric data.
        group_col (Optional[str], optional): Column to split into multiple lines.
        order_x (Optional[List[str]], optional): Explicit order for X-axis categories.
        order_groups (Optional[List[str]], optional): Explicit order for legend items.
        secondary_y_col (Optional[str], optional): Name of a specific group to plot on a right Y-axis.
        secondary_y_title (Optional[str], optional): Title for the right Y-axis.
        line_width (int, optional): Thickness of the lines.
        marker_size (int, optional): Diameter of data point markers.
        show_markers (bool, optional): Toggle markers at data points.
        interpolation (str, optional): Line shape ('linear', 'spline', etc.).
        hover_data_cols (Optional[List[str]], optional): Additional info for tooltips.
        title (str, optional): Chart title.
        x_title (Optional[str], optional): X-axis label.
        y_title (Optional[str], optional): Primary Y-axis label.
        text_format (str, optional): Formatting for numeric values.
        y_as_pct (bool, optional): If True, formats the Y-axis as percentages.
        line_colors (Union[List[str], Dict[str, str], None], optional): Custom line colors.
        font_family (str, optional): Font family for the chart.
        height (int, optional): Figure height.
        width (int, optional): Figure width.
        show_legend (bool, optional): Toggle the legend.
        output_path (Optional[str], optional): Destination for HTML export.

    Returns:
        go.Figure: The line chart figure.
    """
    
    d = df.copy()

    # --- Sorting Logic ---
    if order_x:
        d[x_col] = pd.Categorical(d[x_col], categories=order_x, ordered=True)
    if group_col and order_groups:
        d[group_col] = pd.Categorical(d[group_col], categories=order_groups, ordered=True)

    sort_criteria = [x_col]
    if group_col: sort_criteria.append(group_col)
    d = d.sort_values(by=sort_criteria)

    # --- Color Configuration ---
    if group_col:
        unique_groups = d[group_col].cat.categories if hasattr(d[group_col], 'cat') else sorted(d[group_col].unique())
        if isinstance(line_colors, dict): color_map = line_colors
        else:
            colors = line_colors if isinstance(line_colors, list) else DEFAULT_COLORS
            color_map = {g: colors[i % len(colors)] for i, g in enumerate(unique_groups)}
    else:
        group_col = "_dummy_group_"
        d[group_col] = y_col
        unique_groups = [y_col]
        color_map = {y_col: (line_colors[0] if isinstance(line_colors, list) and line_colors else DEFAULT_COLORS[0])}

    # --- INITIALIZE FIGURE WITH SECONDARY Y SUPPORT ---
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    groups_to_plot = [g for g in unique_groups if g in d[group_col].values]

    for g in groups_to_plot:
        subset = d[d[group_col] == g]
        if subset.empty: continue

        # Determinamos si este grupo va al eje secundario
        is_secondary = (str(g) == str(secondary_y_col))

        customdata = None
        hovertemplate_extras = ""
        if hover_data_cols:
            custom_values = [subset[c].values for c in hover_data_cols]
            customdata = np.stack(custom_values, axis=-1)
            for i, col_name in enumerate(hover_data_cols):
                hovertemplate_extras += f"<br><b>{col_name}:</b> %{{customdata[{i}]}}"

        mode = "lines+markers" if show_markers else "lines"

        fig.add_trace(
            go.Scatter(
                name=str(g),
                x=subset[x_col],
                y=subset[y_col],
                mode=mode,
                line=dict(color=color_map.get(g, "#666666"), width=line_width, shape=interpolation),
                marker=dict(size=marker_size, color=color_map.get(g, "#666666")),
                customdata=customdata,
                hovertemplate=(
                    f"<b>{x_col}:</b> %{{x}}<br>"
                    f"<b>{str(g)}:</b> %{{y:{text_format}}}<br>"
                    f"{hovertemplate_extras}<extra></extra>"
                )
            ),
            secondary_y=is_secondary
        )

    # --- Layout Update ---
    fig.update_layout(
        title={'text': f"<b>{title}<b>", 'y': 0.95, 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top'},
        font=dict(family=font_family, color="black"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=40, r=40, t=80, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1) if show_legend else dict(visible=False),
        height=height, width=width,
        hovermode="x unified"
    )

    fig.update_xaxes(title_text=x_title or x_col, showline=True, linecolor="black", linewidth=1, ticks="outside")
    
    # Eje Y Principal
    fig.update_yaxes(
        title_text=y_title or y_col, showline=True, linecolor="black", linewidth=1,
        showgrid=True, gridcolor="rgba(0,0,0,0.1)", tickformat=".1%" if y_as_pct else None,
        secondary_y=False
    )

    # Eje Y Secundario (solo se activa si existe secondary_y_col)
    if secondary_y_col:
        fig.update_yaxes(
            title_text=secondary_y_title or secondary_y_col,
            showline=True, linecolor="black", linewidth=1,
            showgrid=False, # No mostrar grilla doble para evitar confusión
            secondary_y=True
        )

    if output_path:
        fig.write_html(output_path, include_plotlyjs="cdn")

    return fig


def plot_statistical_strip(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    category_order: Optional[List[str]] = None,
    color_map: Optional[Dict[str, str]] = None,
    show_boxplot: bool = True,
    show_mean_ci: bool = True,
    show_global_mean: bool = True,
    show_counts: bool = True,
    title: str = "",
    x_title: Optional[str] = None,
    y_title: Optional[str] = None,
    point_opacity: float = 0.6,
    point_size: int = 7,
    box_opacity: float = 0.25,
    height: int = 600,
    width: int = 1000,
    filename: Optional[str] = None
) -> go.Figure:
    """
    Advanced statistical chart combining raw data points, distributions, and averages.

    Layer structure:
    1. Strip plot: Raw data points (showing individual variance).
    2. Boxplot: IQR distribution in the background.
    3. Error bars: Mean value with a 95% Confidence Interval (IC95%).
    4. Reference lines: Global average and sample size indicators (n=...).

    Args:
        df (pd.DataFrame): Input dataset.
        x_col (str): Categorical column for grouping.
        y_col (str): Numeric column for statistical analysis.
        category_order (Optional[List[str]], optional): Horizontal ordering of categories.
        color_map (Optional[Dict[str, str]], optional): Group-specific colors.
        show_boxplot (bool, optional): Toggle IQR box visibility.
        show_mean_ci (bool, optional): Toggle Mean + 95% CI diamond/bars.
        show_global_mean (bool, optional): Toggle dotted global mean line.
        show_counts (bool, optional): Toggle 'n=...' labels at the bottom.
        title (str, optional): Chart title.
        x_title/y_title (Optional[str]): Axis labels.
        point_opacity/point_size/box_opacity (float/int): Tuning for visual clusters.
        height/width (int): Figure dimensions.
        filename (Optional[str], optional): HTML save destination.

    Returns:
        go.Figure: The statistical summary figure.
    """
    
    # 1. Preparación de datos
    d = df.copy()
    
    # Si no se pasa orden, usar el natural o alfabético
    if category_order:
        d[x_col] = pd.Categorical(d[x_col], categories=category_order, ordered=True)
        # Filtrar datos que no estén en las categorías especificadas (limpieza)
        d = d[d[x_col].isin(category_order)]
    else:
        # Obtener categorías únicas ordenadas si existen
        if hasattr(d[x_col], 'cat'):
            category_order = list(d[x_col].cat.categories)
        else:
            category_order = sorted(d[x_col].unique().astype(str))

    d = d.sort_values(by=x_col)

    # Configuración de colores por defecto si no se pasa mapa
    if color_map is None:
        colors = px.colors.qualitative.Plotly
        color_map = {cat: colors[i % len(colors)] for i, cat in enumerate(category_order)}

    # 2. Cálculo de Estadísticas
    # observed=True maneja categorías vacías correctamente en pandas nuevos
    g = d.groupby(x_col, observed=True)[y_col]
    stats = g.agg(["count", "mean", "median", "std"]).reindex(category_order)
    
    # Error Estándar e Intervalo de Confianza (95%)
    stats["sem"] = stats["std"] / np.sqrt(stats["count"].clip(lower=1))
    stats["ci95"] = 1.96 * stats["sem"]
    
    mean_global = d[y_col].mean()

    # 3. Construcción del Gráfico
    
    # Capa 1: Strip Plot (Puntos)
    # Usamos px.strip como base porque maneja muy bien el 'jitter' (dispersión lateral)
    fig = px.strip(
        d,
        x=x_col, 
        y=y_col, 
        color=x_col,
        category_orders={x_col: category_order},
        color_discrete_map=color_map,
        template="plotly_white"
    )
    
    # Ajustar estilo de los puntos
    fig.update_traces(
        marker=dict(size=point_size, opacity=point_opacity, line=dict(width=0)),
        hovertemplate=f"<b>{y_col}</b>: %{{y:.2f}}<br>{x_col}: %{{x}}<extra></extra>"
    )

    # Capa 2: Boxplots (Fondo)
    if show_boxplot:
        for cat in category_order:
            sub = d.loc[d[x_col] == cat, y_col]
            if sub.empty: continue
            
            c_color = color_map.get(cat, "#333333")
            
            fig.add_trace(
                go.Box(
                    x=[cat] * len(sub), 
                    y=sub,
                    name=f"{cat} (IQR)",
                    marker_color=c_color,
                    line=dict(color=c_color),
                    fillcolor="rgba(0,0,0,0)",  # Transparente, solo contorno
                    opacity=box_opacity,
                    boxmean=False,              # No mostrar media del box (usamos la nuestra)
                    showlegend=False,
                    hoverinfo="skip"            # Evitar ruido en el hover
                )
            )

    # Capa 3: Media ± IC95% (Top)
    if show_mean_ci:
        fig.add_trace(
            go.Scatter(
                x=stats.index.tolist(),
                y=stats["mean"],
                mode="markers",
                name="Media ± IC95%",
                marker=dict(symbol="diamond", size=point_size + 5, color="black"),
                error_y=dict(
                    type="data", 
                    array=stats["ci95"], 
                    thickness=1.5, 
                    width=6, 
                    color="black"
                ),
                hovertemplate="<b>Media</b>: %{y:.2f}<br>IC95%: ±%{customdata:.2f}<extra></extra>",
                customdata=stats["ci95"].values,
                showlegend=True
            )
        )

    # Capa 4: Línea Global
    if show_global_mean:
        fig.add_hline(
            y=mean_global,
            line=dict(color="black", width=1, dash="dot"),
            annotation_text=f"Promedio Global: {mean_global:.2f}",
            annotation_position="top right",
            annotation_font_color="black"
        )

    # Capa 5: Conteos (n=...)
    if show_counts:
        y_min_chart = d[y_col].min()
        # Calculamos un pequeño offset visual hacia abajo (5% del rango)
        y_range = d[y_col].max() - y_min_chart
        y_pos_n = y_min_chart - (y_range * 0.05)

        for cat in category_order:
            val_n = stats.loc[cat, "count"]
            if pd.isna(val_n) or val_n == 0: continue
            
            n_text = int(val_n)
            fig.add_annotation(
                x=cat, 
                y=y_min_chart, # Anclado al mínimo
                yshift=-25,    # Desplazado en píxeles hacia abajo
                showarrow=False,
                text=f"n={n_text}",
                font=dict(color="black", size=11)
            )

    # 4. Layout Final
    fig.update_layout(
        title={'text': title, 'y':0.95, 'x':0.5, 'xanchor': 'center', 'yanchor': 'top'},
        xaxis_title=x_title or x_col,
        yaxis_title=y_title or y_col,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="black", family="Inter, Arial, sans-serif"),
        margin=dict(t=80, r=30, b=80, l=60),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        height=height,
        width=width
    )
    
    fig.update_xaxes(showline=True, linecolor="black", mirror=True)
    fig.update_yaxes(showline=True, linecolor="black", mirror=True, gridcolor="rgba(0,0,0,0.1)")

    if filename:
        fig.write_html(filename, include_plotlyjs="cdn", full_html=True)

    return fig


def plot_pie(
    df_prepared: pd.DataFrame,
    label_col: str = "label",
    value_col: str = "slice_value",
    hover_col: str = "hover_html",
    title: str = "",
    output_path: Optional[str] = None,
    width: int = 800,
    height: int = 500,
    corporate_colors: Optional[List[str]] = None
) -> go.Figure:
    """
    Renders a standard corporate-styled Pie (Donut) chart.

    Args:
        df_prepared (pd.DataFrame): Dataframe already aggregated and formatted 
            (usually by a specific data preparation utility).
        label_col (str): Column for slice names.
        value_col (str): Column for slice sizes.
        hover_col (str): Column containing HTML strings for tooltips.
        title (str, optional): Chart title.
        output_path (Optional[str], optional): HTML export path.
        width/height (int): Figure dimensions.
        corporate_colors (Optional[List[str]], optional): Color sequence for slices.

    Returns:
        go.Figure: The pie chart figure.
    """

    # Paleta por defecto si no se pasa una
    if not corporate_colors:
        corporate_colors = DEFAULT_COLORS

    # Asegurar que alcanzan los colores repitiendo la lista
    colors = corporate_colors * (len(df_prepared) // len(corporate_colors) + 1)

    fig = go.Figure(
        data=[go.Pie(
            labels=df_prepared[label_col],
            values=df_prepared[value_col],

            # Estilo Visual
            marker=dict(colors=colors[:len(df_prepared)]),
            textinfo="label+percent",
            textposition="inside",
            sort=False, # Asumimos que el DF ya viene ordenado por prepare_pie_data

            # Hover Info (HTML pre-construido)
            hovertext=df_prepared[hover_col],
            hovertemplate="%{hovertext}<br>Participación: %{percent}<extra></extra>"
        )]
    )

    # Layout Corporativo Estándar
    fig.update_layout(
        title={"text": f"<b>{title}<b>", "x": 0.5, "xanchor": "center", "font": dict(color="black", size=20)},
        font=dict(family="Inter, Arial, sans-serif", size=12, color="black"),
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=60, b=20),
        width=width,
        height=height
    )

    if output_path:
        fig.write_html(output_path, include_plotlyjs="cdn")

    return fig



def plot_time_heatmap(
        df: pd.DataFrame,
        x_col: str,
        y_col: str,
        value_col: str,  # Columna numérica (tiempo)
        input_unit: str = "hours",  # 'seconds', 'minutes', 'hours'

        # --- Agregación y Orden ---
        aggfunc: str = "mean",
        x_order: Optional[List[str]] = None,
        y_order: Optional[List[str]] = None,

        # --- Colores ---
        colorscale: Union[str, List[Tuple[float, str]]] = DEFAULT_COLORS,
        zmin: Optional[float] = None,
        zmax: Optional[float] = None,
        zmid: Optional[float] = None,

        # --- Diseño y Textos ---
        title: str = "",
        font_color: str = "black",
        center_font_size: int = 12,
        width: int = 800,
        height: int = 500,
        transparent_bg: bool = True,
        output_path: Optional[str] = None
) -> go.Figure:
    """
    Temporal Heatmap where numeric values are rendered as HH:MM:SS timestamps.

    Args:
        df (pd.DataFrame): Tidy dataset.
        x_col (str): Header category.
        y_col (str): Legend/Row category.
        value_col (str): Numeric column representing durations.
        input_unit (str): Time unit of value_col ('seconds', 'minutes', 'hours').
        aggfunc (str): How to consolidate values in each cell.
        x_order (Optional[List[str]]): Explicit sorting for X-axis.
        y_order (Optional[List[str]]): Explicit sorting for Y-axis.
        colorscale (Union[str, list]): Visual color map.
        zmin (Optional[float]): Minimum value for the color scale.
        zmax (Optional[float]): Maximum value for the color scale.
        zmid (Optional[float]): Midpoint value for the color scale.
        title (str): Header text.
        font_color (str): Text color for labels.
        center_font_size (int): Size of the timestamp in cells.
        width (int): Canvas width.
        height (int): Canvas height.
        transparent_bg (bool): Toggle background transparency.
        output_path (Optional[str]): HTML save path.

    Returns:
        go.Figure: The time-formatted heatmap.
    """

    # --- 1. Helper: Conversor de Tiempo ---
    def _to_hms(val: float, unit: str) -> str:
        if not np.isfinite(val):
            return ""

        # Convertir todo a segundos primero
        if unit == "seconds":
            total_seconds = val
        elif unit == "minutes":
            total_seconds = val * 60
        elif unit == "hours":
            total_seconds = val * 3600
        else:
            raise ValueError("input_unit debe ser 'seconds', 'minutes' u 'hours'")

        # Calcular HH:MM:SS
        total_seconds = int(round(total_seconds))
        sign = "-" if total_seconds < 0 else ""
        total_seconds = abs(total_seconds)

        m, s = divmod(total_seconds, 60)
        h, m = divmod(m, 60)

        return f"{sign}{h:02d}:{m:02d}:{s:02d}"

    # --- 2. Preparación de Datos (Pivot) ---
    pt = pd.pivot_table(df, index=y_col, columns=x_col, values=value_col, aggfunc=aggfunc)

    # --- 3. Ordenamiento ---
    if x_order is None: x_order = sorted(pt.columns.astype(str).tolist())
    if y_order is None: y_order = sorted(pt.index.astype(str).tolist())

    pt = pt.reindex(index=y_order, columns=x_order)

    # Matriz Z (Valores numéricos para el color)
    z = pt.values.astype(float)

    # Matriz de Texto (HH:MM:SS para mostrar)
    text_matrix = np.empty(z.shape, dtype=object)
    for i in range(z.shape[0]):
        for j in range(z.shape[1]):
            text_matrix[i, j] = _to_hms(z[i, j], input_unit)

    # --- 4. Configuración de Escala de Color ---
    if zmin is None and np.nanmin(z) is not np.nan: zmin = np.nanmin(z)
    if zmax is None and np.nanmax(z) is not np.nan: zmax = np.nanmax(z)

    # --- 5. Configuración de la Barra de Color (Colorbar) ---
    # Generamos ticks manuales para que la leyenda muestre HH:MM:SS en lugar de números
    cb_kwargs = dict(
        title=input_unit.capitalize() if not title else "",
        title_font=dict(color=font_color),
        tickfont=dict(color=font_color),
        outlinecolor=font_color,
        outlinewidth=1,
        thickness=15
    )

    if zmin is not None and zmax is not None and np.isfinite([zmin, zmax]).all():
        # Crear 5 ticks distribuidos uniformemente
        tick_vals = np.linspace(zmin, zmax, 5)
        tick_text = [_to_hms(t, input_unit) for t in tick_vals]
        cb_kwargs.update(tickvals=tick_vals.tolist(), ticktext=tick_text)

    # --- 6. Creación de Figura ---
    fig = go.Figure(go.Heatmap(
        z=z,
        x=x_order,
        y=y_order,
        colorscale=colorscale,
        zmin=zmin, zmax=zmax, zmid=zmid,
        colorbar=cb_kwargs,
        customdata=text_matrix,  # Pasamos el texto formateado
        # El hover muestra las coordenadas y el tiempo formateado
        hovertemplate=(
            f"<b>{x_col}:</b> %{{x}}<br>"
            f"<b>{y_col}:</b> %{{y}}<br>"
            f"<b>Tiempo:</b> %{{customdata}}<extra></extra>"
        ),
        xgap=1, ygap=1
    ))

    # --- 7. Anotaciones (Texto Central) ---
    annotations = []
    for i, yy in enumerate(y_order):
        for j, xx in enumerate(x_order):
            txt = text_matrix[i, j]
            if txt:
                annotations.append(dict(
                    x=xx, y=yy,
                    text=txt,
                    showarrow=False,
                    font=dict(color=font_color, size=center_font_size, family="Arial")
                ))

    fig.update_layout(annotations=annotations)

    # --- 8. Estilo y Layout (Corporate/Clean) ---
    axis_style = dict(
        showline=True, linecolor=font_color, linewidth=1.5, mirror=True,
        tickfont=dict(color=font_color),
        title_font=dict(color=font_color),
        ticks="outside", ticklen=5,
        gridcolor="rgba(0,0,0,0)"
    )

    fig.update_xaxes(**axis_style, title=x_col, side="bottom")
    fig.update_yaxes(**axis_style, title=y_col, autorange="reversed")

    layout_args = dict(
        title=dict(text=f"<b>{title}<b>", x=0.5, font=dict(color=font_color, size=18)),
        margin=dict(l=60, r=60, t=80, b=60),
        width=width,
        height=height,
        font=dict(family="Inter, Arial, sans-serif", color=font_color),
    )

    if transparent_bg:
        layout_args.update(dict(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)"))

    fig.update_layout(**layout_args)

    if output_path:
        fig.write_html(output_path, include_plotlyjs="cdn")

    return fig


def plot_scatter(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    group_col: Optional[str] = None,
    size_col: Optional[str] = None,
    order_groups: Optional[List[str]] = None,
    hover_data_cols: Optional[List[str]] = None,
    title: str = "",
    x_title: Optional[str] = None,
    y_title: Optional[str] = None,
    text_format: str = ".1f",
    marker_size: int = 10,
    marker_opacity: float = 0.8,
    marker_colors: Union[List[str], Dict[str, str], None] = None,
    height: int = 600,
    width: int = 1000,
    font_family: str = "Inter, Arial, sans-serif",
    output_path: Optional[str] = None
) -> go.Figure:
    """
    Renders an interactive Scatter or Bubble chart.

    Args:
        df (pd.DataFrame): Data source.
        x_col (str): Column for horizontal positioning.
        y_col (str): Column for vertical positioning.
        group_col (Optional[str], optional): Column for color grouping.
        size_col (Optional[str], optional): Column for bubble size (enables Bubble Chart mode).
        order_groups (Optional[List[str]], optional): Sequence for colors/legend.
        hover_data_cols (Optional[List[str]], optional): Additional tooltips.
        title (str): Chart title.
        x_title (Optional[str]): X-axis label.
        y_title (Optional[str]): Y-axis label.
        text_format (str): Numeric formatting for tooltips.
        marker_size (int): Base size for points (used if size_col is None).
        marker_opacity (float): Transparency (0 to 1).
        marker_colors (Union[List[str], Dict[str, str], None]): Selection of colors.
        height (int): Figure height.
        width (int): Figure width.
        font_family (str): UI font.
        output_path (Optional[str]): HTML file path.

    Returns:
        go.Figure: The scatter/bubble figure.
    """
    
    # 1. Copia y Limpieza
    cols_to_check = [x_col, y_col]
    if group_col: cols_to_check.append(group_col)
    d = df.dropna(subset=cols_to_check).copy()

    if not group_col:
        group_col = "_dummy_group_"
        d[group_col] = "Total"

    # 2. Orden de Grupos
    actual_groups = d[group_col].unique().astype(str)
    if order_groups:
        final_group_order = [g for g in order_groups if g in actual_groups]
        final_group_order.extend([g for g in actual_groups if g not in final_group_order])
    else:
        final_group_order = sorted(actual_groups)

    # 3. Colores
    default_colors = DEFAULT_COLORS
    if isinstance(marker_colors, dict):
        color_map = marker_colors
    elif isinstance(marker_colors, list):
        color_map = {g: marker_colors[i % len(marker_colors)] for i, g in enumerate(final_group_order)}
    else:
        color_map = {g: default_colors[i % len(default_colors)] for i, g in enumerate(final_group_order)}

    fig = go.Figure()

    for g in final_group_order:
        subset = d[d[group_col].astype(str) == str(g)].copy()
        if subset.empty: continue

        # --- Lógica de Hover Data (CORREGIDA) ---
        # Definir columnas base: [0]=X, [1]=Y, [2]=Grupo
        hover_base_cols = [x_col, y_col, group_col]
        extra_cols = hover_data_cols if hover_data_cols else []
        all_hover_cols = hover_base_cols + extra_cols

        # Pre-formatear valores numéricos a strings AQUÍ para evitar errores en Plotly
        c_values = []
        for col_name in all_hover_cols:
            vals = subset[col_name]
            
            # Si es la columna Y, aplicamos el formato numérico deseado
            if col_name == y_col and pd.api.types.is_numeric_dtype(vals):
                formatted = vals.map(lambda x: f"{x:{text_format}}")
                c_values.append(formatted.values)
            # Si es otra columna numérica (ej. X o extras), formateamos genérico o dejamos tal cual
            elif pd.api.types.is_numeric_dtype(vals):
                # Intentamos redondear ligeramente para que no salgan chorros de decimales
                formatted = vals.map(lambda x: f"{x:.4g}" if isinstance(x, (float, int)) else str(x))
                c_values.append(formatted.values)
            else:
                # Texto normal
                c_values.append(vals.astype(str).values)
        
        # Crear matriz para customdata
        customdata = np.stack(c_values, axis=-1)

        # Construir el template del tooltip
        # Nota: Ya no usamos :{text_format} dentro del string de Plotly porque los datos YA son texto
        hovertemplate_suffix = ""
        for i, c_name in enumerate(extra_cols):
            clean_name = c_name.replace("_", " ").title()
            # Índice empieza en 3 porque 0,1,2 son base
            hovertemplate_suffix += f"<br><b>{clean_name}:</b> %{{customdata[{i+3}]}}"

        # Tamaño de burbujas
        if size_col:
            sizes = subset[size_col]
            sizemode, sizeref = 'area', 2.0 * max(sizes) / (40.**2) if max(sizes) > 0 else 1
        else:
            sizes, sizemode, sizeref = marker_size, 'diameter', 1

        fig.add_trace(go.Scatter(
            name=str(g),
            x=subset[x_col],
            y=subset[y_col],
            mode='markers',
            marker=dict(
                size=sizes, sizemode=sizemode, sizeref=sizeref,
                color=color_map.get(g, "#333"), opacity=marker_opacity,
                line=dict(width=1, color="white")
            ),
            customdata=customdata,
            # Template limpio sin lógica de formato interna
            hovertemplate=(
                f"<b>{group_col}:</b> %{{customdata[2]}}<br>"
                f"<b>{x_title or x_col}:</b> %{{customdata[0]}}<br>"
                f"<b>{y_title or y_col}:</b> %{{customdata[1]}}" 
                f"{hovertemplate_suffix}<extra></extra>"
            )
        ))

    # 5. Layout (Igual que antes)
    fig.update_layout(
        title={'text': f"<b>{title}<b>", 'y':0.95, 'x':0.5, 'xanchor':'center', 'yanchor':'top'},
        font=dict(family=font_family, color="black"),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=40, r=40, t=80, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, title_text=""),
        height=height, width=width,
        xaxis=dict(title_text=x_title or x_col, showline=True, linecolor="black", showgrid=True, gridcolor="rgba(0,0,0,0.1)"),
        yaxis=dict(title_text=y_title or y_col, showline=True, linecolor="black", showgrid=True, gridcolor="rgba(0,0,0,0.1)")
    )

    if output_path:
        fig.write_html(output_path, include_plotlyjs="cdn")

    return fig



def _tint(hex_color: str, factor: float) -> str:
    """
    Lightens a hex color by a given factor.

    Args:
        hex_color (str): The hex color string (e.g., "#1C8074").
        factor (float): Lightening factor (0.0 to 1.0). 1.0 results in white.

    Returns:
        str: The lightened hex color string.
    """



def plot_sunburst(
    df: pd.DataFrame,
    path_cols: List[str],
    value_col: Optional[str] = None,
    metric_col: Optional[str] = None,
    equalize_at_level: Optional[int] = None,
    maxdepth: Optional[int] = None,
    title: str = "",
    palette: Optional[List[str]] = None,
    tint_factors: List[float] = [0.30, 0.55, 0.75, 0.85],
    width: int = 800,
    height: int = 600,
    font_family: str = "Inter, Arial, sans-serif",
    output_path: Optional[str] = None
) -> go.Figure:
    """
    Generates a hierarchical Sunburst chart with custom level-based styling.

    Supports:
    - Custom hierarchies defined by path_cols.
    - Level equalization: Forcing nodes in a specific level to have equal size 
      for easier visibility, while still aggregating their values upwards.
    - Double metrics: Mapping sectors to counts/sums (size) and showing an 
      average value of another metric in tooltips.

    Args:
        df (pd.DataFrame): Tidy data.
        path_cols (List[str]): List of column names defining the hierarchy Levels (e.g., ["Region", "City"]).
        value_col (Optional[str], optional): Numeric column for sector sizing. If None, uses counts.
        metric_col (Optional[str], optional): Numeric column to calculate averages for the text labels and tooltips.
        equalize_at_level (Optional[int], optional): Level index (1-based) where slices will be visually normalized.
        maxdepth (Optional[int], optional): How many levels to show at once.
        title (str, optional): Header text.
        palette (Optional[List[str]], optional): Base colors for the first level.
        tint_factors (List[float], optional): Lightening steps for deeper hierarchy levels.
        width/height (int): Dimensions.
        font_family (str): CSS-like font definition.
        output_path (Optional[str], optional): HTML file destination.

    Returns:
        go.Figure: The sunburst figure.
    """
    
    # 1. Preparación de Datos
    d = df.copy()
    
    # Definir columna de valor (tamaño)
    if value_col:
        d[value_col] = pd.to_numeric(d[value_col], errors="coerce").fillna(0)
        val_c = value_col
    else:
        d["_count_"] = 1
        val_c = "_count_"

    # Definir columna de métrica (promedio)
    has_metric = metric_col is not None
    if has_metric:
        d[metric_col] = pd.to_numeric(d[metric_col], errors="coerce")
    
    pal = (palette or DEFAULT_COLORS)[:]

    # 2. Construcción de Nodos (Agregación Manual)
    # Necesario manual para poder manipular los valores con equalize_at
    all_nodes = []
    
    def _mk_id(parts: List[str]) -> str:
        # Crea ID único: L1:Valor|L2:Valor...
        return "|".join([f"L{i+1}:{str(v)}" for i, v in enumerate(parts)])

    # Iterar por profundidad de la jerarquía
    for depth in range(1, len(path_cols) + 1):
        keys = path_cols[:depth]
        
        # Agregación
        agg_dict = {val_c: "sum"}
        if has_metric:
            agg_dict[metric_col] = "sum"
            # Necesitamos conteo para sacar promedio ponderado después
            agg_dict["_rows_"] = "count" if value_col else "sum" # Si value_col es None, _count_ ya es 1
            if value_col is None: 
                # Si estamos contando filas, el conteo de filas es la suma de _count_
                d["_rows_"] = 1 

        cols_to_group = keys + ([metric_col] if has_metric else []) + [val_c]
        if has_metric and "_rows_" not in d.columns: d["_rows_"] = 1

        g = d.groupby(keys, dropna=False).agg(agg_dict).reset_index()
        
        # Propiedades del Nodo
        g["value"] = g[val_c]
        g["id"] = g[keys].astype(str).agg(lambda r: _mk_id(list(r.values)), axis=1)
        g["level"] = f"L{depth}"
        
        # Etiqueta y Padre
        if depth == 1:
            g["parent"] = "root"
            g["label"] = g[path_cols[0]].astype(str)
        else:
            parent_keys = path_cols[:depth-1]
            g["parent"] = g[parent_keys].astype(str).agg(lambda r: _mk_id(list(r.values)), axis=1)
            g["label"] = g[path_cols[depth-1]].astype(str)
        
        # Métrica Promedio
        if has_metric:
            # Evitar división por cero
            g["avg_metric"] = np.where(g["_rows_"] > 0, g[metric_col] / g["_rows_"], np.nan)
        else:
            g["avg_metric"] = np.nan

        all_nodes.append(g)

    # 3. Nodo Raíz
    nodes = pd.concat(all_nodes, ignore_index=True)
    
    # Calcular totales para la raíz
    root_value = nodes.loc[nodes["level"]=="L1", "value"].sum()
    root_avg = np.nan
    
    if has_metric:
        total_metric = nodes.loc[nodes["level"]=="L1", metric_col].sum()
        total_rows = nodes.loc[nodes["level"]=="L1", "_rows_"].sum()
        root_avg = total_metric / total_rows if total_rows > 0 else np.nan

    root = pd.DataFrame([{
        "id": "root", 
        "parent": "", 
        "label": "Total",
        "value": root_value,
        "avg_metric": root_avg,
        "level": "root"
    }])
    
    nodes = pd.concat([root, nodes], ignore_index=True)

    # 4. Lógica de Igualación (Equalize At)
    if equalize_at_level is not None:
        target_lvl = f"L{equalize_at_level}"
        # Forzar valor 1 en el nivel objetivo
        nodes.loc[nodes["level"] == target_lvl, "value"] = 1.0

        # Recalcular hacia arriba (padres = suma de hijos)
        for k in range(equalize_at_level - 1, 0, -1):
            lvl = f"L{k}"
            next_lvl = f"L{k+1}"
            # Sumar hijos agrupa por padre
            sums = nodes[nodes["level"] == next_lvl].groupby("parent")["value"].sum()
            # Mapear suma al ID del padre
            idx = nodes["level"] == lvl
            nodes.loc[idx, "value"] = nodes.loc[idx, "id"].map(sums).fillna(0).values

        # Recalcular raíz
        nodes.loc[nodes["level"]=="root", "value"] = nodes.loc[nodes["level"]=="L1", "value"].sum()

    # 5. Asignación de Colores (Base + Tints)
    id_to_color = {"root": "#FFFFFF"} # Raíz blanca o neutra
    
    # Asignar colores base a L1
    l1_nodes = nodes[nodes["level"] == "L1"]["id"].unique()
    for i, node_id in enumerate(l1_nodes):
        id_to_color[node_id] = pal[i % len(pal)]
    
    def get_color(node_id: str, level: str) -> str:
        if level == "root": return id_to_color["root"]
        
        # Encontrar el ancestro L1
        parts = node_id.split("|")
        root_ancestor = parts[0] # L1:Valor
        base_color = id_to_color.get(root_ancestor, pal[0])
        
        if level == "L1": return base_color
        
        # Aplicar tinte basado en profundidad
        depth = int(level[1:])
        tint_idx = (depth - 2) % len(tint_factors)
        return _tint(base_color, tint_factors[tint_idx])

    node_colors = [get_color(i, lvl) for i, lvl in zip(nodes["id"], nodes["level"])]

    # 6. Construcción de Hover Template
    # Personalizar tooltip dependiendo de si hay métrica o no
    hover_lines = [
        "<b>%{label}</b>",
        "Valor: %{value:.2f}",
        "% del Raíz: %{percentRoot:.1%}",
        "% del Padre: %{percentParent:.1%}"
    ]
    if has_metric:
        # Insertar métrica en segunda posición
        hover_lines.insert(1, f"{metric_col} (Prom): %{{customdata[0]:.2f}}")
    
    hovertemplate = "<br>".join(hover_lines) + "<extra></extra>"
    
    # Texto dentro del sector
    texttemplate = "%{label}"
    if has_metric:
        texttemplate += "<br>%{customdata[0]:.2f}"

    # 7. Generar Figura
    fig = go.Figure(go.Sunburst(
        ids=nodes["id"],
        labels=nodes["label"],
        parents=nodes["parent"],
        values=nodes["value"],
        branchvalues="total",
        maxdepth=maxdepth,
        marker=dict(colors=node_colors, line=dict(color="#FFFFFF", width=1)),
        # Pasamos avg_metric en customdata[0]
        customdata=np.c_[nodes["avg_metric"].fillna(0).values],
        hovertemplate=hovertemplate,
        texttemplate=texttemplate,
        insidetextorientation="radial"
    ))

    # 8. Layout
    layout = dict(
        title=dict(text=title, x=0.5, font=dict(size=18)),
        font=dict(family=font_family, size=12, color="black"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=60, b=20),
        width=width, 
        height=height,
    )
    fig.update_layout(**layout)

    if output_path:
        fig.write_html(output_path, include_plotlyjs="cdn")

    return fig




def plot_correlation_grid(
    df: pd.DataFrame,
    corr_cols: List[str],
    group_col: str,
    group_order: Optional[List[str]] = None,
    method: str = "pearson",
    min_periods: int = 3,
    decimals: int = 2,
    colorscale: str = DEFAULT_COLORS,
    show_values: bool = True,
    n_cols: int = 2,
    title: str = "",
    height: int = 800,
    width: int = 1000,
    output_path: Optional[str] = None
) -> go.Figure:
    """
    Generates a trellis grid of correlation heatmaps (upper triangles only) for sub-groups.

    Args:
        df (pd.DataFrame): Input data.
        corr_cols (List[str]): Numerical columns to correlate.
        group_col (str): Column to split data into sub-plots.
        group_order (Optional[List[str]], optional): Sequence of sub-plots.
        method (str, optional): Correlation type ('pearson', 'spearman', 'kendall').
        min_periods (int, optional): Minimum samples needed for a valid correlation.
        decimals (int, optional): Rounding precision for the displayed values.
        colorscale (str, optional): Diverging palette for the heatmaps.
        show_values (bool, optional): If True, overlays the correlation coefficient as text.
        n_cols (int, optional): Number of maps per row in the grid.
        title (str, optional): Main chart title.
        height/width (int): Grid dimensions.
        output_path (Optional[str], optional): HTML export path.

    Returns:
        go.Figure: The correlation grid figure.
    """

    # 1. Preparación de Grupos
    d = df.copy()
    
    # Obtener grupos únicos ordenados
    if group_order:
        unique_groups = [g for g in group_order if g in d[group_col].unique()]
    else:
        unique_groups = sorted(d[group_col].unique().astype(str))
    
    # 2. Configuración de la Grilla
    num_plots = len(unique_groups)
    num_rows = math.ceil(num_plots / n_cols)
    
    fig = make_subplots(
        rows=num_rows, 
        cols=n_cols,
        subplot_titles=unique_groups,
        horizontal_spacing=0.1,
        vertical_spacing=0.15 / num_rows if num_rows > 1 else 0.1
    )

    # 3. Iterar sobre grupos y crear Heatmaps
    for i, group_name in enumerate(unique_groups):
        # Calcular posición en la grilla (1-based)
        row = (i // n_cols) + 1
        col = (i % n_cols) + 1
        
        # Filtrar datos
        sub = d[d[group_col] == group_name][corr_cols].dropna()
        
        # Validar suficiencia de datos
        if len(sub) < min_periods:
            fig.add_annotation(
                text="Datos insuficientes",
                xref=f"x{i+1}", yref=f"y{i+1}", # Referencia relativa al subplot
                showarrow=False,
                row=row, col=col
            )
            continue
            
        # Calcular Correlación
        df_corr = sub.corr(method=method)
        
        # --- Lógica de Triángulo Superior ---
        # 1. Crear máscara para el triángulo superior (incluyendo diagonal k=0)
        mask = np.triu(np.ones_like(df_corr, dtype=bool))
        
        # 2. Aplicar máscara (poner NaN en lo que no queremos ver)
        z_vals = df_corr.where(mask).values
        
        # 3. Invertir eje Y para visualización estándar de matriz (origen arriba)
        # Esto es necesario porque Plotly dibuja de abajo hacia arriba por defecto
        z_disp = z_vals[::-1] 
        y_labels = df_corr.index[::-1].tolist()
        x_labels = df_corr.columns.tolist()
        
        # 4. Texto de valores
        if show_values:
            text_vals = np.round(z_disp, decimals).astype(str)
            text_vals[np.isnan(z_disp)] = "" # Limpiar NaNs del texto
        else:
            text_vals = None

        # Crear Traza
        heatmap = go.Heatmap(
            z=z_disp,
            x=x_labels,
            y=y_labels,
            coloraxis="coloraxis", # Usar escala compartida
            text=text_vals,
            texttemplate="%{text}" if show_values else None,
            hovertemplate="<b>x:</b> %{x}<br><b>y:</b> %{y}<br><b>Corr:</b> %{z:.3f}<extra></extra>",
            showscale=False # La escala se maneja globalmente en layout
        )
        
        fig.add_trace(heatmap, row=row, col=col)
        
        # Ajustes de ejes específicos para este subplot
        # Ocultar etiquetas del eje Y si no es la primera columna (opcional, para limpieza)
        if col > 1:
            fig.update_yaxes(showticklabels=False, row=row, col=col)

    # 4. Layout Global
    fig.update_layout(
        title=dict(text=title, x=0.5),
        width=width,
        height=height,
        template="plotly_white",
        # Configuración de la barra de color compartida
        coloraxis=dict(
            colorscale=colorscale,
            cmin=-1, 
            cmax=1,
            colorbar=dict(
                title=f"Corr ({method})",
                thickness=15,
                len=0.6,
            )
        )
    )

    if output_path:
        fig.write_html(output_path, include_plotlyjs="cdn")
        
    return fig




def corporate_colorscale() -> List[List[Union[float, str]]]:
    """
    Generates a continuous Plotly colorscale based on the CORPORATE_COLORS palette.

    Returns:
        List[List[Union[float, str]]]: A mapping of offsets (0 to 1) to hex colors.
    """
    n = len(DEFAULT_COLORS)
    return [[i / (n - 1), c] for i, c in enumerate(DEFAULT_COLORS)]


def plot_corr_triangle(
    df: pd.DataFrame,
    value_cols: Sequence[str],
    *,
    method: str = "pearson",
    title: str = "Correlación (Pearson) — triángulo superior",
    decimals: int = 2,
    width: int = 750,
    height: int = 650,
    font_family: str = "Inter, Arial, sans-serif",
    salida_html: Optional[str] = None
) -> go.Figure:
    """
    Renders a standalone Correlation Heatmap highlighting the upper triangle.

    Features a polished design with:
    - Colored upper triangle using the corporate color scale.
    - Clean white lower triangle.
    - Black grid separation and high-contrast labels.

    Args:
        df (pd.DataFrame): Source data.
        value_cols (Sequence[str]): List of columns to include in the matrix.
        method (str, optional): Correlation method.
        title (str, optional): Matrix title.
        decimals (int, optional): Formatting precision.
        width/height (int): Dimensions.
        font_family (str): UI Font.
        salida_html (Optional[str], optional): HTML file destination.

    Returns:
        go.Figure: The correlation triangle figure.
    """

    # --- columnas numéricas ---
    d = df[list(value_cols)].apply(pd.to_numeric, errors="coerce")
    d = d.dropna(how="all")

    if d.shape[1] < 2:
        raise ValueError("Se requieren al menos dos columnas numéricas para correlación.")

    # --- correlación ---
    corr = d.corr(method=method)
    corr_vals = corr.values.astype(float)

    n = corr_vals.shape[0]
    mask_upper = np.triu(np.ones_like(corr_vals, dtype=bool))   # superior (incluye diagonal)
    mask_lower = np.tril(np.ones_like(corr_vals, dtype=bool), k=-1)  # estrictamente inferior

    # matriz para triángulo superior
    z_upper = corr_vals.copy()
    z_upper[~mask_upper] = np.nan

    # matriz para triángulo inferior (solo usamos un valor constante)
    z_lower = np.zeros_like(corr_vals, dtype=float)
    z_lower[~mask_lower] = np.nan   # dejamos NaN fuera del triángulo inferior

    # textos
    text_upper = np.empty_like(z_upper, dtype=object)
    for i in range(n):
        for j in range(n):
            if not np.isnan(z_upper[i, j]):
                text_upper[i, j] = f"{z_upper[i, j]:.{decimals}f}"
            else:
                text_upper[i, j] = ""

    colorscale = corporate_colorscale()

    # --- Heatmap triángulo inferior (blanco) ---
    heat_lower = go.Heatmap(
        z=z_lower,
        x=corr.columns,
        y=corr.index,
        colorscale=[[0, "#FFFFFF"], [1, "#FFFFFF"]],  # siempre blanco
        zmin=0,
        zmax=1,
        showscale=False,
        hoverinfo="skip",
        xgap=1,
        ygap=1,
    )

    # --- Heatmap triángulo superior ---
    heat_upper = go.Heatmap(
        z=z_upper,
        x=corr.columns,
        y=corr.index,
        zmin=-1,
        zmax=1,
        colorscale=colorscale,
        colorbar=dict(title="r", titleside="right"),
        text=text_upper,
        texttemplate="%{text}",
        textfont=dict(size=13, color="#0b2530"),
        hovertemplate="x=%{x}<br>y=%{y}<br>r=%{z:.3f}<extra></extra>",
        xgap=1,
        ygap=1,
    )

    fig = go.Figure(data=[heat_lower, heat_upper])

    fig.update_xaxes(showgrid=False, tickangle=0)
    fig.update_yaxes(showgrid=False, autorange="reversed")

    fig.update_layout(
        template="plotly_white",
        title=dict(text=title, x=0.5, xanchor="center"),
        font=dict(family=font_family, size=14, color="#1F2937"),
        width=width,
        height=height,
        margin=dict(l=60, r=80, t=80, b=40),
        plot_bgcolor="#000000", 
        paper_bgcolor="#FFFFFF",
    )

    if salida_html:
        pio.write_html(fig, salida_html, include_plotlyjs="cdn", full_html=True)

    return fig


def plot_cdf(
    df: pd.DataFrame, 
    value_col: str, 
    group_col: Optional[str] = None, 
    width: int = 1000,
    height: int = 600,
    title: Optional[str] = None,
    xaxis_title: Optional[str] = None,
    yaxis_title: Optional[str] = None,
    output_path: Optional[str] = None, 
    custom_colors: Optional[List[str]] = DEFAULT_COLORS,
) -> go.Figure:
    """
    Generates a Cumulative Distribution Function (CDF) plot for numeric data.

    Calculates the empirical distribution and renders it as one or more lines, 
    optionally grouped by a categorical variable.

    Args:
        df (pd.DataFrame): Data input.
        value_col (str): Numeric column to calculate the distribution for.
        group_col (Optional[str], optional): Column for splitting data into separate series.
        width (int, optional): Figure width.
        height (int, optional): Figure height.
        title (Optional[str], optional): Header text. Auto-generated if None.
        xaxis_title (Optional[str], optional): X-axis label. Defaults to value_col.
        yaxis_title (Optional[str], optional): Y-axis label. Defaults to "Probabilidad acumulada".
        output_path (Optional[str], optional): HTML export path.
        custom_colors (Optional[List[str]], optional): Color sequence for lines.

    Returns:
        go.Figure: The CDF plot figure.
    """
    
    # Trabajar con una copia para no alterar el original
    data = df.copy()
    
    if custom_colors:
        colors = custom_colors
    else:
        colors = px.colors.qualitative.Plotly

    fig = go.Figure()

    if group_col:
        try:
            categorias = sorted(data[group_col].dropna().unique())
        except:
            categorias = data[group_col].dropna().unique()
        legend_title_text = group_col
    else:
        group_col = "_dummy_group"
        data[group_col] = "Todos los datos"
        categorias = ["Todos los datos"]
        legend_title_text = "Grupo"

    for i, cat in enumerate(categorias):
        subset = data.loc[data[group_col] == cat, value_col].dropna().sort_values()
        n = len(subset)
        
        if n == 0:
            continue
        y_cdf = np.arange(1, n + 1) / n

        line_color = colors[i % len(colors)]

        fig.add_trace(go.Scatter(
            x=subset,
            y=y_cdf,
            mode='lines',
            name=str(cat),
            line=dict(color=line_color, width=2)
        ))

   
    if title:
        final_title = title
    else:
        final_title = f"Distribución acumulada (CDF) de {value_col}"
        if group_col != "_dummy_group":
            final_title += f" por {group_col}"
            
    # 2. Títulos de Ejes
    final_x_title = xaxis_title if xaxis_title else value_col
    final_y_title = yaxis_title if yaxis_title else "Probabilidad acumulada"

    # Layout actualizado
    fig.update_layout(
        width=width,  
        height=height,
        title=dict(
            text=final_title,
            font=dict(size=20)
        ),
        xaxis=dict(
            title=final_x_title, # <--- Nuevo
            title_font=dict(size=20),
            tickfont=dict(size=20),
            showgrid=True,
            zeroline=False,
            showline=True,         
            mirror=True,           
            linecolor='black',
            linewidth=2
        ),
        yaxis=dict(
            title=final_y_title, # <--- Nuevo
            title_font=dict(size=20),
            tickfont=dict(size=20),
            showgrid=True,
            zeroline=False,
            showline=True,         
            mirror=True,           
            linecolor='black',
            linewidth=2,
            range=[-0.02, 1.02]
        ),
        legend_title=dict(text=legend_title_text, font=dict(size=20)),
        hovermode="x unified",
        template=None,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        legend=dict(font=dict(size=20))
    )

    if output_path:
        fig.write_html(output_path, include_plotlyjs='cdn', full_html=True)
        
    return fig

def create_subplot_grid(
    figures: List[go.Figure],
    rows: int,
    cols: int,
    titles: Optional[List[str]] = None,
    shared_x: bool = False,
    shared_y: bool = False,
    main_title: str = "",
    height: int = 800,
    width: int = 1000,
    output_path: Optional[str] = None
) -> go.Figure:
    """
    Toma una lista de figuras de Plotly (go.Figure) y las organiza en una grilla,
    unificando la leyenda para evitar duplicados.
    """
    
    # 1. Crear la grilla vacía con soporte para eje secundario si fuera necesario
    fig = make_subplots(
        rows=rows, 
        cols=cols, 
        subplot_titles=titles,
        shared_xaxes=shared_x,
        shared_yaxes=shared_y,
        vertical_spacing=0.15, 
        horizontal_spacing=0.08
    )

    fig_idx = 0
    for r in range(1, rows + 1):
        for c in range(1, cols + 1):
            if fig_idx < len(figures):
                current_fig = figures[fig_idx]
                
                for trace in current_fig.data:
                    # --- UNIFICACIÓN DE LEYENDA ---
                    # Vincula trazas con el mismo nombre para que actúen en bloque
                    trace.legendgroup = trace.name
                    
                    # Solo mostramos la leyenda de la primera figura para evitar 
                    # que se repita el mismo nombre N veces en el panel lateral.
                    if fig_idx > 0:
                        trace.showlegend = False
                    
                    fig.add_trace(trace, row=r, col=c)
                
                fig_idx += 1

    # 2. Configuración estética global
    fig.update_layout(
        title_text=f"<b>{main_title}</b>",
        height=height,
        width=width,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, Arial, sans-serif", color="black"),
        showlegend=True,
        legend=dict(
            orientation="v", 
            yanchor="top", 
            y=1, 
            xanchor="left", 
            x=1.02
        )
    )

    # Estilizar todos los ejes de la grilla de forma masiva
    fig.update_xaxes(showline=True, linecolor="black", showgrid=False)
    fig.update_yaxes(showline=True, linecolor="black", showgrid=True, gridcolor="rgba(0,0,0,0.1)")

    if output_path:
        fig.write_html(output_path, include_plotlyjs="cdn")

    return fig

def plot_distribution(
    df: pd.DataFrame,
    value_col: str,
    group_col: str = None,
    # --- Parámetros Estadísticos ---
    nbins: int = None,          # Número de barras (bins). Si es None, Plotly decide.
    histnorm: str = None,       # 'percent', 'probability', 'density', o None (conteo puro)
    marginal: str = "box",      # 'box', 'violin', 'rug', o None (para quitarlo)
    opacity: float = 0.6,       # Transparencia para ver superposiciones
    
    # --- Parámetros de Estilo y Dimensiones ---
    width: int = 1000,
    height: int = 600,
    title: str = None,
    xaxis_title: str = None,
    yaxis_title: str = None,
    custom_colors: list = DEFAULT_COLORS,
    
    # --- Salida ---
    output_path: str = None
):
    """
    Grafica la distribución de una variable numérica (Histograma + Marginal)
    manteniendo el estilo corporativo estricto.

    Parámetros:
    -----------
    marginal : str
        Gráfico pequeño arriba del histograma ('box', 'violin', 'rug').
    histnorm : str
        Si es 'percent', el eje Y muestra %. Si es None, muestra conteo (n).
    """

    # Copia para no afectar original
    data = df.copy()

    # Gestión de Colores
    if custom_colors is None:
        custom_colors = px.colors.qualitative.Plotly

    # Títulos Automáticos
    if title is None:
        title = f"Distribución de {value_col}"
        if group_col:
            title += f" por {group_col}"
    
    if xaxis_title is None:
        xaxis_title = value_col
    
    if yaxis_title is None:
        yaxis_title = "Porcentaje" if histnorm == 'percent' else "Frecuencia (n)"

    # --- Construcción del Gráfico (Usando px por su potencia con histogramas) ---
    fig = px.histogram(
        data,
        x=value_col,
        color=group_col,
        nbins=nbins,
        histnorm=histnorm,
        marginal=marginal, # El gráfico pequeño de arriba
        barmode='overlay', # Superponer grupos en lugar de apilar (stack)
        opacity=opacity,
        color_discrete_sequence=custom_colors,
        category_orders={group_col: sorted(data[group_col].dropna().unique())} if group_col else None
    )

    # --- Aplicación del Estilo (Idéntico a tu plot_cdf) ---
    fig.update_layout(
        width=width,
        height=height,
        title=dict(
            text=title,
            font=dict(size=20)
        ),
        xaxis=dict(
            title=xaxis_title,
            title_font=dict(size=20),
            tickfont=dict(size=20),
            showgrid=True,
            zeroline=False,
            showline=True,
            mirror=True,
            linecolor='black',
            linewidth=2
        ),
        yaxis=dict(
            title=yaxis_title,
            title_font=dict(size=20),
            tickfont=dict(size=20),
            showgrid=True,
            zeroline=False,
            showline=True,
            mirror=True,
            linecolor='black',
            linewidth=2
        ),
        legend_title=dict(
            text=group_col if group_col else "Grupo",
            font=dict(size=20)
        ),
        legend=dict(font=dict(size=20)),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        hovermode="x unified"
    )

    if output_path:
        fig.write_html(output_path, include_plotlyjs='cdn', full_html=True)

    return fig




def plot_dynamic_trends(
    df, 
    date_col, 
    value_col, 
    category_col=None, 
    windows=[7, 15, 30], 
    title_prefix="Análisis de Tendencia",
    y_label="Valor",
    color_palette=None,
    width=None,   
    height=None
):
    """
    Genera gráficos de series de tiempo con medias móviles dinámicas.
    Permite controlar el tamaño (ancho y alto).

    Parámetros:
    - width: Ancho del gráfico en píxeles (ej. 1000). Si es None, es automático.
    - height: Altura del gráfico en píxeles (ej. 600). Si es None, es automático.
    """
    
    # 1. Configuración de Colores Corporativos (Default)
    if color_palette is None:
        color_palette = DEFAULT_COLORS
    
    # 2. Preparación de datos
    df_proc = df.copy()
    df_proc[date_col] = pd.to_datetime(df_proc[date_col])
    
    if category_col:
        items = df_proc[category_col].unique()
        df_proc = df_proc.sort_values(by=[category_col, date_col])
    else:
        items = ["Total"]
        df_proc[category_col] = "Total"
        df_proc = df_proc.sort_values(by=[date_col])

    # 3. Iteración y Graficación
    for item in items:
        if category_col and category_col in df_proc.columns:
            df_subset = df_proc[df_proc[category_col] == item].copy()
        else:
            df_subset = df_proc.copy()
            
        fig = go.Figure()

        # A. Datos Crudos
        fig.add_trace(go.Scatter(
            x=df_subset[date_col], 
            y=df_subset[value_col],
            mode='markers+lines',
            name='Dato Real (Diario)',
            line=dict(color='lightgrey', width=1),
            opacity=0.9
        ))

        # B. Ventanas Móviles
        for i, w in enumerate(windows):
            col_ma_name = f"MA_{w}"
            df_subset[col_ma_name] = df_subset[value_col].rolling(window=w, min_periods=1).mean()
            color_actual = color_palette[i % len(color_palette)]
            
            fig.add_trace(go.Scatter(
                x=df_subset[date_col], 
                y=df_subset[col_ma_name],
                mode='lines',
                name=f'Media Móvil ({w} per)',
                line=dict(color=color_actual, width=2.5)
            ))

        # C. Layout con Width y Height
        item_title = f": {item}" if item != "Total" else ""
        
        fig.update_layout(
            title={
                'text': f"<b>{title_prefix}{item_title}</b>",
                'y':0.9,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top'
            },
            xaxis_title="Tiempo",
            yaxis_title=y_label,
            hovermode="x unified",
            template="plotly_white",
            
            # --- DIMENSIONES ---
            width=width,   # Aplica el ancho
            height=height, # Aplica la altura
            
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        return fig