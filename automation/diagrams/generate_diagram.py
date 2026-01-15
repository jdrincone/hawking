import graphviz
import os

def generate_process_diagram():
    # Create a new Digraph
    dot = graphviz.Digraph('FeedProductionProcess', comment='Process Flow Diagram', format='svg')
    
    # Global attributes
    dot.attr(rankdir='TB', nodesep='0.5', ranksep='0.8')
    dot.attr('node', fontname='Arial', fontsize='10')
    dot.attr('edge', fontname='Arial', fontsize='9')

    # Define Styles
    styles = {
        'input': {'fillcolor': '#E9F3F1', 'color': '#248374', 'style': 'filled', 'shape': 'box'},
        'process': {'fillcolor': '#F2F2F2', 'color': '#333333', 'style': 'filled', 'shape': 'box'},
        'output': {'fillcolor': '#D3E6E3', 'color': '#248374', 'style': 'filled', 'shape': 'box'},
        'dataPoint': {'fillcolor': '#EFF4EF', 'color': '#9CB79A', 'style': 'filled,dashed', 'shape': 'note'}
    }

    # Helper to apply styles
    def apply_style(node_id, style_type):
        dot.node(node_id, **styles[style_type])

    # --- Bloque 1: Entradas Macro ---
    with dot.subgraph(name='cluster_ENTRADAS_MACRO') as c:
        c.attr(label='PUNTO 1: ENTRADAS AL SISTEMA', style='dotted', color='grey')
        c.node('A', 'Báscula \nMaterias Primas')
        c.node('B', 'Silos de Almacenaje', shape='cylinder')
        c.edge('A', 'B')
        
    apply_style('A', 'input')
    apply_style('B', 'process')

    dot.node('D', 'Molienda y\nDosificación', shape='diamond')
    apply_style('D', 'process')
    dot.edge('B', 'D')

    # --- Bloque 2: Inicio del proceso crítico ---
    with dot.subgraph(name='cluster_PUNTO_CRITICO_INICIO') as c:
        c.attr(label='PUNTO 2: EL CERO DEL PROCESO', style='dotted', color='grey')
        c.node('E', 'MEZCLADORA PRINCIPAL')
        c.node('DP2', 'DATA POINT 2:\n- Peso Total a producir por día/lotes/op\'s\n- % Humedad Harina Inicial por día/lotes/ops')
        c.edge('E', 'DP2', label='DATOS CLAVE', style='dashed')
        
    apply_style('E', 'process')
    apply_style('DP2', 'dataPoint')
    dot.edge('D', 'E')

    dot.node('F', 'Tolva hacia el acondicionador')
    apply_style('F', 'process')
    dot.edge('E', 'F')

    # --- Bloque 3: Zona de Transformación ---
    with dot.subgraph(name='cluster_ZONA_DE_TRANSFORMACION') as c:
        c.attr(label='PUNTO 3 y 4: LA CAJA NEGRA', style='dotted', color='grey')
        c.node('G', 'Acondicionador')
        c.node('H', 'Caldera / Vapor', shape='box')
        c.node('DP3', 'DATA POINT 3:\n- Flujo de Vapor kg/h por Lote/Ops/Día\n- Temp. Acondicionamiento por Lote/Ops/Día')
        c.node('I', 'Tambor con Dado')
        c.node('J', 'ENFRIADOR / SECADOR')
        c.node('K', 'Atmósfera', shape='ellipse')
        c.node('DP4', 'DATA POINT 4:\n- % Humedad Pellet Final por Lote/Ops/Día\n- Temperatura Salida por día/lotes/ops')
        
        c.edge('F', 'G')
        c.edge('H', 'G', label='GANANCIA DE PESO')
        c.edge('G', 'DP3', label='DATOS CLAVE', style='dashed')
        c.edge('G', 'I')
        c.edge('I', 'J')
        c.edge('J', 'K', label='PÉRDIDA DE PESO\n(Humedad/Calor)')
        c.edge('K', 'F', label='Recuperación Finos')
        c.edge('J', 'DP4', label='DATOS CLAVE', style='dashed')

    apply_style('G', 'process')
    apply_style('H', 'input')
    apply_style('DP3', 'dataPoint')
    apply_style('I', 'process')
    apply_style('J', 'process')
    apply_style('K', 'process')
    apply_style('DP4', 'dataPoint')

    dot.node('L', 'Tamiz o sistema que toma el pellet')
    apply_style('L', 'process')
    dot.edge('J', 'L')
    dot.edge('L', 'F', label='Retorno de Finos', constraint='false')

    dot.node('M', 'Tolvas de Producto Terminado')
    apply_style('M', 'process')
    dot.edge('L', 'M')

    # --- Bloque 4: Salidas Finales ---
    with dot.subgraph(name='cluster_SALIDAS_FINALES') as c:
        c.attr(label='PUNTO 5: SALIDAS VENDIBLES', style='dotted', color='grey')
        c.node('N', 'Ruta de Salida', shape='diamond')
        c.node('O', 'ENSACADORA')
        c.node('DP5A', 'DATA POINT 5A Saco:\n- Peso Promedio Real por Saco (aleatorio)\n- Conteo Total de Sacos')
        c.node('P', 'CARGA A GRANEL')
        c.node('DP5B', 'DATA POINT 5B Granel:\n- Peso Neto Báscula Salida')
        
        c.edge('N', 'O')
        c.edge('O', 'DP5A', label='DATOS CLAVE', style='dashed')
        c.edge('N', 'P')
        c.edge('P', 'DP5B', label='DATOS CLAVE', style='dashed')

    apply_style('N', 'process')
    apply_style('O', 'output')
    apply_style('DP5A', 'dataPoint')
    apply_style('P', 'output')
    apply_style('DP5B', 'dataPoint')
    dot.edge('M', 'N')

    # --- Bloque de Resultados ---
    with dot.subgraph(name='cluster_RESULTADO') as c:
        c.attr(label='ANÁLISIS DE DATOS', style='dotted', color='grey')
        c.node('CALCULO_SACKOFF', 'CÁLCULO SACK-OFF BRUTO\n Entrada vs Salida Total')
        c.node('ANALISIS_MERMA', 'ANÁLISIS DE MERMA TÉCNICA\n Balance de Humedad P2 vs P4')

    # Data connections to analysis
    dot.edge('DP2', 'CALCULO_SACKOFF', style='dotted')
    dot.edge('DP5A', 'CALCULO_SACKOFF', style='dotted')
    dot.edge('DP5B', 'CALCULO_SACKOFF', style='dotted')
    dot.edge('DP3', 'ANALISIS_MERMA', style='dotted')
    dot.edge('DP4', 'ANALISIS_MERMA', style='dotted')

    # Determine paths relative to this script's location
    # Script is at: automation/diagrams/generate_diagram.py
    # Docs images at: docs/images/
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.abspath(os.path.join(base_dir, '../../docs/images'))
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Render and Save
    output_base = os.path.join(output_dir, 'feed_production_process')
    # We use a temporary GV file in the same directory as the script
    gv_base = os.path.join(base_dir, 'feed_production_process.gv')
    
    output_path = dot.render(gv_base, outfile=os.path.join(output_dir, 'feed_production_process.svg'), cleanup=True)
    print(f"Diagram generated at: {output_path}")

if __name__ == "__main__":
    generate_process_diagram()
