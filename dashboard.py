from dash import Dash, dcc, html, Input, Output
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objs as go
import pandas as pd
import numpy as np
import datetime
from dash.exceptions import PreventUpdate

# Cargar los datos

def cargar_datos():
    try:
        print("📥 Cargando datos...")
        clientes_df = pd.read_excel('_data/ClientesRelacionadas.xlsx')
        compras_df = pd.read_excel('_data/ComprasRelacionadas.xlsx')
        ventas_df = pd.read_excel('_data/VentasRelacionadas.xlsx')

        clientes_df.columns = clientes_df.columns.str.lower().str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8')
        compras_df.columns = compras_df.columns.str.lower().str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8')
        ventas_df.columns = ventas_df.columns.str.lower().str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8')

        clientes_df['fecha_registro'] = pd.to_datetime(clientes_df['fecha_registro'])
        compras_df['fechahora_transaccion'] = pd.to_datetime(compras_df['fechahora_transaccion'])
        ventas_df['fechahora_transaccion'] = pd.to_datetime(ventas_df['fechahora_transaccion'])

        print("✅ Datos cargados correctamente")
        return clientes_df, compras_df, ventas_df
    except Exception as e:
        print(f"❌ Error cargando los datos: {e}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

clientes_df, compras_df, ventas_df = cargar_datos()
ventas_con_depto = ventas_df.merge(clientes_df[['id_cliente', 'departamento']], on='id_cliente')
print("🔗 Merge de ventas con departamento completado")

# Inicializar app
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)
server = app.server

# KPI Functions
def calcular_kpis(df_ventas, df_compras, df_clientes):
    try:
        print("🔢 Calculando KPIs...")
        total_ventas = df_ventas['total'].sum()
        total_compras = df_compras['total'].sum()
        clientes_por_depto = df_clientes['departamento'].value_counts()
        producto_mas_vendido = df_ventas.groupby('producto')['cantidad'].sum().idxmax() if not df_ventas.empty else 'N/A'
        producto_mas_valorado = df_ventas.groupby('producto')['total'].sum().idxmax() if not df_ventas.empty else 'N/A'
        promedio_edad = df_clientes['edad'].mean()
        suscritos = df_clientes['suscrito'].astype(str).str.strip().str.lower()
        suscritos = suscritos[suscritos.isin(['sí', 'si', 'no'])]
        porcentaje_suscritos = suscritos.map({'sí': 1, 'si': 1, 'no': 0}).mean() * 100 if not suscritos.empty else 0
        ingreso_mensual_prom = df_clientes['ingreso_mensual'].mean()
        prom_total_compras = df_compras['total'].mean()
        prom_total_ventas = df_ventas['total'].mean()
        print("✅ KPIs calculados")
        return [total_ventas, total_compras, clientes_por_depto, producto_mas_vendido, producto_mas_valorado,
                promedio_edad, porcentaje_suscritos, ingreso_mensual_prom, prom_total_compras, prom_total_ventas]
    except Exception as e:
        print(f"❌ Error al calcular KPIs: {e}")
        return [0]*10

# Función para aplicar filtros
def aplicar_filtros(dfv, dfc, dfcli, start_date, end_date, departamento, metodo, genero):
    print("🎛️ Aplicando filtros...")
    try:
        dfv = dfv[(dfv['fechahora_transaccion'] >= pd.to_datetime(start_date)) & (dfv['fechahora_transaccion'] <= pd.to_datetime(end_date))]
        dfc = dfc[(dfc['fechahora_transaccion'] >= pd.to_datetime(start_date)) & (dfc['fechahora_transaccion'] <= pd.to_datetime(end_date))]
        if departamento:
            dfv = dfv[dfv['departamento'].isin(departamento)]
            dfcli = dfcli[dfcli['departamento'].isin(departamento)]
        if metodo:
            dfv = dfv[dfv['método_pago'].isin(metodo)]
        if genero:
            dfcli = dfcli[dfcli['genero'].isin(genero)]
            dfv = dfv[dfv['id_cliente'].isin(dfcli['id_cliente'])]
        print("✅ Filtros aplicados")
        return dfv, dfc, dfcli
    except Exception as e:
        print(f"❌ Error aplicando filtros: {e}")
        return dfv, dfc, dfcli

# Layout
print("📐 Definiendo layout de la aplicación...")
app.layout = dbc.Container([
    html.H1("📊 Dashboard de Ventas y Compras", className="text-center mt-4 mb-4"),

    dbc.Row([
        dbc.Col([
            html.Label("Rango de Fechas"),
            dcc.DatePickerRange(
                id='filtro-fechas',
                min_date_allowed=ventas_df['fechahora_transaccion'].min(),
                max_date_allowed=ventas_df['fechahora_transaccion'].max(),
                start_date=ventas_df['fechahora_transaccion'].min(),
                end_date=ventas_df['fechahora_transaccion'].max(),
                display_format='DD/MM/YYYY'
            )
        ], width=3),

        dbc.Col([
            html.Label("Departamento"),
            dcc.Dropdown(
                id='filtro-departamento',
                options=[{'label': d, 'value': d} for d in clientes_df['departamento'].unique()],
                multi=True
            )
        ], width=3),

        dbc.Col([
            html.Label("Método de Pago"),
            dcc.Dropdown(
                id='filtro-metodo',
                options=[{'label': m, 'value': m} for m in ventas_df['metodo_pago'].unique()],
                multi=True
            )
        ], width=3),

        dbc.Col([
            html.Label("Género"),
            dcc.Dropdown(
                id='filtro-genero',
                options=[{'label': g, 'value': g} for g in clientes_df['genero'].dropna().unique()],
                multi=True
            )
        ], width=3),
    ], className="mb-4"),

    dbc.Tabs([
        dbc.Tab(label='📊 Resumen de KPIs', tab_id='tab-kpis'),
        dbc.Tab(label='📈 Gráficas', tab_id='tab-graficas'),
    ], id='tabs', active_tab='tab-kpis', className="mb-4"),

    html.Div(id='contenido-tab'),

    dcc.Interval(
        id='interval-tiempo-real',
        interval=1000,
        n_intervals=0
    )
], fluid=True)

# Callback principal y callback para subtabs (también omitidos por brevedad)

# Callback para manejar los sub-tabs
@app.callback(
    Output('contenido-sub-tab', 'children'),
    Input('sub-tabs', 'value'),
    Input('filtro-fechas', 'start_date'),
    Input('filtro-fechas', 'end_date'),
    Input('filtro-departamento', 'value'),
    Input('filtro-metodo', 'value'),
    Input('filtro-genero', 'value'),
)
def mostrar_contenido_subtab(subtab, start_date, end_date, departamento, metodo, genero):
    print(f"🛎️ Callback activado con subtab: {subtab}")
    print(f"📅 Fechas: {start_date} - {end_date}")
    print(f"📍 Departamento: {departamento}, 💳 Método: {metodo}, 🚻 Género: {genero}")
    try:
        dfv, dfc, dfcli = aplicar_filtros(ventas_con_depto.copy(), compras_df.copy(), clientes_df.copy(), start_date, end_date, departamento, metodo, genero)

        if subtab == 'graf-6':
            print("📊 Generando gráfico por método y departamento")
            if dfv.empty:
                print("⚠️ No hay datos para el gráfico")
                return html.Div("No hay datos para mostrar con los filtros actuales.")

            df_grouped = dfv.groupby(['departamento', 'método_pago'])['total'].sum().reset_index()
            print(f"📈 Datos agrupados: {df_grouped.head()}")

            fig = px.bar(df_grouped,
                         x='departamento',
                         y='total',
                         color='método_pago',
                         barmode='group',
                         title='Ventas por Método de Pago y Departamento')

            print("✅ Gráfico generado")
            return dcc.Graph(figure=fig)

        print("ℹ️ Sub-tab no reconocido o sin acción específica")
        return html.Div(f"Contenido de la sub-tab: {subtab}")
    except Exception as e:
        print(f"❌ Error en el callback de sub-tab: {e}")
        return html.Div("Se produjo un error al generar el contenido.")

if __name__ == '__main__':
    app.run(debug=True)