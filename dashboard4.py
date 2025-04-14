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
    clientes_df = pd.read_excel('_data/ClientesRelacionadas.xlsx')
    compras_df = pd.read_excel('_data/ComprasRelacionadas.xlsx')
    ventas_df = pd.read_excel('_data/VentasRelacionadas.xlsx')

    clientes_df.columns = clientes_df.columns.str.lower()
    compras_df.columns = compras_df.columns.str.lower()
    ventas_df.columns = ventas_df.columns.str.lower()

    clientes_df['fecha_registro'] = pd.to_datetime(clientes_df['fecha_registro'])
    compras_df['fechahora_transaccion'] = pd.to_datetime(compras_df['fechahora_transaccion'])
    ventas_df['fechahora_transaccion'] = pd.to_datetime(ventas_df['fechahora_transaccion'])

    return clientes_df, compras_df, ventas_df

clientes_df, compras_df, ventas_df = cargar_datos()

ventas_con_depto = ventas_df.merge(clientes_df[['id_cliente', 'departamento']], on='id_cliente')

# Inicializar app
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

# KPI Functions
def calcular_kpis(df_ventas, df_compras, df_clientes):
    total_ventas = df_ventas['total'].sum()
    total_compras = df_compras['total'].sum()
    clientes_por_depto = df_clientes['departamento'].value_counts()
    producto_mas_vendido = df_ventas.groupby('producto')['cantidad'].sum().idxmax() if not df_ventas.empty else 'N/A'
    producto_mas_valorado = df_ventas.groupby('producto')['total'].sum().idxmax() if not df_ventas.empty else 'N/A'
    promedio_edad = df_clientes['edad'].mean()
    porcentaje_suscritos = df_clientes['suscrito'].astype(str).str.lower().map({'sí': 1, 'si': 1, 'no': 0}).mean() * 100
    ingreso_mensual_prom = df_clientes['ingreso_mensual'].mean()
    prom_total_compras = df_compras['total'].mean()
    prom_total_ventas = df_ventas['total'].mean()
    return [total_ventas, total_compras, clientes_por_depto, producto_mas_vendido, producto_mas_valorado,
            promedio_edad, porcentaje_suscritos, ingreso_mensual_prom, prom_total_compras, prom_total_ventas]

# Layout
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
        ], width=4),

        dbc.Col([
            html.Label("Departamento"),
            dcc.Dropdown(
                id='filtro-departamento',
                options=[{'label': d, 'value': d} for d in clientes_df['departamento'].unique()],
                multi=True
            )
        ], width=4),

        dbc.Col([
            html.Label("Método de Pago"),
            dcc.Dropdown(
                id='filtro-metodo',
                options=[{'label': m, 'value': m} for m in ventas_df['método_pago'].unique()],
                multi=True
            )
        ], width=4),
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

# Callback principal
@app.callback(
    Output('contenido-tab', 'children'),
    Input('tabs', 'active_tab'),
    Input('interval-tiempo-real', 'n_intervals'),
    Input('filtro-fechas', 'start_date'),
    Input('filtro-fechas', 'end_date'),
    Input('filtro-departamento', 'value'),
    Input('filtro-metodo', 'value')
)
def render_tab(tab, n_intervals, start_date, end_date, departamento, metodo):
    dfv = ventas_con_depto.copy()
    dfc = compras_df.copy()
    dfcli = clientes_df.copy()

    dfv = dfv[(dfv['fechahora_transaccion'] >= pd.to_datetime(start_date)) & (dfv['fechahora_transaccion'] <= pd.to_datetime(end_date))]
    dfc = dfc[(dfc['fechahora_transaccion'] >= pd.to_datetime(start_date)) & (dfc['fechahora_transaccion'] <= pd.to_datetime(end_date))]
    if departamento:
        dfv = dfv[dfv['departamento'].isin(departamento)]
        dfcli = dfcli[dfcli['departamento'].isin(departamento)]
    if metodo:
        dfv = dfv[dfv['método_pago'].isin(metodo)]

    if tab == 'tab-kpis':
        (total_ventas, total_compras, clientes_por_depto, producto_mas_vendido, producto_mas_valorado,
         promedio_edad, porcentaje_suscritos, ingreso_mensual_prom, prom_total_compras, prom_total_ventas) = calcular_kpis(dfv, dfc, dfcli)

        return dbc.Container([
            dbc.Row([
                dbc.Col(dbc.Card([dbc.CardHeader("💰 Total Ventas"), dbc.CardBody(f"${total_ventas:,.2f}")], color="primary", inverse=True), width=4),
                dbc.Col(dbc.Card([dbc.CardHeader("🛒 Total Compras"), dbc.CardBody(f"${total_compras:,.2f}")], color="info", inverse=True), width=4),
                dbc.Col(dbc.Card([dbc.CardHeader("🧍‍♂️ Clientes por Depto"), dbc.CardBody(html.Ul([html.Li(f"{k}: {v}") for k, v in clientes_por_depto.items()]))], color="secondary", inverse=True), width=4),
            ], className="mb-4"),

            dbc.Row([
                dbc.Col(dbc.Card([dbc.CardHeader("📦 Producto más Vendido"), dbc.CardBody(producto_mas_vendido)], color="success", inverse=True), width=4),
                dbc.Col(dbc.Card([dbc.CardHeader("💎 Producto más Valorado"), dbc.CardBody(producto_mas_valorado)], color="warning", inverse=True), width=4),
                dbc.Col(dbc.Card([dbc.CardHeader("🎂 Edad Promedio"), dbc.CardBody(f"{promedio_edad:.2f} años")], color="dark", inverse=True), width=4),
            ], className="mb-4"),

            dbc.Row([
                dbc.Col(dbc.Card([dbc.CardHeader("✅ % Clientes Suscritos"), dbc.CardBody(f"{porcentaje_suscritos:.2f}%")], color="danger", inverse=True), width=4),
                dbc.Col(dbc.Card([dbc.CardHeader("📈 Ingreso Promedio"), dbc.CardBody(f"${ingreso_mensual_prom:,.2f}")], color="secondary", inverse=True), width=4),
                dbc.Col(dbc.Card([dbc.CardHeader("📉 Prom. Total Compras"), dbc.CardBody(f"${prom_total_compras:,.2f}")], color="info", inverse=True), width=4),
            ], className="mb-4"),

            dbc.Row([
                dbc.Col(dbc.Card([dbc.CardHeader("📊 Prom. Total Ventas"), dbc.CardBody(f"${prom_total_ventas:,.2f}")], color="primary", inverse=True), width=4),
            ], className="mb-4")
        ])

    elif tab == 'tab-graficas':
        return html.Div([
            dcc.Tabs([
                dcc.Tab(label='🗺️ Ventas por Departamento', value='graf-1'),
                dcc.Tab(label='📆 Ventas en el Tiempo', value='graf-2'),
                dcc.Tab(label='🥧 Gráfico de Pastel', value='graf-3'),
                dcc.Tab(label='⏱️ Ventas Tiempo Real', value='graf-4'),
                dcc.Tab(label='📦 Gráfico 3D', value='graf-5'),
            ], id='sub-tabs', value='graf-1'),
            html.Div(id='contenido-sub-tab')
        ])

    else:
        raise PreventUpdate

@app.callback(
    Output('contenido-sub-tab', 'children'),
    Input('sub-tabs', 'value'),
    Input('interval-tiempo-real', 'n_intervals'),
    Input('filtro-fechas', 'start_date'),
    Input('filtro-fechas', 'end_date'),
    Input('filtro-departamento', 'value'),
    Input('filtro-metodo', 'value')
)
def render_sub_tab(sub_tab, n_intervals, start_date, end_date, departamento, metodo):
    dfv = ventas_con_depto.copy()
    dfc = compras_df.copy()

    dfv = dfv[(dfv['fechahora_transaccion'] >= pd.to_datetime(start_date)) & (dfv['fechahora_transaccion'] <= pd.to_datetime(end_date))]
    dfc = dfc[(dfc['fechahora_transaccion'] >= pd.to_datetime(start_date)) & (dfc['fechahora_transaccion'] <= pd.to_datetime(end_date))]
    if departamento:
        dfv = dfv[dfv['departamento'].isin(departamento)]
    if metodo:
        dfv = dfv[dfv['método_pago'].isin(metodo)]

    if sub_tab == 'graf-1':
        fig = px.bar(dfv.groupby('departamento')['total'].sum().reset_index(), x='departamento', y='total')
        return dcc.Graph(figure=fig)

    elif sub_tab == 'graf-2':
        fig = px.line(dfv.groupby('fechahora_transaccion')['total'].sum().reset_index(), x='fechahora_transaccion', y='total')
        return dcc.Graph(figure=fig)

    elif sub_tab == 'graf-3':
        deptos = dfv['departamento'].value_counts().reset_index()
        deptos.columns = ['departamento', 'count']
        fig = px.pie(deptos, names='departamento', values='count')
        return dcc.Graph(figure=fig)

    elif sub_tab == 'graf-4':
        sim_data = dfv.sample(n=20) if len(dfv) > 20 else dfv
        fig = px.bar(sim_data, x='id_venta', y='total', title=f"Simulación Tiempo Real #{n_intervals}")
        return dcc.Graph(figure=fig)

    elif sub_tab == 'graf-5':
        fig = px.scatter_3d(dfv, x='fechahora_transaccion', y='total', z='producto',
                            color='departamento', title="Gráfico 3D Ventas")
        return dcc.Graph(figure=fig)

    else:
        raise PreventUpdate

if __name__ == '__main__':
    app.run(debug=True)