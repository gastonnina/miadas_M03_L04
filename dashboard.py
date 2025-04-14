
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

    if df_clientes['suscrito'].dtype == object:
        porcentaje_suscritos = df_clientes['suscrito'].astype(str).str.lower().map({'sí': 1, 'si': 1, 'no': 0}).mean() * 100
    else:
        porcentaje_suscritos = 0

    ingreso_mensual_prom = df_clientes['ingreso_mensual'].mean()
    prom_total_compras = df_compras['total'].mean()
    prom_total_ventas = df_ventas['total'].mean()
    return [total_ventas, total_compras, clientes_por_depto, producto_mas_vendido, producto_mas_valorado,
            promedio_edad, porcentaje_suscritos, ingreso_mensual_prom, prom_total_compras, prom_total_ventas]

# Layout
app.layout = dbc.Container([
    html.H1("Dashboard de Ventas y Compras", className="text-center mt-4 mb-4"),

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

    dbc.Row([
        dbc.Col(dbc.Card([dbc.CardHeader("Total Ventas"), dbc.CardBody(id='kpi-total-ventas')], color="primary", inverse=True), width=4),
        dbc.Col(dbc.Card([dbc.CardHeader("Total Compras"), dbc.CardBody(id='kpi-total-compras')], color="info", inverse=True), width=4),
        dbc.Col(dbc.Card([dbc.CardHeader("Clientes por Depto"), dbc.CardBody(id='kpi-clientes')], color="secondary", inverse=True), width=4),
    ], className="mb-4"),

    dbc.Tabs([
        dbc.Tab(label='Ventas por Departamento', tab_id='tab-1'),
        dbc.Tab(label='Ventas en el Tiempo', tab_id='tab-2'),
        dbc.Tab(label='Gráfico de Pastel', tab_id='tab-3'),
        dbc.Tab(label='Ventas Tiempo Real', tab_id='tab-4'),
        dbc.Tab(label='Gráfico 3D', tab_id='tab-5'),
    ], id='tabs', active_tab='tab-1', className="mb-4"),

    html.Div(id='contenido-tab'),

    dcc.Interval(
        id='interval-tiempo-real',
        interval=1000,
        n_intervals=0
    )
], fluid=True)

@app.callback(
    Output('kpi-total-ventas', 'children'),
    Output('kpi-total-compras', 'children'),
    Output('kpi-clientes', 'children'),
    Input('filtro-fechas', 'start_date'),
    Input('filtro-fechas', 'end_date'),
    Input('filtro-departamento', 'value'),
    Input('filtro-metodo', 'value')
)
def actualizar_kpis(start_date, end_date, departamento, metodo):
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

    total_ventas, total_compras, clientes_por_depto, *_ = calcular_kpis(dfv, dfc, dfcli)
    return (
        f"${total_ventas:,.2f}",
        f"${total_compras:,.2f}",
        html.Ul([html.Li(f"{k}: {v}") for k, v in clientes_por_depto.items()])
    )

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
    df = ventas_con_depto.copy()
    df = df[(df['fechahora_transaccion'] >= pd.to_datetime(start_date)) & (df['fechahora_transaccion'] <= pd.to_datetime(end_date))]
    if departamento:
        df = df[df['departamento'].isin(departamento)]
    if metodo:
        df = df[df['método_pago'].isin(metodo)]

    if tab == 'tab-1':
        fig = px.bar(df.groupby('departamento')['total'].sum().reset_index(), x='departamento', y='total')
        return dcc.Graph(figure=fig)

    elif tab == 'tab-2':
        fig = px.line(df.groupby('fechahora_transaccion')['total'].sum().reset_index(), x='fechahora_transaccion', y='total')
        return dcc.Graph(figure=fig)

    elif tab == 'tab-3':
        deptos = df['departamento'].value_counts().reset_index()
        deptos.columns = ['departamento', 'count']
        fig = px.pie(deptos, names='departamento', values='count')
        return dcc.Graph(figure=fig)

    elif tab == 'tab-4':
        sim_data = df.sample(n=20) if len(df) > 20 else df
        fig = px.bar(sim_data, x='id_venta', y='total', title=f"Simulación Tiempo Real #{n_intervals}")
        return dcc.Graph(figure=fig)

    elif tab == 'tab-5':
        fig = px.scatter_3d(df, x='fechahora_transaccion', y='total', z='producto',
                            color='departamento', title="Gráfico 3D Ventas")
        return dcc.Graph(figure=fig)

    else:
        raise PreventUpdate

if __name__ == '__main__':
    app.run(debug=True)
