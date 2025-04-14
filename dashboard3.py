from dash import Dash, dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.express as px
import pandas as pd
import datetime
from dash.exceptions import PreventUpdate

# Cargar datos
def cargar_datos():
    clientes_df = pd.read_excel('ClientesRelacionadas.xlsx')
    compras_df = pd.read_excel('ComprasRelacionadas.xlsx')
    ventas_df = pd.read_excel('VentasRelacionadas.xlsx')

    clientes_df.columns = clientes_df.columns.str.lower()
    compras_df.columns = compras_df.columns.str.lower()
    ventas_df.columns = ventas_df.columns.str.lower()

    clientes_df['fecha_registro'] = pd.to_datetime(clientes_df['fecha_registro'])
    compras_df['fechahora_transaccion'] = pd.to_datetime(compras_df['fechahora_transaccion'])
    ventas_df['fechahora_transaccion'] = pd.to_datetime(ventas_df['fechahora_transaccion'])

    return clientes_df, compras_df, ventas_df

clientes_df, compras_df, ventas_df = cargar_datos()

ventas_con_depto = ventas_df.merge(clientes_df[['id_cliente', 'departamento', 'genero']], on='id_cliente')

# App
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

app.layout = dbc.Container([
    html.H1("Dashboard de Ventas y Compras", className="text-center mt-4 mb-4"),

    # Filtros
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
                options=[{'label': m, 'value': m} for m in ventas_df['método_pago'].unique()],
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

    # Tabs de primer nivel
    dbc.Tabs([
        dbc.Tab(label='Resumen de KPIs', tab_id='tab-kpis'),
        dbc.Tab(label='Gráficas', tab_id='tab-graficas'),
        dbc.Tab(label='Tiempo Real', tab_id='tab-tiempo-real'),
    ], id='tabs-principal', active_tab='tab-kpis', className="mb-4"),

    # Contenido del tab principal
    html.Div(id='contenido-tab-principal'),

    # Intervalo
    dcc.Interval(id='interval-tiempo-real', interval=1000, n_intervals=0)
], fluid=True)

# Callback para renderizar el contenido principal
@app.callback(
    Output('contenido-tab-principal', 'children'),
    Input('tabs-principal', 'active_tab')
)
def mostrar_tab_principal(tab):
    if tab == 'tab-kpis':
        return html.Div("Aquí irán los KPIs...")
    elif tab == 'tab-graficas':
        return html.Div([
            dbc.Tabs([
                dbc.Tab(label='Ventas por Departamento', tab_id='subtab-depto'),
                dbc.Tab(label='Ventas en el Tiempo', tab_id='subtab-tiempo'),
                dbc.Tab(label='Gráfico de Pastel', tab_id='subtab-pastel'),
                dbc.Tab(label='Gráfico 3D', tab_id='subtab-3d'),
            ], id='tabs-secundario', active_tab='subtab-depto', className="mb-4"),
            html.Div(id='contenido-subtabs')
        ])
    elif tab == 'tab-tiempo-real':
        return html.Div("Aquí irá la simulación en tiempo real.")

# Callback para subtabs dentro de gráficas
@app.callback(
    Output('contenido-subtabs', 'children'),
    Input('tabs-secundario', 'active_tab'),
    Input('filtro-fechas', 'start_date'),
    Input('filtro-fechas', 'end_date'),
    Input('filtro-departamento', 'value'),
    Input('filtro-metodo', 'value'),
    Input('filtro-genero', 'value'),
)
def mostrar_subtabs(subtab, start_date, end_date, departamento, metodo, genero):
    df = ventas_con_depto.copy()

    df = df[(df['fechahora_transaccion'] >= pd.to_datetime(start_date)) & (df['fechahora_transaccion'] <= pd.to_datetime(end_date))]

    if departamento:
        df = df[df['departamento'].isin(departamento)]
    if metodo:
        df = df[df['método_pago'].isin(metodo)]
    if genero:
        df = df[df['genero'].isin(genero)]

    if df.empty:
        return dbc.Alert("No hay datos con los filtros aplicados.", color="warning")

    if subtab == 'subtab-depto':
        fig = px.bar(df.groupby('departamento')['total'].sum().reset_index(), x='departamento', y='total', title='Ventas por Departamento')
        return dcc.Graph(figure=fig)

    elif subtab == 'subtab-tiempo':
        fig = px.line(df.groupby(df['fechahora_transaccion'].dt.date)['total'].sum().reset_index(),
                      x='fechahora_transaccion', y='total', title='Ventas en el Tiempo')
        return dcc.Graph(figure=fig)

    elif subtab == 'subtab-pastel':
        fig = px.pie(df, names='método_pago', values='total', title='Distribución por Método de Pago')
        return dcc.Graph(figure=fig)

    elif subtab == 'subtab-3d':
        fig = px.scatter_3d(df, x='cantidad', y='total', z='precio_unitario', color='departamento', title='Gráfico 3D de Ventas')
        return dcc.Graph(figure=fig)

    return html.Div("Selecciona una subpestaña.")
