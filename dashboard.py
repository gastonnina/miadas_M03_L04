from dash import Dash, dcc, html, Input, Output
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objs as go
import pandas as pd
import numpy as np
import datetime
from dash.exceptions import PreventUpdate
import random
from datetime import datetime
import uuid

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
print("Columnas en ventas_df:", ventas_df.columns.tolist())


# Opciones simuladas
productos = ['Laptop', 'Smartphone', 'Tablet', 'Audífonos', 'Monitor']
metodos_pago = ['Tarjeta', 'Efectivo', 'Transferencia']
estados_venta = ['Completado', 'Pendiente', 'Cancelado']

# DataFrame global para simular ventas
ventas_simuladas = pd.DataFrame(columns=[
    'id_transaccion', 'id_cliente', 'producto', 'fechahora_transaccion',
    'cantidad', 'precio_unitario', 'total', 'id_venta',
    'metodo_pago', 'estado_venta'
])


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
        suscritos = suscritos[suscritos.isin(['sí', 'si','true', 'no', 'false'])]
        porcentaje_suscritos = suscritos.map({'sí': 1, 'si': 1, 'true': 1, 'no': 0, 'false': 0}).mean() * 100 if not suscritos.empty else 0
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
def aplicar_filtros(dfv, dfc, dfcli, start_date, end_date, departamento, metodo, genero, producto):
    print("🎛️ Aplicando filtros...")
    try:
        dfv = dfv[(dfv['fechahora_transaccion'] >= pd.to_datetime(start_date)) & (dfv['fechahora_transaccion'] <= pd.to_datetime(end_date))]
        dfc = dfc[(dfc['fechahora_transaccion'] >= pd.to_datetime(start_date)) & (dfc['fechahora_transaccion'] <= pd.to_datetime(end_date))]
        if departamento:
            dfv = dfv[dfv['departamento'].isin(departamento)]
            dfcli = dfcli[dfcli['departamento'].isin(departamento)]
        if metodo:
            dfv = dfv[dfv['metodo_pago'].isin(metodo)]
        if genero:
            dfcli = dfcli[dfcli['genero'].isin(genero)]
            dfv = dfv[dfv['id_cliente'].isin(dfcli['id_cliente'])]
        if producto:
            dfv = dfv[dfv['producto'].isin(producto)]
            dfc = dfc[dfc['producto'].isin(producto)]
        print("✅ Filtros aplicados")
        return dfv, dfc, dfcli
    except Exception as e:
        print(f"❌ Error aplicando filtros: {e}")
        return dfv, dfc, dfcli

# Layout
print("📐 Definiendo layout de la aplicación...")
app.layout = dbc.Container([
    html.H1("📊 Dashboard Interactivo", className="text-center my-4"),

     dbc.Row([
        dbc.Col([
            html.H5("📅 Rangos de Fecha", className="mb-1"),
            dcc.DatePickerRange(
                id='filtro-fechas',
                start_date=ventas_df['fechahora_transaccion'].min().date(),
                end_date=ventas_df['fechahora_transaccion'].max().date(),
                display_format='DD/MM/YYYY',
                className='mb-2'
            ),
        ], width=2),
        dbc.Col([
            dbc.Label("📍 Departamento"),
            dcc.Dropdown(
                options=[{"label": d, "value": d} for d in sorted(clientes_df['departamento'].dropna().unique())],
                multi=True,
                id='filtro-departamento',
                className='mb-2'
            ),
        ], width=3),
        dbc.Col([
            dbc.Label("💳 Método de Pago"),
            dcc.Dropdown(
                options=[{"label": m, "value": m} for m in sorted(ventas_df['metodo_pago'].dropna().unique())],
                multi=True,
                id='filtro-metodo',
                className='mb-2'
            ),
        ], width=3),
        dbc.Col([
            dbc.Label("🚻 Género"),
            dcc.Dropdown(
                options=[{"label": g, "value": g} for g in sorted(clientes_df['genero'].dropna().unique())],
                multi=True,
                id='filtro-genero',
                className='mb-2'
            )
        ], width=1),
        dbc.Col([
            html.Label("📦 Producto:"),
            dcc.Dropdown(
                id='filtro-producto',
                options=[{'label': p, 'value': p} for p in sorted(ventas_df['producto'].dropna().unique())],
                multi=True
            ),
        ], width=3),
        dcc.Tabs(id='main-tabs', value='kpis', children=[
            dcc.Tab(label='📈 KPIs', value='kpis'),
            dcc.Tab(label='📊 Gráficas', value='graficas', children=[
                dcc.Tabs(id='sub-tabs', value='graf-ventas-depto', children=[
                    dcc.Tab(label='📊 Ventas por Depto', value='graf-ventas-depto'),
                    dcc.Tab(label='📈 Ventas en el tiempo', value='graf-ventas-tiempo'),
                    dcc.Tab(label='🥧 Clientes por Depto', value='graf-clientes-depto'),
                    dcc.Tab(label='📈 Compras en el tiempo', value='graf-compras-tiempo'),
                    dcc.Tab(label='💳 Ventas por Método', value='graf-metodo-pago'),
                    dcc.Tab(label='🧊 Gráfico 3D', value='graf-3d')
                ]),
                html.Div(id='contenido-sub-tab', className='mt-4')
            ]),
            dcc.Tab(label='⏱️ Ventas en tiempo real', value='tiempo-real')
        ]),
        html.Div(id='contenido-tab', className='mt-4'),
        dcc.Interval(id='intervalo-tiempo-real', interval=1000, n_intervals=0),
        html.Footer(
            html.Div(f"Creado con ❤️ por Gaston Nina · {datetime.now().year}", className="text-center text-muted my-4"),
            className="mt-5"
        )
    ])
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
    Input('main-tabs', 'value'),
    Input('filtro-fechas', 'start_date'),
    Input('filtro-fechas', 'end_date'),
    Input('filtro-departamento', 'value'),
    Input('filtro-metodo', 'value'),
    Input('filtro-genero', 'value'),
    Input('filtro-producto', 'value'),
    Input('intervalo-tiempo-real', 'n_intervals'),
)
def mostrar_contenido_tab(tab, start_date, end_date, departamento, metodo, genero, producto, n):
    if tab == 'kpis':
        dfv, dfc, dfcli = aplicar_filtros(ventas_con_depto.copy(), compras_df.copy(), clientes_df.copy(), start_date, end_date, departamento, metodo, genero, producto)
        kpis = calcular_kpis(dfv, dfc, dfcli)
        clientes_table = dbc.Table.from_dataframe(kpis[2].reset_index().rename(columns={'index': 'Departamento', 'departamento': 'Cantidad'}), striped=True, bordered=True, hover=True, className="mt-3")

        return html.Div([
            dbc.Row([
                dbc.Col(dbc.Card([dbc.CardHeader("💰 Total de Ventas"), dbc.CardBody(f"Bs. {kpis[0]:,.2f}")]), width=4),
                dbc.Col(dbc.Card([dbc.CardHeader("🛒 Total de Compras"), dbc.CardBody(f"Bs. {kpis[1]:,.2f}")]), width=4),
                dbc.Col(dbc.Card([dbc.CardHeader("🔥 Más Vendido (Cantidad)"), dbc.CardBody(kpis[3])]), width=4),
                dbc.Col(dbc.Card([dbc.CardHeader("💎 Más Vendido (Monto)"), dbc.CardBody(kpis[4])]), width=4),
                dbc.Col(dbc.Card([dbc.CardHeader("🎂 Edad Promedio"), dbc.CardBody(f"{kpis[5]:.1f} años")]), width=4),
                dbc.Col(dbc.Card([dbc.CardHeader("📬 % Suscritos"), dbc.CardBody(f"{kpis[6]:.1f}%")]), width=4),
                dbc.Col(dbc.Card([dbc.CardHeader("💼 Ingreso Promedio"), dbc.CardBody(f"Bs. {kpis[7]:,.2f}")]), width=4),
                dbc.Col(dbc.Card([dbc.CardHeader("📦 Promedio Compras"), dbc.CardBody(f"Bs. {kpis[8]:,.2f}")]), width=4),
                dbc.Col(dbc.Card([dbc.CardHeader("📈 Promedio Ventas"), dbc.CardBody(f"Bs. {kpis[9]:,.2f}")]), width=4),
            ], className="g-3"),
            html.Hr(),
            html.H4("🧭 Clientes por Departamento"),
            clientes_table
        ])
    elif tab == 'graficas':
        # Aquí iría el contenido de gráficas
        return html.Div([
            # DIV VACIO
        ])
    if tab == 'tiempo-real':
        try:
            print(f"🔄 Actualización tiempo real #{n}")
            global ventas_simuladas

            # Generar una nueva fila con datos aleatorios
            producto = random.choice(productos)
            cantidad = random.randint(1, 5)
            precio_unitario = round(random.uniform(100, 1000), 2)
            total = round(cantidad * precio_unitario, 2)

            nueva_fila = {
                'id_transaccion': str(uuid.uuid4())[:8],
                'id_cliente': random.randint(1000, 9999),
                'producto': producto,
                'fechahora_transaccion': datetime.now(),
                'cantidad': cantidad,
                'precio_unitario': precio_unitario,
                'total': total,
                'id_venta': random.randint(10000, 99999),
                'metodo_pago': random.choice(metodos_pago),
                'estado_venta': random.choice(estados_venta)
            }

            # Agregar al DataFrame
            ventas_simuladas = pd.concat(
                [ventas_simuladas, pd.DataFrame([nueva_fila])], ignore_index=True)

            # Mostrar últimas 20
            ventas_actuales = ventas_simuladas.sort_values(
                by='fechahora_transaccion', ascending=False).head(20)

            # Crear gráfico
            fig = px.line(
                ventas_actuales.sort_values('fechahora_transaccion'),
                x='fechahora_transaccion',
                y='total',
                title='📈 Total por transacción',
                markers=True,
                hover_data=['producto', 'cantidad', 'precio_unitario', 'metodo_pago', 'estado_venta']
            )
            fig.update_layout(
                template='plotly_dark',
                xaxis_title='Fecha y Hora',
                yaxis_title='Total Venta'
            )
            return dcc.Graph(figure=fig)
        except Exception as e:
            print(f"❌ Error en actualización tiempo real: {e}")
            return html.Div("❌ Error al actualizar ventas en tiempo real")
    raise PreventUpdate
    return None

@app.callback(
    Output('contenido-sub-tab', 'children'),
    Input('sub-tabs', 'value'),
    Input('filtro-fechas', 'start_date'),
    Input('filtro-fechas', 'end_date'),
    Input('filtro-departamento', 'value'),
    Input('filtro-metodo', 'value'),
    Input('filtro-genero', 'value'),
    Input('filtro-producto', 'value'),
)
def mostrar_contenido_subtab(subtab, start_date, end_date, departamento, metodo, genero, producto):
    print(f"🛎️ Callback activado con subtab: {subtab}")
    print(f"📅 Fechas: {start_date} - {end_date}")
    print(f"📍 Departamento: {departamento}, 💳 Método: {metodo}, 🚻 Género: {genero}, 🚻 Producto: {producto}")
    try:
        dfv, dfc, dfcli = aplicar_filtros(ventas_con_depto.copy(), compras_df.copy(), clientes_df.copy(), start_date, end_date, departamento, metodo, genero, producto)

        if dfv.empty or dfc.empty or dfcli.empty:
            print("⚠️ Uno de los dataframes está vacío después de los filtros")
            return html.Div("No hay datos para mostrar con los filtros actuales.")

        if subtab == 'graf-ventas-depto':
            df_grouped = dfv.groupby('departamento')['total'].sum().reset_index()
            print(df_grouped.head())
            fig = px.bar(df_grouped, x='departamento', y='total', title='Ventas por Departamento')
            return dcc.Graph(figure=fig)

        elif subtab == 'graf-ventas-tiempo':
            df_line = dfv.groupby(dfv['fechahora_transaccion'].dt.date)['total'].sum().reset_index()
            print(df_line.head())
            fig = px.line(df_line, x='fechahora_transaccion', y='total', title='Ventas a lo largo del tiempo')
            return dcc.Graph(figure=fig)

        elif subtab == 'graf-clientes-depto':
            df_pie = dfcli['departamento'].value_counts().reset_index()
            df_pie.columns = ['departamento', 'cantidad']
            print(df_pie.head())
            fig = px.pie(df_pie, names='departamento', values='cantidad', title='Clientes por Departamento')
            return dcc.Graph(figure=fig)

        elif subtab == 'graf-compras-tiempo':
            df_line = dfc.groupby(dfc['fechahora_transaccion'].dt.date)['total'].sum().reset_index()
            print(df_line.head())
            fig = px.line(df_line, x='fechahora_transaccion', y='total', title='Compras a lo largo del tiempo')
            return dcc.Graph(figure=fig)

        elif subtab == 'graf-metodo-pago':
            df_grouped = dfv.groupby('metodo_pago')['total'].sum().reset_index()
            print(df_grouped.head())
            fig = px.bar(df_grouped, x='metodo_pago', y='total', title='Total de Ventas por Método de Pago')
            return dcc.Graph(figure=fig)

        elif subtab == 'graf-3d':
            print("🧊 Generando gráfico 3D...")
            columnas_disponibles = dfv.columns.tolist()
            print("Columnas disponibles en dfv:", columnas_disponibles)

            if 'producto' in dfv.columns:
                z_col = 'producto'
            elif 'cantidad' in dfv.columns:
                z_col = 'cantidad'
            else:
                return html.Div("❌ No hay columnas adecuadas para el eje Z del gráfico 3D.")

            print(dfv[['fechahora_transaccion', 'total', z_col]].head())
            fig = px.scatter_3d(dfv, x='fechahora_transaccion', y='total', z=z_col,
                                title='Relación 3D: Fecha vs Monto vs ' + z_col,
                                color='departamento')
            return dcc.Graph(figure=fig)

    except Exception as e:
        print(f"❌ Error en mostrar_contenido_subtab: {e}")
        return html.Div("Ocurrió un error al generar la visualización.")

    return html.Div("Selecciona un gráfico válido.")

if __name__ == '__main__':
    app.run(debug=True)
