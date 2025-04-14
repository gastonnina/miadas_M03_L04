
# 📊 Dashboard Interactivo de Ventas y Compras

Este proyecto es un dashboard web interactivo creado con [Dash](https://dash.plotly.com/), diseñado para visualizar datos de clientes, compras y ventas a partir de archivos Excel. Incluye filtros dinámicos, KPIs, visualizaciones 2D, 3D y simulación en tiempo real.

---

## 🧰 Tecnologías utilizadas

- Python 3.12
- Dash + Plotly
- Dash Bootstrap Components
- PDM (gestor de dependencias moderno)
- Pandas
- Docker (opcional)

---

## 📁 Estructura del Proyecto

```
├── dashboard.py
├── ClientesRelacionadas.xlsx
├── ComprasRelacionadas.xlsx
├── VentasRelacionadas.xlsx
├── pyproject.toml
├── .gitignore
├── .dockerignore
├── Dockerfile
└── README.md
```

---

## 🚀 Ejecución local (sin Docker)

### 1. Crear entorno virtual

```bash
python3.12 -m venv .venv
source .venv/bin/activate  # En Windows: .venv\Scripts\activate
```

### 2. Instalar PDM y dependencias

```bash
pip install pdm
pdm install
```

### 3. Ejecutar el dashboard

```bash
pdm run python dashboard.py
```

---

## 🐳 Ejecución con Docker

### 1. Construir la imagen

```bash
docker build -t dashboard-app .
```

### 2. Ejecutar el contenedor

```bash
docker run -p 8050:8050 -v $(pwd):/app dashboard-app
```

> **Nota:** Asegúrate de que los archivos `.xlsx` estén en el mismo directorio para que sean accesibles desde el contenedor.

---

## 📊 Funcionalidades

- **Filtros dinámicos**: por fecha, departamento y método de pago.
- **KPIs**: Total de ventas, compras, clientes por departamento, producto más vendido, edad promedio, porcentaje de suscritos, etc.
- **Visualizaciones**:
  - Gráfico de barras por departamento
  - Evolución temporal de ventas
  - Gráfico de pastel por distribución
  - Gráfico en tiempo real
  - Gráfico 3D interactivo

---

## 📝 Créditos

Desarrollado por [Tu Nombre]. Proyecto académico de visualización de datos con Dash y Python.