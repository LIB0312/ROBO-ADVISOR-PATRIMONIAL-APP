import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import scipy.optimize as opt
from datetime import datetime, timedelta

st.set_page_config(page_title="Easy Help", layout="wide")

st.title("Easy Help")
st.markdown("Plataforma cuantitativa para la asignacion optima de capital, analisis de la Frontera Eficiente y gestion de riesgos extremos (VaR & Drawdown).")

st.sidebar.header("Parametros del Inversionista")

perfil_riesgo = st.sidebar.select_slider(
    "Tolerancia al Riesgo:",
    options=["Muy Conservador", "Conservador", "Moderado", "Agresivo", "Muy Agresivo"],
    value="Moderado"
)

st.sidebar.subheader("Universo de Inversion")
opcion_activos = st.sidebar.radio(
    "Seleccione el mercado:",
    ("ETFs Globales (Diversificado)", "Mercado Mexicano (IPC)", "Personalizado (Yahoo Finance)")
)

if opcion_activos == "ETFs Globales (Diversificado)":
    tickers = {
        'Bonos Tesoro (SHV)': 'SHV',
        'Bonos Corp (LQD)': 'LQD',
        'S&P 500 (SPY)': 'SPY',
        'Mercados Emergentes (VWO)': 'VWO'
    }
elif opcion_activos == "Mercado Mexicano (IPC)":
    tickers = {
        'Walmex': 'WALMEX.MX',
        'America Movil': 'AMXB.MX',
        'Banorte': 'GFNORTEO.MX',
        'Cemex': 'CEMEXCPO.MX'
    }
else:
    st.sidebar.markdown("---")
    input_usuario = st.sidebar.text_input(
        "Ingrese los Tickers (separados por coma):", 
        value="AAPL, MSFT, NVDA, GOOGL"
    )
    
    lista_tickers = [t.strip().upper() for t in input_usuario.split(',') if t.strip()]
    
    if len(lista_tickers) < 2:
        st.sidebar.warning("Ingresa al menos 2 activos para poder diversificar el portafolio.")
        lista_tickers = ["AAPL", "MSFT"]
        
    tickers = {t: t for t in lista_tickers}

horizonte = st.sidebar.slider("Anios de datos historicos para el modelo:", 1, 5, 3)

@st.cache_data
def cargar_datos(tickers_dict, years):
    lista_tickers = list(tickers_dict.values())
    hoy = datetime.today()
    inicio = hoy - timedelta(days=365 * years) 
    
    # Descarga directa
    data = yf.download(lista_tickers, start=inicio, end=hoy, interval="1d")
    
    if isinstance(data.columns, pd.MultiIndex):
        # Extraemos solo los precios de cierre y aplanamos el MultiIndex
        df = data['Close']
    else:
        # Si no es MultiIndex, es un DataFrame simple de un activo
        df = data[['Close']]
        df.columns = [lista_tickers[0]]
    df = df.rename(columns={v: k for k, v in tickers_dict.items()})
    
    # Rellenamos huecos (días festivos) y eliminamos los que realmente no tienen datos
    return df.ffill().dropna()
    
    # Proteccion contra diferentes formatos de yfinance
    if isinstance(df.columns, pd.MultiIndex):
        if 'Close' in df.columns.get_level_values(0):
            df = df['Close']
        else:
            df = df.xs('Close', level=1, axis=1)
    elif 'Close' in df.columns:
        df = df[['Close']]
        
    # Renombrar columnas
    df = df.rename(columns={v: k for k, v in tickers_dict.items()})
    return df.ffill().dropna()

with st.spinner('Procesando datos y calculando matrices...'):
    precios = cargar_datos(tickers, horizonte)
    
    if precios.empty or precios.shape[1] < 2:
        st.error("Error al procesar los datos. Asegurate de que los Tickers ingresados existan en Yahoo Finance y que haya al menos 2 activos validos.")
        st.stop()
    
    rendimientos_diarios = precios.pct_change().dropna()
    rend_esperado_anual = rendimientos_diarios.mean() * 252
    matriz_cov_anual = rendimientos_diarios.cov() * 252

num_portafolios = 10000
num_activos = len(tickers)

pesos_aleatorios = np.random.random((num_portafolios, num_activos))
pesos_aleatorios = pesos_aleatorios / np.sum(pesos_aleatorios, axis=1)[:, np.newaxis]

rendimientos_simulados = np.dot(pesos_aleatorios, rend_esperado_anual)
riesgos_simulados = np.sqrt(np.einsum('ij,jk,ik->i', pesos_aleatorios, matriz_cov_anual.values, pesos_aleatorios))
sharpe_ratios = rendimientos_simulados / riesgos_simulados

if perfil_riesgo == "Muy Conservador":
    idx_optimo = np.argmin(riesgos_simulados)
elif perfil_riesgo == "Muy Agresivo":
    idx_optimo = np.argmax(rendimientos_simulados)
else:
    riesgo_min = np.min(riesgos_simulados)
    riesgo_max = np.max(riesgos_simulados)
    
    if perfil_riesgo == "Conservador":
        riesgo_objetivo = riesgo_min + (riesgo_max - riesgo_min) * 0.25
        idx_optimo = np.argmin(np.abs(riesgos_simulados - riesgo_objetivo))
    elif perfil_riesgo == "Moderado":
        idx_optimo = np.argmax(sharpe_ratios)
    elif perfil_riesgo == "Agresivo":
        riesgo_objetivo = riesgo_min + (riesgo_max - riesgo_min) * 0.75
        idx_optimo = np.argmin(np.abs(riesgos_simulados - riesgo_objetivo))

pesos_recomendados = pesos_aleatorios[idx_optimo]
rend_recomendado = rendimientos_simulados[idx_optimo]
riesgo_recomendado = riesgos_simulados[idx_optimo]

rend_historico_portafolio = rendimientos_diarios.dot(pesos_recomendados)

var_95 = np.percentile(rend_historico_portafolio, 5)

crecimiento_acumulado = (1 + rend_historico_portafolio).cumprod()
picos_historicos = crecimiento_acumulado.cummax()
drawdowns = (crecimiento_acumulado - picos_historicos) / picos_historicos
max_drawdown = drawdowns.min()

tab1, tab2, tab3 = st.tabs(["Optimizacion y Frontera", "Gestion de Riesgos", "Backtesting y Desempeno"])

with tab1:
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Frontera Eficiente de Markowitz")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=riesgos_simulados, y=rendimientos_simulados, mode='markers',
            marker=dict(color=sharpe_ratios, colorscale='Viridis', showscale=True, size=5, opacity=0.4),
            name='Universo Posible'
        ))
        fig.add_trace(go.Scatter(
            x=[riesgo_recomendado], y=[rend_recomendado], mode='markers+text',
            marker=dict(color='red', size=14, symbol='star', line=dict(width=2, color='DarkSlateGrey')),
            text=["Portafolio Sugerido"], textposition="top center", name='Sugerencia'
        ))
        fig.update_layout(xaxis_title="Volatilidad Esperada (Riesgo)", yaxis_title="Rendimiento Esperado Anual", height=450, template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Asignacion de Capital")
        st.info(f"Estrategia: {perfil_riesgo}")
        st.metric(label="Rendimiento Proyectado", value=f"{rend_recomendado*100:.2f}%")
        st.metric(label="Riesgo (Volatilidad)", value=f"{riesgo_recomendado*100:.2f}%")
        
        fig_pie = go.Figure(data=[go.Pie(labels=list(tickers.keys()), values=pesos_recomendados, hole=.4, textinfo='label+percent')])
        fig_pie.update_layout(height=350, showlegend=False, margin=dict(t=0, b=0, l=0, r=0))
        st.plotly_chart(fig_pie, use_container_width=True)

with tab2:
    st.subheader("Metricas Avanzadas de Riesgo del Portafolio Sugerido")
    st.markdown("Evaluacion del riesgo de perdida extrema para administracion de riesgos institucionales.")
    
    rc1, rc2, rc3 = st.columns(3)
    with rc1:
        st.error(f"Valor en Riesgo (VaR 95%) Diario:\n {var_95*100:.2f}%")
    with rc2:
        st.warning(f"Caida Maxima:\n {max_drawdown*100:.2f}%")
    with rc3:
        st.success(f"Ratio de Sharpe:\n {rend_recomendado/riesgo_recomendado:.2f}")

    st.markdown("---")
    st.subheader("Distribucion de Rendimientos Diarios")
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Histogram(x=rend_historico_portafolio, nbinsx=100, name="Rendimientos", marker_color='indigo'))
    fig_hist.add_vline(x=var_95, line_dash="dash", line_color="red", annotation_text="VaR 95%")
    fig_hist.update_layout(height=350, template="plotly_white", xaxis_title="Rendimiento Diario", yaxis_title="Frecuencia")
    st.plotly_chart(fig_hist, use_container_width=True)

with tab3:
    st.subheader("Backtesting Historico (Base 100)")
    st.markdown("Evolucion del capital si se hubieran invertido $100 en el portafolio sugerido frente a los activos individuales.")
    
    precios_base100 = (precios / precios.iloc[0]) * 100
    
    crecimiento_portafolio = (1 + rend_historico_portafolio).cumprod() * 100
    precios_base100['PORTAFOLIO SUGERIDO'] = crecimiento_portafolio
    
    st.line_chart(precios_base100, height=400)
