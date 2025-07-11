import streamlit as st
import requests
import pandas as pd

st.set_page_config(
    page_title="Predicción de Demanda Redondos",
    layout="wide",
    page_icon="🔴"
)

st.markdown("""
    <style>
    .block-container {padding-top: 1.5rem;}
    .custom-title {
        color: #d32c2f; font-weight: 900;
        font-size: 2.1rem; letter-spacing: -1px;
        margin-bottom: 0.7rem; font-family: Segoe UI, Arial;
    }
    .logo-img {display: block; margin-left: auto; margin-right: auto;}
    .sidebar-content {font-size: 1rem;}
    .stTextInput > div > div > input {font-size: 1.1rem;}
    </style>
""", unsafe_allow_html=True)

# ---- LOGO Y TÍTULO ----
st.image("logo_redondos.png", width=110)
st.markdown('<div class="custom-title">🔮 Predicción de Demanda Redondos</div>', unsafe_allow_html=True)
st.markdown('<div style="font-size:1.17rem; color:#b30f21; font-weight:500; margin-bottom: 0.4rem;">Solicita tu pronóstico de materiales de forma inteligente</div>', unsafe_allow_html=True)
st.markdown("---")

# ---- SIDEBAR CONTROL ----
with st.sidebar:
    st.markdown('<div class="sidebar-content">', unsafe_allow_html=True)
    api_key = st.text_input("🔑 API Key Azure ML", type="password")
    st.markdown("----")
    xls_pred = st.file_uploader("Carga Excel para predicción masiva (columnas: material, fecha)", type=["xls", "xlsx"])
    st.markdown("---")
    st.write("Creado por Heidi Guevara – Redondos")
    st.markdown('</div>', unsafe_allow_html=True)

# ---- FUNCIÓN DE PREDICCIÓN ----
def predecir_demanda(fecha, materiales, api_key):
    url = "https://rdosml-xysue.eastus.inference.ml.azure.com/score"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    payload = {
        "forecast_date": fecha,
        "materials": materiales
    }
    response = requests.post(url, headers=headers, json=payload)
    if response.status_code == 200:
        try:
            resultado = response.json()
            return pd.DataFrame(resultado['predictions'])
        except Exception as e:
            return f"Error procesando la respuesta del modelo: {e}"
    else:
        return f"Error llamando al modelo Azure ML: {response.text}"

# ---- CONSULTA PUNTUAL ----
st.header("Predicción puntual")
with st.form("consulta_puntual"):
    material = st.text_input("Material", placeholder="Ej: POLLO, PAVO, etc.")
    fecha = st.text_input("Fecha de pronóstico", placeholder="YYYY-MM-DD")
    submit = st.form_submit_button("Predecir")
    if submit:
        if not (api_key and material and fecha):
            st.warning("Completa todos los campos y la API Key.")
        else:
            materiales = [m.strip().upper() for m in material.split(",") if m.strip()]
            result = predecir_demanda(fecha, materiales, api_key)
            if isinstance(result, pd.DataFrame):
                st.success(f"Predicción de demanda para {', '.join(materiales)} el {fecha}:")
                st.dataframe(result)
            else:
                st.error(result)

# ---- CONSULTA MASIVA (EXCEL) ----
st.header("Predicción masiva por Excel")
if xls_pred is not None:
    if not api_key:
        st.warning("Ingresa tu API Key para continuar.")
    else:
        df = pd.read_excel(xls_pred)
        if "material" in df.columns and "fecha" in df.columns:
            for f, group in df.groupby("fecha"):
                materiales = list(group["material"].astype(str).unique())
                result = predecir_demanda(f, materiales, api_key)
                st.markdown(f"**Materiales:** {', '.join(materiales)} | **Fecha:** {f}")
                if isinstance(result, pd.DataFrame):
                    st.dataframe(result)
                else:
                    st.error(result)
        else:
            st.error("El Excel debe tener columnas 'material' y 'fecha'.")

st.info("Este asistente está dedicado 100% a predicción de demanda vía Azure ML.\n\nContacta a Analytics Redondos para casos especiales.")
