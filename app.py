import streamlit as st
import requests
import pandas as pd
import openai

# ================== CONFIGURACIÓN BÁSICA ==================
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

# ================== SIDEBAR CONTROL ==================
with st.sidebar:
    st.markdown('<div class="sidebar-content">', unsafe_allow_html=True)
    api_key = st.text_input("🔑 API Key Azure ML", type="password")
    openai_api_key = st.text_input("🔑 API Key OpenAI", type="password")
    st.markdown("----")
    xls_pred = st.file_uploader("Carga Excel para predicción masiva (columnas: material, fecha)", type=["xls", "xlsx"])
    st.markdown("---")
    st.write("Creado por Heidi Guevara – Redondos")
    st.markdown('</div>', unsafe_allow_html=True)

# ================== FUNCIÓN DE PREDICCIÓN ==================
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

# ================== UI PRINCIPAL: PREDICCIÓN ==================
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

# ================== MÓDULO EXTRA: CHAT IA GENERATIVA ==================
st.markdown("---")
st.header("🤖 Chat IA Generativa – Copiloto de Demanda")

# Historial en sesión (para chat)
if "chat_historial" not in st.session_state:
    st.session_state["chat_historial"] = []

prompt_sistema = """
Eres un asistente experto en análisis de demanda para la empresa Redondos. Responde de manera clara, ejecutiva y en lenguaje natural. 
- Si el usuario te pregunta sobre predicciones de demanda para un material y fecha, devuelve el resultado de la predicción y explícalo.
- Si te piden recomendaciones, analiza los resultados y sugiere acciones concretas.
"""

def obtener_prediccion_natural(pregunta, api_key_aml):
    # Busca fecha y material en la pregunta con heurística simple (puedes mejorar con NLP)
    import re
    fecha_match = re.search(r"(20\d{2}-\d{2}-\d{2})", pregunta)
    fecha = fecha_match.group(1) if fecha_match else None
    mat_match = re.findall(r"(pollo|pavo|huevo|materiales?\s?\d+|[A-Z0-9_ -]+)", pregunta, re.IGNORECASE)
    materiales = [m.upper().strip() for m in mat_match if len(m.strip()) > 2]
    # Si encontró ambos, pide predicción al modelo
    if fecha and materiales and api_key_aml:
        result = predecir_demanda(fecha, materiales, api_key_aml)
        if isinstance(result, pd.DataFrame):
            df_md = result.to_markdown(index=False)
            return f"Predicción de demanda para {', '.join(materiales)} el {fecha}:\n\n{df_md}"
        else:
            return str(result)
    else:
        return None

with st.form("form_chat_ia"):
    pregunta = st.text_input("Pregunta (ejemplo: ¿Qué demanda se espera para POLLO el 2025-12-31? o pídeme un análisis o recomendación)", key="chat_input")
    enviar_chat = st.form_submit_button("Enviar")
    if enviar_chat and pregunta.strip():
        respuesta_pred = obtener_prediccion_natural(pregunta, api_key)
        if respuesta_pred:
            contexto = f"{respuesta_pred}"
        else:
            contexto = ""
        if openai_api_key:
            openai.api_key = openai_api_key
            prompt_usuario = f"{prompt_sistema}\n\nPregunta usuario: {pregunta}\n\n{contexto}"
            try:
                respuesta_ia = openai.ChatCompletion.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": prompt_sistema},
                        {"role": "user", "content": prompt_usuario}
                    ],
                    max_tokens=800,
                    temperature=0.2
                ).choices[0].message.content.strip()
            except Exception as e:
                respuesta_ia = f"Error al llamar a OpenAI: {e}"
        else:
            respuesta_ia = "Por favor ingresa tu API Key de OpenAI en el sidebar para usar la IA generativa."
        st.session_state["chat_historial"].append({
            "pregunta": pregunta,
            "respuesta": respuesta_ia
        })
        st.rerun()

# Mostrar historial del chat IA
if st.session_state["chat_historial"]:
    st.markdown("#### Historial del Chat IA Generativa")
    for h in reversed(st.session_state["chat_historial"]):
        st.markdown(f"**Tú:** {h['pregunta']}")
        st.markdown(f"**Copiloto IA:** {h['respuesta']}")

# Botón para limpiar historial
if st.button("🧹 Borrar historial de chat IA"):
    st.session_state["chat_historial"] = []
