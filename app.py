import streamlit as st
import requests
import pandas as pd
import json
from openai import OpenAI
import re

# ----- CONFIG -----
st.set_page_config(page_title="Predicción de Demanda Redondos", layout="wide", page_icon="🔮")

st.markdown("""
    <style>
    .custom-title {
        color: #d32c2f; font-weight: 900; font-size: 2.1rem; letter-spacing: -1px;
        margin-bottom: 0.3rem; font-family: Segoe UI, Arial;
    }
    .custom-sub {
        color: #b30f21; font-size:1.17rem; font-weight:500; margin-bottom:0.7rem;
    }
    .chat-bubble-user { color: #d32c2f; font-weight: bold; }
    .chat-bubble-bot { background:#f8f8f8; border-left:4px solid #d32c2f; padding:10px 20px; border-radius:10px; margin-bottom:14px;}
    .btn-clear button {
        background-color: #ececec !important; color: #333 !important; border: none !important;
        font-size: 0.95rem !important; padding: 3px 13px !important; border-radius: 6px !important;
        box-shadow: none !important; margin-bottom: 13px !important; margin-top: 3px; margin-left: 8px;
        transition: background 0.17s;
    }
    .btn-clear button:hover { background-color: #d3d3d3 !important; color: #d32c2f !important; }
    </style>
""", unsafe_allow_html=True)

st.image("logo_redondos.png", width=110)
st.markdown('<div class="custom-title">🔮 Predicción de Demanda Redondos</div>', unsafe_allow_html=True)
st.markdown('<div class="custom-sub">Consulta puntual, masiva y conversación con IA Generativa</div>', unsafe_allow_html=True)
st.markdown("---")

# ----- SIDEBAR: API Keys y Carga -----
with st.sidebar:
    st.markdown("🔑 <b>API Key Azure ML</b>", unsafe_allow_html=True)
    azureml_api_key = st.text_input(" ", type="password", key="azureml_api")
    st.markdown("---")
    st.markdown("🔑 <b>API Key OpenAI (Chat IA)</b>", unsafe_allow_html=True)
    openai_api_key = st.text_input(" ", type="password", key="openai_api")
    st.markdown("---")
    st.write("Carga Excel para predicción masiva\n(columnas: material, fecha)")
    excel_file = st.file_uploader("Drag and drop file here", type=["xls", "xlsx"])
    st.markdown("---")
    st.write("Creado por Heidi Guevara – Redondos")

# ----- PARÁMETROS -----
AZURE_ML_URL = "https://rdosml-xysue.eastus.inference.ml.azure.com/score"   # actualiza con tu endpoint real

# ----- FUNCIÓN PARA LLAMAR AZURE ML -----
def call_azureml(materiales, forecast_date, api_key):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    body = {
        "forecast_date": forecast_date,
        "materials": materiales if isinstance(materiales, list) else [materiales]
    }
    try:
        response = requests.post(AZURE_ML_URL, headers=headers, json=body)
        response.raise_for_status()
        return response.json()["predictions"]
    except Exception as e:
        return [{"error": f"Error llamando al modelo Azure ML: {e}"}]

# ============ FLUJO PREDICCIÓN PUNTUAL ============= #
st.header("Predicción puntual")
with st.form("puntual_form", clear_on_submit=False):
    col1, col2 = st.columns(2)
    with col1:
        mat = st.text_input("Material", key="mat_single")
    with col2:
        fecha = st.text_input("Fecha de pronóstico", value="2025-12-20", key="fecha_single")
    submitted = st.form_submit_button("Predecir")
    if submitted and mat and fecha and azureml_api_key:
        resultados = call_azureml(mat, fecha, azureml_api_key)
        if resultados and "error" not in resultados[0]:
            st.success(f"Predicción de demanda para {mat} el {fecha}:")
            df_pred = pd.DataFrame(resultados)
            st.dataframe(df_pred)
        else:
            st.error(resultados[0].get("error", "Error desconocido en predicción."))

# ============ FLUJO MASIVO ============= #
if excel_file and azureml_api_key:
    df_in = pd.read_excel(excel_file)
    if "material" in df_in.columns and "fecha" in df_in.columns:
        materiales = df_in["material"].tolist()
        fecha = df_in["fecha"].iloc[0]  # Asume una sola fecha si es igual para todos
        st.info(f"Prediciendo {len(materiales)} materiales para la fecha {fecha}...")
        resultados = call_azureml(materiales, fecha, azureml_api_key)
        if resultados and "error" not in resultados[0]:
            df_pred = pd.DataFrame(resultados)
            st.dataframe(df_pred)
        else:
            st.error(resultados[0].get("error", "Error en predicción masiva."))

st.markdown("---")

# ============ COPILOTO IA GENERATIVA ============= #
st.header("🤖 Chat IA Generativa (Copiloto)")

if "chat_ia" not in st.session_state:
    st.session_state["chat_ia"] = []

def extraer_material_fecha(pregunta):
    # Busca patrón de material de 6 dígitos y fecha yyyy-mm-dd
    mat = None
    fecha = None
    mat_match = re.search(r'\b\d{6}\b', pregunta)
    if mat_match:
        mat = mat_match.group()
    fecha_match = re.search(r'\d{4}-\d{2}-\d{2}', pregunta)
    if fecha_match:
        fecha = fecha_match.group()
    return mat, fecha

with st.form("copiloto_form", clear_on_submit=True):
    user_question = st.text_input(
        "Pregunta (ejemplo: ¿Qué demanda se espera para el material (precise el código) en la fecha (yyyy-mm-dd)? o solicita alguna recomendación. )",
        key="q_copiloto"
    )
    enviar_ia = st.form_submit_button("Enviar")
    if enviar_ia and user_question and openai_api_key and azureml_api_key:
        mat, fecha = extraer_material_fecha(user_question)
        if mat and fecha:
            # Es una pregunta de predicción cuantitativa
            resultados = call_azureml(mat, fecha, azureml_api_key)
            if resultados and "error" not in resultados[0]:
                pred = resultados[0].get("prediccion_kilos", "No disponible")
                answer = f"La predicción de demanda para el material {mat} el {fecha} es: **{pred} kilos**."
            else:
                answer = resultados[0].get("error", "Error en la predicción.")
        else:
            # Pregunta analítica, usa GPT
            prompt = (
                "Eres un experto en data analytics con profundo conocimiento en supply chain avícola. "
                "Responde de manera ejecutiva, precisa y basada en datos reales provenientes del modelo predictivo conectado a Azure. "
                "Nunca inventes códigos de material ni números, solo analiza o recomienda usando patrones, tendencias o mejores prácticas. "
                "Si tu respuesta es una estimación basada en patrones, indícalo. "
                f"Pregunta del usuario: {user_question}"
            )
            try:
                client = OpenAI(api_key=openai_api_key)
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": prompt},
                        {"role": "user", "content": user_question}
                    ]
                )
                answer = response.choices[0].message.content
            except Exception as e:
                answer = f"Error al llamar a OpenAI: {e}"

        st.session_state["chat_ia"].append({"user": user_question, "bot": answer})

# Historial del chat generativo
st.subheader("Historial del Chat IA Generativa")
for h in reversed(st.session_state["chat_ia"]):
    st.markdown(f"<div class='chat-bubble-user'>Tú: {h['user']}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='chat-bubble-bot'><b>Copiloto IA:</b> {h['bot']}</div>", unsafe_allow_html=True)

# Botón limpiar chat
with st.container():
    st.markdown('<div class="btn-clear">', unsafe_allow_html=True)
    if st.button("🧹 Borrar historial de chat IA"):
        st.session_state["chat_ia"] = []
    st.markdown('</div>', unsafe_allow_html=True)
