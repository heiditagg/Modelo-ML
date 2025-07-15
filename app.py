import streamlit as st
import requests
import pandas as pd
from openai import OpenAI

# --- CONFIGURACIÓN VISUAL ---
st.set_page_config(page_title="Predicción de Demanda Redondos", layout="wide")
st.markdown("""
    <style>
    .custom-title {
        color: #d32c2f; font-weight: 900; font-size: 2.1rem; letter-spacing: -1px;
        margin-bottom: 0.3rem; font-family: Segoe UI, Arial;
    }
    .custom-sub {
        color: #b30f21; font-size:1.17rem; font-weight:500; margin-bottom:0.7rem;
    }
    .chat-bubble-user { font-weight: bold; color: #d32c2f; font-size: 1.1rem; }
    .chat-bubble-bot { background:#fff8e6; border-radius:18px 18px 18px 18px; padding:16px 20px; font-size:1.04rem;}
    </style>
""", unsafe_allow_html=True)

st.image("logo_redondos.png", width=110)
st.markdown('<div class="custom-title"> Predicción de Demanda Redondos</div>', unsafe_allow_html=True)
st.markdown('<div class="custom-sub">Consulta puntual, masiva y conversación con IA Generativa</div>', unsafe_allow_html=True)
st.markdown("---")

# --- ICONOS ---
user_icon = "user_icon.png"
bot_icon = "robot_icon.jpg"

# --- SIDEBAR ---
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

# --- PARÁMETROS ---
AZURE_ML_URL = "https://rdosml-xysue.eastus.inference.ml.azure.com/score"

# --- FUNCIÓN PARA LLAMAR AZURE ML ---
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

# --- FLUJO PREDICCIÓN PUNTUAL ---
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

# --- FLUJO MASIVO ---
if excel_file and azureml_api_key:
    df_in = pd.read_excel(excel_file)
    if "material" in df_in.columns and "fecha" in df_in.columns:
        materiales = df_in["material"].tolist()
        fecha = df_in["fecha"].iloc[0]
        st.info(f"Prediciendo {len(materiales)} materiales para la fecha {fecha}...")
        resultados = call_azureml(materiales, fecha, azureml_api_key)
        if resultados and "error" not in resultados[0]:
            df_pred = pd.DataFrame(resultados)
            st.dataframe(df_pred)
        else:
            st.error(resultados[0].get("error", "Error en predicción masiva."))

st.markdown("---")

# --- CHAT IA GENERATIVA ---
st.header("🤖 Chat IA Generativa (Copiloto)")

if "chat_ia" not in st.session_state:
    st.session_state["chat_ia"] = []

# Input para pregunta generativa
with st.form("copiloto_form", clear_on_submit=True):
    user_question = st.text_input(
        "Pregunta (ejemplo: ¿Cuál es la demanda proyectada para el material 1000110 el 2025-12-31? O solicita una explicación o análisis)",
        key="q_copiloto"
    )
    enviar_ia = st.form_submit_button("Enviar")
    if enviar_ia and user_question and openai_api_key and azureml_api_key:
        # PROMPT personalizado
        prompt = (
            "Eres un experto en analytics y supply chain en la industria avícola. "
            "Contesta solo con datos y resultados del modelo de predicción conectados en la app. "
            "Nunca inventes códigos ni datos. "
            "Si la consulta incluye material y fecha, llama a la función de predicción. "
            "Si la pregunta es de análisis o recomendación, responde de forma ejecutiva y clara."
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

        st.session_state["chat_ia"].append(
            {"role": "user", "user": user_question, "bot": answer}
        )

# --- HISTORIAL DE CHAT VISUAL --- 
st.markdown("### Historial del Chat IA Generativa")
for h in st.session_state["chat_ia"]:
    cols = st.columns([0.09, 0.91])
    # Usuario
    with cols[0]:
        st.image(user_icon, width=58)
    with cols[1]:
        st.markdown(
            f"<div class='chat-bubble-user'>{h['user']}</div>",
            unsafe_allow_html=True
        )
    # Bot
    cols = st.columns([0.09, 0.91])
    with cols[0]:
        st.image(bot_icon, width=58)
    with cols[1]:
        st.markdown(
            f"<div class='chat-bubble-bot'><b>Copiloto IA:</b> {h['bot']}</div>",
            unsafe_allow_html=True
        )

# Botón limpiar chat
st.markdown('<div style="margin-top:1.5em"></div>', unsafe_allow_html=True)
if st.button("🧹 Borrar historial de chat IA"):
    st.session_state["chat_ia"] = []

