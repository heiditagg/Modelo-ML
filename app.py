import streamlit as st
import requests
import pandas as pd
import json
from openai import OpenAI

# ============ CONFIGURACIÓN GENERAL Y ESTILO ============

st.set_page_config(page_title="Predicción de Demanda Redondos", layout="wide", page_icon="🔮")

st.markdown("""
    <style>
    .custom-title { color: #d32c2f; font-weight: 900; font-size: 2.1rem; letter-spacing: -1px; margin-bottom: 0.3rem; font-family: Segoe UI, Arial;}
    .custom-sub { color: #b30f21; font-size:1.17rem; font-weight:500; margin-bottom:0.7rem;}
    .msg-row {display: flex; align-items: flex-start; margin-bottom: 17px;}
    .msg-icon {width:48px; height:48px; border-radius:16px; margin-right:15px; display:flex; align-items:center; justify-content:center;}
    .msg-icon.user {background:#FF7376;}
    .msg-icon.bot {background:#FFD86A;}
    .msg-bubble {background:#fafbfc; border-radius:16px 16px 16px 16px; padding: 15px 22px; font-size:1.15rem; color:#313133;}
    .msg-bubble-user {background:#fafbfc; color:#d32c2f; font-weight:700;}
    .msg-bubble-bot {background:#f8f8f8;}
    .btn-clear button {background-color: #ececec !important; color: #333 !important; border: none !important; font-size: 0.95rem !important; padding: 3px 13px !important; border-radius: 6px !important; box-shadow: none !important; margin-bottom: 13px !important; margin-top: 3px; margin-left: 8px; transition: background 0.17s;}
    .btn-clear button:hover { background-color: #d3d3d3 !important; color: #d32c2f !important; }
    </style>
""", unsafe_allow_html=True)

# ============ LOGO Y TÍTULO PRINCIPAL ============
st.image("logo_redondos.png", width=110)
st.markdown('<div class="custom-title">🔮 Predicción de Demanda Redondos</div>', unsafe_allow_html=True)
st.markdown('<div class="custom-sub">Consulta puntual, masiva y conversación con IA Generativa</div>', unsafe_allow_html=True)
st.markdown("---")

# ============ SIDEBAR: API KEYS Y CARGA ============
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

# ============ AZURE ML ENDPOINT ============
AZURE_ML_URL = "https://rdosml-xysue.eastus.inference.ml.azure.com/score"   # cambia por el real si hace falta

def call_azureml(materiales, forecast_date, api_key):
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    body = {"forecast_date": forecast_date, "materials": materiales if isinstance(materiales, list) else [materiales]}
    try:
        response = requests.post(AZURE_ML_URL, headers=headers, json=body)
        response.raise_for_status()
        return response.json()["predictions"]
    except Exception as e:
        return [{"error": f"Error llamando al modelo Azure ML: {e}"}]

# ============ PREDICCIÓN PUNTUAL ============
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

# ============ PREDICCIÓN MASIVA ============
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

# ============ COPILOTO IA GENERATIVA ============ #
st.header("🤖 Chat IA Generativa (Copiloto)")

# Inicialización del historial
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
        # Prompt para la IA
        prompt = (
            f"Eres un experto en analytics y supply chain en la industria avícola. "
            f"Contesta usando los resultados del modelo de predicción. "
            f"Pregunta del usuario: {user_question}. "
            f"Si la consulta incluye un material y una fecha, llama a la función de predicción de demanda con esos datos y responde con el valor. "
            f"Si la pregunta es analítica o requiere recomendación, genera una respuesta ejecutiva y útil, usando IA generativa."
        )
        # Ejecución de la IA generativa
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
            {"user": user_question, "bot": answer}
        )

# ============ HISTORIAL CON ICONOS PERSONALIZADOS ============ #
# Usa tus propios íconos en la carpeta del proyecto, ejemplo: "user_icon.png", "bot_icon.png"
st.subheader("Historial del Chat IA Generativa")
for h in st.session_state["chat_ia"]:
    st.markdown(
        f"""
        <div class='msg-row'>
            <div class='msg-icon user'>
                <img src='https://cdn-icons-png.flaticon.com/512/4140/4140047.png' width="30"/>
            </div>
            <div class='msg-bubble msg-bubble-user'>{h['user']}</div>
        </div>
        <div class='msg-row'>
            <div class='msg-icon bot'>
                <img src='https://cdn-icons-png.flaticon.com/512/4712/4712039.png' width="30"/>
            </div>
            <div class='msg-bubble msg-bubble-bot'><b>Copiloto IA:</b> {h['bot']}</div>
        </div>
        """, unsafe_allow_html=True
    )

# ============ BOTÓN LIMPIAR ============ #
with st.container():
    st.markdown('<div class="btn-clear">', unsafe_allow_html=True)
    if st.button("🧹 Borrar historial de chat IA"):
        st.session_state["chat_ia"] = []
    st.markdown('</div>', unsafe_allow_html=True)



