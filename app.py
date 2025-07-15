import streamlit as st
import requests
import pandas as pd
from openai import OpenAI

# =========== CONFIGURACIÓN GENERAL ===============
st.set_page_config(page_title="Predicción de Demanda Redondos", layout="wide", page_icon="🔮")
st.markdown("""
    <style>
    .bubble-user {
        background: #d32c2f18; color: #d32c2f; 
        padding: 14px 18px; border-radius: 15px 15px 2px 15px;
        margin-bottom: 4px; margin-left: 42px; 
        text-align: left; font-weight: 600;
        position: relative;
    }
    .bubble-bot {
        background: #f8f8f8; color: #232323; 
        padding: 14px 18px; border-radius: 15px 15px 15px 2px;
        margin-bottom: 18px; margin-right: 42px;
        border-left: 5px solid #d32c2f;
        text-align: left; 
        position: relative;
    }
    .icon-user, .icon-bot {
        width: 36px; height: 36px; border-radius: 50%;
        display: inline-flex; align-items: center; 
        justify-content: center; font-size: 23px;
        position: absolute; left: -45px; top: 7px; background: #fff;
    }
    .icon-user { color: #d32c2f; border: 2px solid #d32c2f;}
    .icon-bot { color: #222; border: 2px solid #ccc;}
    .stTextInput>div>div>input {font-size: 1.13rem;}
    .stButton>button {font-size: 1.11rem;}
    </style>
""", unsafe_allow_html=True)
st.image("logo_redondos.png", width=110)
st.markdown('<h2 style="color:#d32c2f; font-weight:900; margin-bottom:2px; letter-spacing:-1px">🔮 Predicción de Demanda Redondos</h2>', unsafe_allow_html=True)
st.markdown('<div style="color:#b30f21; font-size:1.19rem; margin-bottom:0.5rem; font-weight:500;">Consulta puntual, masiva y conversación con IA Generativa</div>', unsafe_allow_html=True)
st.markdown("---")

# =========== SIDEBAR ===========
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

# =========== AZURE ML FUNCTION ===========
AZURE_ML_URL = "https://rdosml-xysue.eastus.inference.ml.azure.com/score"  # Actualiza tu endpoint real aquí

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

# =========== PREDICCIÓN PUNTUAL ===========
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

# =========== PREDICCIÓN MASIVA ===========
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

# =========== CHAT IA GENERATIVA ==============
st.header("😊 Chat IA Generativa (Copiloto)")

if "chat_ia" not in st.session_state:
    st.session_state["chat_ia"] = []

# ---------- Input del usuario ----------
with st.form("copiloto_form", clear_on_submit=True):
    user_question = st.text_input(
        "Pregunta (ejemplo: ¿Cuál es la demanda proyectada para el material 1000110 el 2025-12-31? O solicita una explicación o análisis)",
        key="q_copiloto"
    )
    enviar_ia = st.form_submit_button("Enviar")
    if enviar_ia and user_question and openai_api_key and azureml_api_key:
        prompt = (
            "Eres un experto en data analytics y supply chain en la industria avícola Redondos. "
            "Contesta usando los resultados reales del modelo predictivo si la pregunta es de demanda, SIEMPRE consulta el modelo si piden un número. "
            "Si la pregunta requiere análisis o explicación ejecutiva, responde de manera clara y profesional. "
            "Nunca inventes códigos de material, fechas o resultados que no existan. "
            "Pregunta del usuario: " + user_question
        )

        # ========== Lógica híbrida (predicción real si aplica) ==========
        # Busca si hay código material (solo números) y fecha
        import re
        mat_re = re.findall(r"\b\d{6,}\b", user_question)
        fecha_re = re.findall(r"\d{4}-\d{2}-\d{2}", user_question)
        respuesta_modelo = None

        if mat_re and fecha_re:
            # Si la pregunta pide una predicción puntual, usa el modelo real
            predic = call_azureml(mat_re[0], fecha_re[0], azureml_api_key)
            if predic and "error" not in predic[0]:
                num_pred = predic[0].get("prediccion_kilos", None)
                if num_pred is not None:
                    respuesta_modelo = (
                        f"La demanda proyectada para el material {mat_re[0]} el {fecha_re[0]} es de {num_pred} kg "
                        f"(dato real del modelo ML). ¿Deseas algún análisis sobre este resultado?"
                    )
        # ========== Llama a OpenAI ==========
        try:
            client = OpenAI(api_key=openai_api_key)
            messages = [{"role": "system", "content": prompt}]
            if respuesta_modelo:
                messages.append({"role": "assistant", "content": respuesta_modelo})
            messages.append({"role": "user", "content": user_question})
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages
            )
            answer = response.choices[0].message.content
        except Exception as e:
            answer = f"Error al llamar a OpenAI: {e}"
        # Guarda la conversación
        st.session_state["chat_ia"].append(
            {"user": user_question, "bot": answer}
        )

# =========== HISTORIAL ESTILO BURBUJA ===========
st.subheader("Historial del Chat IA Generativa")
for h in reversed(st.session_state["chat_ia"]):
    # Burbuja usuario
    st.markdown(
        f"""
        <div style="display: flex; align-items: flex-start;">
          <span class="icon-user">🧑</span>
          <div class="bubble-user">Tú: {h['user']}</div>
        </div>
        """, unsafe_allow_html=True
    )
    # Burbuja bot
    st.markdown(
        f"""
        <div style="display: flex; align-items: flex-start;">
          <span class="icon-bot">🤖</span>
          <div class="bubble-bot"><b>Copiloto IA:</b> {h['bot']}</div>
        </div>
        """, unsafe_allow_html=True
    )

# ---------- Botón limpiar chat ----------
with st.container():
    if st.button("🧹 Borrar historial de chat IA"):
        st.session_state["chat_ia"] = []

