import streamlit as st
import requests
import pandas as pd
from openai import OpenAI

# --- ESTILOS VISUALES CHAT ---
st.markdown("""
    <style>
    .chat-container {
        max-width: 750px;
        margin: auto;
        padding: 24px 0 0 0;
    }
    .chat-message {
        display: flex; align-items: flex-start; margin-bottom: 18px;
    }
    .chat-message.user .bubble {
        background: #ff5c5c;
        color: white;
        border-radius: 16px 16px 4px 16px;
        margin-left: 10px;
        margin-right: auto;
    }
    .chat-message.bot .bubble {
        background: #fffbe7;
        color: #4f4f4f;
        border-radius: 16px 16px 16px 4px;
        margin-right: 10px;
        margin-left: auto;
        border: 1px solid #f3dc85;
    }
    .bubble {
        padding: 16px 18px;
        max-width: 76%;
        font-size: 1.09rem;
        box-shadow: 0 1px 8px rgba(0,0,0,0.03);
    }
    .icon {
        font-size: 2.0rem;
        margin-top: 2px;
    }
    </style>
""", unsafe_allow_html=True)

# --- CABECERA APP ---
st.set_page_config(page_title="Predicción de Demanda Redondos", layout="wide", page_icon="🔮")

st.image("logo_redondos.png", width=110)
st.markdown('<div style="color:#d32c2f;font-weight:900;font-size:2.1rem;letter-spacing:-1px;margin-bottom:0.3rem;font-family:Segoe UI,Arial;">🔮 Predicción de Demanda Redondos</div>', unsafe_allow_html=True)
st.markdown('<div style="color:#b30f21;font-size:1.17rem;font-weight:500;margin-bottom:0.7rem;">Consulta puntual, masiva y conversación con IA Generativa</div>', unsafe_allow_html=True)
st.markdown("---")

# --- SIDEBAR PARA CLAVES Y ARCHIVOS ---
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

# --- PARÁMETROS ENDPOINT ---
AZURE_ML_URL = "https://rdosml-xysue.eastus.inference.ml.azure.com/score"   # Actualiza tu endpoint real aquí

# --- FUNCIÓN PARA LLAMAR AL MODELO PREDICTIVO ---
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

# --- FLUJO PREDICCIÓN MASIVA DESDE EXCEL ---
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

# --- INICIA HISTORIAL DEL CHAT IA ---
st.header("🤖 Chat IA Generativa (Copiloto)")

if "chat_ia" not in st.session_state:
    st.session_state["chat_ia"] = []

# --- INPUT CHAT ---
with st.form("copiloto_form", clear_on_submit=True):
    user_question = st.text_input(
        "Pregunta (ejemplo: ¿Qué demanda se espera para el material 1000130 en la fecha 2025-12-31? o pídeme un análisis o recomendación)",
        key="q_copiloto"
    )
    enviar_ia = st.form_submit_button("Enviar")
    if enviar_ia and user_question and openai_api_key and azureml_api_key:
        # Prompt inteligente híbrido
        prompt = (
            f"Eres un experto en data analytics de la industria avícola Redondos. "
            f"Responde SIEMPRE usando datos reales del modelo predictivo conectado a Azure ML si se solicita demanda numérica, y NUNCA inventes materiales ni fechas. "
            f"Si la consulta es para un material y una fecha, llama a la función de predicción y responde SOLO el valor numérico predicho y una breve interpretación ejecutiva. "
            f"Si la consulta es solo analítica o de tendencias, responde como un consultor experto usando IA generativa y aclara si tu respuesta es estimada. "
            f"Evita generalidades y aporta insights de negocio útiles. "
            f"Pregunta del usuario: {user_question}"
        )

        # Si parece una consulta de predicción, intenta llamarla directamente
        import re
        mat_re = re.search(r"(material|código|cod)[\s:]*([0-9]+)", user_question, re.IGNORECASE)
        date_re = re.search(r"(fecha|para|en)\s*(el|la)?\s*([0-9]{4}-[0-9]{2}-[0-9]{2})", user_question, re.IGNORECASE)
        answer = ""
        if mat_re and date_re:
            cod = mat_re.group(2)
            date = date_re.group(3)
            resultados = call_azureml(cod, date, azureml_api_key)
            if resultados and "error" not in resultados[0]:
                predic = resultados[0].get("prediccion_kilos", None)
                if predic is not None:
                    answer = f"La demanda proyectada para el material {cod} el {date} es de **{predic} kilos**. Este dato es resultado del modelo predictivo oficial. Te recomiendo monitorear el comportamiento real para ajustar estrategias de abastecimiento y ventas."
                else:
                    answer = "No se pudo obtener la predicción numérica para ese material y fecha. Por favor verifica los datos."
            else:
                answer = resultados[0].get("error", "No se pudo predecir, revisa los datos ingresados.")
        else:
            # Usa OpenAI solo si NO es una consulta de predicción numérica directa
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

# --- HISTORIAL DE CONVERSACIÓN VISUAL TIPO WHATSAPP ---
def get_icon(role):
    return "🧑" if role == "user" else "🤖"

st.markdown('<div class="chat-container">', unsafe_allow_html=True)
for h in st.session_state["chat_ia"]:
    st.markdown(
        f"""
        <div class="chat-message user">
            <div class="icon">🧑</div>
            <div class="bubble">{h['user']}</div>
        </div>
        """, unsafe_allow_html=True,
    )
    st.markdown(
        f"""
        <div class="chat-message bot">
            <div class="icon">🤖</div>
            <div class="bubble">{h['bot']}</div>
        </div>
        """, unsafe_allow_html=True,
    )
st.markdown('</div>', unsafe_allow_html=True)

# --- BOTÓN LIMPIAR CHAT ---
with st.container():
    st.markdown('<div class="btn-clear">', unsafe_allow_html=True)
    if st.button("🧹 Borrar historial de chat IA"):
        st.session_state["chat_ia"] = []
    st.markdown('</div>', unsafe_allow_html=True)

