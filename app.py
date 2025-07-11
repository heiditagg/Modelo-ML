import streamlit as st
import os
import requests
import pandas as pd
from langchain_community.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader, UnstructuredPowerPointLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

st.set_page_config(
    page_title="Asesor Redondos IA",
    layout="wide",
    page_icon="🔴"
)

# --- ESTILOS VISUALES ---
st.markdown("""
    <style>
    .block-container {padding-top: 1.5rem;}
    .custom-title {
        color: #d32c2f; font-weight: 900;
        font-size: 2.1rem; letter-spacing: -1px;
        margin-bottom: 0.7rem; font-family: Segoe UI, Arial;
    }
    .chat-user {
        font-weight: bold; color: #d32c2f; margin-bottom: 4px;
        font-size: 1.08rem; font-family: Segoe UI, Arial;
        border-bottom: 1px solid #eee; padding-bottom: 3px;
    }
    .chat-bot {
        background: none !important;
        color: #333; font-size: 1.06rem;
        border-left: 3px solid #d32c2f;
        margin-bottom: 24px; padding-left: 14px;
        font-family: Segoe UI, Arial;
    }
    .logo-img {display: block; margin-left: auto; margin-right: auto;}
    .sidebar-content {font-size: 1rem;}
    .stTextInput > div > div > input {font-size: 1.1rem;}
    .chatbox-scroll {
        height: 55vh;
        max-height: 63vh;
        overflow-y: auto;
        padding-right: 10px;
        border-radius: 8px;
        background: #f9f9f9;
        margin-bottom: 16px;
        box-shadow: 0 3px 8px 0 #ededed88;
    }
    .btn-clear button {
        background-color: #ececec !important;
        color: #333 !important;
        border: none !important;
        font-size: 0.95rem !important;
        padding: 3px 13px !important;
        border-radius: 6px !important;
        box-shadow: none !important;
        margin-bottom: 13px !important;
        margin-top: 3px;
        margin-left: 8px;
        transition: background 0.17s;
    }
    .btn-clear button:hover {
        background-color: #d3d3d3 !important;
        color: #d32c2f !important;
    }
    </style>
""", unsafe_allow_html=True)

# ---- LOGO Y TÍTULO ----
st.image("logo_redondos.png", width=110)
st.markdown('<div class="custom-title">🤖 Asesor Redondos IA</div>', unsafe_allow_html=True)
st.markdown('<div style="font-size:1.17rem; color:#b30f21; font-weight:500; margin-bottom: 0.4rem;">Tu asistente IA para soluciones rápidas</div>', unsafe_allow_html=True)
st.markdown("---")

if "historial" not in st.session_state:
    st.session_state["historial"] = []

# ---- PANEL LATERAL: CONTROL ----
with st.sidebar:
    st.markdown('<div class="sidebar-content">', unsafe_allow_html=True)
    openai_api_key = st.text_input("🔑 Ingresa tu API Key de OpenAI:", type="password")
    uploaded_files = st.file_uploader(
        "📑 Sube tus archivos (PDF, Word, PowerPoint)", 
        type=["pdf", "docx", "pptx"], 
        accept_multiple_files=True
    )
    # Para predicción masiva: permite subir Excel con columnas ["material", "fecha"]
    st.markdown("----")
    st.markdown("**Predicción masiva de demanda:**")
    xls_pred = st.file_uploader("Carga tu Excel (material, fecha)", type=["xls", "xlsx"])
    st.markdown("---")
    st.write("Creado por Heidi + ChatGPT 😊")
    st.markdown('</div>', unsafe_allow_html=True)

# ---- Función para predecir demanda vía Azure ML ----
def predecir_demanda(fecha, materiales):
    url = "https://rdosml-xysue.eastus.inference.ml.azure.com/score"
    api_key = "TU_API_KEY"   # <-- Coloca tu API KEY aquí, nunca la expongas en producción real
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
            df = pd.DataFrame(resultado['predictions'])
            return df
        except Exception as e:
            return f"Error procesando la respuesta del modelo: {e}"
    else:
        return f"Error llamando al modelo Azure ML: {response.text}"

# --- PROCESAR DOCUMENTOS Y CHAIN RAG ----
if uploaded_files and openai_api_key:
    os.environ["OPENAI_API_KEY"] = openai_api_key

    all_documents = []
    for uploaded_file in uploaded_files:
        temp_path = f"temp_{uploaded_file.name}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.read())

        if uploaded_file.name.endswith(".pdf"):
            loader = PyPDFLoader(temp_path)
            pages = loader.load()
        elif uploaded_file.name.endswith(".docx"):
            loader = UnstructuredWordDocumentLoader(temp_path)
            pages = loader.load()
        elif uploaded_file.name.endswith(".pptx"):
            loader = UnstructuredPowerPointLoader(temp_path)
            pages = loader.load()
        else:
            pages = []
        all_documents.extend(pages)
        os.remove(temp_path)

    splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=120)
    documents = splitter.split_documents(all_documents)
    embeddings = OpenAIEmbeddings()
    db = FAISS.from_documents(documents, embeddings)
    llm = ChatOpenAI(model_name="gpt-4o", temperature=0.05)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=db.as_retriever(),
        return_source_documents=True   # Importante para detectar si hay fuente relevante
    )
else:
    qa_chain = None

ready = bool(uploaded_files and openai_api_key and qa_chain)

st.success("¡Listo! Haz tus preguntas 👇" if ready else "🔹 Sube archivos (PDF, Word, PPTX) y tu API Key para comenzar.")

with st.container():
    # ---- CAJA DE PREGUNTAS ----
    with st.form("pregunta_form", clear_on_submit=True):
        pregunta = st.text_input(
            "Pregunta al documento o pide predicción de demanda:",
            key="user_pregunta",
            label_visibility="collapsed",
            placeholder="Ej: ¿Cuál será la demanda de POLLO para 2025-12-31? ...",
            disabled=not ready
        )
        enviar = st.form_submit_button("OK", disabled=not ready)
        if enviar and pregunta.strip() != "" and ready:
            # Detectar si es una pregunta de demanda (súper simple, mejora el parsing si quieres)
            if "demanda" in pregunta.lower() and any(word in pregunta.lower() for word in ["para", "de", "del", "en"]):
                # Buscar fecha (año-mes-día) y materiales (palabras en mayúsculas o keywords conocidas)
                import re
                fecha_match = re.search(r"(20\d{2}-\d{2}-\d{2})", pregunta)
                fecha = fecha_match.group(1) if fecha_match else None
                # Materiales: asume palabras en mayúsculas o luego de "de"/"para"
                mat_match = re.findall(r"(?:de|para)\s+([A-Za-z0-9_ -]+)", pregunta)
                materiales = []
                if mat_match:
                    for m in mat_match:
                        materiales += [mat.strip().upper() for mat in m.split(",")]
                # Valida
                if fecha and materiales:
                    pred_df = predecir_demanda(fecha, materiales)
                    if isinstance(pred_df, pd.DataFrame):
                        # Muestra la tabla y guarda el resumen en historial
                        st.dataframe(pred_df)
                        respuesta = f"Predicción de demanda para {', '.join(materiales)} el {fecha}:\n{pred_df.to_markdown(index=False)}"
                        origen = "prediccion_demanda"
                    else:
                        respuesta = pred_df  # mensaje de error
                        origen = "prediccion_demanda"
                else:
                    respuesta = "Por favor indica la fecha (YYYY-MM-DD) y al menos un material. Ejemplo: ¿Cuál será la demanda de POLLO para 2025-12-31?"
                    origen = "prediccion_demanda"
            else:
                # ---- Respuesta combinada: RAG+IA generativa ----
                respuesta = qa_chain(pregunta)
                fuentes = respuesta.get('source_documents', [])
                if fuentes and respuesta['result'].strip() and not respuesta['result'].lower().startswith("no "):
                    origen = "documentos"
                else:
                    # Usa la IA general si no hay respuesta relevante
                    llm_solo = ChatOpenAI(model_name="gpt-4o", temperature=0.1)
                    respuesta_general = llm_solo.predict(pregunta)
                    respuesta['result'] = respuesta_general
                    origen = "conocimiento general"
            st.session_state["historial"].append({
                "pregunta": pregunta,
                "respuesta": respuesta if isinstance(respuesta, str) else respuesta['result'],
                "origen": origen
            })
            st.rerun()

    # ---- HISTORIAL DE CHAT ----
    st.markdown('<div class="chatbox-scroll">', unsafe_allow_html=True)
    if st.session_state["historial"]:
        for h in reversed(st.session_state["historial"]):
            if h["origen"] == "documentos":
                fuente = " (de tus documentos)"
            elif h["origen"] == "conocimiento general":
                fuente = " (de conocimientos generales IA)"
            elif h["origen"] == "prediccion_demanda":
                fuente = " (Predicción de demanda ML)"
            else:
                fuente = ""
            st.markdown(f'<div class="chat-user">Tú: {h["pregunta"]}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="chat-bot"><b>Asesor Redondos IA{fuente}:</b> {h["respuesta"]}</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div style="color:#888; font-size:1rem; margin-top:30px;">Aquí aparecerán tus preguntas y respuestas.</div>', unsafe_allow_html=True)

    # --- Botón de limpiar historial debajo del chat ---
    st.markdown('<div class="btn-clear">', unsafe_allow_html=True)
    if st.button("🧹 Borrar historial de chat", disabled=not ready):
        st.session_state["historial"] = []
    st.markdown('</div>', unsafe_allow_html=True)

# ---- Predicción masiva por Excel ----
if xls_pred is not None and ready:
    df = pd.read_excel(xls_pred)
    # Espera columnas: "material", "fecha"
    if "material" in df.columns and "fecha" in df.columns:
        # Agrupa materiales por fecha
        for f, group in df.groupby("fecha"):
            materiales = list(group["material"].astype(str).unique())
            pred_df = predecir_demanda(f, materiales)
            st.markdown(f"#### Predicción de demanda para {', '.join(materiales)} el {f}:")
            if isinstance(pred_df, pd.DataFrame):
                st.dataframe(pred_df)
            else:
                st.warning(str(pred_df))
    else:
        st.warning("El Excel debe tener las columnas 'material' y 'fecha'.")
