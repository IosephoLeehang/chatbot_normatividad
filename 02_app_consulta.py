#NAplicativo consulta versión 2.0

import streamlit as st
import os
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

# Configuración de rutas limpia
RUTA_FINAL_DB = "./chroma_db_normas"

st.set_page_config(page_title="Consulta de Normas - MIMP", page_icon="⚖️", layout="centered")

st.title("⚖️ Asistente Normativo Inteligente")
st.markdown("##### Consulta de documentos normativos publicados")
st.caption("Conectado de forma eficiente a la base de conocimiento local indexada.")

@st.cache_resource
def inicializar_base_conocimiento(ruta_db):
    if not os.path.exists(ruta_db):
        return None
    try:
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            model_kwargs={"device": "cpu"}
        )
        # Modo lectura directa
        db = Chroma(persist_directory=ruta_db, embedding_function=embeddings)
        return db
    except Exception as e:
        st.error(f"Error crítico al leer la estructura indexada: {e}")
        return None

# Carga en milisegundos usando st.cache_resource
base_vectores = inicializar_base_conocimiento(RUTA_FINAL_DB)

if base_vectores is None:
    st.error(f"❌ La base de datos en `{RUTA_FINAL_DB}` no ha sido generada o está vacía. Por favor, ejecuta primero el script `01_actualizador_db.py` desde tu consola.")
else:
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "¡Hola! Soy tu asistente legal del MIMP. ¿Qué norma, ley o directiva deseas validar hoy?"}
        ]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if user_query := st.chat_input("Ej. ¿Cuáles son las directivas vigentes sobre acoso político?"):
        with st.chat_message("user"):
            st.markdown(user_query)
        st.session_state.messages.append({"role": "user", "content": user_query})

        with st.chat_message("assistant"):
            with st.spinner("Buscando en la base normativa..."):
                documentos_relevantes = base_vectores.similarity_search(user_query, k=4)
                
                if documentos_relevantes:
                    contexto_encontrado = ""
                    fuentes = set()
                    
                    for doc in documentos_relevantes:
                        contexto_encontrado += f"[Fragmento de Norma]: {doc.page_content}\n\n"
                        doc_nombre = doc.metadata.get('documento', 'Norma Oficial')
                        doc_link = doc.metadata.get('enlace', '#')
                        
                        if doc_link != "Texto directo":
                            fuentes.add(f"• [{doc_nombre}]({doc_link})")
                        else:
                            fuentes.add(f"• {doc_nombre} *(Texto Directo)*")
                    
                    llm = ChatGroq(
                        temperature=0.0, 
                        model_name="llama-3.1-8b-instant",
                        groq_api_key=st.secrets["GROQ_API_KEY"]
                    )
                    
                    instruccion = f"""
                    Eres un asistente legal experto en la normatividad del Ministerio de la Mujer y Poblaciones Vulnerables (MIMP) de Perú.
                    Responde a la consulta usando exclusivamente los siguientes fragmentos verdaderos provistos. 
                    Sé conciso, estructurado con viñetas claras y sumamente profesional.
                    Si no está en el contexto, indica amablemente que la base de datos actual no cuenta con el texto exacto.

                    Contexto Legal Registrado:
                    {contexto_encontrado}

                    Pregunta: {user_query}
                    Respuesta:
                    """
                    
                    respuesta_final = llm.invoke(instruccion).content
                    respuesta_final += "\n\n**Fuentes oficiales aplicadas:**\n" + "\n".join(fuentes)
                else:
                    respuesta_final = "No se encontraron registros indexados compatibles con los términos consultados."

                st.markdown(respuesta_final)
                st.session_state.messages.append({"role": "assistant", "content": respuesta_final})