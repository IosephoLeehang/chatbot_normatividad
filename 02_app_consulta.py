#NAplicativo consulta versión 2.1

import streamlit as st
import os
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

# Configuración de rutas limpia
RUTA_FINAL_DB = "./chroma_db_normas"

st.set_page_config(page_title="Consulta de Normas - ON MIMP", page_icon="⚖️", layout="centered")

st.title("⚖️ Asistente Normativo Inteligente")
st.markdown("##### Consulta de documentos normativos publicados")
st.caption("Conectado a la sección Normatividad del Observatorio Nacional de la Violencia contra la Mujer e integrantes del grupo familiar")

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
            {"role": "assistant", "content": "¡Hola! Soy tu asistente legal del Observatorio Nacional. ¿Qué norma, ley o directiva deseas validar hoy?"}
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
                    
                    # NUEVO: Ahora pasamos el nombre y el enlace DENTRO del contexto para que la IA los use.
                    for i, doc in enumerate(documentos_relevantes, 1):
                        doc_nombre = doc.metadata.get('documento', f'Norma Oficial {i}')
                        doc_link = doc.metadata.get('enlace', '#')
                        
                        contexto_encontrado += f"--- Documento {i} ---\n"
                        contexto_encontrado += f"Nombre de la Norma: {doc_nombre}\n"
                        contexto_encontrado += f"Enlace: {doc_link}\n"
                        contexto_encontrado += f"Fragmento: {doc.page_content}\n\n"
                    
                    llm = ChatGroq(
                        temperature=0.0, 
                        model_name="llama-3.1-8b-instant",
                        groq_api_key=st.secrets["GROQ_API_KEY"]
                    )
                    
                    # NUEVO: Instrucciones estrictas para forzar el formato deseado
                    instruccion = f"""
                    Eres un asistente legal experto en la normatividad del Ministerio de la Mujer y Poblaciones Vulnerables (MIMP) de Perú.
                    Tu objetivo es responder a la consulta del usuario basándote EXCLUSIVAMENTE en el "Contexto Legal Registrado" provisto.

                    REGLA DE FORMATO OBLIGATORIA:
                    Por CADA norma o documento relevante que encuentres en el contexto, debes generar una viñeta utilizando ESTRICTAMENTE la siguiente estructura:
                    **[Nombre de la Norma]**: [Texto descriptivo conciso basado en el fragmento] [([Enlace]({'{enlace}'}))]

                    Ejemplo de cómo debe verse tu respuesta:
                    **Ley N° 30364**: Norma matriz que crea el Observatorio para recolectar, sistematizar y monitorear datos sobre la violencia de género. [Ver documento](https://enlace_de_ejemplo.com)

                    Si la información no está en el contexto, indica amablemente que la base de datos actual no cuenta con el texto exacto, sin inventar leyes ni enlaces.

                    Contexto Legal Registrado:
                    {contexto_encontrado}

                    Pregunta: {user_query}
                    Respuesta:
                    """
                    
                    respuesta_final = llm.invoke(instruccion).content
                    
                    # Nota: Ya no concatenamos las fuentes al final porque la IA está forzada a ponerlas en cada línea.
                    
                else:
                    respuesta_final = "No se encontraron registros indexados compatibles con los términos consultados."

                st.markdown(respuesta_final)
                st.session_state.messages.append({"role": "assistant", "content": respuesta_final})