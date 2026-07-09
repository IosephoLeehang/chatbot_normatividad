# Código para desarrollo de la fase 2
# del proyecto de aplicativo de consulta sobre Normatividad

import streamlit as st
import os
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

# Nombre de la carpeta que generó la Fase 1
NOMBRE_CARPETA_DB = "chroma_db_normas"

# Verificamos si la app corre en Streamlit Cloud (Nube) o localmente
if os.path.exists(NOMBRE_CARPETA_DB):
    RUTA_FINAL_DB = os.path.abspath(NOMBRE_CARPETA_DB)
else:
    RUTA_FINAL_DB = r"D:\Proyectos\git_porjects\chatbot_normatividad\chroma_db_normas"

# INTERFAZ DE USUARIO (STREAMLIT)
st.set_page_config(page_title="Consulta de Normas - MIMP", page_icon="⚖️", layout="centered")

st.title("⚖️ Asistente Normativo Inteligente")
st.markdown("##### Haga consultas aquí sobre los documentos normativos publicados")
st.caption("Conectado a la base de conocimiento oficializada del Observatorio.")

# Carga optimizada de la Base de Datos para evitar recargas lentas (Cache)
@st.cache_resource
def inicializar_base_conocimiento(ruta_db):
    if not os.path.exists(ruta_db):
        return None
    try:
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        config_ligera = {"device": "cpu"}
        
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            model_kwargs=config_ligera
        )
        db = Chroma(persist_directory=ruta_db, embedding_function=embeddings)
        return db
    except Exception as e:
        st.error(f"Error al inicializar la base de datos vectorial: {e}")
        return None

# Inicializar la base de datos
base_vectores = inicializar_base_conocimiento(RUTA_FINAL_DB)

if base_vectores is None:
    st.error(f"❌ No se encontró la carpeta de conocimiento en: `{RUTA_FINAL_DB}`. Por favor, ejecuta primero la Fase 1.")
else:
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "¡Hola! Soy tu asistente legal. ¿Qué norma o artículo específico del portal deseas consultar hoy?"}
        ]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if user_query := st.chat_input("Ej. ¿Cuáles son las medidas de protección en la Ley 30364?"):
        with st.chat_message("user"):
            st.markdown(user_query)
        st.session_state.messages.append({"role": "user", "content": user_query})

        with st.chat_message("assistant"):
            with st.spinner("Analizando documentos normativos..."):
                # La búsqueda por similitud ahora será inmediata
                documentos_relevantes = base_vectores.similarity_search(user_query, k=4)
                
                if documentos_relevantes:
                    contexto_encontrado = ""
                    fuentes = set()
                    
                    for doc in documentos_relevantes:
                        contexto_encontrado += f"[Fragmento]: {doc.page_content}\n\n"
                        doc_nombre = doc.metadata.get('documento', 'Norma Oficial')
                        doc_link = doc.metadata.get('enlace', '#')
                        fuentes.add(f"[{doc_nombre}]({doc_link})")
                    
                    llm = ChatGroq(
                        temperature=0.0, 
                        model_name="llama-3.1-8b-instant",
                        groq_api_key=st.secrets["GROQ_API_KEY"]
                    )
                    
                    instruccion = f"""
                    Eres un asistente legal experto en normatividad peruana del MIMP.
                    Basándote estrictamente en los siguientes fragmentos obtenidos de la base de datos oficial, responde de forma clara, ordenada y directa a la pregunta del usuario. 
                    Si la pregunta requiere un listado, estructúralo de forma limpia usando viñetas.
                    No inventes datos que no estén explícitos en el contexto proporcionado.

                    Contexto legal disponible:
                    {contexto_encontrado}

                    Pregunta del usuario: {user_query}
                    Respuesta:
                    """
                    
                    respuesta_final = llm.invoke(instruccion).content
                    respuesta_final += "\n\n**Fuentes oficiales detectadas:**\n" + "\n".join(fuentes)
                    
                else:
                    respuesta_final = "No logré encontrar fragmentos específicos en la base de datos que respondan a tu consulta."

                st.markdown(respuesta_final)
                st.session_state.messages.append({"role": "assistant", "content": respuesta_final})

