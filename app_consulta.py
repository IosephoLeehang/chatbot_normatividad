# Código para desarrollo de la fase 2
# del proyecto de aplicativo de consulta sobre Normatividad

import streamlit as st
import os
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# ==========================================
# CONFIGURACIÓN DE RUTAS (SITUACIÓN 1 Y 2)
# ==========================================
# Nombre de la carpeta que generó la Fase 1
NOMBRE_CARPETA_DB = "chroma_db_normas"

# Verificamos si la app corre en Streamlit Cloud (Nube) o localmente
if os.path.exists(NOMBRE_CARPETA_DB):
    # Situación 1 o Situación 2 (Vectores subidos junto al código en GitHub)
    RUTA_FINAL_DB = os.path.abspath(NOMBRE_CARPETA_DB)
else:
    # Fallback o configuración alternativa si usas rutas absolutas en tu laptop
    RUTA_FINAL_DB = r"D:\Proyectos\git_porjects\chatbot_normatividad\chroma_db_normas"

# ==========================================
# INTERFAZ DE USUARIO (STREAMLIT)
# ==========================================
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
    st.success("✅ Base de conocimiento cargada y lista para consultas.")

    # Historial de chat en la sesión de Streamlit
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "¡Hola! Soy tu asistente legal. ¿Qué norma o artículo específico del portal deseas consultar hoy?"}
        ]

    # Mostrar mensajes anteriores del historial
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Entrada de texto del usuario
    if user_query := st.chat_input("Ej. ¿Cuáles son las medidas de protección en la Ley 30364?"):
        # Mostrar consulta del usuario en pantalla
        with st.chat_message("user"):
            st.markdown(user_query)
        st.session_state.messages.append({"role": "user", "content": user_query})

        # Búsqueda semántica en los vectores recopilados
        with st.chat_message("assistant"):
            with st.spinner("Buscando en los documentos normativos..."):
                # Recuperamos los 3 fragmentos de leyes más similares a la pregunta
                documentos_relevantes = base_vectores.similarity_search(user_query, k=3)
                
                if documentos_relevantes:
                    contexto_encontrado = ""
                    fuentes = set()
                    
                    for doc in documentos_relevantes:
                        contexto_encontrado += f"- {doc.page_content}\n\n"
                        # Extraemos metadatos guardados en la Fase 1
                        doc_nombre = doc.metadata.get('documento', 'Norma Oficial')
                        doc_link = doc.metadata.get('enlace', '#')
                        fuentes.add(f"[{doc_nombre}]({doc_link})")
                    
                    # --- NOTA PARA LA FASE 3 ---
                    # Aquí enviaremos el 'contexto_encontrado' junto con la pregunta al LLM (Ej. OpenAI/Groq)
                    # Por ahora, simulamos la respuesta mostrando los fragmentos puros recuperados
                    respuesta_ia = f"**Información relevante encontrada en las normas:**\n\n{contexto_encontrado}"
                    respuesta_ia += "\n\n**Fuentes oficiales detectadas:**\n" + "\n".join(fuentes)
                else:
                    respuesta_ia = "No logré encontrar fragmentos específicos en la base de datos que respondan exactamente a tu consulta."

                st.markdown(respuesta_ia)
                st.session_state.messages.append({"role": "assistant", "content": respuesta_ia})
