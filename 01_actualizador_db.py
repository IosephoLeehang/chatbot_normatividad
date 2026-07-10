# Actualizador de base de datos versión 1.0
# La ejecución es local no se ejecuta en la nube
# Para generar la BD y luego cargarla a la nube

import os
import json
import time
import gc
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import fitz  # PyMuPDF
from bs4 import BeautifulSoup
from langchain_core.documents import Document 
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import base64

# Configuración de Archivos y Carpetas
ARCHIVO_XLS = "Tabulado normas.xlsx"
DIRECTORIO_DB = "./chroma_db_normas" 
ARCHIVO_CONTROL = "control_procesamiento.json"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36"
}

# Configuración de sesión HTTP resiliente
session = requests.Session()
retry = Retry(connect=3, read=3, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
adapter = HTTPAdapter(max_retries=retry)
session.mount('http://', adapter)
session.mount('https://', adapter)

def cargar_control():
    if os.path.exists(ARCHIVO_CONTROL):
        with open(ARCHIVO_CONTROL, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def guardar_control(control_data):
    with open(ARCHIVO_CONTROL, 'w', encoding='utf-8') as f:
        json.dump(control_data, f, ensure_ascii=False, indent=4)

def extraer_texto_de_pdf(contenido_binario):
    texto = ""
    try:
        with fitz.open(stream=contenido_binario, filetype="pdf") as doc:
            for pagina in doc:
                texto += pagina.get_text() + "\n"
    except Exception as e:
        print(f"  [!] Error parseando PDF binario: {e}")
    return texto

import re

def extraer_texto_de_url(url_o_ruta):
    url_o_ruta = str(url_o_ruta).strip()
    texto_extraido = ""
    
    try:
        # --- CASO A: ES UN ARCHIVO LOCAL EN TU PC ---
        if not url_o_ruta.startswith("http"):
            if os.path.exists(url_o_ruta):
                with fitz.open(url_o_ruta) as doc_pdf:
                    for pagina in doc_pdf:
                        texto_extraido += pagina.get_text() + "\n"
                return texto_extraido
            else:
                print(f"  [!] Archivo local no encontrado en la ruta: {url_o_ruta}")
                return ""

        # --- CASO B: ES UN ENLACE DE GOOGLE DRIVE ---
        if "drive.google.com" in url_o_ruta:
            match = re.search(r'/d/([a-zA-Z0-9_-]+)', url_o_ruta)
            if match:
                file_id = match.group(1)
                url_o_ruta = f"https://drive.google.com/uc?export=download&id={file_id}"

        # --- CASO C: EXTRACCIÓN WEB (Drive Directo o Páginas Normales) ---
        response = session.get(url_o_ruta, headers=HEADERS, timeout=30)
        response.raise_for_status()
        content_type = response.headers.get("Content-Type", "").lower()
        
        # Si es un PDF (como el de Google Drive) lo leemos en memoria
        if "application/pdf" in content_type or "drive.google.com/uc" in url_o_ruta or url_o_ruta.lower().endswith('.pdf'):
            with fitz.open(stream=response.content, filetype="pdf") as doc_pdf:
                for pagina in doc_pdf:
                    texto_extraido += pagina.get_text() + "\n"
        else:
            # Si es una web normal del gobierno
            soup = BeautifulSoup(response.text, "html.parser")
            texto_extraido = soup.get_text(separator="\n").strip()
            
        return texto_extraido

    except Exception as e:
        print(f"  [!] Error crítico extrayendo texto de {url_o_ruta}: {e}")
        return ""

def ejecutar_actualizacion():
    print("🚀 Iniciando servicio de sincronización de Base de Datos Vectorial...")
    
    if not os.path.exists(ARCHIVO_XLS):
        print(f"[!] Cancelado: No existe el archivo Excel '{ARCHIVO_XLS}'")
        return

    df = pd.read_excel(ARCHIVO_XLS).dropna(subset=['Documento'])
    total_filas = len(df)
    
    control_historico = cargar_control()
    documentos_nuevos = []
    nuevo_control = {}
    
    # 1. Identificar estrictamente cambios o nuevos registros
    for index, row in df.iterrows():
        enlace = row.get('Enlace', '')
        nombre_doc = str(row['Documento']) 
        enlace_str = str(enlace).strip() if pd.notna(enlace) else ""
        
        id_registro = enlace_str if enlace_str else nombre_doc
        nuevo_control[id_registro] = nombre_doc
        
        # Verificación incremental
        if id_registro in control_historico and os.path.exists(DIRECTORIO_DB):
            continue

        # Procesar extracción del texto
        texto_extraido = ""
        if enlace_str:
            print(f"[{index + 1}/{total_filas}] Procesando Documento: {nombre_doc[:50]}...")
            texto_extraido = extraer_texto_de_url(enlace_str)

            # Adicional para verificar lo de la Ley 30364
            # =============================================================================
            #if "30364" in str(nombre_doc) or "30364" in str(enlace):
            #    print(f"--- TEXTO LEIDO DE LA WEB --- \n{texto_extraido[:200]}\n-----------------------------")
            # =============================================================================

            time.sleep(1.5)
        else:
            print(f"[{index + 1}/{total_filas}] Registrando Texto Directo: {nombre_doc[:50]}...")

        # 2. Respaldo si la extracción falló o no hubo enlace
        if not texto_extraido.strip():
            texto_extraido = "Contenido referencial proveniente de la matriz."

        # 3. TEXTO ENRIQUECIDO: Fusión de metadatos del Excel + PDF
        texto_enriquecido = f"""
        IDENTIFICADOR DE NORMA: {nombre_doc} | {nombre_doc}
        AÑO DE PUBLICACIÓN OFICIAL: {row.get('Año', 'No especificado')}
        TEMA PRINCIPAL: {row.get('Tema', 'No especificado')}
        SUBTEMA: {row.get('Subtema', 'No especificado')}
        
        Contenido Íntegro de la Norma Legal:
        {texto_extraido}
        """            

        # 4. Creación del documento   
        doc = Document(
              page_content=texto_enriquecido,
              metadata={
                  "tema": str(row.get('Tema', 'N.D.')),
                  "subtema": str(row.get('Subtema', 'N.D.')),
                  "documento": nombre_doc[:70], 
                  "enlace": str(enlace) if pd.notna(enlace) else "Texto directo",
                  "año de publicación": str(row.get('Año', 'N.D.'))
                  }
            )
        documentos_nuevos.append(doc)
            
        # Liberación explícita de variables pesadas en memoria RAM
        del texto_extraido
        gc.collect()

    if not documentos_nuevos:
        print("\n😎 Todo al día: No se detectaron modificaciones en el Excel.")
        guardar_control(nuevo_control)
        return

    print(f"\nSe encontraron {len(documentos_nuevos)} nuevos registros. Segmentando texto...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150, length_function=len)
    fragmentos = text_splitter.split_documents(documentos_nuevos)
    
    print("Inicializando modelo de embeddings...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        model_kwargs={"device": "cpu"}
    )

    print(f"Indexando en base de datos: '{DIRECTORIO_DB}'...")
    vectorstore = Chroma.from_documents(
        documents=fragmentos, 
        embedding=embeddings, 
        persist_directory=DIRECTORIO_DB
    )
    
    guardar_control(nuevo_control)    
    print("✅ ¡Sincronización finalizada exitosamente!")

if __name__ == "__main__":
    ejecutar_actualizacion()