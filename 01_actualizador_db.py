# Actualizador de base de datos versión 1.0

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

def extraer_texto_de_url(url):
    from urllib.parse import urlparse
    try:
        response = session.get(url, headers=HEADERS, timeout=25) 
        response.raise_for_status() 
        content_type = response.headers.get('Content-Type', '').lower()
        
        if 'application/pdf' in content_type or url.lower().endswith('.pdf'):
            return extraer_texto_de_pdf(response.content)
            
        else:
            soup = BeautifulSoup(response.content, 'html.parser')
            enlaces = soup.find_all('a', href=True)
            url_del_pdf = None
            
            for enlace in enlaces:
                href = enlace['href']
                if '.pdf' in href.lower():
                    url_del_pdf = href
                    if not url_del_pdf.startswith('http'):
                        parsed_uri = urlparse(url)
                        base_url = f"{parsed_uri.scheme}://{parsed_uri.netloc}"
                        url_del_pdf = base_url + url_del_pdf
                    break 
            
            if url_del_pdf:
                print(f"    -> Redirigiendo a PDF adjunto: {url_del_pdf[:60]}...")
                pdf_response = session.get(url_del_pdf, headers=HEADERS, timeout=25)
                pdf_response.raise_for_status()
                return extraer_texto_de_pdf(pdf_response.content)
            else:
                for script in soup(["script", "style"]):
                    script.extract()
                return soup.get_text(separator=' ', strip=True)
                
    except Exception as e:
        print(f"  [!] Salto de URL por error de conexión o tiempo de espera en: {url}")
        return None

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
        enlace = row['Enlace']
        nombre_doc = str(row['Documento']) 
        
        # Identificador único basado en URL o Texto Directo
        id_registro = str(enlace).strip() if pd.notna(enlace) and str(enlace).strip().startswith('http') else nombre_doc
        nuevo_control[id_registro] = nombre_doc

        # Verificación inteligente: Si ya existe en control y la DB está creada, se omite
        if id_registro in control_historico and os.path.exists(DIRECTORIO_DB):
            continue

        # Procesar descarga de datos
        texto_extraido = ""
        if pd.notna(enlace) and str(enlace).strip().startswith('http'):
            print(f"[{index + 1}/{total_filas}] Sincronizando URL: {nombre_doc[:50]}...")
            texto_extraido = extraer_texto_de_url(id_registro)
            time.sleep(1.5)  # Delay prudencial antipandemia de bloqueos IP
        else:
            print(f"[{index + 1}/{total_filas}] Registrando Texto Directo: {nombre_doc[:50]}...")
            texto_extraido = nombre_doc 

        if texto_extraido and len(texto_extraido.strip()) > 10: 
            doc = Document(
                page_content=texto_extraido,
                metadata={
                    "tema": str(row.get('Tema', 'N.D.')),
                    "subtema": str(row.get('Subtema', 'N.D.')),
                    "documento": nombre_doc[:70], 
                    "enlace": str(enlace) if pd.notna(enlace) else "Texto directo",
                    "año": str(row.get('Año', 'N.D.'))
                }
            )
            documentos_nuevos.append(doc)
            
        # Liberación explícita de variables pesadas en memoria RAM
        del texto_extraido
        gc.collect()

    if not documentos_nuevos:
        print("\n😎 Todo al día: No se detectaron modificaciones en el Excel. Base de datos sin cambios.")
        guardar_control(nuevo_control)
        return

    print(f"\nSe encontraron {len(documentos_nuevos)} modificaciones/nuevos registros. Segmentando texto...")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    fragmentos = text_splitter.split_documents(documentos_nuevos)
    
    print("Inicializando modelo de embeddings...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        model_kwargs={"device": "cpu"}
    )

    print(f"Indexando de forma incremental en base de datos: '{DIRECTORIO_DB}'...")
    
    # Chroma.from_documents añade registros incrementalmente si la DB ya existe, no la destruye
    vectorstore = Chroma.from_documents(
        documents=fragmentos, 
        embedding=embeddings, 
        persist_directory=DIRECTORIO_DB
    )
    
    guardar_control(nuevo_control)    
    print("✅ ¡Sincronización finalizada exitosamente!")

if __name__ == "__main__":
    ejecutar_actualizacion()