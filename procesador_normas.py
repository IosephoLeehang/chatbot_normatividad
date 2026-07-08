# Para instalar
# python -m pip install pandas requests beautifulsoup4 pymupdf langchain langchain-community langchain-huggingface chromadb sentence-transformers

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import fitz  # PyMuPDF
from bs4 import BeautifulSoup
import os
import time
from langchain_core.documents import Document 
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# 1. Configuración Inicial
ARCHIVO_XLS = "C:\\Proyectos\\Minmuj\\Proy 005\\Informatico\\Pro aplicativo\\Tabulado normas.xlsx"
DIRECTORIO_DB = "./chroma_db_normas" 

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36"
}

# --- NUEVO: Configuración de la Sesión para evitar el error "Max retries exceeded" ---
session = requests.Session()
# Configuramos reintentos: Si falla, reintenta 3 veces. Espera 1s, luego 2s, luego 4s entre intentos.
retry = Retry(connect=3, read=3, backoff_factor=1, status_forcelist=[ 500, 502, 503, 504 ])
adapter = HTTPAdapter(max_retries=retry)
session.mount('http://', adapter)
session.mount('https://', adapter)
# --------------------------------------------------------------------------------------

def extraer_texto_de_pdf(contenido_binario):
    """Función de apoyo para leer el interior de un PDF"""
    texto = ""
    with fitz.open(stream=contenido_binario, filetype="pdf") as doc:
        for pagina in doc:
            texto += pagina.get_text() + "\n"
    return texto

def extraer_texto_de_url(url):
    """Visita la URL, si es web busca un PDF dentro, si no, lee la web."""
    from urllib.parse import urlparse
    
    try:
        # Usamos la 'session' que tiene reintentos automáticos en lugar de 'requests.get'
        response = session.get(url, headers=HEADERS, timeout=20) 
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
                print(f"    -> Se encontró un PDF adjunto, redirigiendo a: {url_del_pdf[:40]}...")
                # También usamos 'session' aquí para la redirección
                pdf_response = session.get(url_del_pdf, headers=HEADERS, timeout=20)
                pdf_response.raise_for_status()
                return extraer_texto_de_pdf(pdf_response.content)
                
            else:
                for script in soup(["script", "style"]):
                    script.extract()
                return soup.get_text(separator=' ', strip=True)
                
    except Exception as e:
        # Si aún después de todos los reintentos falla, capturamos el error sin detener el script
        print(f"  [!] No se pudo acceder a la página después de varios intentos: {url}")
        # print(f"  [Detalle Técnico]: {e}") # Descomenta esto si quieres ver el error feo
        return None

def procesar_excel_y_crear_bd():
    print("Iniciando Fase 1: Lectura masiva de documentos y textos directos...")
    
    df = pd.read_excel(ARCHIVO_XLS)
    documentos_procesados = []
    
    df = df.dropna(subset=['Documento']) 
    total_filas = len(df)
    
    for index, row in df.iterrows():
        enlace = row['Enlace']
        nombre_doc = str(row['Documento']) 
        
        texto_extraido = ""
        
        if pd.notna(enlace) and str(enlace).strip().startswith('http'):
            print(f"[{index + 1}/{total_filas}] Descargando URL: {nombre_doc[:40]}...")
            texto_extraido = extraer_texto_de_url(enlace)
            time.sleep(2) # AUMENTAMOS A 2 SEGUNDOS DE PAUSA para no saturar al servidor
        else:
            print(f"[{index + 1}/{total_filas}] Leyendo texto directo: {nombre_doc[:40]}...")
            texto_extraido = nombre_doc 

        if texto_extraido and len(texto_extraido.strip()) > 10: 
            doc = Document(
                page_content=texto_extraido,
                metadata={
                    "tema": str(row.get('Tema', 'N.D.')),
                    "subtema": str(row.get('Subtema', 'N.D.')),
                    "documento": nombre_doc[:50], 
                    "enlace": str(enlace) if pd.notna(enlace) else "Texto directo",
                    "año": str(row.get('Año', 'N.D.'))
                }
            )
            documentos_procesados.append(doc)
        else:
            print(f"  [-] No se pudo obtener contenido útil de la fila {index + 1}")

    print(f"\n¡Extracción completada! Se procesaron {len(documentos_procesados)} registros exitosamente.")
    
    print("Dividiendo los textos en fragmentos para la Inteligencia Artificial...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200 
    )
    fragmentos = text_splitter.split_documents(documentos_procesados)
    
    print(f"Creando la base de datos vectorial en '{DIRECTORIO_DB}'...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    
    vectorstore = Chroma.from_documents(
        documents=fragmentos, 
        embedding=embeddings, 
        persist_directory=DIRECTORIO_DB
    )
    
    print("\n✅ ¡Fase 1 Completada con éxito!")
    print(f"Tu base de datos está lista y guardada en la carpeta: {DIRECTORIO_DB}")

if __name__ == "__main__":
    procesar_excel_y_crear_bd()