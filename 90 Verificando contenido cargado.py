# Verificando contenido cargado

import os
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

RUTA_DB = "./chroma_db_normas"

# 1. Inicializar el mismo modelo de embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    model_kwargs={"device": "cpu"}
)

# 2. Conectar a la base de datos local
if os.path.exists(RUTA_DB):
    db = Chroma(persist_directory=RUTA_DB, embedding_function=embeddings)
    
    # 3. Texto de búsqueda para verificar
    consulta = "¿En qué año se publicó la Ley 30364?"
    resultados = db.similarity_search(consulta, k=10)
    
    print(f"\n🔍 SE ENCONTRARON {len(resultados)} FRAGMENTOS RELEVANTES:\n")
    for i, doc in enumerate(resultados):
        print(f"--- Fragmento {i+1} ---")
        print(f"Fuente: {doc.metadata.get('documento', 'No definido')}")
        print(f"Enlace/Origen: {doc.metadata.get('enlace', 'No definido')}")
        print(f"Contenido: {doc.page_content[:300]}...\n")
else:
    print("❌ La carpeta de la base de datos no existe en esta ruta.")