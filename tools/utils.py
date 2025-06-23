import os
import fitz  # PyMuPDF
import hashlib
import pickle
import numpy as np
import torch
from typing import List
from tqdm import tqdm
from bs4 import BeautifulSoup
import requests
from sentence_transformers import SentenceTransformer

from openai import OpenAI
from scipy.spatial.distance import cosine

# Constantes
MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
EMBEDDING_FILE = "vector_db.pkl"
DOCUMENT_FOLDER = "data"

# Variables globales
CHUNKS = []
EMBEDDINGS = []

# Chargement modèle Camembert
model = SentenceTransformer(MODEL_NAME)

model.eval()


# ============ Extraction ============

def split_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """Découpe le texte en chunks"""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def extract_chunks_from_pdf(pdf_path: str) -> List[dict]:
    """Extrait chaque page comme un chunk avec métadonnées."""
    chunks = []
    filename = os.path.basename(pdf_path)

    try:
        doc = fitz.open(pdf_path)
        for i, page in enumerate(doc, start=1):
            text = page.get_text().strip()
            if text:
                chunks.append({"file": filename, "page": i, "text": text})
        doc.close()
    except Exception as e:
        print(f"Erreur avec {filename} : {e}")

    return chunks


def extract_chunks_from_folder(folder_path: str) -> List[dict]:
    """Parcourt tous les PDF du dossier et extrait les chunks."""
    all_chunks = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(".pdf"):
            filepath = os.path.join(folder_path, filename)
            all_chunks.extend(extract_chunks_from_pdf(filepath))
    print(f"✅ Extraction terminée : {len(all_chunks)} chunks trouvés.")
    return all_chunks


# ============ Embedding & Base vectorielle ============

def get_folder_hash(folder_path: str) -> str:
    """Crée un hash MD5 unique basé sur le contenu des PDF."""
    hash_md5 = hashlib.md5()
    for filename in sorted(os.listdir(folder_path)):
        if filename.lower().endswith(".pdf"):
            with open(os.path.join(folder_path, filename), "rb") as f:
                while chunk := f.read(8192):
                    hash_md5.update(chunk)
    return hash_md5.hexdigest()


def embed_texts(chunks: List[dict]) -> np.ndarray:
    """Génère les embeddings SentenceTransformer des chunks."""
    texts = [chunk["text"] for chunk in chunks]
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)
    return embeddings



def save_embeddings(chunks: List[dict], embeddings: np.ndarray, folder_hash: str):
    with open(EMBEDDING_FILE, "wb") as f:
        pickle.dump({
            "chunks": chunks,
            "embeddings": embeddings,
            "hash": folder_hash
        }, f)
    print(f"💾 Embeddings sauvegardés dans {EMBEDDING_FILE}")


def load_embeddings(expected_hash: str) -> bool:
    global CHUNKS, EMBEDDINGS
    if os.path.exists(EMBEDDING_FILE):
        with open(EMBEDDING_FILE, "rb") as f:
            data = pickle.load(f)
            if data.get("hash") == expected_hash:
                CHUNKS = data["chunks"]
                EMBEDDINGS = data["embeddings"]
                print(f"📂 Embeddings chargés depuis {EMBEDDING_FILE} ({len(CHUNKS)} chunks)")
                return True
            else:
                print("⚠️ Les fichiers ont changé — recalcul des embeddings.")
    return False


def load_document(folder_path: str = DOCUMENT_FOLDER):
    """Charge ou régénère les embeddings selon les fichiers PDF présents."""
    global CHUNKS, EMBEDDINGS
    folder_hash = get_folder_hash(folder_path)

    if load_embeddings(expected_hash=folder_hash):
        return

    CHUNKS = extract_chunks_from_folder(folder_path)
    EMBEDDINGS = embed_texts(CHUNKS)
    save_embeddings(CHUNKS, EMBEDDINGS, folder_hash)


# ============ Question & Similarité ============

def get_top_k_chunks(question: str, k: int = 3) -> List[dict]:
    """Renvoie les k chunks les plus similaires à la question."""
    question_embedding = model.encode(question, convert_to_numpy=True, normalize_embeddings=True)

    similarities = [
        (1 - cosine(question_embedding, emb), idx)
        for idx, emb in enumerate(EMBEDDINGS)
    ]

    similarities.sort(reverse=True)
    return [CHUNKS[i] for _, i in similarities[:k]]



# ============ LLM & Requête ============

client = OpenAI(
    base_url="http://127.0.0.1:8080",
    api_key="llama-local"
)

def query_llama(question: str, context_chunks: List[dict]) -> str:
    context_text = "\n\n".join(
        f"[{chunk.get('file')} - page {chunk.get('page')}]:\n{chunk['text']}" for chunk in context_chunks
    )
    system_prompt = (
        "Tu es un assistant intelligent. Tu réponds uniquement à la question, "
        "sans réflexion, sans texte explicatif. "
        "Voici les informations disponibles :\n\n"
        f"{context_text}"
    )
    response = client.chat.completions.create(
        model="Qwen3-14B-Q6_K.gguf",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ],
        temperature=0.2,
        max_tokens=3000,
    )
    return response.choices[0].message.content.strip()


# ============ Web scraping utilitaire ============

def scrap_url(url: str, min_paragraph_length: int = 50) -> str:
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, 'html.parser')
        for tag in soup(['script', 'style', 'header', 'footer', 'nav', 'aside']):
            tag.decompose()

        if "wikipedia.org" in url:
            for tag in soup.select(".reference, .infobox"):
                tag.decompose()

        paragraphs = soup.find_all('p')
        text = '\n'.join(
            p.get_text(strip=True) for p in paragraphs if len(p.get_text(strip=True)) >= min_paragraph_length
        )

        return text.strip()
    except Exception as e:
        return f"[ERREUR] Scraping échoué : {str(e)}"