import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import numpy as np
from scipy.spatial.distance import cosine
from openai import OpenAI

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extrait tout le texte d'un PDF en une seule chaîne."""
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text

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



# Charge un modèle local pour encoder du texte en vecteurs
embedder = SentenceTransformer("all-MiniLM-L6-v2")

def embed_texts(chunks: list[str]) -> np.ndarray:
    """Transforme une liste de textes en vecteurs."""
    embeddings = embedder.encode(chunks, convert_to_numpy=True, show_progress_bar=True)
    return embeddings


def get_top_k_chunks(question: str, chunks: list[str], chunk_embeddings: np.ndarray, k: int = 3) -> list[str]:
    """
    Renvoie les k chunks les plus similaires à la question.
    Pour cela on calcul le vecteur de la question et on le compare aux vecteurs des chunks.
    Le calcul repose sur la similarité cosinus, plus la valeur est proche de 1, plus les vecteurs sont similaires.
    """
    question_embedding = embed_texts([question])[0]  # un seul vecteur
    scores = []

    for idx, chunk_vector in enumerate(chunk_embeddings):
        similarity = 1 - cosine(question_embedding, chunk_vector)
        scores.append((similarity, idx))

    # Trier par similarité décroissante
    scores.sort(reverse=True)
    top_chunks = [chunks[idx] for _, idx in scores[:k]]

    return top_chunks


# Configuration du client vers le serveur local
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="llama-local"  # requis même s’il n’est pas utilisé
)

def query_llama(question: str, context_chunks: list[str]) -> str:
    """Construit le prompt et interroge le modèle LLaMA local."""
    context_text = "\n\n".join(context_chunks)
    system_prompt = (
        "Tu es un assistant intelligent. Tu vas répondre à la question de l'utilisateur "
        "en te basant uniquement sur les informations suivantes extraites d'un document :\n\n"
        f"{context_text}"
    )

    response = client.chat.completions.create(
        model="llama",  # adapte au nom de ton modèle si nécessaire
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ],
        temperature=0.2,
        max_tokens=512,
    )

    return response.choices[0].message.content
