import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import numpy as np
from scipy.spatial.distance import cosine
from openai import OpenAI
import os



def extract_chunks_from_folder(folder_path: str = "./documents") -> list[str]:
    """
    Parcourt tous les fichiers PDF d’un dossier et extrait chaque page comme un chunk,
    en utilisant extract_chunks_from_pdf.
    
    Retourne une liste de strings formatés par page.
    """
    all_chunks = []
    total_files = 0

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(".pdf"):
            total_files += 1
            filepath = os.path.join(folder_path, filename)
            chunks = extract_chunks_from_pdf(filepath)
            all_chunks.extend(chunks)

    print(f"\n {total_files} fichier(s) PDF traité(s), {len(all_chunks)} chunk(s) extraits au total.")
    return all_chunks



def extract_chunks_from_pdf(pdf_path: str) -> list[str]:
    """Extrait chaque page d'un pdf comme un chunk entier, formaté avec nom de fichier et numéro de page."""
    chunks = []
    filename = os.path.basename(pdf_path)

    try:
        doc = fitz.open(pdf_path)
        for i, page in enumerate(doc, start=1):
            text = page.get_text().strip()
            if text:  # ignorer les pages vides
                chunk = f"Page {i} du fichier {filename}:\n{text}"
                chunks.append(chunk)
        doc.close()
    except Exception as e:
        print(f"Erreur dans le fichier {filename} : {e}")

    return chunks



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





def embed_texts(chunks: list[dict]) -> np.ndarray:
    """Transforme une liste de chunks (avec 'text') en vecteurs."""
    # Charge un modèle local pour encoder du texte en vecteurs
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    texts = [chunk["text"] for chunk in chunks]
    embeddings = embedder.encode(texts, convert_to_numpy=True, show_progress_bar=True)
    return embeddings



def get_top_k_chunks(question: str, chunks: list[dict], chunk_embeddings: np.ndarray, k: int = 3) -> list[dict]:
    """
    Renvoie les k chunks les plus similaires à la question.
    """
    question_embedding = embed_texts([{"text": question}])[0]  # on simule un chunk dict ici
    scores = []

    for idx, chunk_vector in enumerate(chunk_embeddings):
        similarity = 1 - cosine(question_embedding, chunk_vector)
        scores.append((similarity, idx))

    scores.sort(reverse=True)
    top_chunks = [chunks[idx] for _, idx in scores[:k]]

    return top_chunks



# Configuration du client vers le serveur local
client = OpenAI(
    base_url="http://127.0.0.1:8080",
    api_key="llama-local"  # requis même s’il n’est pas utilisé
)

def query_llama(question: str, context_chunks: list[dict]) -> str:
    """Construit le prompt avec des infos sur la source de chaque chunk."""
    context_text = "\n\n".join(
        f"[{chunk['file']} - page {chunk['page']}]:\n{chunk['text']}" for chunk in context_chunks
    )

    system_prompt = (
        "Tu es un assistant intelligent. Tu vas répondre à la question de l'utilisateur "
        "en te basant uniquement sur les informations suivantes extraites d'un document :\n\n"
        f"{context_text}"
    )

    response = client.chat.completions.create(
        model="Qwen3-14B-Q6_K.gguf",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ],
        temperature=0.2,
        max_tokens=512,
    )

    return response.choices[0].message.content



def scrap_url(url: str, min_paragraph_length: int = 50) -> str:
    """Scrape une page web générique (Wikipédia ou autre)."""
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, 'html.parser')
        for tag in soup(['script', 'style', 'header', 'footer', 'nav', 'aside']):
            tag.decompose()

        # Traitement spécial Wikipédia
        if "wikipedia.org" in url:
            for tag in soup.select(".reference, .infobox"):
                tag.decompose()

        paragraphs = soup.find_all('p')
        text = '\n'.join(
            p.get_text(strip=True) for p in paragraphs if len(p.get_text(strip=True)) >= min_paragraph_length
        )

        return text.strip()

    except Exception as e:
        return f"[ERREUR] Impossible de scrap {url} : {str(e)}"