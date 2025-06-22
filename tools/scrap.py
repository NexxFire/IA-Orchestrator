from bs4 import BeautifulSoup
import requests
from utils import split_text, embed_texts, get_top_k_chunks, query_llama, scrap_url



def answer_question(question: str) -> str:
    """Scrape la première URL détectée dans la question et répond à partir de son contenu."""
    urls = re.findall(r'(https?://\S+)', question)
    if not urls:
        return "[ERREUR] Aucun lien détecté dans la question."

    url = urls[0]
    text = scrap_url(url)
    if text.startswith("[ERREUR]"):
        return text

    chunks = split_text(text)
    embeddings = embed_texts(chunks)
    top_chunks = get_top_k_chunks(question, chunks, embeddings)
    return query_llama(question, top_chunks)
