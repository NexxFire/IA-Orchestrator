from tools.utils import extract_chunks_from_folder, embed_texts, get_top_k_chunks, query_llama

# Variables globales
CHUNKS = []
EMBEDDINGS = []

def load_document(folder_path: str = "./documents") -> None:
    """
    Charge les documents depuis un dossier, extrait les chunks et calcule les embeddings.
    """
    global CHUNKS, EMBEDDINGS

    raw_chunks = extract_chunks_from_folder(folder_path)  # list[str]

    # Ajout de 'file' et 'page' pour compatibilité avec query_llama
    CHUNKS = [
        {"text": chunk, "file": "document.pdf", "page": i + 1}
        for i, chunk in enumerate(raw_chunks)
    ]

    EMBEDDINGS = embed_texts(CHUNKS)
    print(f"[RAG] {len(CHUNKS)} chunks chargés depuis {folder_path}")

def answer_question(question: str) -> str:
    """
    Répond à une question en utilisant RAG (Retrieval-Augmented Generation).
    """
    if len(CHUNKS) == 0 or len(EMBEDDINGS) == 0:
        return "[ERREUR] Les documents n'ont pas été chargés. Appelle load_document() d'abord."

    top_chunks = get_top_k_chunks(question, CHUNKS, EMBEDDINGS, k=3)
    response = query_llama(question, top_chunks)
    return response
