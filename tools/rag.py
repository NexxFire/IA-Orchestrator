from utils import extract_all_chunks_from_folder, embed_texts, get_top_k_chunks, query_llama
import os



def answer_question(question: str) -> str:
    """Répond à une question en utilisant RAG (Retrieval-Augmented Generation)."""
    # 1. Lecture et découpage
    all_chunks = extract_all_chunks_from_folder("./documents")

    # 2. Vectorisation
    embeddings = embed_texts(all_chunks)

    # 3.1 Obtenir les k chunks les plus pertinents
    top_chunks = get_top_k_chunks(question, all_chunks, embeddings, k=3)

    # 3.2 Requête à LLaMA
    response = query_llama(question, top_chunks)
    print("\nRéponse de LLaMA :")
    print(response)