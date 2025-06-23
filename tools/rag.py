from tools.utils import get_top_k_chunks, query_llama, load_document


def answer_question(question: str) -> str:
    """
    Répond à une question en utilisant RAG (Retrieval-Augmented Generation).
    """
    load_document()

    top_chunks = get_top_k_chunks(question, k=5)
    response = query_llama(question, top_chunks)
    return response
