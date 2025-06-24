from utils import extract_text_from_pdf, split_text, embed_texts, get_top_k_chunks, query_llama

# 1. Lecture et découpage
text = extract_text_from_pdf("/home/xi/Documents/LA2/IA/RAG/document.pdf")
chunks = split_text(text)

# 2. Vectorisation
embeddings = embed_texts(chunks)

# 3. Question utilisateur
while True:
    question = input("Pose ta question (ou 'exit' pour quitter) : ")
    if question.lower() == 'exit':
        break

    # 3.1 Obtenir les k chunks les plus pertinents
    top_chunks = get_top_k_chunks(question, k=5)

    # 3.2 Requête à LLaMA
    response = query_llama(question, top_chunks)
    print("\nRéponse de LLaMA :")
    print(response)
