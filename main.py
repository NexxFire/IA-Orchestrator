from orchestrator import handle_question
from tools import rag

rag.load_document("data")  # Chemin OK ici

print("Assistant intelligent (RAG + SCRAP + OS)\n")

while True:
    question = input("Pose ta question (ou 'exit') : ")
    if question.lower() == 'exit':
        break

    response = handle_question(question)
    print("\nRéponse :\n", response)
