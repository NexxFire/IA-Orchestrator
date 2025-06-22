from tools import rag, scrap, scrap_to_file
import re

def classify_question(question: str) -> str:
    save_keywords = [
        "enregistre", "sauvegarde", "sauvegarder",
        "garde", "conserve", "mets-le", "exporte", "archive", "copie"
    ]

    if re.search(r'https?://\S+', question):
        if any(word in question.lower() for word in save_keywords):
            return "scrap_to_file"
        return "scrap"
    return "rag"


def handle_question(question: str) -> str:
    tool = classify_question(question)

    if tool == "scrap_to_file":
        scrap_to_file.answer_question(question)  # Cette fonction gère tout et enregistre la réponse
        return "Réponse enregistrée dans un fichier."

    elif tool == "scrap":
        return scrap.answer_question(question)

    elif tool == "rag":
        return rag.answer_question(question)

    else:
        return "Aucun outil ne correspond à cette requête."
