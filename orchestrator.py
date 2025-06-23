from tools import rag, scrap, scrap_to_file
from tools.utils import query_llama
import json
import re

def prompt_orchestrator_llm(question: str) -> str:
    system = (
        "Tu es un orchestrateur intelligent.\n"
        "Tu dois choisir quelle méthode utiliser parmi :\n"
        "- 'rag' : pour répondre à partir de documents PDF internes\n"
        "- 'scrap' : pour répondre en extrayant des infos depuis une URL\n"
        "- 'scrap_to_file' : pour extraire une URL ET enregistrer le résultat dans un fichier\n\n"
        "Tu DOIS répondre uniquement par un JSON de la forme {\"tool\": \"rag\"} sans aucun autre texte, ni explication, ni balises comme <think>.\n"
        "Respecte STRICTEMENT ce format.\n"
        "Voici la question :"
    )

    response = query_llama(
        question=question,
        context_chunks=[{"file": "orchestrator", "page": 1, "text": system}]
    )

    # Nettoyer la réponse pour garder uniquement le JSON
    # On cherche un JSON dans la réponse
    match = re.search(r'\{.*\}', response, re.DOTALL)
    if not match:
        raise ValueError(f"Réponse non interprétable comme JSON :\n{response}")
    json_str = match.group()
    data = json.loads(json_str)
    return data["tool"].strip().lower()


def handle_question(question: str) -> str:
    try:
        chosen_tool = prompt_orchestrator_llm(question)
        print(f"[Orchestrator] Outil choisi : {chosen_tool}")

        if chosen_tool == "scrap_to_file":
            scrap_to_file.answer_question(question)
            return "Réponse enregistrée dans un fichier."

        elif chosen_tool == "scrap":
            return scrap.answer_question(question)

        elif chosen_tool == "rag":
            # Appelle RAG, qui répond sans <think> grâce au prompt modifié
            return rag.answer_question(question)

        else:
            return f"[Orchestrator] Outil non reconnu : {chosen_tool}"

    except Exception as e:
        return f"[ERREUR orchestrator] : {str(e)}"
