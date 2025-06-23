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
        "Les sujets couverts par le rag sont :\n"
        "- Des généralité sur les bases de données\n"
        "- La modulation AM-P\n"
        "L'eco conception\n"
        "Un pokedex sur les pokemons de la generation 1\n"
        "Le rapport de stage d'alexi a l'iMSA couvrant les différents projets qu'il a realisé\n"
        "Répond STRICTEMENT par un JSON valide de la forme {\"tool\": \"rag\"} sans aucun autre texte,\n"
        "sans explication, ni balises <think>, ni espaces superflus.\n"
        "Ne mets aucun autre caractère.\n"
        "Voici la question :"
    )

    response = query_llama(
        question=question,
        context_chunks=[{"file": "orchestrator", "page": 1, "text": system}]
    )
    print(f"[Orchestrator] Réponse brute du LLM :\n{response}")

    # Extraire strictement le premier objet JSON dans la réponse
    # On recherche la première occurrence d'un JSON objet (entre accolades équilibrées)
    def extract_first_json(text):
        stack = []
        start_idx = None
        for i, char in enumerate(text):
            if char == '{':
                if not stack:
                    start_idx = i
                stack.append('{')
            elif char == '}':
                if stack:
                    stack.pop()
                    if not stack and start_idx is not None:
                        return text[start_idx:i+1]
        return None

    json_str = extract_first_json(response)
    if json_str is None:
        raise ValueError(f"Aucun JSON trouvé dans la réponse :\n{response}")

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"Erreur de parsing JSON : {e}\nJSON brut :\n{json_str}")

    if "tool" not in data:
        raise ValueError(f"Le JSON ne contient pas la clé 'tool' :\n{json_str}")

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
