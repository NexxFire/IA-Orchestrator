import re
from utils import scrap_url



def answer_question(question: str) -> str:
    """
    Récupère l'URL dans la question, scrape la page, sauvegarde le texte brut dans un fichier,
    et confirme la sauvegarde sans faire de génération ou réponse.
    """

    # Extraire l'URL depuis la question
    url_match = re.search(r'https?://\S+', question)
    if not url_match:
        return "[ERREUR] Aucune URL trouvée dans la question."

    url = url_match.group(0)

    # Scraper tout le contenu pertinent de la page
    scraped_text = scrap_url(url)
    if scraped_text.startswith("[ERREUR]"):
        return scraped_text

    # Sauvegarder dans un fichier texte
    output_filename = "page_scrap.txt"
    try:
        with open(output_filename, "w", encoding="utf-8") as f:
            f.write(f"URL scrappée : {url}\n\n")
            f.write(scraped_text)
        print(f"✅ Contenu complet sauvegardé dans le fichier : {output_filename}")
    except Exception as e:
        return f"[ERREUR] Impossible d'écrire le fichier : {str(e)}"

    return f"✅ Le contenu complet de la page a été sauvegardé dans '{output_filename}'."
