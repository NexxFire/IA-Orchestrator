# Introduction

Ce programme implémente une RAG (Retrieval-Augmented Generation), c’est-à-dire une génération de texte assistée par une recherche documentaire, avec un modèle de langage.

## Fonctionnement
1. **Extraction et découpage du texte** : Le programme prend en entrée un fichier PDF, en extrait le contenu, puis le divise en morceaux (chunks) de 500 tokens.
2. **Vectorisation des chunks** : Chaque morceau est converti en un vecteur numérique appelé embedding grâce au modèle pré-entraîné SentenceTransformer all-MiniLM-L6-v2.
Ces embeddings capturent la signification sémantique des morceaux, en traduisant leur contenu textuel en vecteurs de caractéristiques numériques.
3. **Recherche de similarité** : Lorsqu'une question est posée, le programme compare l'embedding de la question avec ceux des morceaux de texte pour identifier les 3 morceaux les plus pertinents.
4. **Génération de réponse** : Les morceaux pertinents sont utilisés pour construire un contexte enrichi autour de la question.
Ce contexte est envoyé avec la question au modèle de langage via l'API de llama-server, qui génère une réponse basée sur ce contexte.

## Installation
```bash
# 1. Créer un environnement virtuel python
python -m venv env
source env/bin/activate  # Sur Linux/Mac
env\Scripts\activate  # Sur Windows

# 2. Installer les dépendances
pip install -r requirements.txt

# 3. Ajouter un fichier à la racine nommé "document.pdf" contenant le document à traiter.

# 4. Lancer le programme (assurez-vous que le serveur llama-server est en cours d'exécution)
python main.py
```
# IA-Orchestrator
