# RAG Project - Assistant Financier BNP Paribas

Assistant conversationnel basé sur **Retrieval-Augmented Generation (RAG)** avec Mistral, LangChain et Chroma DB pour répondre aux questions sur les résultats financiers de **BNP Paribas**.

🚀 **Lien déployé** : https://rag-project-noegruoc.streamlit.app/

## 🛠️ Stack Technologique

- **LLM** : [Mistral AI](https://mistral.ai/) (7B, température 0)
- **Embeddings** : Mistral-Embed
- **Vector DB** : [Chroma](https://www.trychroma.com/)
- **Framework** : [LangChain](https://www.langchain.com/)
- **UI** : [Streamlit](https://streamlit.io/)

## 🚀 Installation Rapide

```bash
# Cloner et installer
git clone <repo>
cd RAG_project
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Configurer la clé API
# Créer .streamlit/secrets.toml avec:
# MISTRAL_API_KEY = "votre-clé"

# Lancer
streamlit run app.py
```

## 📁 Structure

```
RAG_project/
├── app.py              # Application principale
├── requirements.txt    # Dépendances
├── chroma_db/         # Base de données vectorielle
└── README.md
```

## 🔧 Fonctionnalités

✅ **Système RAG** : Recherche vectorielle + génération intelligente  
✅ **Sécurité** : Filtrage des mots-clés sensibles + protection injection  
✅ **Français** : Réponses en français uniquement sur domaine BNP Paribas  
✅ **Persistance** : Base vectorielle sauvegardée  
✅ **Chat** : Historique de conversation

## 🌐 Déploiement

🚀 **Lien déployé** : https://rag-project-noegruoc.streamlit.app/

**Streamlit Cloud** (recommandé)
1. Push sur GitHub
2. [Streamlit Cloud](https://share.streamlit.io) → New app
3. Ajouter secret : `MISTRAL_API_KEY`

## 📚 Ressources

- [LangChain](https://python.langchain.com/)
- [Mistral Docs](https://docs.mistral.ai/)
- [Chroma Docs](https://docs.trychroma.com/)

---

**Statut** : ✅ Production Ready | **Mis à jour** : Janvier 2026
