# Dashboard — Classificateur de mails (POC DataSpace)

Application Streamlit de démonstration du POC : exploration des données,
prédiction en direct et synthèse des performances des modèles.

## Lancer en local

```bash
# Depuis la racine du projet, avec le venv activé
streamlit run dashboard/app.py
```

Prérequis (déjà installés si vous avez suivi le projet) :
- le dataset emails généré : `python scripts/generate_dataset.py`
- les modèles déployables : `python scripts/save_deploy_model.py`

Pour activer la comparaison Ministral, créez `.streamlit/secrets.toml`
(copie de `.streamlit/secrets.toml.example`) avec votre clé `MISTRAL_API_KEY`.

## Contenu

| Onglet | Description |
|--------|-------------|
| 🔎 Exploration | EDA des textes : distribution des catégories, longueur des messages, fréquence des mots, **WordCloud**. Graphiques interactifs (Plotly). |
| 🤖 Prédiction | Saisie d'un message + choix du contexte → catégorie prédite par TF-IDF (local) et, en option, Ministral 8B (API). |
| 📊 Performances | Courbes d'apprentissage et tableau récapitulatif des modèles. |

## Déploiement sur Streamlit Community Cloud

1. Poussez le projet sur un dépôt GitHub.
2. Sur [share.streamlit.io](https://share.streamlit.io) : *New app* → sélectionnez le dépôt.
3. **Main file path** : `dashboard/app.py`
4. **Python dependencies** : indiquez `dashboard/requirements.txt`
   (volontairement léger : pas de torch/transformers, le dashboard n'en a pas besoin).
5. Dans *Advanced settings → Secrets*, collez :
   ```toml
   MISTRAL_API_KEY = "votre_cle"
   ```
6. Déployez. L'app charge les modèles `models_saved/deploy_*.joblib` (légers, versionnés).

### Fichiers nécessaires au cloud (à committer)
- `dashboard/` (code de l'app)
- `src/` (modules data/models)
- `models_saved/deploy_emails.joblib` et `deploy_newsgroups.joblib`
- `data/processed/emails.csv` (pour l'EDA des mails)
- `.streamlit/config.toml` (thème accessible)

## Accessibilité (WCAG 2.1 AA)

- Palette **Okabe-Ito** discriminable par les daltoniens ; aucune information
  portée par la seule couleur (toujours doublée d'un libellé, d'une valeur ou d'une forme).
- Contrastes texte/fond ≥ 4.5:1 (thème clair, texte quasi-noir).
- Contour de focus visible pour la navigation au clavier.
- Textes alternatifs et descriptions textuelles sous chaque visuel (WordCloud inclus).
- Titres hiérarchisés et libellés explicites sur tous les champs de saisie.
