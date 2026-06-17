# POC — Classification automatique de mails de clubs sportifs

Preuve de concept (DataSpace, Projet 9) comparant **5 stratégies** de classification
de textes, sur deux jeux de données, dans une logique produit : *quel modèle déployer
pour quel profil de club ?*

## Le problème

Un SaaS pour clubs sportifs doit classer et prioriser automatiquement les mails entrants
(inscription, sponsor, arbitrage, parent, fédération, logistique, indemnités, divers).
L'enjeu : un **nouveau club arrive avec zéro mail étiqueté**. On mesure donc comment
chaque stratégie se comporte de 8 exemples/classe jusqu'au régime full data.

## Les 5 stratégies comparées

| # | Stratégie | Idée |
|---|-----------|------|
| 1 | TF-IDF + Régression logistique | Baseline classique, rapide, sans GPU |
| 2 | CamemBERT fine-tuné | Modèle de langue français, approche data-driven |
| 3 | SetFit (mpnet / ModernBERT / mmBERT) | Few-shot contrastif, champion à faibles données |
| 4 | Ministral 8B (API) | LLM-as-classifier, zéro entraînement |
| 5 | Hybride SetFit + Ministral | Routage par confiance (seuil τ) |

## Résultats clés

- **SetFit domine le few-shot** : F1 0.95 (emails) dès 8 exemples/classe.
- **CamemBERT est data-hungry** : F1 0.10 à 8-shot → 0.99 en full data.
- **TF-IDF égale les modèles lourds en full data** (0.99) — d'où son choix pour le déploiement.
- **Système hybride** : F1 0.997 en n'escaladant que ~1 % des mails vers le LLM.

## Structure du projet

```
src/            modules data, modèles, évaluation
scripts/        génération dataset + lancement des expériences
notebooks/      4 notebooks d'analyse (EDA, résultats, erreurs, matrice de décision)
dashboard/      application Streamlit (déployée sur le cloud)
results/        résultats agrégés des expériences
models_saved/   modèles légers déployables (TF-IDF)
```

## Dashboard

Application Streamlit avec exploration des données (EDA + WordCloud), prédiction en
direct et synthèse des performances. Voir [dashboard/README.md](dashboard/README.md)
pour le détail et le déploiement.

```bash
streamlit run dashboard/app.py
```

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

## Reproduire les expériences

```bash
python scripts/generate_dataset.py     # génère les mails synthétiques (API Anthropic)
python scripts/run_all.py              # lance toutes les stratégies
```
