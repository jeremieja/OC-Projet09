"""
Entraîne et sauvegarde un modèle TF-IDF + Régression logistique léger et déployable,
pour la prédiction en direct dans le dashboard Streamlit (cloud).

Pourquoi TF-IDF ? CamemBERT et SetFit pèsent plusieurs Go et nécessitent un GPU :
impossibles à déployer sur Streamlit Community Cloud (CPU, RAM limitée). TF-IDF+LR
pèse quelques Mo, tourne sur CPU en millisecondes, et atteint ~0.99 de F1 en full data
sur les emails — parfaitement adapté à une démo produit.

Le modèle est entraîné sur TOUT le dataset (train + test) pour maximiser sa qualité
en production (on n'évalue plus ici, on déploie).

Sortie : models_saved/deploy_<dataset>.joblib

Usage :
    python scripts/save_deploy_model.py
"""
import sys
from pathlib import Path

import joblib

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.email_dataset import load_emails
from src.data.newsgroups import load_newsgroups
from src.models.baseline import build_pipeline
from src.utils.logging import get_logger

logger = get_logger("save_deploy_model")
OUT_DIR = Path(__file__).parents[1] / "models_saved"


def save_one(dataset_name: str) -> None:
    if dataset_name == "newsgroups":
        train_df, test_df = load_newsgroups()
    else:
        train_df, test_df = load_emails()

    # On concatène train + test : en production, plus de données = meilleur modèle
    import pandas as pd
    full = pd.concat([train_df, test_df], ignore_index=True)

    logger.info(f"[{dataset_name}] entraînement sur {len(full)} exemples...")
    pipe = build_pipeline()
    pipe.fit(full["text"].tolist(), full["label"].tolist())

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"deploy_{dataset_name}.joblib"
    joblib.dump(pipe, path)
    size_mb = path.stat().st_size / 1e6
    logger.info(f"[{dataset_name}] sauvegardé : {path} ({size_mb:.1f} Mo)")


def main():
    for ds in ["emails", "newsgroups"]:
        save_one(ds)
    logger.info("Modèles déployables prêts.")


if __name__ == "__main__":
    main()
