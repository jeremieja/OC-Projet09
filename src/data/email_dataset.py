"""
Chargement du dataset de mails synthétiques de clubs sportifs (cas d'usage métier).

Le fichier CSV est généré par scripts/generate_dataset.py avant d'utiliser ce module.
Il contient ~1 800 mails en français répartis sur 8 catégories métier.
"""
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from typing import Tuple

# Chemin absolu vers le CSV généré, calculé relativement à ce fichier
DATA_PATH = Path(__file__).parents[2] / "data" / "processed" / "emails.csv"

# Les 8 catégories métier reflètent les types de mails réels
# que reçoit un club sportif structuré
CLASSES = [
    "inscription",
    "sponsor",
    "arbitrage-officiels",
    "parent",
    "federation",
    "logistique-matchday",
    "indemnites-coachs",
    "divers-administratif",
]


def load_emails(test_size: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Charge le dataset de mails synthétiques depuis le CSV généré.

    Le split train/test est stratifié : chaque classe est représentée
    proportionnellement dans les deux parties, ce qui est crucial avec
    seulement ~225 exemples par classe.

    Retourne deux DataFrames avec colonnes 'text' et 'label'.
    """
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset introuvable : {DATA_PATH}. "
            "Lance d'abord : python scripts/generate_dataset.py"
        )

    # On ne garde que les colonnes utiles pour l'entraînement
    df = pd.read_csv(DATA_PATH)[["text", "label"]].dropna().reset_index(drop=True)

    # stratify=df["label"] garantit que la distribution des classes
    # est identique dans train et test (important pour le few-shot)
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df["label"],
        random_state=random_state,
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def get_label_mapping() -> Tuple[dict, dict]:
    """
    Retourne deux dictionnaires de correspondance label <-> entier.
    Nécessaire pour CamemBERT qui attend des entiers en entrée.
    """
    label2id = {c: i for i, c in enumerate(CLASSES)}
    id2label = {i: c for i, c in enumerate(CLASSES)}
    return label2id, id2label
