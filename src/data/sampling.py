"""
Échantillonnage stratifié reproductible pour les expériences few-shot.

Le protocole expérimental impose 5 régimes (8, 16, 32, 64 exemples/classe + full data)
et 5 seeds différentes par régime, soit 25 combinaisons par modèle par dataset.
Les seeds sont fixées pour garantir la reproductibilité et permettre des comparaisons
équitables entre modèles (chaque modèle voit exactement les mêmes exemples).
"""
import numpy as np
import pandas as pd
from typing import List


def sample_few_shot(
    df: pd.DataFrame,
    n_per_class: int,
    seed: int,
    label_col: str = "label",
) -> pd.DataFrame:
    """
    Tire aléatoirement n_per_class exemples par classe de manière reproductible.

    L'échantillonnage est stratifié : on tire dans chaque classe séparément,
    ce qui garantit l'équilibre même quand les classes sont déséquilibrées.
    numpy.random.default_rng (générateur moderne) est utilisé plutôt que
    np.random.seed (global) pour éviter les effets de bord.
    """
    # Générateur isolé par seed : n'affecte pas les autres appels random du programme
    rng = np.random.default_rng(seed)
    sampled_indices = []

    # sorted() assure un ordre déterministe entre les classes
    for label in sorted(df[label_col].unique()):
        class_indices = df.index[df[label_col] == label].tolist()
        # min() protège contre le cas où une classe a moins de n exemples
        n = min(n_per_class, len(class_indices))
        chosen = rng.choice(class_indices, size=n, replace=False)
        sampled_indices.extend(chosen.tolist())

    return df.loc[sampled_indices].reset_index(drop=True)


def get_train_splits(
    train_df: pd.DataFrame,
    regimes: List[int],
    seeds: List[int],
    label_col: str = "label",
) -> dict:
    """
    Génère tous les sous-ensembles d'entraînement nécessaires aux expériences.

    Retourne un dictionnaire imbriqué : splits[regime][seed] = DataFrame.
    Le régime 'full' retourne l'intégralité du train pour chaque seed
    (les résultats sont identiques entre seeds en full data, mais on garde
    la structure uniforme pour simplifier les boucles dans les scripts).
    """
    splits = {}

    # Régimes few-shot : on sous-échantillonne le train
    for regime in regimes:
        splits[regime] = {
            seed: sample_few_shot(train_df, regime, seed, label_col)
            for seed in seeds
        }

    # Régime full data : on utilise tout le train (pas de sous-échantillonnage)
    splits["full"] = {seed: train_df.copy() for seed in seeds}

    return splits
