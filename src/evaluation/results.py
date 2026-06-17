"""
Sauvegarde et chargement des résultats d'expérience.

Chaque run (1 modèle × 1 dataset × 1 régime × 1 seed) est sauvegardé
dans un fichier JSON séparé dans results/runs/.
Cette approche granulaire permet de relancer des runs individuels
sans écraser les autres, et de charger facilement un sous-ensemble de résultats.

Structure du nom de fichier :
    <modele>__<dataset>__regime<n>__seed<s>.json
"""
import json
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Optional

# Chemins absolus calculés relativement à ce fichier
RUNS_DIR = Path(__file__).parents[2] / "results" / "runs"
AGG_DIR = Path(__file__).parents[2] / "results" / "aggregated"


def save_run(
    results: dict,
    model: str,
    dataset: str,
    regime,
    seed: int,
    extra: Optional[dict] = None,
    tag: Optional[str] = None,
) -> Path:
    """
    Sauvegarde les résultats d'un run dans un fichier JSON.

    Le timestamp permet de détecter si un run a été rejoué.
    Le paramètre extra permet d'ajouter des métadonnées spécifiques
    à certains modèles (ex: backbone pour SetFit, threshold pour le hybride).
    Le paramètre tag rend le nom de fichier unique quand plusieurs résultats
    partagent le même (modèle, dataset, régime, seed) — indispensable pour le
    hybride qui sauvegarde un fichier par seuil τ (sinon ils s'écrasent).
    """
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    run_data = {
        "model": model,
        "dataset": dataset,
        "regime": str(regime),  # str pour uniformiser (int vs "full")
        "seed": seed,
        "timestamp": datetime.now().isoformat(),
        **results,  # on déplie les métriques (f1_macro, accuracy, etc.)
    }
    if extra:
        run_data.update(extra)

    suffix = f"__{tag}" if tag else ""
    filename = f"{model}__{dataset}__regime{regime}__seed{seed}{suffix}.json"
    path = RUNS_DIR / filename

    with open(path, "w", encoding="utf-8") as f:
        json.dump(run_data, f, ensure_ascii=False, indent=2)

    return path


def load_all_runs(model: Optional[str] = None, dataset: Optional[str] = None) -> pd.DataFrame:
    """
    Charge tous les fichiers JSON de runs dans un DataFrame.

    Les filtres optionnels model et dataset permettent de ne charger
    qu'un sous-ensemble des résultats (utile dans les notebooks d'analyse).
    Retourne un DataFrame vide si aucun résultat n'existe encore.
    """
    records = []
    for p in RUNS_DIR.glob("*.json"):
        with open(p, encoding="utf-8") as f:
            records.append(json.load(f))

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    if model:
        df = df[df["model"] == model]
    if dataset:
        df = df[df["dataset"] == dataset]
    return df


def aggregate_results(df: pd.DataFrame) -> pd.DataFrame:
    """
    Agrège les résultats sur les 5 seeds : calcule moyenne et écart-type.

    L'écart-type reflète la variance liée au choix des exemples d'entraînement,
    particulièrement importante en régime few-shot (ex: 8-shot peut varier
    fortement selon quels 8 exemples sont tirés).
    """
    group_cols = ["model", "dataset", "regime"]
    # On n'agrège que les colonnes métriques présentes dans le DataFrame
    metric_cols = [c for c in ["f1_macro", "accuracy"] if c in df.columns]

    agg = df.groupby(group_cols)[metric_cols].agg(["mean", "std"]).round(4)
    # Renomme les colonnes multi-niveaux : ("f1_macro", "mean") -> "f1_macro_mean"
    agg.columns = ["_".join(c) for c in agg.columns]
    return agg.reset_index()


def save_aggregated(df: pd.DataFrame, filename: str = "results_aggregated.csv") -> Path:
    """Sauvegarde le tableau agrégé en CSV pour référence et partage."""
    AGG_DIR.mkdir(parents=True, exist_ok=True)
    path = AGG_DIR / filename
    df.to_csv(path, index=False)
    return path
