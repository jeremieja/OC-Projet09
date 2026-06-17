"""
Expériences Stratégie 5 — Routage hybride SetFit + fallback Ministral.

Pour chaque combinaison (régime, seed), ce script :
  1. Entraîne SetFit (backbone mpnet-multilingual, identique à la stratégie 3)
     sur le sous-ensemble d'entraînement
  2. Récupère les prédictions + probabilités SetFit sur le test
  3. Récupère les prédictions Ministral zero-shot sur le même test
  4. Balaye 7 seuils de confiance τ et calcule pour chacun :
       - F1 macro du système hybride
       - Taux d'escalade (% de mails routés vers Ministral)
       - Coût API moyen par mail

Note : SetFit est ré-entraîné ici (et non rechargé) car run_setfit.py ne persiste
que les métriques, pas les modèles. L'entraînement étant déterministe à seed fixée
et utilisant la même recette (max_steps=1500), les prédictions sont équivalentes à
celles de la stratégie 3 isolée — au seuil τ=0 (aucune escalade), le F1 du système
hybride coïncide donc avec le F1 de SetFit seul.

Le mode zero-shot est utilisé pour Ministral (pas few-shot) pour limiter le coût
et rendre l'architecture réaliste : un nouveau club n'a pas forcément d'exemples
étiquetés disponibles pour le few-shot.

Usage :
    python scripts/run_hybrid.py --dataset emails             # cas d'usage métier
    python scripts/run_hybrid.py                              # les deux datasets
    python scripts/run_hybrid.py --dataset emails --regime 64 --seed 42  # un run (debug)
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.email_dataset import CLASSES as EMAIL_CLASSES
from src.data.email_dataset import load_emails
from src.data.newsgroups import SELECTED_CLASSES as NEWS_CLASSES
from src.data.newsgroups import load_newsgroups
from src.data.sampling import get_train_splits
from src.evaluation.results import RUNS_DIR, save_run
from src.models.hybrid import sweep_thresholds
from src.models.ministral import classify_batch
from src.models.setfit_model import (
    BACKBONE_CAMEMBERT,
    get_predictions_with_proba,
    train_and_evaluate,
)
from src.utils.logging import get_logger

SEEDS = [42, 123, 456, 789, 1024]
REGIMES = [8, 16, 32, 64]
# Seuils τ à balayer : de très permissif (0.3 = peu d'escalades) à très strict (0.9 = beaucoup)
THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
MODEL_NAME = "hybrid_setfit_ministral"

logger = get_logger("run_hybrid")


def _load_or_compute_ministral(dataset_name: str, test_texts: list, categories: list) -> list:
    """
    Recharge les prédictions Ministral zero-shot depuis le JSON de run_ministral.py
    si disponibles (et de la bonne taille), sinon les calcule via l'API.
    Évite de repayer les appels API si la stratégie 4 a déjà tourné.
    """
    import json
    path = RUNS_DIR / f"ministral__{dataset_name}__regimezero_shot__seed0.json"
    if path.exists():
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        preds = data.get("preds")
        if preds and len(preds) == len(test_texts):
            logger.info("Prédictions Ministral rechargées depuis le JSON existant (0 appel API).")
            return preds
        logger.info("JSON Ministral présent mais incompatible — recalcul via API.")

    logger.info("Appel API Ministral zero-shot (une seule fois)...")
    preds, _, _ = classify_batch(test_texts, categories, mode="zero_shot")
    return preds


def run_dataset(dataset_name: str, regimes: list, seeds: list) -> None:
    """Lance les expériences hybrides pour un dataset donné."""
    logger.info(f"Dataset={dataset_name}")

    if dataset_name == "newsgroups":
        train_df, test_df = load_newsgroups()
        categories = NEWS_CLASSES
    else:
        train_df, test_df = load_emails()
        categories = EMAIL_CLASSES

    test_texts = test_df["text"].tolist()
    test_labels = test_df["label"].tolist()

    splits = get_train_splits(train_df, regimes, seeds)

    # Prédictions Ministral zero-shot : identiques pour tous les runs (même test set,
    # temperature=0, prompt identique). On tente d'abord de les recharger depuis le JSON
    # produit par run_ministral.py (gratuit), sinon on appelle l'API une seule fois.
    ministral_preds = _load_or_compute_ministral(dataset_name, test_texts, categories)
    logger.info(f"Ministral zero-shot : {len(ministral_preds)} prédictions prêtes.")

    for regime, seed_splits in splits.items():
        for seed, split_df in seed_splits.items():
            logger.info(f"regime={regime} | seed={seed}")

            # Étape 1 : entraînement SetFit sur ce sous-ensemble
            train_texts = split_df["text"].tolist()
            train_labels = split_df["label"].tolist()
            _, setfit_model = train_and_evaluate(
                train_texts, train_labels, test_texts, test_labels,
                backbone=BACKBONE_CAMEMBERT,
            )

            # Étape 2 : prédictions SetFit + matrice de probabilités sur le test
            # setfit_proba shape : (n_test, n_classes) — la confiance = max de chaque ligne
            setfit_preds, setfit_proba = get_predictions_with_proba(setfit_model, test_texts)

            # Étape 3 : prédictions Ministral déjà calculées (réutilisation)

            # Étape 4 : balayage des seuils τ — aucun entraînement supplémentaire,
            # on combine simplement les prédictions déjà calculées
            threshold_results = sweep_thresholds(
                test_labels, setfit_preds, setfit_proba, ministral_preds, THRESHOLDS
            )

            # Sauvegarde un JSON par seuil τ (7 fichiers par run).
            # tag=tau<valeur> rend chaque nom de fichier unique, sinon les 7 seuils
            # s'écraseraient (le nom de base ne contient pas le seuil).
            for res in threshold_results:
                tau = res["threshold"]
                save_run(res, MODEL_NAME, dataset_name, regime, seed,
                         extra={"threshold": tau},
                         tag=f"tau{tau:.1f}")
                logger.info(
                    f"  tau={tau} | F1={res['f1_macro']:.4f} | "
                    f"escalation={res['escalation_rate']:.2%} | "
                    f"cost/mail=${res['avg_cost_per_mail_usd']:.6f}"
                )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["newsgroups", "emails", "all"], default="all")
    parser.add_argument("--regime", type=int, default=None,
                        help="Limite à un seul régime (utile pour tester rapidement)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Limite à une seule seed (utile pour déboguer)")
    args = parser.parse_args()

    datasets = ["newsgroups", "emails"] if args.dataset == "all" else [args.dataset]
    regimes = [args.regime] if args.regime else REGIMES
    seeds = [args.seed] if args.seed else SEEDS

    for ds in datasets:
        run_dataset(ds, regimes, seeds)


if __name__ == "__main__":
    main()
