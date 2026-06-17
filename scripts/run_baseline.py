"""
Expériences Stratégie 1 — TF-IDF + Régression logistique.

Lance 25 runs par dataset (5 régimes × 5 seeds) et sauvegarde chaque résultat
dans un fichier JSON séparé dans results/runs/.

Protocole :
  - Régimes few-shot : 8, 16, 32, 64 exemples par classe
  - Régime full data : tout le train
  - 5 seeds par régime pour mesurer la variance liée à l'échantillonnage

Durée : ~2 minutes pour les 25 runs (pas de GPU nécessaire).

Usage :
    python scripts/run_baseline.py                  # les deux datasets
    python scripts/run_baseline.py --dataset newsgroups
    python scripts/run_baseline.py --dataset emails
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.email_dataset import load_emails
from src.data.newsgroups import load_newsgroups
from src.data.sampling import get_train_splits
from src.evaluation.results import save_run
from src.models.baseline import train_and_evaluate
from src.utils.logging import get_logger

# Protocole expérimental défini dans la note prévisionnelle
SEEDS = [42, 123, 456, 789, 1024]
REGIMES = [8, 16, 32, 64]
MODEL_NAME = "tfidf_lr"  # identifiant utilisé dans les noms de fichiers JSON

logger = get_logger("run_baseline")


def run_dataset(dataset_name: str) -> None:
    """Lance tous les runs pour un dataset donné."""
    logger.info(f"Dataset: {dataset_name}")

    # Chargement du dataset selon le nom passé en argument
    train_df, test_df = load_newsgroups() if dataset_name == "newsgroups" else load_emails()

    # Le jeu de test ne change pas entre les runs : on l'extrait une seule fois
    test_texts = test_df["text"].tolist()
    test_labels = test_df["label"].tolist()

    # Génère tous les sous-ensembles d'entraînement (régimes × seeds)
    splits = get_train_splits(train_df, REGIMES, SEEDS)
    total = (len(REGIMES) + 1) * len(SEEDS)  # +1 pour le régime "full"
    done = 0

    # Double boucle : pour chaque régime, pour chaque seed
    for regime, seed_splits in splits.items():
        for seed, split_df in seed_splits.items():
            train_texts = split_df["text"].tolist()
            train_labels = split_df["label"].tolist()

            logger.info(f"regime={regime} | seed={seed} | n_train={len(train_texts)}")
            results, _ = train_and_evaluate(train_texts, train_labels, test_texts, test_labels)

            # Sauvegarde immédiate après chaque run (pas de perte si interruption)
            save_run(results, MODEL_NAME, dataset_name, regime, seed)

            done += 1
            logger.info(f"[{done}/{total}] F1={results['f1_macro']:.4f} | Acc={results['accuracy']:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["newsgroups", "emails", "all"], default="all")
    args = parser.parse_args()

    for ds in (["newsgroups", "emails"] if args.dataset == "all" else [args.dataset]):
        run_dataset(ds)


if __name__ == "__main__":
    main()
