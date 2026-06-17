"""
Expériences Stratégie 2 — Fine-tuning CamemBERT.

Lance 25 runs par dataset (5 régimes × 5 seeds). Chaque run fine-tune
CamemBERT depuis zéro sur le sous-ensemble d'entraînement correspondant.

Note sur la durée : CamemBERT est un modèle ~110M paramètres.
Avec la RTX 5070, compter environ :
  - ~30s par run en few-shot (8-64 exemples, peu de gradient steps)
  - ~5-10min pour le régime full data (4000+ exemples × 3 epochs)
  Soit ~30-45min par dataset.

CamemBERT a besoin des mappings label<->entier car Hugging Face Trainer
travaille avec des entiers, pas des chaînes de caractères.

Usage :
    python scripts/run_camembert.py
    python scripts/run_camembert.py --dataset newsgroups
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.email_dataset import get_label_mapping as email_labels
from src.data.email_dataset import load_emails
from src.data.newsgroups import get_label_mapping as news_labels
from src.data.newsgroups import load_newsgroups
from src.data.sampling import get_train_splits
from src.evaluation.results import save_run
from src.models.camembert import train_and_evaluate
from src.utils.logging import get_logger

SEEDS = [42, 123, 456, 789, 1024]
REGIMES = [8, 16, 32, 64]
MODEL_NAME = "camembert"

logger = get_logger("run_camembert")


def run_dataset(dataset_name: str) -> None:
    """Lance tous les runs CamemBERT pour un dataset donné."""
    logger.info(f"Dataset: {dataset_name}")

    # Chargement du dataset ET des mappings label<->entier spécifiques à chaque dataset
    if dataset_name == "newsgroups":
        train_df, test_df = load_newsgroups()
        label2id, id2label = news_labels()
    else:
        train_df, test_df = load_emails()
        label2id, id2label = email_labels()

    test_texts = test_df["text"].tolist()
    test_labels = test_df["label"].tolist()

    splits = get_train_splits(train_df, REGIMES, SEEDS)
    total = (len(REGIMES) + 1) * len(SEEDS)
    done = 0

    for regime, seed_splits in splits.items():
        for seed, split_df in seed_splits.items():
            train_texts = split_df["text"].tolist()
            train_labels = split_df["label"].tolist()

            logger.info(f"regime={regime} | seed={seed} | n_train={len(train_texts)}")
            results, _ = train_and_evaluate(
                train_texts, train_labels, test_texts, test_labels,
                label2id=label2id,
                id2label=id2label,
                # Répertoire temporaire pour les checkpoints Hugging Face Trainer
                # (save_strategy="no" donc rien n'est vraiment écrit, mais le param est requis)
                output_dir=f"models_saved/camembert_{dataset_name}_{regime}_{seed}",
            )
            save_run(results, MODEL_NAME, dataset_name, regime, seed)

            done += 1
            logger.info(f"[{done}/{total}] F1={results['f1_macro']:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["newsgroups", "emails", "all"], default="all")
    args = parser.parse_args()

    for ds in (["newsgroups", "emails"] if args.dataset == "all" else [args.dataset]):
        run_dataset(ds)


if __name__ == "__main__":
    main()
