"""
Expériences Stratégie 4 — Ministral 8B via API (zero-shot et few-shot).

Contrairement aux stratégies 1-3, il n'y a PAS d'entraînement.
On évalue directement le modèle via API Mistral La Plateforme.

Deux modes :
  - zero-shot  : 1 seul run déterministe (temperature=0, résultat reproductible)
  - few-shot 8 : 3 seeds pour mesurer la variance liée au choix des exemples in-context

Coût estimé : ~$0.07 total pour les ~4000 appels (2 datasets × 2 modes × ~1000 tests)

Usage :
    python scripts/run_ministral.py
    python scripts/run_ministral.py --dataset emails --mode zero_shot
    python scripts/run_ministral.py --mode few_shot
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.email_dataset import CLASSES as EMAIL_CLASSES
from src.data.email_dataset import load_emails
from src.data.newsgroups import SELECTED_CLASSES as NEWS_CLASSES
from src.data.newsgroups import load_newsgroups
from src.data.sampling import sample_few_shot
from src.evaluation.results import save_run
from src.models.ministral import evaluate
from src.utils.logging import get_logger

FEW_SHOT_K = 8
FEW_SHOT_SEEDS = [42, 123, 456]
MODEL_NAME = "ministral"

logger = get_logger("run_ministral")


def _already_done(dataset_name: str, regime: str, seed: int) -> bool:
    """Vérifie si ce run a déjà été sauvegardé (reprise en cas d'interruption)."""
    from src.evaluation.results import RUNS_DIR
    path = RUNS_DIR / f"{MODEL_NAME}__{dataset_name}__regime{regime}__seed{seed}.json"
    return path.exists()


def run_dataset(dataset_name: str, mode: str) -> None:
    logger.info(f"Dataset={dataset_name} | mode={mode}")

    if dataset_name == "newsgroups":
        train_df, test_df = load_newsgroups()
        categories = NEWS_CLASSES
        # Domaine neutre adapté aux posts de forums tech anglophones, pour ne pas
        # biaiser le modèle avec un cadrage "clubs sportifs" hors-sujet.
        domain = "messages de forums de discussion thématiques (newsgroups)"
    else:
        train_df, test_df = load_emails()
        categories = EMAIL_CLASSES
        domain = "mails de clubs sportifs français"

    test_texts = test_df["text"].tolist()
    test_labels = test_df["label"].tolist()
    logger.info(f"  {len(test_texts)} exemples de test | {len(categories)} classes | domaine='{domain}'")

    if mode == "zero_shot":
        regime = "zero_shot"
        if _already_done(dataset_name, regime, 0):
            logger.info("  Déjà fait, on passe.")
            return

        logger.info("Lancement zero-shot...")
        results, preds, confs = evaluate(
            test_texts, test_labels, categories, mode="zero_shot", domain=domain
        )
        save_run(results, MODEL_NAME, dataset_name, regime=regime, seed=0,
                 extra={"mode": "zero_shot", "preds": preds, "confidences": confs})
        logger.info(
            f"  F1={results['f1_macro']:.4f} | Acc={results['accuracy']:.4f} | "
            f"Latence moy={results['inference_time_ms']:.0f}ms | "
            f"Coût=${results['total_cost_usd']:.4f}"
        )

    else:  # few_shot
        for seed in FEW_SHOT_SEEDS:
            regime = f"few_shot_{FEW_SHOT_K}"
            if _already_done(dataset_name, regime, seed):
                logger.info(f"  seed={seed} déjà fait, on passe.")
                continue

            examples_df = sample_few_shot(train_df, FEW_SHOT_K, seed)
            examples = examples_df[["text", "label"]].to_dict("records")

            logger.info(f"Few-shot k={FEW_SHOT_K} | seed={seed}")
            results, preds, confs = evaluate(
                test_texts, test_labels, categories, mode="few_shot",
                examples=examples, domain=domain
            )
            save_run(results, MODEL_NAME, dataset_name, regime=regime, seed=seed,
                     extra={"mode": "few_shot", "k": FEW_SHOT_K,
                            "preds": preds, "confidences": confs})
            logger.info(
                f"  F1={results['f1_macro']:.4f} | Acc={results['accuracy']:.4f} | "
                f"Latence moy={results['inference_time_ms']:.0f}ms"
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["newsgroups", "emails", "all"], default="all")
    parser.add_argument("--mode", choices=["zero_shot", "few_shot", "all"], default="all")
    args = parser.parse_args()

    datasets = ["newsgroups", "emails"] if args.dataset == "all" else [args.dataset]
    modes = ["zero_shot", "few_shot"] if args.mode == "all" else [args.mode]

    for ds in datasets:
        for m in modes:
            run_dataset(ds, m)


if __name__ == "__main__":
    main()
