"""
Expériences Stratégie 3 — SetFit (few-shot learning contrastif).

Lance les runs pour les 3 backbones × datasets compatibles × 5 régimes × 5 seeds.
La compatibilité backbone/dataset reflète la spécialisation linguistique :
  - camembert       : français natif → compatible avec les deux datasets
  - modernbert_en   : anglais uniquement → seulement 20 Newsgroups
  - modernbert_ml   : multilingue → seulement les mails français

Cela donne l'observation centrale de la note :
"spécialisation linguistique (CamemBERT) vs modernité architecturale (ModernBERT)".

Usage :
    python scripts/run_setfit.py                                     # tout
    python scripts/run_setfit.py --dataset emails                    # un dataset
    python scripts/run_setfit.py --backbone camembert                # un backbone
    python scripts/run_setfit.py --dataset emails --backbone camembert
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.email_dataset import load_emails
from src.data.newsgroups import load_newsgroups
from src.data.sampling import get_train_splits
from src.evaluation.results import save_run
from src.models.setfit_model import (
    BACKBONE_CAMEMBERT,
    BACKBONE_MODERNBERT_EN,
    BACKBONE_MMBERT,
    train_and_evaluate,
)
from src.evaluation.results import RUNS_DIR
from src.utils.logging import get_logger

SEEDS = [42, 123, 456, 789, 1024]
REGIMES = [8, 16, 32, 64]

# Dictionnaire backbone_key -> (identifiant HuggingFace, datasets compatibles)
# La compatibilité est linguistique :
#   - mpnet-multilingual : multilingue → les deux datasets
#   - ModernBERT-base    : anglais → newsgroups uniquement
#   - mmBERT             : ModernBERT multilingue (2025) → les deux datasets
# Le rôle "ModernBERT multilingue" est tenu par mmBERT (vrai ModernBERT entraîné
# sur 1800+ langues), et non plus par e5-large-instruct (XLM-RoBERTa, mal nommé
# et instable : gradients NaN lors du fine-tuning contrastif sur RTX 5070).
BACKBONES = {
    "camembert":      (BACKBONE_CAMEMBERT,      ["newsgroups", "emails"]),
    "modernbert_en":  (BACKBONE_MODERNBERT_EN,  ["newsgroups"]),
    # mmBERT ~5× plus lent que mpnet sur RTX 5070. Évalué sur les emails uniquement
    # (le cas d'usage métier français), avec un plafond de steps réduit pour rester
    # dans des temps raisonnables. Newsgroups (anglais) est déjà couvert par ModernBERT-EN
    # pour l'angle "modernité architecturale".
    "mmbert":         (BACKBONE_MMBERT,         ["emails"]),
}

# Plafond de steps contrastifs par backbone (1500 par défaut, 600 pour mmBERT qui est lent)
MAX_STEPS_OVERRIDE = {
    "mmbert": 600,
}

logger = get_logger("run_setfit")


def run_dataset(dataset_name: str, backbone_key: str) -> None:
    """Lance les runs SetFit pour un backbone et un dataset donnés."""
    backbone_id, compatible = BACKBONES[backbone_key]

    # Vérification de la compatibilité linguistique avant de lancer
    if dataset_name not in compatible:
        logger.info(f"Skipping {backbone_key} on {dataset_name} (incompatible)")
        return

    # Nom unique du modèle dans les JSONs de résultats : setfit_camembert, setfit_modernbert_en...
    model_name = f"setfit_{backbone_key}"
    logger.info(f"Dataset={dataset_name} | Backbone={backbone_key}")

    train_df, test_df = load_newsgroups() if dataset_name == "newsgroups" else load_emails()
    test_texts = test_df["text"].tolist()
    test_labels = test_df["label"].tolist()

    splits = get_train_splits(train_df, REGIMES, SEEDS)
    total = (len(REGIMES) + 1) * len(SEEDS)
    done = 0

    for regime, seed_splits in splits.items():
        for seed, split_df in seed_splits.items():
            done += 1

            # Reprise automatique : on saute les runs déjà sauvegardés.
            # Permet de relancer le script en n'ajoutant que les backbones manquants
            # (ex: mmbert) sans recalculer les autres.
            out_path = RUNS_DIR / f"{model_name}__{dataset_name}__regime{regime}__seed{seed}.json"
            if out_path.exists():
                logger.info(f"[{done}/{total}] regime={regime} | seed={seed} déjà fait, on passe.")
                continue

            train_texts = split_df["text"].tolist()
            train_labels = split_df["label"].tolist()

            logger.info(f"regime={regime} | seed={seed} | n_train={len(train_texts)}")
            results, _ = train_and_evaluate(
                train_texts, train_labels, test_texts, test_labels,
                backbone=backbone_id,
                max_steps=MAX_STEPS_OVERRIDE.get(backbone_key, 1500),
            )
            # On sauvegarde backbone_key dans le JSON pour distinguer les variantes
            save_run(results, model_name, dataset_name, regime, seed,
                     extra={"backbone_key": backbone_key})

            logger.info(f"[{done}/{total}] F1={results['f1_macro']:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["newsgroups", "emails", "all"], default="all")
    parser.add_argument("--backbone", choices=list(BACKBONES.keys()) + ["all"], default="all")
    args = parser.parse_args()

    datasets = ["newsgroups", "emails"] if args.dataset == "all" else [args.dataset]
    backbones = list(BACKBONES.keys()) if args.backbone == "all" else [args.backbone]

    # Boucle backbone en externe : on charge le même backbone pour tous les datasets
    # (évite de recharger le modèle inutilement entre datasets)
    for bb in backbones:
        for ds in datasets:
            run_dataset(ds, bb)


if __name__ == "__main__":
    main()
