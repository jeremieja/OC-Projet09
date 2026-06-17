"""
Génère un CSV de prédictions pour l'analyse qualitative des erreurs (notebook 03).

Entraîne chaque modèle entraînable sur UNE configuration de référence
(full data, seed 42) puis sauvegarde, pour chaque mail du jeu de test,
la vraie classe et la prédiction de chaque modèle. Ajoute aussi Ministral
zero-shot (pas d'entraînement).

Sortie : results/aggregated/predictions_<dataset>.csv
  colonnes : text, true_label, pred_tfidf_lr, pred_camembert,
             pred_setfit_camembert, pred_ministral

Usage :
    python scripts/dump_predictions.py --dataset emails
    python scripts/dump_predictions.py --dataset newsgroups
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.email_dataset import CLASSES as EMAIL_CLASSES
from src.data.email_dataset import get_label_mapping as email_labels
from src.data.email_dataset import load_emails
from src.data.newsgroups import SELECTED_CLASSES as NEWS_CLASSES
from src.data.newsgroups import get_label_mapping as news_labels
from src.data.newsgroups import load_newsgroups
from src.evaluation.results import AGG_DIR
from src.utils.logging import get_logger

SEED = 42  # configuration de référence pour l'analyse d'erreurs
logger = get_logger("dump_predictions")


def run(dataset_name: str, with_ministral: bool) -> None:
    if dataset_name == "newsgroups":
        train_df, test_df = load_newsgroups()
        label2id, id2label = news_labels()
        categories = NEWS_CLASSES
        domain = "messages de forums de discussion thématiques (newsgroups)"
    else:
        train_df, test_df = load_emails()
        label2id, id2label = email_labels()
        categories = EMAIL_CLASSES
        domain = "mails de clubs sportifs français"

    train_texts = train_df["text"].tolist()
    train_labels = train_df["label"].tolist()
    test_texts = test_df["text"].tolist()
    test_labels = test_df["label"].tolist()

    out = pd.DataFrame({"text": test_texts, "true_label": test_labels})

    # --- 1. TF-IDF + LR ---
    logger.info("TF-IDF + LR...")
    from src.models.baseline import build_pipeline
    pipe = build_pipeline()
    pipe.fit(train_texts, train_labels)
    out["pred_tfidf_lr"] = list(pipe.predict(test_texts))

    # --- 2. CamemBERT ---
    logger.info("CamemBERT (fine-tuning, ~5 min)...")
    from src.models.camembert import train_and_evaluate as camembert_train
    import numpy as np
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    # On réutilise la fonction d'entraînement, mais elle ne renvoie que les métriques.
    # On ré-entraîne et prédit ici pour récupérer les labels prédits.
    _, cam_model = camembert_train(
        train_texts, train_labels, test_texts, test_labels,
        label2id=label2id, id2label=id2label,
        output_dir=f"models_saved/camembert_dump_{dataset_name}",
    )
    # Prédiction avec le modèle entraîné
    from transformers import pipeline as hf_pipeline
    tok = AutoTokenizer.from_pretrained("almanach/camembert-base")
    clf = hf_pipeline("text-classification", model=cam_model, tokenizer=tok,
                      truncation=True, max_length=256, device=0,
                      batch_size=32)
    cam_preds = [id2label[int(r["label"].split("_")[-1])] if r["label"].startswith("LABEL_")
                 else r["label"] for r in clf(test_texts)]
    out["pred_camembert"] = cam_preds

    # --- 3. SetFit (mpnet-multilingual) ---
    logger.info("SetFit (mpnet-multilingual)...")
    from src.models.setfit_model import train_and_evaluate as setfit_train, BACKBONE_CAMEMBERT
    from src.models.setfit_model import get_predictions_with_proba
    # Plafond réduit : les textes newsgroups sont longs → SetFit lent (~5s/it).
    # 600 steps suffisent pour des prédictions représentatives (cf. runs principaux).
    _, setfit_model = setfit_train(
        train_texts, train_labels, test_texts, test_labels,
        backbone=BACKBONE_CAMEMBERT,
        max_steps=600,
    )
    setfit_preds, _ = get_predictions_with_proba(setfit_model, test_texts)
    out["pred_setfit_camembert"] = setfit_preds

    # --- 4. Ministral zero-shot (optionnel : coûte des appels API) ---
    if with_ministral:
        logger.info("Ministral zero-shot (API)...")
        from src.models.ministral import classify_batch
        m_preds, _, _ = classify_batch(test_texts, categories, mode="zero_shot", domain=domain)
        out["pred_ministral"] = m_preds

    AGG_DIR.mkdir(parents=True, exist_ok=True)
    path = AGG_DIR / f"predictions_{dataset_name}.csv"
    out.to_csv(path, index=False, encoding="utf-8")
    logger.info(f"Prédictions sauvegardées : {path} ({len(out)} lignes)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["newsgroups", "emails"], default="emails")
    parser.add_argument("--no-ministral", action="store_true",
                        help="Ne pas appeler Ministral (évite le coût API)")
    args = parser.parse_args()
    run(args.dataset, with_ministral=not args.no_ministral)


if __name__ == "__main__":
    main()
