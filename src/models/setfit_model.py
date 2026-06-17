"""
Stratégie 3 — Méthode few-shot : SetFit (Tunstall et al., 2022).

SetFit (Sentence Transformer Fine-tuning) fonctionne en deux étapes :
  1. Fine-tuning contrastif du Sentence Transformer sur les rares exemples
     étiquetés : on génère des paires (même classe = positif, classes différentes
     = négatif) pour rapprocher/éloigner les embeddings dans l'espace vectoriel.
     Avec 16 exemples par classe sur 8 classes, on obtient des milliers de paires.
  2. Entraînement d'une tête de classification légère (régression logistique)
     sur les embeddings produits par le transformer fine-tuné.

Trois backbones testés :
  - paraphrase-multilingual-mpnet-base-v2 : multilingue de référence (MTEB-fr),
                             testé sur les deux datasets
  - ModernBERT-base (EN)  : architecture 2024, anglais → 20 Newsgroups uniquement
  - mmBERT-base            : ModernBERT entraîné sur 3T tokens / 1800+ langues (2025),
                             combine modernité architecturale ET couverture multilingue,
                             testé sur les deux datasets
"""
import time
from typing import List, Tuple

import numpy as np
from datasets import Dataset
from setfit import SetFitModel
from setfit import Trainer as SetFitTrainer
from setfit import TrainingArguments

# Backbone multilingue de référence — remplace sentence-camembert-large qui crashe
# le GPU RTX 5070 (sm_120) lors du backward pass contrastif (incompatibilité transformers 5.9.x)
BACKBONE_CAMEMBERT = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
# ModernBERT anglais — utilisé sur 20 Newsgroups pour tester l'apport architectural
BACKBONE_MODERNBERT_EN = "answerdotai/ModernBERT-base"
# mmBERT-base (2025) — ModernBERT entraîné sur 3T tokens en 1800+ langues (jhu-clsp)
# Combine modernité architecturale (Flash Attention 2, 8192 tokens) et couverture
# multilingue complète : testé sur les DEUX datasets pour observer le gain vs variantes
BACKBONE_MMBERT = "jhu-clsp/mmBERT-base"


def train_and_evaluate(
    train_texts: List[str],
    train_labels: List[str],
    test_texts: List[str],
    test_labels: List[str],
    backbone: str = BACKBONE_CAMEMBERT,
    num_epochs: int = 1,
    batch_size: int = 16,
    max_steps: int = 1500,
) -> Tuple[dict, SetFitModel]:
    """
    Entraîne SetFit et évalue sur le jeu de test.

    num_epochs=1 suffit généralement pour SetFit car la génération de paires
    contrastives multiplie déjà le signal d'apprentissage. Augmenter les epochs
    risque le surapprentissage avec peu de données.

    body_learning_rate : taux pour le Sentence Transformer (fin)
    head_learning_rate : taux pour la tête logistique (plus élevé, converge vite)
    end_to_end=False : on entraîne d'abord le body seul, puis la tête seule
                       (plus stable avec peu de données qu'un entraînement joint)
    """
    from src.evaluation.metrics import compute_metrics

    # SetFit attend des datasets Hugging Face avec colonnes 'text' et 'label'
    train_ds = Dataset.from_dict({"text": train_texts, "label": train_labels})
    test_ds = Dataset.from_dict({"text": test_texts, "label": test_labels})

    # labels= est requis dans SetFit >= 1.1 quand on part d'un Sentence Transformer brut
    # (pas d'un modèle SetFit pré-entraîné). Sans ce paramètre, SetFit cherche
    # config_setfit.json sur HuggingFace Hub et échoue avec une 404.
    #
    # labels= requis dans SetFit >= 1.1 quand on part d'un Sentence Transformer brut
    # (sinon SetFit cherche config_setfit.json sur le Hub et échoue avec une 404).
    labels = sorted(set(train_labels))
    model = SetFitModel.from_pretrained(backbone, labels=labels)

    # 128 tokens suffit pour les embeddings contrastifs de SetFit (les mots discriminants
    # apparaissent dans les premiers tokens). Réduit le temps d'entraînement ~4-8× vs 512.
    model.model_body.max_seq_length = 128

    args = TrainingArguments(
        num_epochs=num_epochs,
        batch_size=batch_size,
        body_learning_rate=1e-5,  # fine-tuning du transformer (petit pour ne pas tout oublier)
        head_learning_rate=1e-2,  # entraînement de la tête logistique (plus agressif)
        report_to="none",
        # max_steps plafonne le nb d'itérations contrastives. SetFit génère des paires de
        # façon quadratique (n²) : sans plafond, les régimes 64/full explosent (>30 min/run).
        # 1500 steps suffisent à la convergence (cf. papier SetFit) et ramènent chaque run
        # à ~1 min, quel que soit le volume de données. -1 = pas de plafond (few-shot léger).
        max_steps=max_steps,
    )

    trainer = SetFitTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        metric="f1",
        metric_kwargs={"average": "macro"},
    )

    t0 = time.perf_counter()
    trainer.train()
    train_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    preds = model.predict(test_texts)
    inference_time_ms = (time.perf_counter() - t0) / len(test_texts) * 1000

    # predict_proba donne les probabilités de la tête logistique
    # (utilisées pour le système hybride stratégie 5)
    proba = model.predict_proba(test_texts)

    results = compute_metrics(test_labels, list(preds), proba)
    results["train_time_s"] = round(train_time, 3)
    results["inference_time_ms"] = round(inference_time_ms, 4)
    results["backbone"] = backbone  # sauvegardé dans le JSON pour différencier les variantes

    return results, model


def get_predictions_with_proba(
    model: SetFitModel,
    texts: List[str],
) -> Tuple[List[str], np.ndarray]:
    """
    Retourne les prédictions ET les probabilités associées.
    Utilisé par le système hybride (stratégie 5) pour décider
    si un mail doit être escaladé vers Ministral.
    """
    preds = list(model.predict(texts))
    proba = model.predict_proba(texts)  # matrice (n_samples, n_classes)
    return preds, np.array(proba)
