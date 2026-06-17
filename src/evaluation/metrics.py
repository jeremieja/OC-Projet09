"""
Calcul des métriques d'évaluation communes à tous les modèles.

Métrique principale : F1 macro
  - Calcule le F1 score par classe puis fait la moyenne non pondérée
  - Donne autant de poids à chaque classe quelle que soit sa taille
  - Adapté aux classes potentiellement déséquilibrées (ex: "sponsor" < "parent")
  - zero_division=0 évite les warnings quand une classe n'a aucune prédiction
    (peut arriver en régime 8-shot avec peu de données)
"""
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, classification_report
from typing import List, Optional


def compute_metrics(
    true_labels: List[str],
    pred_labels: List[str],
    proba: Optional[np.ndarray] = None,
) -> dict:
    """
    Calcule F1 macro et accuracy à partir des labels prédits et réels.

    Le paramètre proba est accepté pour compatibilité mais pas utilisé ici :
    on le garde pour une éventuelle extension (AUC, calibration ECE, etc.).
    """
    f1_macro = f1_score(true_labels, pred_labels, average="macro", zero_division=0)
    accuracy = accuracy_score(true_labels, pred_labels)

    return {
        "f1_macro": float(f1_macro),
        "accuracy": float(accuracy),
        "n_test": len(true_labels),
    }


def compute_metrics_from_logits(
    true_labels: List[str],
    pred_labels: List[str],
    logits: np.ndarray,
) -> dict:
    """
    Variante qui accepte les logits bruts de CamemBERT.
    Applique softmax pour convertir en probabilités avant le calcul des métriques.
    """
    from scipy.special import softmax
    proba = softmax(logits, axis=1)
    return compute_metrics(true_labels, pred_labels, proba)


def get_classification_report(true_labels: List[str], pred_labels: List[str]) -> str:
    """
    Retourne un rapport détaillé par classe (precision, recall, F1, support).
    Utile pour l'analyse qualitative des erreurs (notebook 03).
    """
    return classification_report(true_labels, pred_labels, zero_division=0)
