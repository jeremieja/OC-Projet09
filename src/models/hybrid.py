"""
Stratégie 5 — Routage hybride par confiance : SetFit + fallback Ministral.

Principe : SetFit classifie localement la majorité des mails.
Pour chaque prédiction, la probabilité issue de la tête logistique de SetFit
donne un score de confiance. Si ce score est inférieur au seuil tau (τ),
le mail est "escaladé" vers Ministral 8B en few-shot.

Avantage : coût quasi nul sur le volume courant (SetFit local gratuit)
           + robustesse sur les cas ambigus (LLM payant seulement si nécessaire).

Le seuil τ devient un levier produit :
  - τ faible (0.3) : peu d'escalades, moins cher mais moins précis sur les cas durs
  - τ élevé (0.9) : beaucoup d'escalades, plus précis mais plus cher

On balaye plusieurs valeurs de τ pour construire la courbe coût/qualité.
"""
from typing import List, Tuple

import numpy as np

# Coût estimé par appel Ministral : ~170 tokens (150 input + 20 output) à $0.10/M tokens
_COST_PER_CALL_USD = 0.10 / 1_000_000 * 170


def route(
    setfit_preds: List[str],
    setfit_proba: np.ndarray,
    ministral_preds: List[str],
    threshold: float,
) -> Tuple[List[str], np.ndarray]:
    """
    Applique le routage par confiance et retourne les prédictions finales.

    La confiance SetFit = max des probabilités sur toutes les classes.
    Un max faible signifie que le modèle hésite entre plusieurs catégories :
    c'est exactement ces cas ambigus qu'on escalade vers Ministral.

    Retourne aussi le masque booléen des mails escaladés
    (utile pour calculer le taux d'escalade et le coût).
    """
    # max par ligne = probabilité de la classe la plus probable
    confidences = np.max(setfit_proba, axis=1)

    # True = mail escaladé vers Ministral, False = SetFit suffit
    escalated = confidences < threshold

    final_preds = [
        ministral_preds[i] if escalated[i] else setfit_preds[i]
        for i in range(len(setfit_preds))
    ]
    return final_preds, escalated


def evaluate_threshold(
    true_labels: List[str],
    setfit_preds: List[str],
    setfit_proba: np.ndarray,
    ministral_preds: List[str],
    threshold: float,
) -> dict:
    """
    Évalue le système hybride pour un seuil τ donné.

    Calcule trois métriques clés du trade-off produit :
    - f1_macro : qualité de classification globale
    - escalation_rate : part des mails routés vers le LLM (= part du coût)
    - avg_cost_per_mail_usd : coût moyen par mail (escalade_rate × coût_appel_API)
    """
    from src.evaluation.metrics import compute_metrics

    final_preds, escalated = route(setfit_preds, setfit_proba, ministral_preds, threshold)

    escalation_rate = float(np.mean(escalated))
    results = compute_metrics(true_labels, final_preds)
    results["threshold"] = threshold
    results["escalation_rate"] = round(escalation_rate, 4)
    # Seule la fraction escaladée est facturée par l'API
    results["avg_cost_per_mail_usd"] = round(escalation_rate * _COST_PER_CALL_USD, 8)

    return results


def sweep_thresholds(
    true_labels: List[str],
    setfit_preds: List[str],
    setfit_proba: np.ndarray,
    ministral_preds: List[str],
    thresholds: List[float] = None,
) -> List[dict]:
    """
    Évalue le système hybride sur une plage de seuils τ.
    Retourne une liste de résultats, un par seuil.
    Permet de tracer la courbe F1 vs taux d'escalade vs coût.
    """
    if thresholds is None:
        thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    return [
        evaluate_threshold(true_labels, setfit_preds, setfit_proba, ministral_preds, tau)
        for tau in thresholds
    ]
