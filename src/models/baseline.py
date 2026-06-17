"""
Stratégie 1 — Baseline classique : TF-IDF + Régression logistique.

Référence pré-deep learning. Sert d'ancre de réalité pour mesurer le gain
apporté par les approches modernes (CamemBERT, SetFit, Ministral).
Avantages : rapide, interprétable, pas de GPU, déployable partout.
"""
import time
from typing import List, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


def build_pipeline(max_features: int = 50_000, C: float = 1.0) -> Pipeline:
    """
    Construit le pipeline TF-IDF + Régression logistique.

    TF-IDF (Term Frequency - Inverse Document Frequency) transforme chaque
    texte en vecteur numérique : chaque dimension correspond à un mot ou
    bigramme, pondéré par sa fréquence dans le document et sa rareté dans
    le corpus (les mots très courants comme "le", "et" sont down-pondérés).

    Paramètres clés :
    - ngram_range=(1,2) : on considère aussi les bigrammes ("club sportif")
    - sublinear_tf=True : applique log(tf) pour atténuer les mots très fréquents
    - min_df=2 : ignore les mots n'apparaissant que dans un seul document
    - class_weight="balanced" : compense les éventuels déséquilibres de classes
    - C : inverse de la régularisation (plus C est grand, moins on régularise)
    """
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=2,
        )),
        ("clf", LogisticRegression(
            C=C,
            max_iter=1000,
            class_weight="balanced",
            solver="lbfgs",
        )),
    ])


def train_and_evaluate(
    train_texts: List[str],
    train_labels: List[str],
    test_texts: List[str],
    test_labels: List[str],
) -> Tuple[dict, Pipeline]:
    """
    Entraîne le pipeline et évalue sur le jeu de test.

    Retourne un dictionnaire de métriques et le pipeline entraîné.
    Le temps d'inférence est mesuré par mail (en ms) pour comparer
    la latence avec les approches deep learning.
    """
    from src.evaluation.metrics import compute_metrics

    pipe = build_pipeline()

    # Mesure du temps d'entraînement complet (TF-IDF fit + LR fit)
    t0 = time.perf_counter()
    pipe.fit(train_texts, train_labels)
    train_time = time.perf_counter() - t0

    # Mesure de la latence d'inférence par mail
    t0 = time.perf_counter()
    preds = pipe.predict(test_texts)
    inference_time_ms = (time.perf_counter() - t0) / len(test_texts) * 1000

    proba = pipe.predict_proba(test_texts)
    results = compute_metrics(test_labels, list(preds), proba)
    results["train_time_s"] = round(train_time, 3)
    results["inference_time_ms"] = round(inference_time_ms, 4)

    return results, pipe
