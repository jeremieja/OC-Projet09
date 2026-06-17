"""
Analyse de la feature importance (globale et locale) pour la note méthodologique.

Produit, sur le dataset emails :
  1. GLOBALE — TF-IDF + LR : les mots au plus fort coefficient par catégorie
     (le modèle est nativement interprétable : coef = poids du mot dans la décision).
  2. LOCALE — TF-IDF + LR : explication d'une prédiction individuelle (contributions).
  3. SetFit (boîte noire) : explication LOCALE via LIME sur un mail, et agrégation
     GLOBALE approximée (mots les plus influents sur un échantillon).

Sorties (dans results/interpretability/) :
  - global_tfidf.json        : top mots par classe (coefficients LR)
  - global_tfidf.png         : heatmap / barres des mots discriminants
  - local_tfidf.json         : contributions pour un mail exemple
  - lime_setfit_local.json   : poids LIME pour un mail (SetFit)
  - lime_setfit_global.json  : importance LIME agrégée (SetFit, échantillon)

Usage :
    python scripts/feature_importance.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.email_dataset import load_emails, CLASSES
from src.utils.logging import get_logger

logger = get_logger("feature_importance")
OUT = Path(__file__).parents[1] / "results" / "interpretability"
OUT.mkdir(parents=True, exist_ok=True)

N_TOP = 12          # nb de mots affichés par catégorie
LIME_SAMPLES = 40   # nb de mails pour l'agrégation LIME globale (coûteux)


# ──────────────────────────────────────────────────────────────────────
# 1 & 2. TF-IDF + LR (nativement interprétable)
# ──────────────────────────────────────────────────────────────────────
def analyse_tfidf(train_df, test_df):
    from src.models.baseline import build_pipeline

    logger.info("Entraînement TF-IDF + LR sur full data (emails)...")
    full = pd.concat([train_df, test_df], ignore_index=True)
    pipe = build_pipeline()
    pipe.fit(full["text"].tolist(), full["label"].tolist())

    vectorizer = pipe.named_steps["tfidf"]
    clf = pipe.named_steps["clf"]
    feature_names = np.array(vectorizer.get_feature_names_out())
    classes = list(clf.classes_)

    # — GLOBALE : pour chaque classe, les mots au plus fort coefficient positif —
    # En multiclasse, clf.coef_ a la forme (n_classes, n_features).
    global_importance = {}
    for i, cls in enumerate(classes):
        coefs = clf.coef_[i]
        top_idx = np.argsort(coefs)[-N_TOP:][::-1]
        global_importance[cls] = [
            {"mot": feature_names[j], "poids": round(float(coefs[j]), 3)}
            for j in top_idx
        ]
    with open(OUT / "global_tfidf.json", "w", encoding="utf-8") as f:
        json.dump(global_importance, f, ensure_ascii=False, indent=2)
    logger.info(f"Importance globale TF-IDF sauvegardée ({len(classes)} classes).")

    # Heatmap globale (mots × poids, regroupés par classe) → image pour la note
    _plot_global_tfidf(global_importance)

    # — LOCALE : explication d'UN mail (contributions = tfidf(mot) × coef(mot, classe)) —
    sample_text = test_df.iloc[0]["text"]
    sample_true = test_df.iloc[0]["label"]
    x_vec = vectorizer.transform([sample_text])
    pred = pipe.predict([sample_text])[0]
    pred_idx = classes.index(pred)

    contributions = (x_vec.toarray()[0]) * clf.coef_[pred_idx]
    nz = np.nonzero(x_vec.toarray()[0])[0]
    contrib_list = sorted(
        [{"mot": feature_names[j], "contribution": round(float(contributions[j]), 4)}
         for j in nz],
        key=lambda d: abs(d["contribution"]), reverse=True
    )[:N_TOP]

    local = {
        "mail": sample_text[:400],
        "vraie_categorie": sample_true,
        "prediction": pred,
        "contributions": contrib_list,
    }
    with open(OUT / "local_tfidf.json", "w", encoding="utf-8") as f:
        json.dump(local, f, ensure_ascii=False, indent=2)
    logger.info(f"Importance locale TF-IDF sauvegardée (mail prédit '{pred}').")

    return pipe, sample_text


def _plot_global_tfidf(global_importance):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    classes = list(global_importance.keys())
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    axes = axes.flatten()
    for ax, cls in zip(axes, classes):
        items = global_importance[cls][:8]
        mots = [d["mot"] for d in items][::-1]
        poids = [d["poids"] for d in items][::-1]
        ax.barh(mots, poids, color="#0072B2")
        ax.set_title(cls, fontsize=11, fontweight="bold")
        ax.tick_params(labelsize=9)
    for ax in axes[len(classes):]:
        ax.axis("off")
    fig.suptitle("Mots les plus discriminants par catégorie (coefficients TF-IDF + LR)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(OUT / "global_tfidf.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    logger.info("Figure global_tfidf.png générée.")


# ──────────────────────────────────────────────────────────────────────
# 3. SetFit via LIME (boîte noire)
# ──────────────────────────────────────────────────────────────────────
def analyse_setfit_lime(train_df, test_df, sample_text):
    from lime.lime_text import LimeTextExplainer
    from src.models.setfit_model import train_and_evaluate, BACKBONE_CAMEMBERT

    logger.info("Entraînement SetFit (mpnet) sur full data pour LIME...")
    # On entraîne sur un sous-ensemble pour limiter le temps (LIME refait bcp d'inférences)
    _, model = train_and_evaluate(
        train_df["text"].tolist(), train_df["label"].tolist(),
        test_df["text"].tolist(), test_df["label"].tolist(),
        backbone=BACKBONE_CAMEMBERT, max_steps=600,
    )

    class_names = sorted(train_df["label"].unique())

    def predict_proba(texts):
        return np.array(model.predict_proba(texts))

    explainer = LimeTextExplainer(class_names=class_names)

    # — LOCALE : explication d'un mail —
    logger.info("LIME : explication locale d'un mail (SetFit)...")
    pred = model.predict([sample_text])[0]
    exp = explainer.explain_instance(
        sample_text, predict_proba, num_features=N_TOP,
        labels=[class_names.index(pred)],
    )
    weights = exp.as_list(label=class_names.index(pred))
    local = {
        "mail": sample_text[:400],
        "prediction": str(pred),
        "mots_influents": [{"mot": w, "poids": round(float(p), 4)} for w, p in weights],
    }
    with open(OUT / "lime_setfit_local.json", "w", encoding="utf-8") as f:
        json.dump(local, f, ensure_ascii=False, indent=2)

    # — GLOBALE approximée : agrégation des poids LIME sur un échantillon —
    logger.info(f"LIME : agrégation globale sur {LIME_SAMPLES} mails (peut être long)...")
    from collections import defaultdict
    agg = defaultdict(float)
    sample = test_df.sample(min(LIME_SAMPLES, len(test_df)), random_state=42)
    for txt in sample["text"]:
        p = model.predict([txt])[0]
        e = explainer.explain_instance(
            txt, predict_proba, num_features=8, labels=[class_names.index(p)]
        )
        for word, weight in e.as_list(label=class_names.index(p)):
            agg[word] += abs(weight)
    top_global = sorted(agg.items(), key=lambda kv: kv[1], reverse=True)[:20]
    glob = {"mots_influents_globaux": [
        {"mot": w, "importance_cumulee": round(v, 3)} for w, v in top_global
    ]}
    with open(OUT / "lime_setfit_global.json", "w", encoding="utf-8") as f:
        json.dump(glob, f, ensure_ascii=False, indent=2)
    logger.info("LIME SetFit (local + global) sauvegardé.")


def main():
    train_df, test_df = load_emails()
    pipe, sample_text = analyse_tfidf(train_df, test_df)
    analyse_setfit_lime(train_df, test_df, sample_text)
    logger.info(f"Terminé. Résultats dans {OUT}")


if __name__ == "__main__":
    main()
