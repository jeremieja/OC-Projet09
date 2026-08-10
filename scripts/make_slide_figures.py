"""
Génère les figures du POC destinées au support de soutenance (PNG haute résolution).

Choix de visualisation :
  - Courbes d'apprentissage → forme « ligne » : la donnée décrit une évolution le long
    d'un axe ordonné (nombre d'exemples par classe).
  - Comparaison full data → forme « barres » : la donnée décrit des magnitudes comparées.
  - Système hybride → deux panneaux distincts plutôt qu'un double axe : le F1 et le taux
    d'escalade sont deux mesures d'échelles différentes.

Accessibilité : palette Okabe-Ito (conçue pour les déficiences de vision des couleurs),
doublée d'un encodage secondaire par forme de marqueur, afin que l'identité d'une série
ne repose jamais sur la seule couleur. Textes en gris neutre, grille discrète.

Sortie : results/figures/*.png
Usage  : python scripts/make_slide_figures.py
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

ROOT = Path(__file__).parents[1]
OUT = ROOT / "results" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# ── Paramètres visuels ────────────────────────────────────────────────────────
# Palette Okabe-Ito : quatre teintes maximalement séparées, adaptées au daltonisme.
COLORS = {
    "tfidf_lr":             "#0072B2",  # bleu
    "camembert":            "#E69F00",  # orange
    "setfit_camembert":     "#009E73",  # vert
    "setfit_mmbert":        "#CC79A7",  # rose
    "setfit_modernbert_en": "#CC79A7",  # rose (jamais affiché avec mmBERT)
}
# Encodage secondaire : chaque série a une forme de marqueur distincte.
MARKERS = {
    "tfidf_lr": "o", "camembert": "s", "setfit_camembert": "D",
    "setfit_mmbert": "^", "setfit_modernbert_en": "^",
}
LABELS = {
    "tfidf_lr": "TF-IDF + Régression log.",
    "camembert": "CamemBERT",
    "setfit_camembert": "SetFit (mpnet)",
    "setfit_mmbert": "SetFit (mmBERT)",
    "setfit_modernbert_en": "SetFit (ModernBERT)",
}

INK = "#1A1A1A"      # texte principal
INK_SOFT = "#555555"  # texte secondaire
GRID = "#D9D9D9"      # grille discrète

plt.rcParams.update({
    "font.size": 13,
    "axes.titlesize": 16,
    "axes.labelsize": 13,
    "axes.edgecolor": GRID,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": INK_SOFT,
    "ytick.color": INK_SOFT,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})


def _style_axes(ax):
    """Grille discrète, cadre allégé : les données priment sur le décor."""
    ax.grid(True, color=GRID, linewidth=0.8, alpha=0.7)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)


def load_agg() -> pd.DataFrame:
    return pd.read_csv(ROOT / "results" / "aggregated" / "results_aggregated.csv")


# ── 1 & 2. Courbes d'apprentissage ────────────────────────────────────────────
def learning_curves(df: pd.DataFrame, dataset: str, models: list, fname: str, titre: str):
    sub = df[(df["dataset"] == dataset) & df["regime"].astype(str).str.isdigit()].copy()
    sub["regime"] = sub["regime"].astype(int)

    fig, ax = plt.subplots(figsize=(9, 5.2), dpi=200)
    first_points = {}
    for m in models:
        d = sub[sub["model"] == m].sort_values("regime")
        if d.empty:
            continue
        ax.plot(d["regime"], d["f1_macro_mean"],
                color=COLORS[m], marker=MARKERS[m], markersize=9,
                linewidth=2.5, label=LABELS[m],
                markeredgecolor="white", markeredgewidth=1.2)
        # Bande de confiance (± écart-type sur les 5 tirages)
        ax.fill_between(d["regime"],
                        d["f1_macro_mean"] - d["f1_macro_std"].fillna(0),
                        d["f1_macro_mean"] + d["f1_macro_std"].fillna(0),
                        color=COLORS[m], alpha=0.12, linewidth=0)
        first_points[m] = d.iloc[0]["f1_macro_mean"]

    # Étiquetage sélectif : on annote uniquement le meilleur et le moins bon modèle
    # au régime le plus contraint (8 exemples), là où l'écart porte le message.
    # Annoter chaque point rendrait la figure illisible (étiquettes superposées).
    if first_points:
        for m in (max(first_points, key=first_points.get),
                  min(first_points, key=first_points.get)):
            v = first_points[m]
            ax.annotate(f"{v:.2f}", (8, v),
                        textcoords="offset points", xytext=(12, 6),
                        color=COLORS[m], fontsize=13, fontweight="bold")

    ax.set_xscale("log", base=2)
    ax.set_xticks([8, 16, 32, 64])
    ax.set_xticklabels(["8", "16", "32", "64"])
    ax.set_xlabel("Exemples étiquetés par catégorie")
    ax.set_ylabel("F1 macro")
    ax.set_ylim(0, 1.05)
    ax.set_title(titre, fontweight="bold", pad=14)
    ax.legend(frameon=False, loc="lower right", fontsize=11)
    _style_axes(ax)
    fig.tight_layout()
    fig.savefig(OUT / fname, bbox_inches="tight")
    plt.close(fig)
    print(f"  {fname}")


# ── 3. Comparaison en full data ───────────────────────────────────────────────
def fulldata_bars(df: pd.DataFrame, fname: str):
    sub = df[df["regime"] == "full"].copy()

    # Les variantes de socle SetFit diffèrent selon le corpus : ModernBERT est
    # anglophone (donc évalué sur Newsgroups), mmBERT est multilingue et a été
    # réservé au cas d'usage métier. On affiche dans chaque panneau les modèles
    # réellement évalués sur ce corpus, plutôt qu'une case vide trompeuse.
    ORDRE = {
        "emails": ["tfidf_lr", "camembert", "setfit_camembert", "setfit_mmbert"],
        "newsgroups": ["tfidf_lr", "camembert", "setfit_camembert", "setfit_modernbert_en"],
    }

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), dpi=200)
    for ax, ds, titre in zip(axes, ["emails", "newsgroups"],
                             ["Mails de clubs sportifs", "20 Newsgroups"]):
        order = ORDRE[ds]
        d = sub[sub["dataset"] == ds]
        d = d[d["model"].isin(order)].copy()
        d["ord"] = d["model"].map({m: i for i, m in enumerate(order)})
        d = d.sort_values("ord")
        if d.empty:
            continue
        bars = ax.barh([LABELS[m] for m in d["model"]], d["f1_macro_mean"],
                       color=[COLORS[m] for m in d["model"]], height=0.6)
        # Valeur en bout de barre : lecture directe sans quadrillage
        for b, v in zip(bars, d["f1_macro_mean"]):
            ax.text(v + 0.015, b.get_y() + b.get_height() / 2, f"{v:.2f}",
                    va="center", fontsize=11, color=INK, fontweight="bold")
        ax.set_xlim(0, 1.12)
        ax.set_title(titre, fontweight="bold", fontsize=14)
        ax.set_xlabel("F1 macro")
        _style_axes(ax)
        ax.grid(axis="y", visible=False)

    fig.suptitle("Performance avec l'intégralité des données d'entraînement",
                 fontweight="bold", fontsize=16, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / fname, bbox_inches="tight")
    plt.close(fig)
    print(f"  {fname}")


# ── 4. Système hybride : deux panneaux (jamais de double axe) ─────────────────
def hybrid_tradeoff(fname: str):
    from src.evaluation.results import load_all_runs
    runs = load_all_runs()
    h = runs[(runs["model"] == "hybrid_setfit_ministral")
             & (runs["dataset"] == "emails")
             & (runs["regime"] == "full")].copy()
    if h.empty:
        print("  (pas de données hybrides, figure ignorée)")
        return
    h["threshold"] = h["threshold"].astype(float)
    g = h.groupby("threshold").agg(f1=("f1_macro", "mean"),
                                   esc=("escalation_rate", "mean")).reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), dpi=200)
    # Panneau gauche : qualité
    axes[0].plot(g["threshold"], g["f1"], color="#0072B2", marker="o",
                 markersize=9, linewidth=2.5, markeredgecolor="white", markeredgewidth=1.2)
    axes[0].set_title("Qualité du système", fontweight="bold", fontsize=14)
    axes[0].set_ylabel("F1 macro")
    axes[0].set_ylim(0.985, 1.0)
    # Panneau droit : coût (part des mails envoyés au LLM)
    axes[1].plot(g["threshold"], g["esc"] * 100, color="#E69F00", marker="s",
                 markersize=9, linewidth=2.5, markeredgecolor="white", markeredgewidth=1.2)
    axes[1].set_title("Sollicitation du LLM payant", fontweight="bold", fontsize=14)
    axes[1].set_ylabel("Mails escaladés (%)")

    for ax in axes:
        ax.set_xlabel("Seuil de confiance τ")
        _style_axes(ax)

    fig.suptitle("Système hybride : le seuil τ arbitre entre qualité et coût",
                 fontweight="bold", fontsize=16, y=1.03)
    fig.tight_layout()
    fig.savefig(OUT / fname, bbox_inches="tight")
    plt.close(fig)
    print(f"  {fname}")


# ── 5. Exploration du dataset métier ──────────────────────────────────────────
def eda_emails(fname: str):
    from src.data.email_dataset import load_emails
    train, test = load_emails()
    full = pd.concat([train, test], ignore_index=True)
    full["n_mots"] = full["text"].str.split().str.len()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), dpi=200)

    counts = full["label"].value_counts().sort_values()
    axes[0].barh(counts.index, counts.values, color="#0072B2", height=0.65)
    axes[0].set_title("Un corpus parfaitement équilibré", fontweight="bold", fontsize=14)
    axes[0].set_xlabel("Nombre de mails")
    for i, v in enumerate(counts.values):
        axes[0].text(v + 3, i, str(v), va="center", fontsize=10, color=INK)
    axes[0].set_xlim(0, counts.max() * 1.15)

    axes[1].hist(full["n_mots"], bins=40, color="#009E73", edgecolor="white", linewidth=0.6)
    axes[1].set_title("Longueur des mails", fontweight="bold", fontsize=14)
    axes[1].set_xlabel("Nombre de mots")
    axes[1].set_ylabel("Nombre de mails")
    med = full["n_mots"].median()
    axes[1].axvline(med, color=INK_SOFT, linestyle="--", linewidth=1.5)
    axes[1].annotate(f"médiane : {med:.0f} mots", (med, axes[1].get_ylim()[1] * 0.9),
                     xytext=(8, 0), textcoords="offset points",
                     color=INK_SOFT, fontsize=11)

    for ax in axes:
        _style_axes(ax)
    axes[0].grid(axis="y", visible=False)

    fig.tight_layout()
    fig.savefig(OUT / fname, bbox_inches="tight")
    plt.close(fig)
    print(f"  {fname}")


def lime_local(fname: str, n_mots: int = 8):
    """
    Explication locale d'une prédiction SetFit : quels mots poussent vers la
    catégorie prédite, et lesquels l'en éloignent.

    Forme retenue : barres divergentes. La donnée porte une POLARITÉ (pour /
    contre), pas seulement une magnitude — l'axe zéro devient le point de
    lecture. Deux teintes seulement, et le sens est aussi donné par le côté
    de l'axe : l'information ne repose donc jamais sur la seule couleur.
    """
    src = ROOT / "results" / "interpretability" / "lime_setfit_local.json"
    d = json.loads(src.read_text(encoding="utf-8"))

    mots = d["mots_influents"][:n_mots]
    mots = sorted(mots, key=lambda m: m["poids"])
    noms = [m["mot"] for m in mots]
    poids = [m["poids"] for m in mots]

    POUR = "#0072B2"     # bleu : pousse vers la catégorie prédite
    CONTRE = "#D55E00"   # vermillon : pousse contre

    fig, ax = plt.subplots(figsize=(9, 4.6), dpi=200)
    couleurs = [POUR if p > 0 else CONTRE for p in poids]
    barres = ax.barh(noms, poids, color=couleurs, height=0.62)

    # Valeur au bout de chaque barre, du côté où elle pointe.
    for b, p in zip(barres, poids):
        decalage = 0.004 if p > 0 else -0.004
        ax.text(b.get_width() + decalage, b.get_y() + b.get_height() / 2,
                f"{p:+.3f}".replace(".", ","),
                va="center", ha="left" if p > 0 else "right",
                fontsize=11, color=INK_SOFT)

    ax.axvline(0, color=INK_SOFT, linewidth=1.2)
    _style_axes(ax)
    ax.grid(axis="y", visible=False)

    marge = max(abs(min(poids)), max(poids)) * 1.35
    ax.set_xlim(-marge, marge)
    ax.set_xlabel("Poids attribué par LIME")

    # Une bande libre est réservée au-dessus des barres pour nommer les deux
    # pôles : sous l'axe, ils entreraient en collision avec les graduations.
    haut = len(noms) - 1
    ax.set_ylim(-0.7, haut + 1.15)
    ax.text(-marge * 0.97, haut + 0.62, "◀  pousse vers une autre catégorie",
            color=CONTRE, fontsize=11.5, fontweight="bold", ha="left", va="center")
    ax.text(marge * 0.97, haut + 0.62, "pousse vers la catégorie prédite  ▶",
            color=POUR, fontsize=11.5, fontweight="bold", ha="right", va="center")

    ax.set_title(f"Un mail prédit « {d['prediction']} » : ce qui a pesé",
                 fontweight="bold", pad=16)

    fig.savefig(OUT / fname, bbox_inches="tight")
    plt.close(fig)


def main():
    df = load_agg()
    print("Figures générées :")
    learning_curves(df, "emails",
                    ["tfidf_lr", "camembert", "setfit_camembert", "setfit_mmbert"],
                    "fig_courbes_emails.png",
                    "Mails de clubs sportifs : SetFit domine à faibles données")
    learning_curves(df, "newsgroups",
                    ["tfidf_lr", "camembert", "setfit_camembert", "setfit_modernbert_en"],
                    "fig_courbes_newsgroups.png",
                    "20 Newsgroups : le même constat se vérifie")
    fulldata_bars(df, "fig_fulldata.png")
    hybrid_tradeoff("fig_hybride.png")
    eda_emails("fig_eda_emails.png")
    lime_local("fig_lime_local.png")
    print(f"\nDossier : {OUT}")


if __name__ == "__main__":
    main()
