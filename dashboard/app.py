"""
Dashboard Streamlit — POC Classification de mails de clubs sportifs (DataSpace, Projet 9).

Conforme aux spécifications dashboard :
  1. EDA des données non structurées texte : ≥2 analyses statistiques interactives
     (longueur des textes, fréquence de mots) + un WordCloud.
  2. Sélection de données en entrée du moteur de prédiction (choix dataset + saisie texte).
  3. Affichage du résultat de la prédiction.
  4. Déployable sur le cloud (modèle TF-IDF léger + Ministral via API).
  5. Accessibilité WCAG 2.1 AA (contrastes, libellés, pas d'info portée par la seule couleur,
     textes alternatifs sur les images, structure de titres).

Lancement local :
    streamlit run dashboard/app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Permet d'importer src/ et les modules du dashboard
ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

from dashboard.theme import (  # noqa: E402
    OKABE_ITO, COLOR_SUCCESS, COLOR_INFO, apply_accessible_layout, color_for_label,
)
from dashboard.text_analysis import (  # noqa: E402
    add_text_stats, top_words, build_wordcloud_png,
)
from src.data.email_dataset import CLASSES as EMAIL_CLASSES  # noqa: E402
from src.data.newsgroups import SELECTED_CLASSES as NEWS_CLASSES  # noqa: E402

# ──────────────────────────────────────────────────────────────────────────────
# Configuration de la page
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Classificateur de mails — POC DataSpace",
    page_icon="📧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS minimal pour renforcer l'accessibilité : focus visible au clavier,
# taille de police de base confortable.
st.markdown(
    """
    <style>
      /* Contour de focus visible pour la navigation clavier (WCAG 2.4.7) */
      *:focus-visible { outline: 3px solid #0072B2 !important; outline-offset: 2px; }
      /* Taille de police de base lisible */
      html, body, [class*="css"] { font-size: 16px; }
    </style>
    """,
    unsafe_allow_html=True,
)

DATASET_LABELS = {
    "emails": "Mails de clubs sportifs (français)",
    "newsgroups": "20 Newsgroups (anglais — validation scientifique)",
}
CLASSES_BY_DATASET = {"emails": EMAIL_CLASSES, "newsgroups": NEWS_CLASSES}


# ──────────────────────────────────────────────────────────────────────────────
# Chargement des données et modèles (mis en cache)
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_dataset(name: str) -> pd.DataFrame:
    """Charge le CSV complet d'un dataset pour l'EDA. Cache pour éviter les relectures."""
    if name == "emails":
        path = ROOT / "data" / "processed" / "emails.csv"
        if path.exists():
            return pd.read_csv(path)
        return pd.DataFrame()
    else:
        # Newsgroups : on reconstruit via sklearn (téléchargé/caché localement)
        from src.data.newsgroups import load_newsgroups
        train, test = load_newsgroups()
        return pd.concat([train, test], ignore_index=True)


@st.cache_resource(show_spinner=False)
def load_deploy_model(name: str):
    """Charge le pipeline TF-IDF déployable (léger). Cache ressource (objet lourd)."""
    path = ROOT / "models_saved" / f"deploy_{name}.joblib"
    if path.exists():
        return joblib.load(path)
    return None


@st.cache_data(show_spinner=False)
def load_results() -> pd.DataFrame:
    """
    Charge les résultats agrégés des expériences (pour l'onglet Performances).

    On lit en priorité le CSV agrégé léger (results_aggregated.csv), versionné et
    déployé sur le cloud. En local, si les runs bruts existent, on les recharge
    pour avoir le détail complet (fallback développeur).
    """
    csv_path = ROOT / "results" / "aggregated" / "results_aggregated.csv"
    if csv_path.exists():
        return pd.read_csv(csv_path)
    # Fallback local : recalcul depuis les runs bruts s'ils sont présents
    try:
        from src.evaluation.results import load_all_runs, aggregate_results
        runs = load_all_runs()
        return aggregate_results(runs) if not runs.empty else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


# ──────────────────────────────────────────────────────────────────────────────
# Barre latérale — navigation
# ──────────────────────────────────────────────────────────────────────────────
st.sidebar.title("📧 Classificateur de mails")
st.sidebar.caption("POC DataSpace — Projet 9")

page = st.sidebar.radio(
    "Navigation principale",
    ["🔎 Exploration des données", "🤖 Prédiction en direct", "📊 Performances des modèles"],
    help="Choisissez la section du dashboard à afficher.",
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Accessibilité** : palette adaptée au daltonisme, navigation clavier, "
    "textes alternatifs sur les visuels."
)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — EXPLORATION DES DONNÉES (EDA)
# ══════════════════════════════════════════════════════════════════════════════
def page_eda():
    st.title("🔎 Exploration des données")
    st.markdown(
        "Analyse exploratoire des **données textuelles** : distribution des catégories, "
        "longueur des messages, mots les plus fréquents et nuage de mots."
    )

    dataset = st.selectbox(
        "Jeu de données à explorer",
        options=list(DATASET_LABELS.keys()),
        format_func=lambda k: DATASET_LABELS[k],
        help="Sélectionnez le jeu de données dont vous voulez voir les statistiques.",
    )

    df = load_dataset(dataset)
    if df.empty:
        st.warning(
            "Jeu de données indisponible. Pour les mails, lancez d'abord "
            "`python scripts/generate_dataset.py`."
        )
        return

    df = add_text_stats(df)
    labels = CLASSES_BY_DATASET[dataset]

    # — Indicateurs clés (describe() façon produit) —
    st.subheader("Vue d'ensemble")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Messages", f"{len(df):,}".replace(",", " "))
    c2.metric("Catégories", df["label"].nunique())
    c3.metric("Longueur médiane", f"{int(df['n_mots'].median())} mots")
    c4.metric("Longueur max", f"{int(df['n_mots'].max())} mots")

    # — Statistiques descriptives détaillées (type describe()) —
    with st.expander("Statistiques descriptives détaillées (describe)"):
        st.dataframe(
            df[["n_mots", "n_caracteres"]].describe().round(1),
            use_container_width=True,
        )

    st.markdown("---")

    # — Graphique interactif 1 : distribution des catégories —
    st.subheader("1. Distribution des catégories")
    counts = df["label"].value_counts().reset_index()
    counts.columns = ["categorie", "nombre"]
    fig1 = px.bar(
        counts, x="nombre", y="categorie", orientation="h",
        text="nombre",
        color="categorie",
        color_discrete_sequence=OKABE_ITO,
    )
    fig1.update_traces(textposition="outside")
    fig1.update_layout(showlegend=False)  # redondant avec l'axe Y → on masque
    apply_accessible_layout(fig1, "Nombre de messages par catégorie", height=380)
    fig1.update_xaxes(title="Nombre de messages")
    fig1.update_yaxes(title="Catégorie")
    st.plotly_chart(fig1, use_container_width=True)
    st.caption(
        "Figure 1 — Histogramme horizontal du nombre de messages par catégorie. "
        f"Le jeu compte {len(df)} messages répartis sur {df['label'].nunique()} catégories."
    )

    # — Graphique interactif 2 : longueur des messages par catégorie —
    st.subheader("2. Longueur des messages (analyse statistique)")
    fig2 = px.box(
        df, x="label", y="n_mots",
        color="label", color_discrete_sequence=OKABE_ITO,
        points=False,
    )
    fig2.update_layout(showlegend=False)
    apply_accessible_layout(fig2, "Distribution de la longueur des messages (en mots)", height=420)
    fig2.update_xaxes(title="Catégorie", tickangle=-30)
    fig2.update_yaxes(title="Nombre de mots")
    st.plotly_chart(fig2, use_container_width=True)
    st.caption(
        "Figure 2 — Boîtes à moustaches de la longueur (en mots) par catégorie : "
        "médiane, quartiles et valeurs extrêmes."
    )

    # — Graphique interactif 3 : fréquence des mots —
    st.subheader("3. Mots les plus fréquents (analyse statistique)")
    col_a, col_b = st.columns([1, 2])
    with col_a:
        cat_filter = st.selectbox(
            "Filtrer par catégorie (optionnel)",
            options=["Toutes"] + sorted(labels),
            help="Affiche les mots fréquents pour une catégorie précise ou l'ensemble.",
        )
        n_words = st.slider("Nombre de mots à afficher", 10, 40, 20, step=5)

    texts = df["text"] if cat_filter == "Toutes" else df.loc[df["label"] == cat_filter, "text"]
    freq = top_words(texts.tolist(), n=n_words)

    with col_b:
        fig3 = px.bar(
            freq.sort_values("frequence"), x="frequence", y="mot",
            orientation="h", text="frequence",
            color_discrete_sequence=[COLOR_INFO],
        )
        fig3.update_traces(textposition="outside")
        apply_accessible_layout(
            fig3, f"Top {n_words} mots — {cat_filter}", height=max(380, n_words * 18)
        )
        fig3.update_xaxes(title="Fréquence")
        fig3.update_yaxes(title="Mot")
        st.plotly_chart(fig3, use_container_width=True)
    st.caption(
        f"Figure 3 — Fréquence d'apparition des {n_words} mots les plus courants "
        f"(catégorie : {cat_filter}), hors mots vides."
    )

    # — WordCloud (exigé par la spec pour les données texte) —
    st.subheader("4. Nuage de mots (WordCloud)")
    with st.spinner("Génération du nuage de mots…"):
        png = build_wordcloud_png(texts.tolist())
    # alt-text descriptif pour l'accessibilité (lecteurs d'écran)
    st.image(
        png,
        caption=f"Figure 4 — Nuage de mots de la catégorie « {cat_filter} ». "
                "La taille de chaque mot est proportionnelle à sa fréquence.",
        use_container_width=True,
    )
    st.markdown(
        f"_Description textuelle (accessibilité)_ : les mots les plus fréquents de cette "
        f"sélection sont **{', '.join(freq['mot'].head(8))}**."
    )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — PRÉDICTION EN DIRECT
# ══════════════════════════════════════════════════════════════════════════════
def page_prediction():
    st.title("🤖 Prédiction en direct")
    st.markdown(
        "Saisissez un message, choisissez le contexte, et obtenez la **catégorie prédite** "
        "par le modèle local (TF-IDF + Régression logistique) — et, en option, par "
        "le LLM Ministral 8B via API."
    )

    # Exemples pré-remplis (point de départ de la démo, avant tout tirage aléatoire)
    examples = {
        "emails": "Bonjour, je souhaite inscrire mon fils de 12 ans dans votre club de "
                  "football pour la saison prochaine. Quels documents faut-il fournir et "
                  "quel est le montant de la cotisation ? Merci d'avance.",
        "newsgroups": "The new graphics card supports hardware acceleration for 3D rendering "
                      "and OpenGL. Has anyone benchmarked it against the previous generation?",
    }

    # — État persistant entre les interactions (session_state) —
    if "pred_dataset" not in st.session_state:
        st.session_state.pred_dataset = "emails"
    if "pred_text" not in st.session_state:
        st.session_state.pred_text = examples["emails"]
    if "pred_true_label" not in st.session_state:
        st.session_state.pred_true_label = None  # vraie catégorie si le mail vient du dataset

    def _on_dataset_change():
        """Quand on change de contexte, on repart de l'exemple par défaut."""
        st.session_state.pred_text = examples[st.session_state.pred_dataset]
        st.session_state.pred_true_label = None

    def _draw_random():
        """Pioche un message aléatoire dans le dataset sélectionné + mémorise sa vraie classe."""
        sample_df = load_dataset(st.session_state.pred_dataset)
        if not sample_df.empty:
            row = sample_df.sample(1).iloc[0]
            st.session_state.pred_text = str(row["text"])
            st.session_state.pred_true_label = str(row["label"])

    # — Sélection des données d'entrée (exigée par la spec) —
    col1, col2 = st.columns([1, 1])
    with col1:
        dataset = st.selectbox(
            "Contexte de classification",
            options=list(DATASET_LABELS.keys()),
            format_func=lambda k: DATASET_LABELS[k],
            key="pred_dataset",
            on_change=_on_dataset_change,
            help="Détermine la liste de catégories et le modèle utilisé.",
        )
    with col2:
        use_ministral = st.toggle(
            "Comparer avec Ministral 8B (API)",
            value=False,
            help="Nécessite une clé MISTRAL_API_KEY. Appelle le LLM en plus du modèle local.",
        )

    labels = CLASSES_BY_DATASET[dataset]

    # — Zone de saisie liée au session_state (modifiable par le bouton aléatoire) —
    mail_text = st.text_area(
        "Texte du message à classer",
        height=180,
        key="pred_text",
        help="Saisissez un message, ou tirez-en un au hasard dans le jeu de données.",
    )

    # — Boutons d'action —
    b1, b2 = st.columns([1, 1])
    with b1:
        predict = st.button("Classer le message", type="primary", use_container_width=True)
    with b2:
        st.button("🎲 Tirer un mail au hasard", on_click=_draw_random,
                  use_container_width=True,
                  help="Remplace le texte par un message aléatoire du jeu de données.")

    # — Si le mail provient du dataset, on affiche sa vraie catégorie (utile pour la démo) —
    if st.session_state.pred_true_label:
        st.info(
            f"📂 Message tiré du jeu de données — **vraie catégorie : "
            f"{st.session_state.pred_true_label}**. "
            "Voyons si le modèle la retrouve !"
        )

    if not predict:
        return
    if not mail_text.strip():
        st.warning("Veuillez saisir un message avant de lancer la prédiction.")
        return

    st.markdown("---")
    col_local, col_llm = st.columns(2) if use_ministral else (st.container(), None)

    # — Modèle local TF-IDF + LR —
    with col_local:
        st.subheader("Modèle local — TF-IDF + Régression logistique")
        model = load_deploy_model(dataset)
        if model is None:
            st.error(
                "Modèle déployable introuvable. Lancez d'abord "
                "`python scripts/save_deploy_model.py`."
            )
        else:
            pred = model.predict([mail_text])[0]
            proba = model.predict_proba([mail_text])[0]
            classes = list(model.classes_)
            conf = float(np.max(proba))

            # Résultat principal : texte + couleur (jamais la couleur seule)
            st.markdown(f"### Catégorie prédite : **{pred}**")
            st.progress(conf, text=f"Confiance : {conf:.0%}")

            # Si on connaît la vraie catégorie (mail tiré du dataset), on indique
            # explicitement si la prédiction est correcte (icône + texte, pas couleur seule).
            true_label = st.session_state.get("pred_true_label")
            if true_label:
                if pred == true_label:
                    st.success(f"✓ Correct — la vraie catégorie était bien « {true_label} ».")
                else:
                    st.error(f"✗ Incorrect — la vraie catégorie était « {true_label} ».")

            # Distribution complète des probabilités (graphique interactif)
            proba_df = pd.DataFrame({"categorie": classes, "probabilite": proba})
            proba_df = proba_df.sort_values("probabilite")
            fig = px.bar(
                proba_df, x="probabilite", y="categorie", orientation="h",
                text=proba_df["probabilite"].map("{:.0%}".format),
                color_discrete_sequence=[COLOR_SUCCESS],
            )
            fig.update_traces(textposition="outside")
            apply_accessible_layout(fig, "Probabilité par catégorie", height=320)
            fig.update_xaxes(title="Probabilité", range=[0, 1], tickformat=".0%")
            fig.update_yaxes(title="")
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Prédiction instantanée, hors-ligne, sans coût.")

    # — Ministral 8B (optionnel) —
    if use_ministral and col_llm is not None:
        with col_llm:
            st.subheader("LLM — Ministral 8B (API)")
            domain = ("mails de clubs sportifs français" if dataset == "emails"
                      else "messages de forums de discussion thématiques (newsgroups)")
            with st.spinner("Appel à l'API Mistral…"):
                try:
                    from src.models.ministral import classify_batch
                    preds, confs, _ = classify_batch(
                        [mail_text], labels, mode="zero_shot",
                        domain=domain, show_progress=False,
                    )
                    st.markdown(f"### Catégorie prédite : **{preds[0]}**")

                    # Indicateur correct/incorrect si le mail vient du dataset (cohérence
                    # avec le modèle local) — icône + texte, jamais la couleur seule.
                    true_label = st.session_state.get("pred_true_label")
                    if true_label:
                        if preds[0] == true_label:
                            st.success(f"✓ Correct — vraie catégorie « {true_label} ».")
                        else:
                            st.error(f"✗ Incorrect — vraie catégorie « {true_label} ».")

                    conf_llm = min(max(float(confs[0]), 0.0), 1.0)  # borne [0,1] pour st.progress
                    st.progress(conf_llm, text=f"Confiance auto-déclarée : {conf_llm:.0%}")
                    st.caption(
                        "Classé par un grand modèle de langage, sans entraînement. "
                        "⚠️ La confiance est **auto-déclarée** par le LLM (souvent ~50 %) : "
                        "elle est indicative et moins fiable qu'une probabilité calculée."
                    )
                except Exception as exc:
                    st.error(f"Appel Ministral impossible : {exc}")
                    st.caption(
                        "Vérifiez que MISTRAL_API_KEY est définie dans les secrets de l'app."
                    )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — PERFORMANCES DES MODÈLES
# ══════════════════════════════════════════════════════════════════════════════
def page_performances():
    st.title("📊 Performances des modèles")
    st.markdown(
        "Synthèse des expériences : comment chaque stratégie se comporte selon le "
        "nombre d'exemples d'entraînement disponibles."
    )

    df = load_results()
    if df.empty:
        st.warning("Aucun résultat trouvé. Lancez les scripts d'expérience d'abord.")
        return

    dataset = st.selectbox(
        "Jeu de données",
        options=list(DATASET_LABELS.keys()),
        format_func=lambda k: DATASET_LABELS[k],
    )

    # Le CSV agrégé fournit directement f1_macro_mean et f1_macro_std par
    # (model, dataset, regime). On filtre sur les régimes numériques (few-shot)
    # et les modèles entraînables (hors Ministral/hybride, sans courbe d'apprentissage).
    sub = df[(df["dataset"] == dataset) & df["regime"].astype(str).str.isdigit()].copy()
    sub = sub[~sub["model"].str.startswith(("ministral", "hybrid"))]
    if sub.empty:
        st.info("Pas de courbes d'apprentissage pour ce jeu de données.")
        return
    sub["regime"] = sub["regime"].astype(int)

    all_models = sorted(sub["model"].unique())
    markers = ["circle", "square", "diamond", "triangle-up", "x", "star"]
    fig = go.Figure()
    for i, model in enumerate(all_models):
        mc = sub[sub["model"] == model].sort_values("regime")
        color = OKABE_ITO[i % len(OKABE_ITO)]
        fig.add_trace(go.Scatter(
            x=mc["regime"], y=mc["f1_macro_mean"],
            mode="lines+markers",
            name=model,
            line=dict(color=color, width=2),
            marker=dict(size=8, symbol=markers[i % len(markers)]),  # motif != couleur seule
            error_y=dict(type="data", array=mc["f1_macro_std"].fillna(0), visible=True, thickness=1),
        ))
    apply_accessible_layout(fig, f"Courbes d'apprentissage — {DATASET_LABELS[dataset]}", height=480)
    fig.update_xaxes(title="Exemples par catégorie (échelle log)", type="log")
    fig.update_yaxes(title="F1 macro", range=[0, 1], tickformat=".0%")
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Chaque point = F1 macro moyen sur 5 tirages ; les barres verticales montrent "
        "l'écart-type. Les marqueurs ont des formes distinctes (accessibilité)."
    )

    # Tableau récap full data
    full = df[(df["dataset"] == dataset) & (df["regime"] == "full")]
    if not full.empty:
        st.subheader("Performance avec toutes les données (full data)")
        tab = full[["model", "f1_macro_mean"]].sort_values("f1_macro_mean", ascending=False).copy()
        tab["f1_macro_mean"] = tab["f1_macro_mean"].map("{:.1%}".format)
        tab.columns = ["Modèle", "F1 macro"]
        st.dataframe(tab, use_container_width=True, hide_index=True)


# ──────────────────────────────────────────────────────────────────────────────
# Routage
# ──────────────────────────────────────────────────────────────────────────────
if page.startswith("🔎"):
    page_eda()
elif page.startswith("🤖"):
    page_prediction()
else:
    page_performances()
