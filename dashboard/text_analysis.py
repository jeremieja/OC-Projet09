"""
Fonctions d'analyse de texte pour l'onglet EDA du dashboard.

Données non structurées de type texte → l'EDA se formalise par (spec dashboard) :
  - au moins deux analyses statistiques (longueur des textes, fréquence de mots)
  - un WordCloud
le tout via des graphiques interactifs.
"""
from __future__ import annotations

import re
from collections import Counter
from io import BytesIO

import pandas as pd

# Stop-words FR + EN minimaux (mots vides à exclure des fréquences/WordCloud).
# On reste volontairement simple et transparent plutôt que d'ajouter une dépendance.
_STOPWORDS = {
    # français
    "le", "la", "les", "un", "une", "des", "de", "du", "et", "à", "a", "au", "aux",
    "en", "dans", "pour", "par", "sur", "avec", "sans", "ce", "cet", "cette", "ces",
    "que", "qui", "quoi", "dont", "où", "je", "tu", "il", "elle", "nous", "vous",
    "ils", "elles", "on", "se", "sa", "son", "ses", "mes", "mon", "ma", "tes", "ton",
    "ne", "pas", "plus", "est", "sont", "été", "être", "avoir", "ai", "as", "ont",
    "vos", "votre", "nos", "notre", "leur", "leurs", "y", "d", "l", "s", "n", "c",
    "j", "m", "t", "qu", "si", "ou", "mais", "donc", "car", "aussi", "très", "bien",
    "cela", "comme", "fait", "faire", "merci", "bonjour", "cordialement",
    # anglais
    "the", "a", "an", "of", "to", "in", "is", "are", "was", "were", "be", "been",
    "and", "or", "but", "for", "on", "with", "as", "at", "by", "from", "this", "that",
    "it", "i", "you", "he", "she", "we", "they", "my", "your", "his", "her", "our",
    "not", "no", "do", "does", "did", "have", "has", "had", "will", "would", "can",
    "could", "should", "if", "so", "than", "then", "there", "here", "what", "which",
    "who", "how", "all", "any", "some", "more", "out", "up", "about", "just", "like",
}

_WORD_RE = re.compile(r"\b[a-zàâäéèêëïîôöùûüç]{3,}\b", re.IGNORECASE)


def add_text_stats(df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
    """Ajoute des colonnes statistiques par texte (longueur en mots et caractères)."""
    out = df.copy()
    out["n_mots"] = out[text_col].str.split().str.len()
    out["n_caracteres"] = out[text_col].str.len()
    return out


def top_words(texts: list[str], n: int = 20, extra_stopwords: set[str] | None = None) -> pd.DataFrame:
    """
    Retourne les n mots les plus fréquents (hors stop-words), avec leur compte.
    Sert à l'analyse statistique 'fréquence de mots' et alimente un graphique interactif.
    """
    stop = _STOPWORDS | (extra_stopwords or set())
    counter: Counter = Counter()
    for t in texts:
        for w in _WORD_RE.findall(str(t).lower()):
            if w not in stop:
                counter[w] += 1
    common = counter.most_common(n)
    return pd.DataFrame(common, columns=["mot", "frequence"])


def build_wordcloud_png(texts: list[str], extra_stopwords: set[str] | None = None) -> bytes:
    """
    Génère un nuage de mots (WordCloud) et le retourne en PNG (bytes) pour st.image.
    On passe par une image car le WordCloud n'est pas un graphique Plotly natif.
    """
    from wordcloud import WordCloud

    stop = _STOPWORDS | (extra_stopwords or set())
    text_blob = " ".join(str(t) for t in texts)

    wc = WordCloud(
        width=1000,
        height=400,
        background_color="white",   # fond blanc = contraste maximal (accessibilité)
        stopwords=stop,
        colormap="viridis",          # colormap perceptuellement uniforme
        max_words=120,
        collocations=False,
    ).generate(text_blob)

    buf = BytesIO()
    wc.to_image().save(buf, format="PNG")
    return buf.getvalue()
