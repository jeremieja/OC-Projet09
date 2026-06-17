"""
Thème visuel et helpers d'accessibilité (WCAG 2.1 niveau AA) pour le dashboard.

Choix d'accessibilité appliqués :
- Palette qualitative à fort contraste, distinguable par les daltoniens
  (palette "Okabe-Ito", standard scientifique pour l'accessibilité couleur).
- Jamais d'information portée par la SEULE couleur : on ajoute toujours
  une étiquette texte, une valeur chiffrée ou un motif.
- Contrastes texte/fond ≥ 4.5:1 (seuil WCAG AA pour le texte normal).
"""
from __future__ import annotations

# Palette Okabe-Ito : 8 couleurs discriminables même en cas de daltonisme
# (deutéranopie, protanopie, tritanopie). Référence accessibilité reconnue.
OKABE_ITO = [
    "#0072B2",  # bleu
    "#E69F00",  # orange
    "#009E73",  # vert
    "#CC79A7",  # rose
    "#56B4E9",  # bleu ciel
    "#D55E00",  # vermillon
    "#F0E442",  # jaune
    "#999999",  # gris
]

# Template Plotly clair, fonds neutres, bon contraste
PLOTLY_TEMPLATE = "plotly_white"

# Couleurs sémantiques pour les messages (contraste AA garanti sur fond blanc)
COLOR_SUCCESS = "#1B7F4B"   # vert foncé
COLOR_WARNING = "#B85C00"   # orange foncé
COLOR_ERROR = "#C0392B"     # rouge foncé
COLOR_INFO = "#0072B2"      # bleu


def color_for_label(label: str, all_labels: list[str]) -> str:
    """Couleur stable et accessible pour une étiquette donnée."""
    idx = sorted(all_labels).index(label) if label in all_labels else 0
    return OKABE_ITO[idx % len(OKABE_ITO)]


def apply_accessible_layout(fig, title: str, height: int = 400):
    """
    Applique une mise en forme accessible commune à une figure Plotly :
    titre explicite, police lisible, légende claire, marges aérées.
    """
    fig.update_layout(
        title=dict(text=title, font=dict(size=18)),
        template=PLOTLY_TEMPLATE,
        height=height,
        font=dict(size=14),                 # police d'au moins 14px pour la lisibilité
        legend=dict(font=dict(size=13)),
        margin=dict(l=40, r=20, t=60, b=40),
    )
    return fig
