"""
Chargement du dataset 20 Newsgroups (validation scientifique).

On retient 8 classes sur 20 pour reproduire un benchmark connu et garantir
la reproductibilité. scikit-learn télécharge automatiquement le dataset
lors du premier appel.
"""
import pandas as pd
from sklearn.datasets import fetch_20newsgroups
from typing import Tuple

# Les 8 classes retenues parmi les 20 disponibles.
# Choix équilibré entre sciences, informatique, sport et politique
# pour couvrir des thématiques bien distinctes.
SELECTED_CLASSES = [
    "sci.med",
    "sci.space",
    "comp.graphics",
    "comp.os.ms-windows.misc",
    "rec.autos",
    "rec.sport.hockey",
    "talk.politics.guns",
    "talk.religion.misc",
]


def load_newsgroups(test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Charge le dataset 20 Newsgroups avec les 8 classes sélectionnées.

    Le paramètre remove=("headers", "footers", "quotes") supprime les métadonnées
    des emails (expéditeur, date, citations) pour forcer le modèle à apprendre
    sur le contenu réel, pas sur des artefacts de mise en forme.

    Retourne deux DataFrames avec colonnes 'text' et 'label'.
    """
    # scikit-learn utilise les splits train/test officiels du benchmark
    train_raw = fetch_20newsgroups(
        subset="train",
        categories=SELECTED_CLASSES,
        remove=("headers", "footers", "quotes"),
    )
    test_raw = fetch_20newsgroups(
        subset="test",
        categories=SELECTED_CLASSES,
        remove=("headers", "footers", "quotes"),
    )

    # Conversion en DataFrame avec labels textuels (plus lisible que les indices).
    # IMPORTANT : on indexe via train_raw.target_names (ordre réel de sklearn, qui
    # trie TOUJOURS les catégories par ordre alphabétique), et NON via SELECTED_CLASSES
    # (notre ordre custom). Sinon les labels sont décalés par rapport aux textes —
    # bug silencieux qui fausse toute la vérité-terrain.
    train_df = pd.DataFrame({
        "text": train_raw.data,
        "label": [train_raw.target_names[i] for i in train_raw.target],
    })
    test_df = pd.DataFrame({
        "text": test_raw.data,
        "label": [test_raw.target_names[i] for i in test_raw.target],
    })

    # Suppression des documents quasi-vides après nettoyage des métadonnées
    # (certains posts ne contiennent que des headers, il n'en reste rien après strip)
    train_df = train_df[train_df["text"].str.strip().str.len() > 20].reset_index(drop=True)
    test_df = test_df[test_df["text"].str.strip().str.len() > 20].reset_index(drop=True)

    return train_df, test_df


def get_label_mapping() -> Tuple[dict, dict]:
    """
    Retourne deux dictionnaires de correspondance label <-> entier.
    Nécessaire pour CamemBERT qui attend des entiers en entrée.

    On trie SELECTED_CLASSES par ordre alphabétique pour rester cohérent avec
    l'ordre des labels produits par load_newsgroups (qui suit target_names sklearn).
    """
    classes = sorted(SELECTED_CLASSES)
    label2id = {c: i for i, c in enumerate(classes)}
    id2label = {i: c for i, c in enumerate(classes)}
    return label2id, id2label
