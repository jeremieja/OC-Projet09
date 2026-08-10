"""
Construit les archives de rendu.

  Livrable 2 — notebooks, code de préparation et d'entraînement, jeu de données
  Livrable 4 — code du tableau de bord déployé

Les scripts qui fabriquent les documents de rendu (note méthodologique, diaporama,
note de révision) sont exclus : ils ne font pas partie de la preuve de concept.

Usage : python scripts/make_livrables.py
"""
import zipfile
from pathlib import Path

RACINE = Path(__file__).parents[1]
MOIS = "082026"

# Outillage de rédaction : hors périmètre du POC, exclu des deux archives.
SCRIPTS_EXCLUS = {
    "generate_plan_previsionnel.py",
    "generate_note_methodo.py",
    "generate_biblio_doc.py",
    "generate_revision_soutenance.py",
    "generate_presentation.py",
    "eclater_slide_sources.py",
    "make_livrables.py",
}

IGNORES = {"__pycache__", ".ipynb_checkpoints", ".git", ".venv"}


def a_garder(chemin: Path) -> bool:
    if any(p in IGNORES for p in chemin.parts):
        return False
    if chemin.suffix in {".pyc", ".pyo"}:
        return False
    if chemin.parent.name == "scripts" and chemin.name in SCRIPTS_EXCLUS:
        return False
    return True


def ajouter(zf: zipfile.ZipFile, motif: str, base: Path = RACINE) -> int:
    """Ajoute les fichiers correspondant au motif, chemins relatifs à la racine."""
    n = 0
    for f in sorted(base.glob(motif)):
        if f.is_file() and a_garder(f):
            zf.write(f, f.relative_to(RACINE))
            n += 1
    return n


def construire(nom: str, contenus: list[tuple[str, str]]) -> Path:
    sortie = RACINE / nom
    total = 0
    with zipfile.ZipFile(sortie, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        print(f"\n{nom}")
        for motif, libelle in contenus:
            n = ajouter(zf, motif)
            total += n
            etat = f"{n:4d} fichier(s)" if n else "   ABSENT (!)"
            print(f"   {etat}  {libelle}")
    print(f"   -> {total} fichiers, {sortie.stat().st_size / 1e6:.1f} Mo")
    return sortie


def main():
    # ── Livrable 2 ───────────────────────────────────────────────────────
    # Les notebooks importent src/ et lisent les résultats d'expériences :
    # sans eux, ni le code d'entraînement ni les analyses ne sont exploitables.
    construire(f"Jambon_Jeremie_2_notebook_{MOIS}.zip", [
        ("notebooks/*.ipynb", "les quatre notebooks d'analyse"),
        ("src/**/*.py", "préparation des données et implémentation des modèles"),
        ("scripts/*.py", "points d'entrée des expériences"),
        ("data/processed/emails.csv", "le jeu de données métier"),
        ("data/generated/*.json", "les mails bruts, catégorie par catégorie"),
        ("results/runs/*.json", "les résultats des expériences"),
        ("results/aggregated/*.csv", "résultats agrégés et prédictions"),
        ("requirements.txt", "dépendances légères"),
        ("requirements-dev.txt", "dépendances d'entraînement"),
        ("README.md", "présentation du projet"),
    ])

    # ── Livrable 4 ───────────────────────────────────────────────────────
    # Le tableau de bord importe src/ et charge le modèle, le corpus et les
    # résultats agrégés : les quatre sont indispensables pour qu'il démarre.
    construire(f"Jambon_Jeremie_4_dashboard_code_{MOIS}.zip", [
        ("dashboard/*.py", "l'application Streamlit"),
        ("dashboard/README.md", "notice de déploiement"),
        ("dashboard/requirements.txt", "dépendances de l'application"),
        ("src/**/*.py", "modules importés par l'application"),
        ("models_saved/deploy_*.joblib", "le modèle servi en ligne"),
        ("data/processed/emails.csv", "le corpus affiché et prédit"),
        ("results/runs/*.json", "les résultats des courbes d'apprentissage"),
        ("results/aggregated/results_aggregated.csv", "le tableau récapitulatif"),
        (".streamlit/*", "configuration et modèle de secrets"),
        ("requirements.txt", "dépendances du déploiement"),
        ("README.md", "présentation du projet"),
    ])


if __name__ == "__main__":
    main()
