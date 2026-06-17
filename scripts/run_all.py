"""
Orchestrateur du pipeline expérimental complet.

Lance les 6 étapes dans l'ordre logique en utilisant subprocess pour que
chaque script s'exécute dans son propre processus (isolation mémoire,
logs propres, code de retour exploitable).

Ordre d'exécution :
  1. generate_dataset.py  — génère les mails synthétiques (skippé si CSV existe)
  2. run_baseline.py      — TF-IDF + LR (rapide, ~2 min)
  3. run_camembert.py     — CamemBERT fine-tuning (~30-45 min/dataset)
  4. run_setfit.py        — SetFit × 3 backbones (~20-30 min/dataset)
  5. run_ministral.py     — Ministral API (~20 min, coût ~$0.07)
  6. run_hybrid.py        — Système hybride (~20 min, inclut re-entraînement SetFit)

En cas d'échec d'une étape, le pipeline s'arrête avec le code d'erreur
du script fautif (comportement fail-fast).

Usage :
    python scripts/run_all.py                          # pipeline complet
    python scripts/run_all.py --skip-generate          # si emails.csv déjà présent
    python scripts/run_all.py --dataset newsgroups     # uniquement 20 Newsgroups
"""
import argparse
import subprocess
import sys
from pathlib import Path

SCRIPTS = Path(__file__).parent
# Chemin vers le CSV des emails : sert à détecter si la génération est nécessaire
DATA_PATH = Path(__file__).parents[1] / "data" / "processed" / "emails.csv"


def run(script: str, extra_args: list = None) -> None:
    """
    Lance un script Python dans un sous-processus et attend sa fin.
    Arrête tout le pipeline si le script retourne un code d'erreur non nul.
    sys.executable garantit que le même interpréteur (et .venv) est utilisé.
    """
    cmd = [sys.executable, str(SCRIPTS / script)] + (extra_args or [])
    print(f"\n{'='*60}\nRunning: {' '.join(cmd)}\n{'='*60}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"[ERROR] {script} failed with code {result.returncode}")
        sys.exit(result.returncode)


def main():
    parser = argparse.ArgumentParser(description="Lance le pipeline expérimental complet")
    parser.add_argument("--dataset", choices=["newsgroups", "emails", "all"], default="all")
    parser.add_argument(
        "--skip-generate", action="store_true",
        help="Saute la génération du dataset (emails.csv déjà présent)"
    )
    args = parser.parse_args()

    ds_args = ["--dataset", args.dataset]

    # Génération du dataset emails si nécessaire (détection automatique)
    if not args.skip_generate and not DATA_PATH.exists():
        run("generate_dataset.py")

    # Exécution séquentielle des expériences
    run("run_baseline.py", ds_args)
    run("run_camembert.py", ds_args)
    run("run_setfit.py", ds_args)
    run("run_ministral.py", ds_args)
    run("run_hybrid.py", ds_args)

    print("\nAll experiments complete. Results saved in results/runs/")


if __name__ == "__main__":
    main()
