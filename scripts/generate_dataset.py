"""
Point d'entrée pour la génération du dataset synthétique de mails de clubs sportifs.

Ce script appelle src/data/generate_emails.py qui effectue les appels à l'API Anthropic.
La génération est reprise automatiquement si elle a été interrompue : les catégories
déjà générées sont détectées via leurs fichiers JSON dans data/generated/ et sautées.

Durée estimée : ~1h20 pour 225 emails × 8 catégories avec claude-haiku.
Coût estimé : ~1.50-2.00 $ avec prompt caching activé.

Usage :
    python scripts/generate_dataset.py                        # génération complète
    python scripts/generate_dataset.py --n-per-class 10      # test rapide (80 emails)
    python scripts/generate_dataset.py --model claude-sonnet-4-6  # modèle plus puissant
"""
import argparse
import sys
from pathlib import Path

# Permet d'importer les modules src/ sans installer le projet comme package
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.generate_emails import generate_full_dataset


def main():
    parser = argparse.ArgumentParser(description="Génère le dataset de mails synthétiques")
    parser.add_argument(
        "--n-per-class", type=int, default=225,
        help="Nombre d'emails à générer par catégorie (défaut : 225, soit 1800 au total)"
    )
    parser.add_argument(
        "--model", type=str, default="claude-haiku-4-5-20251001",
        help="Modèle Anthropic utilisé pour la génération (défaut : Haiku, le moins cher)"
    )
    args = parser.parse_args()

    print(f"Generating {args.n_per_class} emails/class with {args.model}")
    generate_full_dataset(n_per_class=args.n_per_class, model=args.model)
    print("Dataset generation complete.")


if __name__ == "__main__":
    main()
