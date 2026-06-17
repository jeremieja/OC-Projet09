"""
Generates the synthetic sports club emails dataset via Anthropic API (Claude).
Produces ~1800 emails across 8 business categories with controlled variation
in length, tone, urgency, sender persona, and sport.

Uses prompt caching on the system prompt to reduce cost across the 1800 calls.
"""
import json
import os
import random
from pathlib import Path

import anthropic
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

RAW_PATH = Path(__file__).parents[2] / "data" / "generated" / "emails_raw.json"
PROCESSED_PATH = Path(__file__).parents[2] / "data" / "processed" / "emails.csv"

CATEGORIES = {
    "inscription": (
        "Demande d'inscription d'un joueur, renouvellement de licence, "
        "questions sur les modalités d'adhésion, les tarifs ou les documents requis."
    ),
    "sponsor": (
        "Proposition de partenariat ou de sponsoring d'une entreprise locale, "
        "négociation de contreparties, demande de visibilité."
    ),
    "arbitrage-officiels": (
        "Désignation d'arbitres, confirmation de disponibilités, "
        "indemnités d'arbitrage, problèmes ou incidents lors d'un match officiel."
    ),
    "parent": (
        "Message d'un parent concernant son enfant : absence à l'entraînement, "
        "blessure, transport, comportement, questions sur le programme."
    ),
    "federation": (
        "Communication officielle émanant d'une fédération sportive : "
        "convocation, qualification, mise à jour réglementaire, championnat."
    ),
    "logistique-matchday": (
        "Organisation pratique d'un match à domicile ou en déplacement : "
        "réservation de terrain, vestiaires, buvette, feuille de match, arbitres."
    ),
    "indemnites-coachs": (
        "Demande ou confirmation de paiement d'indemnités kilométriques "
        "ou de vacation pour un entraîneur ou un dirigeant bénévole."
    ),
    "divers-administratif": (
        "Demande de documents administratifs, attestation, certificat médical, "
        "convocation à une AG, budget annuel, demande de subvention mairie."
    ),
}

SPORTS = [
    "football", "basketball", "handball", "rugby", "volleyball",
    "tennis de table", "natation", "athlétisme", "judo", "badminton",
]
LENGTHS = [
    "très court (2-4 lignes)",
    "court (5-7 lignes)",
    "moyen (8-12 lignes)",
    "long (15-20 lignes)",
]
TONES = ["formel et poli", "informel et direct", "urgent", "hésitant et maladroit"]
PERSONAS = [
    "un parent d'enfant licencié",
    "un entraîneur bénévole",
    "un dirigeant du club",
    "un joueur adulte",
    "un représentant fédéral",
    "un responsable administratif municipal",
    "un responsable d'entreprise locale",
]

# Cached system prompt — sent once, reused across all calls (prompt caching)
_SYSTEM_PROMPT = (
    "Tu es un générateur de données d'entraînement pour un système de classification "
    "de mails de clubs sportifs français. Tu génères des mails authentiques et variés "
    "selon les instructions de l'utilisateur. "
    "Réponds UNIQUEMENT avec le corps du mail, sans balise, sans commentaire."
)


def _build_user_prompt(
    category: str, description: str, sport: str, length: str, tone: str, persona: str
) -> str:
    return (
        f"Génère un mail envoyé à un club de {sport}.\n"
        f"Catégorie : {category} — {description}\n"
        f"Longueur : {length}\n"
        f"Ton : {tone}\n"
        f"Expéditeur : {persona}\n\n"
        "Le mail doit être en français, réaliste, avec les imperfections naturelles "
        "d'un vrai message (fautes légères, formulations maladroites selon le ton)."
    )


def _generate_category(
    client: anthropic.Anthropic,
    category: str,
    description: str,
    n_samples: int,
    model: str,
    rng: random.Random,
) -> list[dict]:
    results = []

    for _ in tqdm(range(n_samples), desc=category, leave=False):
        sport = rng.choice(SPORTS)
        length = rng.choice(LENGTHS)
        tone = rng.choice(TONES)
        persona = rng.choice(PERSONAS)

        user_content = _build_user_prompt(category, description, sport, length, tone, persona)

        try:
            response = client.messages.create(
                model=model,
                max_tokens=500,
                system=[
                    {
                        "type": "text",
                        "text": _SYSTEM_PROMPT,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                messages=[{"role": "user", "content": user_content}],
            )
            text = response.content[0].text.strip()
            results.append({
                "text": text,
                "label": category,
                "sport": sport,
                "length_target": length,
                "tone": tone,
                "persona": persona,
            })
        except Exception as exc:
            tqdm.write(f"  [WARN] Failed sample for '{category}': {exc}")

    return results


def generate_full_dataset(
    n_per_class: int = 225,
    model: str = "claude-haiku-4-5-20251001",
) -> None:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "ANTHROPIC_API_KEY not set. Copy .env.example to .env and fill it in."
        )

    client = anthropic.Anthropic(api_key=api_key)
    rng = random.Random(42)
    all_emails = []

    RAW_PATH.parent.mkdir(parents=True, exist_ok=True)

    for category, description in CATEGORIES.items():
        category_path = RAW_PATH.parent / f"{category}.json"

        # Skip already-generated categories (resume support)
        if category_path.exists():
            with open(category_path, encoding="utf-8") as f:
                existing = json.load(f)
            print(f"[{category}] already done ({len(existing)} samples), skipping.")
            all_emails.extend(existing)
            continue

        print(f"\n[{category}] generating {n_per_class} samples...")
        emails = _generate_category(client, category, description, n_per_class, model, rng)
        all_emails.extend(emails)
        print(f"  -> {len(emails)} generated (target: {n_per_class})")

        # Save per-category immediately so progress is never lost
        with open(category_path, "w", encoding="utf-8") as f:
            json.dump(emails, f, ensure_ascii=False, indent=2)

    if not all_emails:
        raise RuntimeError("No emails were generated. Check your ANTHROPIC_API_KEY and model name.")

    with open(RAW_PATH, "w", encoding="utf-8") as f:
        json.dump(all_emails, f, ensure_ascii=False, indent=2)
    print(f"\nRaw JSON saved: {RAW_PATH}  ({len(all_emails)} total)")

    df = pd.DataFrame(all_emails)[["text", "label", "sport", "tone", "persona"]]
    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_PATH, index=False, encoding="utf-8")
    print(f"Processed CSV saved: {PROCESSED_PATH}")
