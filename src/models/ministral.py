"""
Stratégie 4 — Zero/few-shot via LLM API : Ministral 8B (Mistral AI).

Approche "LLM-as-classifier" : on n'entraîne pas le modèle, on lui demande
directement de classifier via une instruction en langage naturel.

Deux modes testés :
  - zero-shot : instruction seule, aucun exemple (0 donnée étiquetée nécessaire)
  - few-shot  : 8 exemples en in-context learning (le modèle voit des exemples
                directement dans le prompt, sans modifier ses poids)

Avantage principal : déploiement instantané chez un nouveau club sans aucune donnée.
Inconvénient : latence ~500ms-2s par mail et coût API récurrent.

Note technique : on appelle l'API REST Mistral directement via httpx plutôt que
le SDK `mistralai`. Le package SDK est systématiquement corrompu à l'installation
sur cette machine (Windows Defender supprime les fichiers principaux du package).
L'endpoint REST est simple et stable, ce contournement est donc robuste.
"""
import json
import os
import time
from typing import List, Optional, Tuple

import httpx
from dotenv import load_dotenv

# tqdm est optionnel : utile pour la barre de progression CLI, inutile dans le
# dashboard. On fournit un fallback no-op pour que le module fonctionne même
# si tqdm n'est pas installé (ex: environnement de déploiement minimal).
try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    class tqdm:  # type: ignore
        """Remplacement minimal de tqdm quand le paquet est absent."""
        def __init__(self, iterable=None, **kwargs):
            self._iterable = iterable if iterable is not None else []
        def __iter__(self):
            return iter(self._iterable)
        def set_postfix(self, *args, **kwargs):
            pass
        @staticmethod
        def write(msg, *args, **kwargs):
            print(msg)

load_dotenv()

MODEL_ID = "ministral-8b-latest"
API_URL = "https://api.mistral.ai/v1/chat/completions"

# Coût estimé par appel (input ~200 tokens + output ~20 tokens @ $0.10/M tokens)
_COST_PER_CALL_USD = 0.10 / 1_000_000 * 220

# Prompt système : format JSON strict pour faciliter le parsing automatique
# et forcer le modèle à exprimer sa confiance (utile pour le système hybride).
_SYSTEM_PROMPT = (
    "Tu es un classificateur de {domain}. "
    "Classe chaque texte dans exactement une des catégories suivantes :\n{categories}\n\n"
    'Réponds UNIQUEMENT avec un objet JSON : {{"categorie": "<nom>", "confiance": <float 0-1>}}'
    "\nAucune explication."
)


# Domaine par défaut (cas d'usage métier). run_ministral.py passe un domaine adapté
# à chaque dataset pour ne pas biaiser le modèle (ex: newsgroups = forums de discussion).
DEFAULT_DOMAIN = "mails de clubs sportifs français"


def _system(categories: List[str], domain: str = DEFAULT_DOMAIN) -> str:
    """Injecte le domaine et la liste des catégories dans le prompt système."""
    return _SYSTEM_PROMPT.format(
        domain=domain,
        categories="\n".join(f"- {c}" for c in categories),
    )


def _zero_shot_messages(text: str, categories: List[str], domain: str = DEFAULT_DOMAIN) -> list:
    """Messages pour une classification zero-shot (instruction seule, aucun exemple)."""
    return [
        {"role": "system", "content": _system(categories, domain)},
        {"role": "user", "content": f"Texte :\n\n{text}"},
    ]


def _few_shot_messages(text: str, categories: List[str], examples: List[dict],
                       domain: str = DEFAULT_DOMAIN) -> list:
    """
    Messages pour une classification few-shot (in-context learning).
    Les exemples sont insérés comme des échanges user/assistant simulés :
    le modèle apprend le format attendu avant de voir le vrai texte.
    """
    messages = [{"role": "system", "content": _system(categories, domain)}]
    for ex in examples:
        messages.append({"role": "user", "content": f"Texte :\n\n{ex['text']}"})
        messages.append({
            "role": "assistant",
            "content": json.dumps({"categorie": ex["label"], "confiance": 1.0}, ensure_ascii=False),
        })
    messages.append({"role": "user", "content": f"Texte :\n\n{text}"})
    return messages


def _parse_response(raw: str, categories: List[str]) -> Tuple[str, float]:
    """
    Parse la réponse JSON du modèle (label + confiance).
    En cas d'échec, tente une correspondance approximative par sous-chaîne.
    """
    try:
        parsed = json.loads(raw)
        label = parsed.get("categorie", "").strip()
        confidence = float(parsed.get("confiance", 0.5))
    except (json.JSONDecodeError, ValueError, TypeError):
        label = ""
        confidence = 0.5

    if label not in categories:
        raw_lower = raw.lower()
        label = next((c for c in categories if c.lower() in raw_lower), categories[0])

    return label, confidence


def _call_with_retry(
    client: httpx.Client,
    api_key: str,
    messages: list,
    max_tokens: int,
    temperature: float,
    max_retries: int = 3,
) -> Tuple[str, float]:
    """
    Appel API REST avec retry exponentiel en cas d'erreur réseau ou rate limit.
    Retourne (réponse brute, latence en ms).
    """
    payload = {
        "model": MODEL_ID,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    for attempt in range(max_retries):
        try:
            t0 = time.perf_counter()
            resp = client.post(API_URL, json=payload, headers=headers, timeout=60.0)
            resp.raise_for_status()
            latency_ms = (time.perf_counter() - t0) * 1000
            content = resp.json()["choices"][0]["message"]["content"].strip()
            return content, latency_ms
        except (httpx.HTTPError, KeyError, json.JSONDecodeError) as e:
            if attempt < max_retries - 1:
                wait = 2 ** attempt  # backoff exponentiel : 1s, 2s, 4s
                tqdm.write(f"  [RETRY {attempt+1}/{max_retries}] {e} — attente {wait}s")
                time.sleep(wait)
            else:
                raise


def classify_batch(
    texts: List[str],
    categories: List[str],
    mode: str = "zero_shot",
    examples: Optional[List[dict]] = None,
    api_key: Optional[str] = None,
    max_tokens: int = 60,
    temperature: float = 0.0,
    show_progress: bool = True,
    domain: str = DEFAULT_DOMAIN,
) -> Tuple[List[str], List[float], List[float]]:
    """
    Classifie une liste de textes via l'API Ministral avec barre de progression.

    temperature=0.0 : réponses déterministes, garantit la reproductibilité.
    max_tokens=60 : la réponse JSON attendue est courte, limite coût et latence.

    Retourne : (prédictions, confiances, latences en ms)
    """
    api_key = api_key or os.getenv("MISTRAL_API_KEY")
    # Fallback : sur Streamlit Cloud, la clé est fournie via st.secrets et non l'env.
    if not api_key:
        try:
            import streamlit as st
            api_key = st.secrets.get("MISTRAL_API_KEY")
        except Exception:
            api_key = None
    if not api_key:
        raise EnvironmentError("MISTRAL_API_KEY non définie (ni dans .env, ni dans st.secrets).")

    preds, confidences, latencies = [], [], []
    total_cost = 0.0

    desc = f"Ministral {mode}"
    bar = tqdm(texts, desc=desc, unit="mail") if show_progress else texts

    # Un seul client httpx réutilisé (connexions persistantes = plus rapide)
    with httpx.Client() as client:
        for text in bar:
            messages = (
                _few_shot_messages(text, categories, examples, domain)
                if mode == "few_shot" and examples
                else _zero_shot_messages(text, categories, domain)
            )

            raw, latency_ms = _call_with_retry(client, api_key, messages, max_tokens, temperature)
            label, confidence = _parse_response(raw, categories)

            preds.append(label)
            confidences.append(confidence)
            latencies.append(latency_ms)
            total_cost += _COST_PER_CALL_USD

            if show_progress:
                bar.set_postfix(
                    latence=f"{latency_ms:.0f}ms",
                    cout=f"${total_cost:.4f}",
                    refresh=False,
                )

    return preds, confidences, latencies


def evaluate(
    texts: List[str],
    true_labels: List[str],
    categories: List[str],
    mode: str = "zero_shot",
    examples: Optional[List[dict]] = None,
    api_key: Optional[str] = None,
    domain: str = DEFAULT_DOMAIN,
) -> Tuple[dict, List[str], List[float]]:
    """
    Évalue Ministral sur un jeu de test complet et retourne les métriques.
    Point d'entrée principal utilisé par scripts/run_ministral.py.
    """
    from src.evaluation.metrics import compute_metrics

    preds, confidences, latencies = classify_batch(
        texts, categories, mode=mode, examples=examples, api_key=api_key, domain=domain
    )

    results = compute_metrics(true_labels, preds)
    results["inference_time_ms"] = round(sum(latencies) / len(latencies), 2)
    results["p50_latency_ms"] = round(sorted(latencies)[len(latencies) // 2], 2)
    results["p95_latency_ms"] = round(sorted(latencies)[int(len(latencies) * 0.95)], 2)
    results["total_cost_usd"] = round(len(texts) * _COST_PER_CALL_USD, 6)
    results["mode"] = mode

    return results, preds, confidences
