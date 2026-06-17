"""
Configuration du logger partagé entre tous les scripts d'expérience.

Format des messages : HH:MM:SS | nom_script | NIVEAU | message
Exemple : 22:26:04 | run_baseline | INFO | Dataset: newsgroups
"""
import logging
import sys


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Crée ou récupère un logger nommé avec un handler stdout.

    La vérification 'if not logger.handlers' évite d'ajouter
    plusieurs handlers si la fonction est appelée plusieurs fois
    avec le même nom (ce qui doublerait les messages).
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(
            "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
            datefmt="%H:%M:%S",
        ))
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger
