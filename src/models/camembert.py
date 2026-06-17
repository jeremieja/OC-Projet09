"""
Stratégie 2 — Fine-tuning classique : CamemBERT.

CamemBERT est un modèle BERT entraîné sur 138 Go de texte français (Common Crawl).
On ajoute une tête de classification linéaire et on fine-tune l'ensemble
sur nos données étiquetées. Représente l'approche "data-driven" :
très bonnes performances si le club dispose de suffisamment de données.
"""
import time
from typing import List, Tuple

import numpy as np
from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

# Modèle de base depuis Hugging Face Hub (almanach = équipe créatrice de CamemBERT)
MODEL_NAME = "almanach/camembert-base"


def _tokenize(texts: List[str], labels: List[int], tokenizer, max_length: int) -> Dataset:
    """
    Convertit les textes bruts en tokens numériques attendus par CamemBERT.

    La tokenisation découpe chaque texte en sous-mots (subwords) et ajoute
    les tokens spéciaux [CLS] et [SEP]. truncation=True coupe les textes
    trop longs à max_length tokens (les mails dépassent rarement 256 tokens).
    """
    ds = Dataset.from_dict({"text": texts, "label": labels})
    return ds.map(
        lambda b: tokenizer(b["text"], truncation=True, max_length=max_length),
        batched=True,
        remove_columns=["text"],  # supprime la colonne texte brut après tokenisation
    )


def train_and_evaluate(
    train_texts: List[str],
    train_labels: List[str],
    test_texts: List[str],
    test_labels: List[str],
    label2id: dict,
    id2label: dict,
    output_dir: str = "models_saved/camembert_tmp",
    num_epochs: int = 3,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    max_length: int = 256,
) -> Tuple[dict, object]:
    """
    Fine-tune CamemBERT sur les données d'entraînement et évalue sur le test.

    Le fine-tuning modifie tous les poids du modèle (pas seulement la tête),
    ce qui lui permet de s'adapter au vocabulaire et au style des mails sportifs.
    Le taux d'apprentissage 2e-5 est standard pour le fine-tuning BERT
    (assez petit pour ne pas "oublier" les représentations pré-entraînées).
    """
    from src.evaluation.metrics import compute_metrics

    # Chargement du tokeniseur et du modèle pré-entraîné
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(label2id),
        label2id=label2id,
        id2label=id2label,
        ignore_mismatched_sizes=True,  # ignore le fait que la tête de classif change de taille
    )

    # Conversion des labels textuels en entiers (CamemBERT attend des entiers)
    train_labels_int = [label2id[l] for l in train_labels]
    test_labels_int = [label2id[l] for l in test_labels]

    train_ds = _tokenize(train_texts, train_labels_int, tokenizer, max_length)
    test_ds = _tokenize(test_texts, test_labels_int, tokenizer, max_length)

    # Le DataCollator padde dynamiquement les séquences à la longueur max du batch
    # (plus efficace que de padder à max_length fixe pour tous les exemples)
    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        eval_strategy="epoch",       # évalue à la fin de chaque epoch
        save_strategy="no",          # ne sauvegarde pas les checkpoints intermédiaires
        load_best_model_at_end=False, # on garde le modèle final (pas le meilleur)
        report_to="none",            # désactive WandB/TensorBoard
        logging_steps=50,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        data_collator=collator,
    )

    t0 = time.perf_counter()
    trainer.train()
    train_time = time.perf_counter() - t0

    # Inférence sur tout le test set d'un coup (plus rapide qu'appel par appel)
    t0 = time.perf_counter()
    predictions = trainer.predict(test_ds)
    inference_time_ms = (time.perf_counter() - t0) / len(test_texts) * 1000

    # argmax sur les logits pour obtenir la classe prédite
    preds_int = np.argmax(predictions.predictions, axis=-1)
    pred_labels = [id2label[p] for p in preds_int]

    results = compute_metrics(test_labels, pred_labels)
    results["train_time_s"] = round(train_time, 3)
    results["inference_time_ms"] = round(inference_time_ms, 4)

    return results, model
