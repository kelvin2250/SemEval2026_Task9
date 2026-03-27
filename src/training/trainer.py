import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)

from src.data.dataset import PolarizationDataset
from src.data.loading import load_csv_for_task
from src.training.metrics import compute_classification_metrics
from src.utils.processing import clean_text

logger = logging.getLogger(__name__)


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / np.sum(exp_values, axis=1, keepdims=True)


def _predict_and_save(
    trainer: Trainer,
    tokenizer: AutoTokenizer,
    data_file_path: str,
    output_pred_path: Path,
    output_prob_path: Path,
    max_len: int,
    compute_eval_metrics: bool,
) -> None:
    if not Path(data_file_path).exists():
        logger.warning("Prediction input not found: %s", data_file_path)
        return

    df = pd.read_csv(data_file_path)
    if "id" not in df.columns or "text" not in df.columns:
        logger.warning("Prediction input missing required columns id/text: %s", data_file_path)
        return

    df["text"] = df["text"].apply(clean_text)
    labels = df["polarization"].values if "polarization" in df.columns else np.zeros(len(df), dtype=int)
    dataset = PolarizationDataset(df["text"].values, labels, tokenizer, max_len)

    prediction_output = trainer.predict(dataset)
    logits = np.asarray(prediction_output.predictions)
    probs = _softmax(logits)
    preds = np.argmax(probs, axis=1)

    pred_df = pd.DataFrame({"id": df["id"], "polarization": preds})
    prob_df = pd.DataFrame({"id": df["id"], "prob_class1": probs[:, 1]})

    output_pred_path.parent.mkdir(parents=True, exist_ok=True)
    output_prob_path.parent.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(output_pred_path, index=False)
    prob_df.to_csv(output_prob_path, index=False)

    if compute_eval_metrics and "polarization" in df.columns:
        logger.info(
            "Saved predictions for %s with eval labels available.",
            data_file_path,
        )

    logger.info("Saved prediction file: %s", output_pred_path)
    logger.info("Saved probability file: %s", output_prob_path)


def run_st1_training(
    train_file_path: str,
    output_dir_path: Path,
    result_dir_path: Path,
    lang: str,
    dev_file_path: str,
    test_file_path: str,
    model_name: str,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    warmup_ratio: float,
    seed: int,
    max_len: int,
    val_size: float,
) -> None:
    set_seed(seed)

    logger.info("Loading data from %s", train_file_path)
    df = load_csv_for_task(train_file_path, task="st1")

    logger.info("Preprocessing text")
    df["text"] = df["text"].apply(clean_text)

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df["text"].values,
        df["polarization"].values,
        test_size=val_size,
        random_state=seed,
        stratify=df["polarization"].values,
    )

    logger.info("Loading tokenizer: %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    train_dataset = PolarizationDataset(train_texts, train_labels, tokenizer, max_len)
    val_dataset = PolarizationDataset(val_texts, val_labels, tokenizer, max_len)

    logger.info("Loading model: %s", model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    training_args = TrainingArguments(
        output_dir=str(output_dir_path),
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        num_train_epochs=epochs,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        logging_dir=str(output_dir_path / "logs"),
        seed=seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_classification_metrics,
    )

    trainer.train()

    final_model_dir = output_dir_path / "final_model"
    trainer.save_model(str(final_model_dir))
    tokenizer.save_pretrained(str(final_model_dir))
    logger.info("Training complete. Model saved to %s", final_model_dir)

    # Export prediction artifacts for downstream usage in result/public/<lang>.
    result_dir_path.mkdir(parents=True, exist_ok=True)
    _predict_and_save(
        trainer=trainer,
        tokenizer=tokenizer,
        data_file_path=dev_file_path,
        output_pred_path=result_dir_path / f"pred_{lang}_dev.csv",
        output_prob_path=result_dir_path / f"pred_{lang}_dev_probs.csv",
        max_len=max_len,
        compute_eval_metrics=True,
    )
    _predict_and_save(
        trainer=trainer,
        tokenizer=tokenizer,
        data_file_path=test_file_path,
        output_pred_path=result_dir_path / f"pred_{lang}.csv",
        output_prob_path=result_dir_path / f"pred_{lang}_probs.csv",
        max_len=max_len,
        compute_eval_metrics=False,
    )
