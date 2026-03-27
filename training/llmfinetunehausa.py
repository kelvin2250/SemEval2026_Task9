import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

TEXT_COL = "text"
LABEL_COL = "polarization"

LANG_DEFAULTS = {
    "eng": {
        "train_file": "dataset/subtask1/train/eng.csv",
        "output_dir": "outputs/train_llm/eng",
    },
    "hau": {
        "train_file": "dataset/subtask1/train/hau.csv",
        "output_dir": "outputs/train_llm/hau",
    },
}


def compute_metrics(eval_pred):
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro"),
        "f1_weighted": f1_score(labels, preds, average="weighted"),
        "precision": precision_score(labels, preds, average="binary", zero_division=0),
        "recall": recall_score(labels, preds, average="binary", zero_division=0),
    }


def _resolve(path: str) -> Path:
    return Path(path).resolve()


def _prepare_tokenized_dataset(df: pd.DataFrame, tokenizer, max_len: int) -> Dataset:
    dataset = Dataset.from_pandas(df.reset_index(drop=True))

    def tokenize_fn(batch):
        return tokenizer(
            batch[TEXT_COL],
            truncation=True,
            max_length=max_len,
        )

    remove_columns = [col for col in [TEXT_COL, "id"] if col in dataset.column_names]
    tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=remove_columns)

    if LABEL_COL in tokenized.column_names:
        tokenized = tokenized.rename_column(LABEL_COL, "labels")

    return tokenized


def main():
    parser = argparse.ArgumentParser(description="QLoRA fine-tuning for SemEval Task 9 (English/Hausa)")
    parser.add_argument("--lang", type=str, choices=["eng", "hau"], default="eng")
    parser.add_argument("--train_file", type=str, default=None, help="Training CSV path")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory")
    parser.add_argument("--hf_token", type=str, required=True, help="Hugging Face token")
    parser.add_argument("--model_id", type=str, default="meta-llama/Meta-Llama-3.1-8B")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_len", type=int, default=256)
    parser.add_argument("--val_size", type=float, default=0.2)
    args = parser.parse_args()

    defaults = LANG_DEFAULTS[args.lang]
    train_file = _resolve(args.train_file or defaults["train_file"])
    output_dir = _resolve(args.output_dir or defaults["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)

    logger.info("Language: %s", args.lang)
    logger.info("Loading data from %s", train_file)
    df = pd.read_csv(train_file)

    df_train, df_val = train_test_split(
        df,
        test_size=args.val_size,
        random_state=args.seed,
        stratify=df[LABEL_COL],
    )

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    logger.info("Loading tokenizer: %s", args.model_id)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=True, token=args.hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info("Loading model: %s", args.model_id)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_id,
        num_labels=2,
        device_map="auto",
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        token=args.hf_token,
    )

    model.config.pad_token_id = tokenizer.pad_token_id
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="SEQ_CLS",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    train_tok = _prepare_tokenized_dataset(df_train, tokenizer, args.max_len)
    val_tok = _prepare_tokenized_dataset(df_val, tokenizer, args.max_len)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        overwrite_output_dir=True,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        warmup_ratio=0.05,
        weight_decay=0.01,
        evaluation_strategy="steps",
        eval_steps=80,
        save_strategy="steps",
        save_steps=80,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        logging_steps=20,
        report_to="none",
        fp16=False,
        bf16=True,
        optim="paged_adamw_8bit",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tok,
        eval_dataset=val_tok,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=4)],
    )

    logger.info("Starting LLM fine-tuning...")
    trainer.train()

    logger.info("Saving model to %s", output_dir)
    trainer.model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))


if __name__ == "__main__":
    main()
