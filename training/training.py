import argparse
import logging
import sys
from pathlib import Path

# Allow running as: python training/training.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training.trainer import run_st1_training

# Configure Logger 
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


LANG_DEFAULTS = {
    "eng": {
        "train_file": "dataset/subtask1/train/eng.csv",
        "dev_file": "dataset/subtask1/dev/eng.csv",
        "test_file": "dataset/subtask1/test/eng.csv",
        "model_name": "roberta-base",
        "output_dir": "outputs/train/eng",
        "result_dir": "result/public/eng",
    },
    "hau": {
        "train_file": "dataset/subtask1/train/hau.csv",
        "dev_file": "dataset/subtask1/dev/hau.csv",
        "test_file": "dataset/subtask1/test/hau.csv",
        "model_name": "Davlan/xlm-roberta-base-finetuned-hausa",
        "output_dir": "outputs/train/hau",
        "result_dir": "result/public/hau",
    },
}


def _resolve_path(path: str) -> Path:
    return Path(path).resolve()


def _ensure_dir(path: str) -> Path:
    output_dir = Path(path).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir

def main():
    parser = argparse.ArgumentParser(description="SemEval Task 9 Polarization Detection Training Script")
    parser.add_argument("--lang", type=str, default="eng", choices=["eng", "hau"], help="Language for defaults")
    
    # Data params
    parser.add_argument("--train_file", type=str, default=None, help="Path to training CSV file")
    parser.add_argument("--dev_file", type=str, default=None, help="Path to dev CSV for prediction export")
    parser.add_argument("--test_file", type=str, default=None, help="Path to test CSV for prediction export")
    parser.add_argument("--model_name", type=str, default=None, help="HuggingFace model name")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory")
    parser.add_argument("--result_dir", type=str, default=None, help="Directory for prediction CSV artifacts")
    
    # Training params
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.06)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--val_size", type=float, default=0.2)
    
    args = parser.parse_args()

    defaults = LANG_DEFAULTS[args.lang]

    default_train_file = defaults["train_file"]
    default_dev_file = defaults["dev_file"]
    default_test_file = defaults["test_file"]
    default_model_name = defaults["model_name"]
    default_output_dir = defaults["output_dir"]
    default_result_dir = defaults["result_dir"]

    args.train_file = args.train_file or default_train_file
    args.dev_file = args.dev_file or default_dev_file
    args.test_file = args.test_file or default_test_file
    args.model_name = args.model_name or default_model_name
    args.output_dir = args.output_dir or default_output_dir
    args.result_dir = args.result_dir or default_result_dir

    train_file_path = _resolve_path(args.train_file)
    output_dir_path = _ensure_dir(args.output_dir)
    result_dir_path = _ensure_dir(args.result_dir)
    dev_file_path = _resolve_path(args.dev_file)
    test_file_path = _resolve_path(args.test_file)
    
    logger.info(f"Starting training with args: {args}")

    run_st1_training(
        train_file_path=str(train_file_path),
        output_dir_path=output_dir_path,
        result_dir_path=result_dir_path,
        lang=args.lang,
        dev_file_path=str(dev_file_path),
        test_file_path=str(test_file_path),
        model_name=args.model_name,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        seed=args.seed,
        max_len=args.max_len,
        val_size=args.val_size,
    )

if __name__ == "__main__":
    main()
