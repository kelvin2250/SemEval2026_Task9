import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


# Paper weights (Table 4) and standardized result layout.
ENSEMBLE_DEFAULTS: Dict[str, Dict[str, object]] = {
    "hau": {
        "files_with_weights": [
            ("result/public/hau/df_1.csv", 0.20),
            ("result/public/hau/df_2.csv", 0.20),
            ("result/public/hau/df_3.csv", 0.30),
            ("result/public/hau/original.csv", 0.30),
        ],
        "output_pred": "result/ensemble/hau/pred_hau_ensemble.csv",
        "output_prob": "result/ensemble/hau/pred_hau_ensemble_prob.csv",
        "threshold": 0.5,
    },
    "eng": {
        "files_with_weights": [
            ("result/public/eng/llm_500_full.csv", 0.40),
            ("result/public/eng/llm_850.csv", 0.25),
            ("result/public/eng/llm_fulldata.csv", 0.10),
            ("result/public/eng/rbase-300-full1.csv", 0.15),
            ("result/public/eng/rbase-fulldata.csv", 0.10),
        ],
        "output_pred": "result/ensemble/eng/pred_eng_ensemble.csv",
        "output_prob": "result/ensemble/eng/pred_eng_ensemble_prob.csv",
        "threshold": 0.5,
    },
}


def weighted_soft_voting(
    files_with_weights: List[Tuple[str, float]],
    output_pred: str,
    output_prob: str,
    threshold: float = 0.5,
) -> None:
    merged_df = None
    total_weight = 0.0

    for i, (file_path, weight) in enumerate(files_with_weights):
        path = Path(file_path)
        if not path.exists():
            print(f"Warning: File not found: {path}. Skipping.")
            continue

        df = pd.read_csv(path)
        if "id" not in df.columns or "prob_class1" not in df.columns:
            print(f"Error: File {path} missing 'id' or 'prob_class1'. Skipping.")
            continue

        df["id"] = df["id"].astype(str)

        if merged_df is None:
            merged_df = df[["id"]].copy()
            merged_df["weighted_sum_prob"] = 0.0

        temp_df = df[["id", "prob_class1"]].rename(columns={"prob_class1": f"prob_{i}"})
        merged_df = merged_df.merge(temp_df, on="id", how="inner")
        merged_df["weighted_sum_prob"] += merged_df[f"prob_{i}"] * weight
        total_weight += weight

    if merged_df is None or total_weight == 0:
        raise ValueError("No valid input files were processed for ensemble.")

    merged_df["final_prob"] = merged_df["weighted_sum_prob"] / total_weight
    merged_df["polarization"] = (merged_df["final_prob"] >= threshold).astype(int)

    pred_df = merged_df[["id", "polarization"]]
    prob_df = merged_df[["id", "final_prob"]].rename(columns={"final_prob": "prob_class1"})

    output_pred_path = Path(output_pred)
    output_prob_path = Path(output_prob)
    output_pred_path.parent.mkdir(parents=True, exist_ok=True)
    output_prob_path.parent.mkdir(parents=True, exist_ok=True)

    pred_df.to_csv(output_pred_path, index=False)
    prob_df.to_csv(output_prob_path, index=False)


def build_ensemble_inputs(lang: str):
    lang_cfg = ENSEMBLE_DEFAULTS[lang]
    files_with_weights = [(str(Path(path).resolve()), float(weight)) for path, weight in lang_cfg["files_with_weights"]]
    output_pred = str(Path(lang_cfg["output_pred"]).resolve())
    output_prob = str(Path(lang_cfg["output_prob"]).resolve())
    threshold = float(lang_cfg["threshold"])
    return files_with_weights, output_pred, output_prob, threshold


def main():
    parser = argparse.ArgumentParser(description="Weighted soft-voting ensemble (single-file mode)")
    parser.add_argument("--lang", type=str, default="hau", choices=["eng", "hau"], help="Language setup")
    parser.add_argument("--threshold", type=float, default=None, help="Override threshold (default from paper: 0.5)")
    parser.add_argument("--output_pred", type=str, default=None, help="Override prediction output path")
    parser.add_argument("--output_prob", type=str, default=None, help="Override probability output path")
    args = parser.parse_args()

    files_with_weights, output_pred, output_prob, threshold = build_ensemble_inputs(args.lang)
    if args.threshold is not None:
        threshold = args.threshold
    if args.output_pred:
        output_pred = str(Path(args.output_pred).resolve())
    if args.output_prob:
        output_prob = str(Path(args.output_prob).resolve())

    weighted_soft_voting(files_with_weights, output_pred, output_prob, threshold)


if __name__ == "__main__":
    main()
