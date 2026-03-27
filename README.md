# SemEval 2026 Task 9: Multilingual Polarization Detection (AJT Pipeline)

This repository contains our implementation for **SemEval 2026 Task 9 (POLARDETECT)**.
The system is designed around an **AJT workflow**:

1. **Augment**: generate synthetic training samples.
2. **Judge**: filter generated samples by quality.
3. **Train**: train ST1 classifiers for English and Hausa.
4. **Ensemble**: combine model probabilities with weighted soft voting.

## Overview

Current implementation focus:

- **ST1 (binary polarization classification)** for English and Hausa.
- Augmentation for:
   - English ST1 (`label 0` and `label 1` generation).
   - Hausa ST2-style multi-label augmentation used for downstream ST1 training mix.
- Weighted ensemble based on paper configuration (Table 4 style weights).

## Project Structure

```text
dataset/
   subtask1/
   subtask2/
   augmentation/
      english/
      hausa/

augmentation/
   augmentEnglish.py
   augmentHausa.py

training/
   training.py
   llmfinetunehausa.py

ensemble/
   ensemble.py

result/
   public/
      eng/
      hau/
   ensemble/
      eng/
      hau/

src/
   augmentation/
   data/
   training/
   utils/

tests/
```

## Requirements

- Python 3.10+ (tested with Python 3.13)
- Install dependencies:

```bash
pip install -r requirements.txt
```

## Data Requirements

### ST1 expected files

- `dataset/subtask1/train/eng.csv`
- `dataset/subtask1/dev/eng.csv`
- `dataset/subtask1/test/eng.csv`
- `dataset/subtask1/train/hau.csv`
- `dataset/subtask1/dev/hau.csv`
- `dataset/subtask1/test/hau.csv`

### ST2 expected files (for Hausa augmentation)

- `dataset/subtask2/train/hau.csv`

### Schema

- ST1 columns: `id, text, polarization`
- ST2 columns: `id, text, political, racial/ethnic, religious, gender/sexual, other`

## End-to-End Pipeline

## 1. Augmentation + Judge

### 1.1 English augmentation

Generate label-0 or label-1 augmented data:

```bash
python augmentation/augmentEnglish.py --label 0
python augmentation/augmentEnglish.py --label 1
```

Default outputs:

- `dataset/augmentation/english/augmented_data_label0.csv`
- `dataset/augmentation/english/augmented_data_label1.csv`
- Judge score files are saved with suffix `_judge_scores.csv`

### 1.2 Hausa augmentation

```bash
python augmentation/augmentHausa.py
```

Default output:

- `dataset/augmentation/hausa/df_3.csv`
- Judge score file: `dataset/augmentation/hausa/df_3_judge_scores.csv`

## 2. Training + Prediction Export

Train ST1 model and automatically export prediction CSV files for both dev and test.

### 2.1 English

```bash
python training/training.py --lang eng
```

### 2.2 Hausa

```bash
python training/training.py --lang hau
```

What training writes:

- Model directory:
   - `outputs/train/eng/final_model`
   - `outputs/train/hau/final_model`
- Prediction artifacts:
   - `result/public/eng/pred_eng_dev.csv`
   - `result/public/eng/pred_eng_dev_probs.csv`
   - `result/public/eng/pred_eng.csv`
   - `result/public/eng/pred_eng_probs.csv`
   - `result/public/hau/pred_hau_dev.csv`
   - `result/public/hau/pred_hau_dev_probs.csv`
   - `result/public/hau/pred_hau.csv`
   - `result/public/hau/pred_hau_probs.csv`

## 3. Ensemble

Run weighted soft voting:

```bash
python ensemble/ensemble.py --lang eng
python ensemble/ensemble.py --lang hau
```

Default ensemble inputs:

- English uses:
   - `result/public/eng/llm_500_full.csv`
   - `result/public/eng/llm_850.csv`
   - `result/public/eng/llm_fulldata.csv`
   - `result/public/eng/rbase-300-full1.csv`
   - `result/public/eng/rbase-fulldata.csv`
- Hausa uses:
   - `result/public/hau/df_1.csv`
   - `result/public/hau/df_2.csv`
   - `result/public/hau/df_3.csv`
   - `result/public/hau/original.csv`

Default ensemble outputs:

- `result/ensemble/eng/pred_eng_ensemble.csv`
- `result/ensemble/eng/pred_eng_ensemble_prob.csv`
- `result/ensemble/hau/pred_hau_ensemble.csv`
- `result/ensemble/hau/pred_hau_ensemble_prob.csv`

## Ensemble Logic

Soft-voting probability:

`P_ens = (sum_i w_i * P_i) / (sum_i w_i)`

Decision rule:

- predict `1` if `P_ens >= 0.5`
- else predict `0`

Paper-style weights in current code:

- Hausa: `[0.20, 0.20, 0.30, 0.30]`
- English: `[0.40, 0.25, 0.10, 0.15, 0.10]`

## Optional: LLM Fine-tuning Script

QLoRA entrypoint:

```bash
python training/llmfinetunehausa.py --lang hau --hf_token <YOUR_HF_TOKEN>
```

Note: this path requires `peft` and suitable GPU setup.

## Testing

Run all unit tests:

```bash
python -m unittest discover -s tests -p "test_*.py"
```

## Common Issues

1. `No module named 'google'`
- Install `google-generativeai` and ensure environment is active.

2. Gemini call failed in augmentation/judge
- Ensure `GEMINI_API_KEY` is set in environment or `.env`.

3. Missing ensemble input files
- Verify all expected CSV files exist under `result/public/eng` or `result/public/hau`.

## Results Snapshot

Reported in paper draft:

- Hausa: rank 1, Macro-F1 0.8336
- English: Top 10, Macro-F1 0.8092
