"""
Judge module for augmentation quality filtering.

Functions for evaluating and filtering augmented data based on quality metrics:
- Relevance: Does the sample match the task definition?
- Correctness: Is the label assigned correctly?
- Style: Does it match the original data style?
"""

import json
import pandas as pd
from tqdm import tqdm
from src.utils.gemini import call_gemini, extract_json


JUDGE_MODEL = "gemini-2.5-pro"
BATCH_SIZE = 20
RELEVANCE_THRESHOLD = 0.7  # 0-1 score for keeping sample


def get_judge_prompt(texts, labels, task="st1"):
    """
    Generate a prompt for LLM-based quality judgment.
    
    Args:
        texts: List of augmented text samples
        labels: List of corresponding labels
        task: "st1" for polarization, "st2" for multi-label
    
    Returns:
        str: Prompt for judge LLM
    """
    texts_block = "\n".join([f"{i+1}. [{label}] {text}" for i, (text, label) in enumerate(zip(texts, labels))])
    
    if task == "st1":
        return f"""
### ROLE
You are a professional quality auditor for data augmentation in NLP research.

### TASK: JUDGE AUGMENTED DATA QUALITY
Evaluate the following augmented text samples. For EACH sample, score it from 0.0 to 1.0 based on:

**Relevance (40%):**
- Does the text fit the task definition?
- For Label 0: Text should NOT create group polarization (no "us vs. them")
- For Label 1: Text MUST show clear polarization/hostility toward groups

**Correctness (40%):**
- Is the assigned label correct for the text?
- Is the label forced or naturally fitting?

**Style (20%):**
- Does it match social media style (tone, length, vocabulary)?
- Avoid overly academic or formal tone

### JUDGED SAMPLES:
{texts_block}

### OUTPUT FORMAT
Return ONLY a JSON array with scores for each sample (in same order):
[
  {{"index": 1, "score": 0.85, "reason": "Clear non-polarized discussion"}},
  {{"index": 2, "score": 0.42, "reason": "Too academic in tone"}},
  ...
]
"""
    else:  # task == "st2"
        return f"""
### ROLE
You are a professional quality auditor for data augmentation in NLP research.

### TASK: JUDGE AUGMENTED DATA QUALITY
Evaluate the following augmented text samples. For EACH sample, score it from 0.0 to 1.0 based on:

**Relevance (40%):**
- Does the text clearly exhibit the assigned polarization types?
- Are the label categories (political, racial/ethnic, religious, gender/sexual, other) correctly represented?

**Correctness (40%):**
- Is the assigned label combo correct for the text?
- Does it naturally fit the polarization described?

**Style (20%):**
- Does it match social media style (tone, length, vocabulary)?
- Appropriate language register for the platform

### JUDGED SAMPLES:
{texts_block}

### OUTPUT FORMAT
Return ONLY a JSON array with scores for each sample (in same order):
[
  {{"index": 1, "score": 0.88, "reason": "Clear political polarization"}},
  {{"index": 2, "score": 0.51, "reason": "Ambiguous label assignment"}},
  ...
]
"""


def judge_batch(texts, labels, task="st1", judge_model=JUDGE_MODEL, threshold=RELEVANCE_THRESHOLD):
    """
    Judge a batch of samples using LLM.
    
    Args:
        texts: List of text samples
        labels: List of labels
        task: "st1" or "st2"
        judge_model: LLM model name
        threshold: Minimum score to keep (0-1)
    
    Returns:
        list: List of dicts with keys: {index, text, label, score, reason, keep}
    """
    if not texts:
        return []
    
    prompt = get_judge_prompt(texts, labels, task)
    
    try:
        response = call_gemini(
            model_name=judge_model,
            prompt=prompt,
            temperature=0.3,  # Low temp for consistent judgement
            max_retries=2,
            response_mime_type="application/json",
        )
        
        scores = extract_json(response)
        if not isinstance(scores, list):
            # Sometimes LLM wraps in a dict
            if isinstance(scores, dict) and "scores" in scores:
                scores = scores["scores"]
            else:
                return []
        
        # Attach original data and filter decision
        results = []
        for item in scores:
            if "index" in item and 0 <= item["index"] - 1 < len(texts):
                idx = item["index"] - 1
                score = item.get("score", 0.0)
                results.append({
                    "index": item["index"],
                    "text": texts[idx],
                    "label": labels[idx],
                    "score": float(score),
                    "reason": item.get("reason", ""),
                    "keep": float(score) >= threshold
                })
        
        return results
    
    except Exception as e:
        print(f"Error during judging: {e}")
        return []


def judge_augmented_dataframe(df, task="st1", threshold=RELEVANCE_THRESHOLD):
    """
    Judge all rows in a DataFrame.
    
    Args:
        df: DataFrame with 'text' and 'polarization' columns (or 'text' and 5 label columns for ST2)
        task: "st1" or "st2"
        threshold: Minimum score to keep (0-1)
    
    Returns:
        tuple: (filtered_df, scores_df)
            - filtered_df: Only samples with score >= threshold
            - scores_df: All samples with judgment scores
    """
    if df.empty:
        return df.copy(), pd.DataFrame()
    
    all_results = []
    
    # Process in batches
    for batch_start in tqdm(range(0, len(df), BATCH_SIZE), desc="Judging"):
        batch_end = min(batch_start + BATCH_SIZE, len(df))
        batch_df = df.iloc[batch_start:batch_end]
        
        texts = batch_df["text"].tolist()
        
        if task == "st1":
            labels = batch_df["polarization"].tolist()
        else:  # st2
            # For ST2, create label combo string
            label_cols = ["political", "racial/ethnic", "religious", "gender/sexual", "other"]
            labels = batch_df[label_cols].astype(str).agg("".join, axis=1).tolist()
        
        batch_results = judge_batch(texts, labels, task=task, threshold=threshold)
        all_results.extend(batch_results)
    
    if not all_results:
        empty_df = df.copy()
        empty_df["judge_score"] = 0.0
        empty_df["judge_reason"] = ""
        return empty_df[empty_df["judge_score"] >= threshold], empty_df
    
    # Create scores dataframe
    scores_df = pd.DataFrame(all_results)
    scores_df = scores_df.sort_values("index").reset_index(drop=True)
    
    # Create filtered dataframe (only high-quality samples)
    filtered_df = pd.DataFrame([r for r in all_results if r["keep"]])
    
    if filtered_df.empty:
        print(f"⚠️  Warning: No samples passed quality threshold ({threshold}). Returning top 50%.")
        # If too strict, at least keep top 50%
        threshold_adjusted = sorted([r["score"] for r in all_results])[len(all_results) // 2]
        filtered_df = pd.DataFrame([r for r in all_results if r["score"] >= threshold_adjusted])
    
    return filtered_df, scores_df


def judge_csv(input_csv, task="st1", threshold=RELEVANCE_THRESHOLD, output_scores_csv=None):
    """
    Judge augmented data from CSV file.
    
    Args:
        input_csv: Path to input CSV
        task: "st1" or "st2"
        threshold: Minimum score to keep
        output_scores_csv: Optional path to save detailed scores
    
    Returns:
        tuple: (filtered_df, scores_df)
    """
    df = pd.read_csv(input_csv)
    print(f"Loaded {len(df)} samples from {input_csv}")
    
    filtered_df, scores_df = judge_augmented_dataframe(df, task=task, threshold=threshold)
    
    print(f"✓ Passed: {len(filtered_df)} samples")
    print(f"✗ Filtered: {len(scores_df) - len(filtered_df)} samples")
    print(f"Score distribution: min={scores_df['score'].min():.2f}, max={scores_df['score'].max():.2f}, mean={scores_df['score'].mean():.2f}")
    
    if output_scores_csv:
        scores_df.to_csv(output_scores_csv, index=False)
        print(f"Saved detailed scores to {output_scores_csv}")
    
    return filtered_df, scores_df
