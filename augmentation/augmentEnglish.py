import os
import json
import pandas as pd
import time
import argparse
import sys
from pathlib import Path
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.gemini import call_gemini, extract_json
from prompts.promptST1_eng import get_prompt_label0, get_prompt_label1
from src.augmentation.judge import judge_csv

# Configuration
MODEL_NAME = "gemini-2.0-flash"
DEFAULT_INPUT_FILE = "./dataset/subtask1/train/eng.csv"
DEFAULT_BATCH_SIZE = 20
MAX_RETRIES = 3

def get_system_prompt(label, samples_json, n_samples=20):
    """
    Get system prompt from extracted modules.
    """
    if label == 0:
        return get_prompt_label0(n_samples, samples_json)
    elif label == 1:
        return get_prompt_label1(n_samples, samples_json)
    else:
        raise ValueError("Label must be 0 or 1")

def get_augmented_batch(batch_df, label, n_to_generate):
    # Convert batch to JSON list
    samples_list = batch_df[['text', 'polarization']].to_dict(orient='records')
    samples_json = json.dumps(samples_list, ensure_ascii=False)
    
    # instruct the LLM to generate exactly n_to_generate samples
    sys_prompt = get_system_prompt(label, samples_json, n_samples=n_to_generate)
    
    content = call_gemini(
        model_name=MODEL_NAME,
        prompt=sys_prompt,
        temperature=0.75,
        max_retries=MAX_RETRIES,
        response_mime_type="application/json",
    )

    data = extract_json(content)

    # Handle cases where the model wraps the list in a key.
    if isinstance(data, dict):
        for key in data:
            if isinstance(data[key], list):
                return data[key]
    return data if isinstance(data, list) else []

def process_augmentation(input_file, output_file, label, num_samples, batch_size, ref_mode="random"):
    """
    Main processing loop for data augmentation.
    """
    print(f"Loading data from {input_file}...")
    if not os.path.exists(input_file):
        print(f"Error: Input file {input_file} not found.")
        return

    df = pd.read_csv(input_file)
    
    # Filter by label to get the pool of samples for reference
    target_pool = df[df['polarization'] == label]
    print(f"Found {len(target_pool)} total samples with label {label} in pool.")

    if ref_mode == "sequential":
        # In sequential mode, if num_samples is not specified, run until pool is exhausted
        estimated_max = (len(target_pool) // 20) * batch_size
        if num_samples is None or num_samples <= 0:
            num_samples = estimated_max
        print(f"Sequential mode: Processing pool in chunks of 20.")
    else:
        if num_samples is None or num_samples <= 0:
            num_samples = 100 # Default if not specified correctly
    
    print(f"Goal: Generate {num_samples} new samples in batches of {batch_size}.")
    print(f"Reference Mode: {ref_mode}")
    print(f"Output will be saved to: {output_file}")

    # Ensure output directory exists
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Initialize output file with header if it doesn't exist or is empty
    if not os.path.exists(output_file) or os.path.getsize(output_file) == 0:
        pd.DataFrame(columns=['text', 'polarization']).to_csv(output_file, index=False)
        print(f"Created new output file with columns ['text', 'polarization']")
    else:
        # If file exists, check if it has the correct header
        try:
            temp_df = pd.read_csv(output_file, nrows=0)
            if not all(col in temp_df.columns for col in ['text', 'polarization']):
                print(f"File {output_file} exists but missing headers. Re-creating...")
                pd.DataFrame(columns=['text', 'polarization']).to_csv(output_file, index=False)
        except Exception:
            pd.DataFrame(columns=['text', 'polarization']).to_csv(output_file, index=False)

    # Processing loop
    total_generated = 0
    ref_idx = 0
    pbar = tqdm(total=num_samples, desc="Augmenting")

    while total_generated < num_samples:
        # Step 1: Select reference samples based on mode
        current_request_size = min(batch_size, num_samples - total_generated)
        
        if ref_mode == "sequential":
            if ref_idx >= len(target_pool):
                print("\nReached end of reference pool.")
                break
            batch_reference = target_pool.iloc[ref_idx : ref_idx + 20]
            ref_idx += 20
        else:
            # Random mode
            batch_reference = target_pool.sample(n=min(len(target_pool), 20))
            
        if len(batch_reference) == 0:
            break
        
        new_samples = []
        success = False

        for attempt in range(1, MAX_RETRIES + 1):
            new_samples = get_augmented_batch(batch_reference, label, n_to_generate=current_request_size)

            # Validate output
            if (
                isinstance(new_samples, list)
                and len(new_samples) > 0
                and all(isinstance(x, dict) for x in new_samples)
                and all("text" in x and "polarization" in x for x in new_samples)
            ):
                success = True
                break
            else:
                print(f"\nRetry {attempt}/{MAX_RETRIES} for batch (invalid output)")
                time.sleep(1.5)

        if not success:
            print(f"\n❌ Batch FAILED after {MAX_RETRIES} retries, skipping.")
            # Safety break to avoid infinite loop
            if total_generated == 0 and attempt == MAX_RETRIES:
                break
            continue

        # Save results
        new_df = pd.DataFrame(new_samples)
        # Ensure label matches requested (sometimes models might hallucinate label)
        new_df['polarization'] = label
        
        new_df.to_csv(output_file, mode='a', header=False, index=False)
        
        total_generated += len(new_samples)
        pbar.update(len(new_samples))

    pbar.close()
    print(f"\nDone! Total samples generated and saved: {total_generated}")


def judge_augmented_data(csv_path, task="st1", threshold=0.7, save_scores=True):
    """
    Judge augmented data and filter by quality threshold.
    
    Args:
        csv_path: Path to augmented CSV file
        task: "st1" for this script
        threshold: Quality score threshold (0-1)
        save_scores: Whether to save detailed scores
    
    Returns:
        pd.DataFrame: Filtered (high-quality) augmented data
    """
    if not os.path.exists(csv_path):
        print(f"⚠️  File {csv_path} not found. Skipping judge step.")
        return None
    
    print(f"\n{'='*60}")
    print(f"JUDGE PHASE: Evaluating data quality...")
    print(f"{'='*60}")
    
    scores_csv = csv_path.replace(".csv", "_judge_scores.csv") if save_scores else None
    filtered_df, scores_df = judge_csv(csv_path, task=task, threshold=threshold, output_scores_csv=scores_csv)
    
    # Save filtered data back to the original file (overwrite)
    if not filtered_df.empty:
        # Keep only relevant columns for final output
        if task == "st1":
            filtered_df = filtered_df[["text", "label"]]
            filtered_df.columns = ["text", "polarization"]
        
        filtered_df.to_csv(csv_path, index=False)
        print(f"✓ Saved {len(filtered_df)} high-quality samples to {csv_path}")
    else:
        print(f"⚠️  No samples passed quality threshold. Keeping all data.")
    
    return filtered_df

def main():
    parser = argparse.ArgumentParser(description="Data Augmentation Tool for SemEval Task 9")
    
    parser.add_argument(
        "--label", 
        type=int, 
        choices=[0, 1], 
        required=True, 
        help="Target label to augment (0 for Non-Polarized, 1 for Polarized)"
    )
    
    parser.add_argument(
        "--input", 
        type=str, 
        default=DEFAULT_INPUT_FILE, 
        help=f"Path to input CSV file (default: {DEFAULT_INPUT_FILE})"
    )
    
    parser.add_argument(
        "--output", 
        type=str, 
        help="Path to output CSV file. If not provided, a default name based on label will be used."
    )
    
    parser.add_argument(
        "--samples", 
        type=int, 
        default=None, 
        help="Number of samples to process from the end of the dataset (default: All)"
    )
    
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=DEFAULT_BATCH_SIZE, 
        help=f"Batch size for API requests (default: {DEFAULT_BATCH_SIZE})"
    )

    parser.add_argument(
        "--ref_mode",
        type=str,
        choices=["random", "sequential"],
        default="random",
        help="How to select reference samples from input file (random or sequential)"
    )

    args = parser.parse_args()

    # Determine default output filename if not provided
    if not args.output:
        output_dir = "dataset/augmentation/english"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        if args.label == 0:
            args.output = os.path.join(output_dir, "augmented_data_label0.csv")
        else:
            args.output = os.path.join(output_dir, "augmented_data_label1.csv")

    process_augmentation(
        input_file=args.input,
        output_file=args.output,
        label=args.label,
        num_samples=args.samples,
        batch_size=args.batch_size,
        ref_mode=args.ref_mode
    )
    
    # Judge augmented data for quality
    judge_augmented_data(args.output, task="st1", threshold=0.7, save_scores=True)

if __name__ == "__main__":
    main()


# python unified_augmentation.py --label 1 --output my_custom_data.csv --samples 100 --batch_size 10
# python unified_augmentation.py --label 1 --output my_custom_data.csv --samples 100 --batch_size 10