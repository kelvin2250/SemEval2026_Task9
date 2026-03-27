import os
import time
import random
import argparse
import sys
from pathlib import Path
import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.gemini import call_gemini, extract_json
from src.augmentation.judge import judge_csv

# Import prompts trực tiếp từ file của Hausa
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from prompts.promptST2_hau import get_paraphrasing_prompt, get_balancing_plan

# CONFIG
GEN_MODEL_NAME = "gemini-2.0-flash"
LABEL_COLS = ["political", "racial/ethnic", "religious", "gender/sexual", "other"]
LANG = "hau"  # Cố định ngôn ngữ là Hausa
OUTPUT_FILE = "dataset/augmentation/hausa/df_3.csv"
INPUT_PATH = f"./dataset/subtask2/train/{LANG}.csv"

def generate_batch(original_texts, labels, n_samples, get_paraphrasing_prompt):
    user_prompt = get_paraphrasing_prompt(original_texts, labels, n_samples=n_samples)

    content = call_gemini(
        model_name=GEN_MODEL_NAME,
        prompt=user_prompt,
        temperature=0.85
    )
    return extract_json(content) or []

# MAIN EXECUTION
def main():
    # Giữ argparse nhưng loại bỏ lựa chọn ngôn ngữ, có thể thêm các tham số khác nếu cần sau này
    parser = argparse.ArgumentParser(description="Augmentation script for Hausa only")
    args = parser.parse_args()

    OUTPUT_COLUMNS = ["id", "text", "label_combo"] + LABEL_COLS
    output_dir = os.path.dirname(OUTPUT_FILE)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Load original dataset (Hausa)
    try:
        df_train = pd.read_csv(INPUT_PATH)
        print(f"Loaded {LANG} dataset with {len(df_train)} rows.")
    except FileNotFoundError:
        print(f"Error: File not found at {INPUT_PATH}")
        return

    df_polar = df_train.copy()
    df_polar["label_combo"] = df_polar[LABEL_COLS].astype(str).agg("".join, axis=1)

    current_counts = df_polar["label_combo"].value_counts().to_dict()
    plans = get_balancing_plan()

    # Initialize
    generated_counts = {}
    if not os.path.exists("test_aug"):
        os.makedirs("test_aug")

    # Resume logic
    if os.path.exists(OUTPUT_FILE):
        try:
            print(f"Checking existing file: {OUTPUT_FILE}")
            df_existing = pd.read_csv(OUTPUT_FILE, dtype={'label_combo': str}, on_bad_lines='skip')
            
            if set(OUTPUT_COLUMNS).issubset(df_existing.columns):
                df_existing = df_existing[OUTPUT_COLUMNS]
                if "label_combo" in df_existing.columns:
                    df_existing["label_combo"] = df_existing["label_combo"].astype(str).str.zfill(5)
                    generated_counts = df_existing["label_combo"].value_counts().to_dict()
                print(f"Resuming... Found {len(df_existing)} existing rows.")
            else:
                print("Existing file has missing columns. Backing up and creating new.")
                os.rename(OUTPUT_FILE, OUTPUT_FILE + f".bak_{int(time.time())}")
                pd.DataFrame(columns=OUTPUT_COLUMNS).to_csv(OUTPUT_FILE, index=False)
                
        except Exception as e:
            print(f"Error reading existing file: {e}. Backing up and creating new.")
            os.rename(OUTPUT_FILE, OUTPUT_FILE + f".bak_{int(time.time())}")
            pd.DataFrame(columns=OUTPUT_COLUMNS).to_csv(OUTPUT_FILE, index=False)
    else:
        pd.DataFrame(columns=OUTPUT_COLUMNS).to_csv(OUTPUT_FILE, index=False)
        print(f"Created new output file: {OUTPUT_FILE}")

    total_generated_session = 0

    try:
        for plan in plans:
            combo = plan["combo"]
            target = plan["target_count"]
            desc = plan["desc"]
            
            existing_origin = current_counts.get(combo, 0)
            existing_generated = generated_counts.get(combo, 0)
            
            needed = target - existing_origin - existing_generated

            print(f"\nCombo {combo} ({desc}) | Origin: {existing_origin} | Generated: {existing_generated} | Target: {target}")
            if needed <= 0:
                print("✓ Skip (enough data)")
                continue

            subset = df_polar[df_polar["label_combo"] == combo]
            if subset.empty:
                print(f"No source data for combo {combo}. Skipping.")
                continue
            
            source_samples = subset.to_dict('records')
            pbar = tqdm(total=needed, desc=f"Augmenting {combo}")
            generated_count = 0

            while generated_count < needed:
                try:
                    num_available = len(source_samples)
                    n_fewshot = min(num_available, 15)
                    max_gen_batch = 15
                    
                    batch_samples = random.sample(source_samples, n_fewshot)
                    original_texts = [s['text'] for s in batch_samples]
                    
                    sample = batch_samples[0]
                    labels = [sample[col] for col in LABEL_COLS]
                    
                    batch_size = min(needed - generated_count, max_gen_batch)
                    items = generate_batch(original_texts, labels, batch_size, get_paraphrasing_prompt)
                    
                    if not isinstance(items, list):
                        items = []

                    batch_new_rows = []
                    for item in items:
                        text = item.get("text", "").strip()
                        if not text: continue

                        row = {
                            "id": f"aug_{LANG}_{combo}_{int(time.time())}_{total_generated_session}",
                            "text": text,
                            "label_combo": combo
                        }
                        for i, col in enumerate(LABEL_COLS):
                            row[col] = int(combo[i])
                        
                        batch_new_rows.append(row)
                        generated_count += 1
                        total_generated_session += 1
                        pbar.update(1)
                        if generated_count >= needed: break
                    
                    if batch_new_rows:
                        pd.DataFrame(batch_new_rows, columns=OUTPUT_COLUMNS).to_csv(OUTPUT_FILE, mode='a', header=False, index=False)
                
                except Exception as e:
                    print(f"\nError in batch: {e}")
                    time.sleep(5)
                time.sleep(0.5)
            pbar.close()

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    print(f"\nDONE. Total generated: {total_generated_session}")
    
    # Judge augmented data for quality
    judge_augmented_hausa(OUTPUT_FILE, threshold=0.7, save_scores=True)


def judge_augmented_hausa(csv_path, threshold=0.7, save_scores=True):
    """
    Judge Hausa augmented data and filter by quality threshold.
    
    Args:
        csv_path: Path to augmented CSV file
        threshold: Quality score threshold (0-1)
        save_scores: Whether to save detailed scores
    
    Returns:
        pd.DataFrame: Filtered (high-quality) augmented data
    """
    if not os.path.exists(csv_path):
        print(f"⚠️  File {csv_path} not found. Skipping judge step.")
        return None
    
    print(f"\n{'='*60}")
    print(f"JUDGE PHASE: Evaluating Hausa data quality...")
    print(f"{'='*60}")
    
    scores_csv = csv_path.replace(".csv", "_judge_scores.csv") if save_scores else None
    filtered_df, scores_df = judge_csv(csv_path, task="st2", threshold=threshold, output_scores_csv=scores_csv)
    
    # Save filtered data back to the original file (overwrite)
    if not filtered_df.empty:
        # Keep only relevant columns for final output
        output_cols = ["id", "text", "label_combo"] + LABEL_COLS
        filtered_df = filtered_df[output_cols]
        
        filtered_df.to_csv(csv_path, index=False)
        print(f"✓ Saved {len(filtered_df)} high-quality samples to {csv_path}")
    else:
        print(f"⚠️  No samples passed quality threshold. Keeping all data.")
    
    return filtered_df

if __name__ == "__main__":
    main()