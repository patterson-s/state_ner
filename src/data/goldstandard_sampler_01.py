import random
import spacy
import pandas as pd
from pathlib import Path
import json
import datetime
import re

# Load spaCy model for tokenization
nlp = spacy.load("en_core_web_lg")

def clean_text(text):
    """Remove unnecessary line breaks within sentences."""
    return re.sub(r"(?<!\n)\n(?!\n)", " ", text)  # Keeps paragraph breaks, removes single line breaks

def load_previous_samples(log_file_path):
    """Load previously sampled doc_ids from the log file."""
    if not log_file_path.exists():
        return set()
    
    with open(log_file_path, "r", encoding="utf-8") as log_file:
        logs = [json.loads(line) for line in log_file]
        previous_doc_ids = {doc_id for log in logs for doc_id in log.get("sampled_doc_ids", [])}
    return previous_doc_ids

def sample_gold_standard(source_file, output_dir, log_file):
    # Ask the user for the number of speeches to sample
    num_speeches = int(input("Enter the number of speeches to sample for the gold standard: "))
    
    # Load speeches from CSV
    df = pd.read_csv(source_file)
    
    # Load previously sampled doc_ids
    previous_doc_ids = load_previous_samples(log_file)
    
    # Filter out previously sampled speeches
    available_df = df[~df["doc_id"].isin(previous_doc_ids)]
    if available_df.empty:
        print("No new speeches available for sampling.")
        return

    # Sample speeches from available pool
    sampled_df = available_df.sample(n=min(num_speeches, len(available_df)))
    
    # Save the sampled speeches to the output directory with an iterative naming convention
    iteration = len(list(Path(output_dir).glob("sample_*"))) + 1
    sample_folder = Path(output_dir) / f"sample_{iteration}"
    sample_folder.mkdir(parents=True, exist_ok=True)

    sample_output_path = sample_folder / "goldstandard_sample.jsonl"
    with open(sample_output_path, "w", encoding="utf-8") as f:
        for _, row in sampled_df.iterrows():
            cleaned_text = clean_text(row["text"])
            f.write(json.dumps({"text": cleaned_text, "doc_id": row["doc_id"]}) + "\n")

    # Log the sample details
    log_data = {
        "annotation_round": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "num_speeches_sampled": num_speeches,
        "sampled_doc_ids": list(sampled_df["doc_id"])
    }
    
    with open(log_file, "a", encoding="utf-8") as log:
        log.write(json.dumps(log_data) + "\n")

    print(f"Gold standard sample saved to {sample_output_path}")

# Parameters
source_file = r"C:/Users/spatt/Desktop/diss_3/state_ner/data/raw/ungdc_1946-2022.csv"
output_dir = r"C:/Users/spatt/Desktop/diss_3/state_ner/goldstandard"
log_file = Path(output_dir) / "goldstandard_log.json"

# Run the function
sample_gold_standard(source_file, output_dir, log_file)
