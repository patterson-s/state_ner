import random
import spacy
import pandas as pd
from pathlib import Path
import json
import datetime
import re

# Load spaCy model with sentencizer
nlp = spacy.load("en_core_web_lg")

# Add the sentencizer directly by name and customize punctuation
nlp.add_pipe("sentencizer", config={"punct_chars": ["\n", ".", "!", "?"]})

def clean_text(text):
    """Remove unnecessary line breaks within sentences."""
    # This regular expression keeps paragraph breaks (double newlines) but removes single line breaks
    return re.sub(r"(?<!\n)\n(?!\n)", " ", text)

def sample_and_annotate(source_file, output_dir, log_file):
    # Ask the user for the number of speeches to sample
    num_speeches = int(input("Enter the number of speeches to sample: "))
    
    # Create output directory if it doesn’t exist
    log_dir = Path(output_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

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
    
    # Process sampled speeches and collect sentences
    sentences = []
    for _, row in sampled_df.iterrows():
        cleaned_text = clean_text(row["text"])  # Clean text to handle line breaks
        doc = nlp(cleaned_text)  # Segment text into sentences
        sentences.extend([{"text": sent.text} for sent in doc.sents])
    
    # Save sentences for Prodigy annotation
    sample_output_path = log_dir / "state_demonym_01_samples.jsonl"
    with open(sample_output_path, "w", encoding="utf-8") as f:
        for sentence in sentences:
            f.write(json.dumps(sentence) + "\n")
    
    # Log the annotation round details
    log_data = {
        "annotation_round": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "num_speeches_sampled": num_speeches,
        "sampled_doc_ids": list(sampled_df["doc_id"])
    }
    
    with open(log_file, "a", encoding="utf-8") as log:
        log.write(json.dumps(log_data) + "\n")

def load_previous_samples(log_file_path):
    """Load previously sampled doc_ids from the log file."""
    if not log_file_path.exists():
        return set()
    
    with open(log_file_path, "r", encoding="utf-8") as log_file:
        logs = [json.loads(line) for line in log_file]
        previous_doc_ids = {doc_id for log in logs for doc_id in log.get("sampled_doc_ids", [])}
    return previous_doc_ids

# Parameters
source_file = r"C:/Users/spatt/Desktop/diss_3/state_ner/data/raw/ungdc_1946-2022.csv"
output_dir = r"C:/Users/spatt/Desktop/diss_3/state_ner/data/annotations"
log_file = Path(output_dir) / "annotation_log.json"

# Run the function
sample_and_annotate(source_file, output_dir, log_file)
