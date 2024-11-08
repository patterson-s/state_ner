import spacy
import pandas as pd
import json
from pathlib import Path

def sample_and_save_ner(csv_path, output_path, sample_size=10):
    """
    Sample speeches, run NER, and save results in Prodigy-compatible format
    """
    # Load spaCy model
    nlp = spacy.load('en_core_web_lg')
    
    # Read CSV and sample rows
    df = pd.read_csv(csv_path)
    sample_df = df.sample(n=sample_size, random_state=42)
    
    # Prepare data for Prodigy
    examples = []
    for idx, row in sample_df.iterrows():
        doc = nlp(row['text'])
        
        # Create Prodigy task for this text
        task = {
            "text": row['text'],
            "meta": {"doc_id": row['doc_id'], "year": row['year']},
            "spans": [
                {
                    "start": ent.start_char,
                    "end": ent.end_char,
                    "label": ent.label_,
                    "token_start": ent.start,
                    "token_end": ent.end
                }
                for ent in doc.ents if ent.label_ == 'GPE'
            ]
        }
        examples.append(task)
    
    # Save to JSONL file
    output_file = Path(output_path) / "explore_ner_1.jsonl"
    with output_file.open('w', encoding='utf-8') as f:
        for example in examples:
            f.write(json.dumps(example) + '\n')
    
    print(f"Saved {len(examples)} examples to {output_file}")

if __name__ == "__main__":
    csv_path = "data/raw/ungdc_1946-2022.csv"
    output_path = "data/annotations"  # Using the annotations directory we created earlier
    sample_and_save_ner(csv_path, output_path)