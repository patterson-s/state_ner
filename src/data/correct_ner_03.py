import spacy
import pandas as pd
import json
from pathlib import Path

def sample_and_save_ner(csv_path, output_path, sample_size=5):
    """
    Sample speeches, run NER, and save results in Prodigy-compatible format.
    """
    # Load the latest custom-trained spaCy model (model-v4)
    nlp = spacy.load("C:/Users/spatt/Desktop/diss_3/state_ner/models/model-v4")
    
    # Read CSV and sample new rows
    df = pd.read_csv(csv_path)
    sample_df = df.sample(n=sample_size, random_state=None)  # New random sample each time
    
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
    
    # Save to JSONL file with updated naming convention
    output_file = Path(output_path) / "correct_ner_03.jsonl"
    with output_file.open('w', encoding='utf-8') as f:
        for example in examples:
            f.write(json.dumps(example) + '\n')
    
    print(f"Saved {len(examples)} new examples to {output_file}")

if __name__ == "__main__":
    csv_path = "data/raw/ungdc_1946-2022.csv"
    output_path = "data/annotations"  # Same annotations directory
    sample_and_save_ner(csv_path, output_path)
