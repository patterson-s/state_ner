import spacy
from spacy.pipeline import EntityRuler
import json
from pathlib import Path

def add_entity_ruler_to_model(model_path, patterns_path, output_path):
    # Load the model
    nlp = spacy.load(model_path)

    # Add EntityRuler before the NER component if not already added
    if "entity_ruler" not in nlp.pipe_names:
        ruler = nlp.add_pipe("entity_ruler", before="ner")
    else:
        ruler = nlp.get_pipe("entity_ruler")

    # Load patterns from JSONL file
    patterns = []
    with open(patterns_path, "r", encoding="utf-8") as f:
        for line in f:
            patterns.append(json.loads(line))
    ruler.add_patterns(patterns)

    # Ensure the output directory exists
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save the modified model to the specified output path
    nlp.to_disk(output_path)
    print(f"Model with EntityRuler saved to {output_path}")

# Specify paths here directly
model_path = "models/model-v4"  # Path to the existing model
patterns_path = "pattern/demonym_gpe_v2.jsonl"  # Path to the patterns file
output_path = "state_ner/models/model-v4_with_ruler"  # Path to save the updated model

# Run the function
add_entity_ruler_to_model(model_path, patterns_path, output_path)
