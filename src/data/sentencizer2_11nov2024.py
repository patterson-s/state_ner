import spacy

# Load the model you want to update (model-v3 in this case)
nlp = spacy.load("C:/Users/spatt/Desktop/diss_3/state_ner/models/model-v3")

# Add a sentencizer if it's not already in the pipeline
if "sentencizer" not in nlp.pipe_names:
    nlp.add_pipe("sentencizer", before="ner")

# Save the updated model back to model-v3
nlp.to_disk("C:/Users/spatt/Desktop/diss_3/state_ner/models/model-v3")
