import spacy

# Load the existing model
#nlp = spacy.load("C:/Users/spatt/Desktop/diss_3/state_ner/models/model-last")
nlp = spacy.load("C:/Users/spatt/Desktop/diss_3/state_ner/models/model-v4")

# Add a sentencizer if it's not already in the pipeline
if "sentencizer" not in nlp.pipe_names:
    nlp.add_pipe("sentencizer", before="ner")

# Save the updated model with sentence boundaries enabled
#nlp.to_disk("C:/Users/spatt/Desktop/diss_3/state_ner/models/model-last-sentencizer")
nlp.to_disk("C:/Users/spatt/Desktop/diss_3/state_ner/models/model-v4")
