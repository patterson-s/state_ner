import spacy

def test_spacy_gpe():
    # Load the model
    nlp = spacy.load('en_core_web_lg')
    
    # Test text with some obvious GPEs
    test_text = """
    California and Texas are the two most populous states in America. 
    New York City is the largest city in the United States. 
    Washington is both a state and the location of the capital.
    """
    
    # Process the text
    doc = nlp(test_text)
    
    # Print all entities and their labels
    print("All entities found:")
    for ent in doc.ents:
        print(f"Text: {ent.text}\tLabel: {ent.label_}")
    
    # Print just the GPEs
    print("\nJust GPEs:")
    gpes = [ent.text for ent in doc.ents if ent.label_ == 'GPE']
    print(gpes)

if __name__ == "__main__":
    test_spacy_gpe()