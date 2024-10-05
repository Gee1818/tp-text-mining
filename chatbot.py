import spacy
from spacy.pipeline import EntityRuler
import pandas as pd
import streamlit as st

df = pd.read_csv("car_brands_models.csv")

makes = df['Brand']
models = df['Model']


# Load Spanish NER model
nlp = spacy.load("es_core_news_sm")

# Create an instance of the EntityRuler
ruler = EntityRuler(nlp)

# Define patterns for the new entities
patrones = []
for make in makes:
	patrones.append({"label": "MAKE", "pattern": make.lower()})
for model in models:
	patrones.append({"label": "MODEL", "pattern": model.lower()})


# Add the EntityRuler to the spaCy pipeline
ruler = nlp.add_pipe("entity_ruler", before="ner")
ruler.add_patterns(patrones)

def extract_car_info(text):
    doc = nlp(text.lower())
    make = None
    model = None
    
    print(doc.ents)
    
    for ent in doc.ents:
    	print(ent.label_)

    # Example entity labels for car makes and models
    for ent in doc.ents:
        if ent.label_ == 'MAKE':  # Adjust based on your model
            make = ent.text
        elif ent.label_ == 'MODEL':  # Adjust based on your model
            model = ent.text

    return make if make else 'Unknown', model if model else 'Unknown'

def main():
    st.title("Car Information Chatbot")
    st.write("¡Hola! Cuéntame sobre tu coche. Escribe 'Salir' para terminar la conversación.")

    # Create a text input box for user input
    user_input = st.text_input("Tú:", "")

    if user_input:
        if user_input.lower() == 'salir':
            st.write("Chatbot: ¡Adiós!")
        else:
            make, model = extract_car_info(user_input)
            st.write(f"Chatbot: El fabricante de tu coche es {make.capitalize()} y el modelo es {model.capitalize()}.")

if __name__ == "__main__":
    main()

