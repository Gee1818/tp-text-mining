import random
import json
import nltk
import pandas as pd

import torch

from model import NeuralNet
from utils import bag_of_words, tokenize


# Cargar modelo conversacional
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open("intents_orig.json", "r") as json_data:
    intents = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE, map_location=device)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data["all_words"]
tags = data["tags"]
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()


# Cargar datos de catalogo
catalog = pd.read_csv("fricrot_data.csv")

brands = catalog["Marca"].unique()
models = catalog["Modelo"].unique()

brands = [brand.lower() for brand in brands]
models = [model.lower() for model in models]

car_brand = None
car_model = None


# Iniciar interaccion
bot_name = "FricBot"
print(f"Hola, soy {bot_name}. ¿En qué te puedo ayudar? (escribí 'quit' para salir)")

while True:
    intersection = None
    sentence = input("Vos: ")
    if sentence == "quit":
        break

    sentence = tokenize(sentence)

    # Distancia de Levenshtein (por si hay typos)
    best_fit_model = None
    least_ratio = 0.3 # el nro de operaciones es <= al 30% de len(word)
    for word in sentence:
        for modelo in models:
            ratio = nltk.edit_distance(modelo, word) / len(word)
            if ratio <= least_ratio:
                best_fit_model = modelo
                least_ratio = ratio



    intersection_brand = set(sentence).intersection(brands)
    #intersection_model = set(sentence).intersection(models)
    intersection_model = best_fit_model
    
    if intersection_model:
        #car_model = intersection_model.pop() # ignora multiples coincidencias
        car_model = intersection_model
        print(
            f"{bot_name}: He detectado que el modelo de tu auto es {car_model.capitalize()}.", 
            "Escribe 's' para confirmar y buscaré el repuesto que aplica a tu vehículo"
        )
    elif intersection_brand:
        car_brand = intersection_brand.pop() # ignora multiples coincidencias
        print(
            f"{bot_name}: He detectado que la marca de tu auto es {car_brand.capitalize()}.", 
            "¿Podrías decirme el modelo?"
        )
    elif sentence == ["s"] and car_model != None:
        print(f"{bot_name}: Estos son los repuestos que tenemos para tu vehículo.")
        print(catalog[catalog.Modelo == car_model.upper()][["Marca", "Modelo", "Anio", "Linea", "Producto"]])

    else:
        X = bag_of_words(sentence, all_words)
        X = X.reshape(1, X.shape[0])
        X = torch.from_numpy(X).to(device)

        output = model(X)
        _, predicted = torch.max(output, dim=1)

        tag = tags[predicted.item()]

        probs = torch.softmax(output, dim=1)
        prob = probs[0][predicted.item()]
        if prob.item() > 0.75:
            intent = intents["intents"][tag]
            print(f"{bot_name}: {random.choice(intent['responses'])}")
            #for intent in intents["intents"]:
                #if tag == intent["tag"]:
                    #print(f"{bot_name}: {random.choice(intent['responses'])}")
        else:
            print(f"{bot_name}: Lo siento, pero no entiendo.")
