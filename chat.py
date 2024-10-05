import random
import json
import pandas as pd

import torch

from model import NeuralNet
from utils import bag_of_words, tokenize

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open("intents.json", "r") as json_data:
    intents = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data["all_words"]
tags = data["tags"]
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "FricBot"
print("Hola, soy FricBot. En que te puedo ayudar? (escribí 'quit' para salir)")

catalog = pd.read_csv("car_brands_models.csv")

brands = catalog["Brand"].unique()
models = catalog["Model"].unique()

brands = [brand.lower() for brand in brands]
models = [model.lower() for model in models]

car_brand = None
car_model = None

while True:
    intersection = None
    # sentence = "do you use credit cards?"
    sentence = input("Vos: ")
    if sentence == "quit":
        break

    sentence = tokenize(sentence)

    intersection_brand = set(sentence).intersection(brands)

    intersection_model = set(sentence).intersection(models)
    if intersection_model:
        car_model = intersection_model.pop()

        print(
            f"{bot_name}: He detectado que el modelo de tu auto es {car_model}. Escribe 's' para confirmar y buscare el repuesto que aplica a tu vehiculo"
        )
    elif intersection_brand:
        car_brand = intersection_brand.pop()

        print(
            f"{bot_name}: He detectado que la marca de tu auto es {car_brand}. ¿Podrías decirme el modelo?"
        )
    elif sentence == ["s"] and car_model != None and car_brand != None:
        print('Buscando repuestos para tu auto...')

    else:
        X = bag_of_words(sentence, all_words)
        X = X.reshape(1, X.shape[0])
        X = torch.from_numpy(X).to(device)

        output = model(X)
        _, predicted = torch.max(output, dim=1)

        tag = tags[predicted.item()]

        # print(tag)

        probs = torch.softmax(output, dim=1)
        prob = probs[0][predicted.item()]
        if prob.item() > 0.75:
            for intent in intents["intents"]:
                if tag == intent["tag"]:
                    print(f"{bot_name}: {random.choice(intent['responses'])}")
        else:
            print(f"{bot_name}: I do not understand...")
