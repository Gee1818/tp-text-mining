import pandas as pd
import openpyxl as xl
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate
import streamlit as st
import numpy as np
import re

FILE = 'ReporteProductos2.csv'

#MODEL = 'llama3.2'
MODEL = 'llama3.1'
#MODEL = 'mistral'


df = pd.read_csv(FILE, dtype=str)

df['Mes_inicial'] = df['Ano_inicial'].str.split('-').str[0]

df["Ano_inicial"] = df['Ano_inicial'].str.split('-').str[1]

df['Mes_fin'] = df['Ano_fin'].str.split('-').str[0]

df["Ano_fin"] = df['Ano_fin'].str.split('-').str[1]

#df= df[['Codigo', 'Marca', 'Modelo', 'Posicion','Ano_inicial', 'Ano_fin']]

st.title("FricBot")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

model = ChatOllama(model=MODEL, temperature=0)

parser = StrOutputParser()

template = """
Sos un asistente a clientes llamado FricBot que asiste a los clientes
para encontrar un producto de la empresa Fric Rot.

Para asistir a los clientes es necesario que interactues de manera clara y consisa con el cliente 
hasta que tengas informacion de la marca, modelo y año de su vehículo. No es necesaria informacion
acerca del modelo especifico o la linea del vehículo o combustible.

A medida que obetengas esta informacion debes anexar el siguiente texto a la pregunta del cliente:
'Tu vehiculo es el siguiente:
Marca: 'marca'
Modelo: 'modelo'
Año: 'ano''

Respuestas anteriores: {context}
Marca actual de respuestas anteriores: {marca}
Modelo actual de respuestas anteriores: {modelo}
Año actual de respuestas anteriores: {ano}

Pregunta: {question}
"""


prompt = PromptTemplate.from_template(template)

chain = prompt | model | parser

marca, modelo, ano = None, None, None
context = ""


# React to user input
if question := st.chat_input("Escriba su consulta aquí."):
    # Display user message in chat message container
    st.chat_message("user").markdown(question)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": question})

    response = chain.invoke({
        "context": context,
        "marca": marca,
        "modelo": modelo,
        "ano": ano,
        "question": question
    })

    context = " ".join(response)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})

    # Extracting the car brand, model, and year using regular expressions
    marca_match = re.search(r"Marca:\s*(\w+)", response, re.DOTALL)
    modelo_match = re.search(r"Modelo:\s*(\w+)", response)
    ano_match = re.search(r"Año:\s*(\d+)", response)

    if marca_match:
        marca = marca_match.group(1)
    if modelo_match:
        modelo = modelo_match.group(1)
    if ano_match:
        ano = ano_match.group(1)

    if marca != None and modelo != None and ano != None:
        break

filtered_df = df[(df['Marca'] == marca.upper()) & (df['Modelo'] == modelo.upper()) & (df['Ano_inicial'] <= ano) & (df['Ano_fin'] >= ano)]

# Display assistant response in chat message container
with st.chat_message("assistant"):
    st.dataframe(filtered_df)
