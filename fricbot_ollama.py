import pandas as pd
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate
import streamlit as st
import numpy as np
import re

FILE = 'ReporteProductos_aplicaciones.csv'

#MODEL = 'llama3.2'
#MODEL = 'llama3.1'
MODEL = 'mistral'


df = pd.read_csv(FILE, dtype=str)

df_atributos = pd.read_csv('ReporteProductos_atributos.csv', dtype=str)

print(df_atributos.columns)

df["Ano_inicial"] = df['Ano_inicial'].str.split('-').str[1]
df["Ano_fin"] = df['Ano_fin'].str.split('-').str[1]

# Modificacion: reemplazar nan por un valor extremadamente alto
df['Ano_fin_filtro'] = df['Ano_fin'].replace(np.nan, '2999')

df = df.drop(columns=['Linea'])

df = df.merge(df_atributos[["Codigo", "TipoProducto", "Linea"]], on='Codigo', how='left')

st.set_page_config(
    layout="wide",
    page_title="FricBot",
    page_icon=":robot_face:"
)
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
Respuestas anteriores: {context}

Marca actual: {marca}
Modelo actual: {modelo}
Año actual: {ano}

Sos un asistente a clientes llamado FricBot que asiste a los clientes
para encontrar un producto de la empresa Fric Rot.

Para asistir a los clientes es necesario que interactues de manera clara y concisa con el cliente 
hasta que tengas informacion de la marca, modelo y año de su vehículo. No es necesaria informacion
acerca del modelo especifico o la linea del vehículo o combustible.

Pregunta: {question}

Cuando tengas informacion sobre la marca, modelo y año del vehiculo
debes anexar el siguiente texto a la pregunta del cliente:
'Tu vehiculo es el siguiente:

* Marca: {marca}
* Modelo: {modelo}
* Año: {ano}'
"""


prompt = PromptTemplate.from_template(template)

chain = prompt | model | parser

if "marca" not in st.session_state:
    st.session_state.marca = None
if "modelo" not in st.session_state:
    st.session_state.modelo = None
if "ano" not in st.session_state:
    st.session_state.ano = None
if "context" not in st.session_state:
    st.session_state.context = ""



# React to user input
if question := st.chat_input("Escriba su consulta aquí."):
    # Display user message in chat message container
    st.chat_message("user").markdown(question)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": question})

    response = chain.invoke({
        "context": st.session_state.context,
        "marca": st.session_state.marca,
        "modelo": st.session_state.modelo,
        "ano": st.session_state.ano,
        "question": question
    })
    
    st.session_state.context = f"""
        Respuestas anteriores: {response}
        Marca actual: {st.session_state.marca}
        Modelo actual: {st.session_state.modelo}
        Año actual: {st.session_state.ano}
        """

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
        st.session_state.marca = marca_match.group(1).upper()
    if modelo_match:
        st.session_state.modelo = modelo_match.group(1).upper()
    if ano_match:
        st.session_state.ano = ano_match.group(1).upper()

if st.session_state.marca != None and st.session_state.modelo != None and st.session_state.ano != None:
    filtered_df = df[
    (df['Marca'] == st.session_state.marca.upper()) & 
    (df['Modelo'] == st.session_state.modelo.upper()) & 
    (df['Ano_inicial'] <= st.session_state.ano) & 
    (df['Ano_fin_filtro'] >= st.session_state.ano)  # Use st.session_state.ano here
]

    filtered_df = filtered_df[['Codigo', "TipoProducto", "Linea", 'Marca', 'Modelo', 'Posicion', 'Ano_inicial', 'Ano_fin']]

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown('He encontrado los siguientes productos para tu vehículo:')
        st.dataframe(data = filtered_df, hide_index=True)
