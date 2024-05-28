import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle

# Título de la aplicación
st.title('Predicción de Hongos Venenosos')

# Cargar el modelo entrenado
model = pickle.load(open('mushroom_random_forest.model', 'rb'))

# Crear un formulario para introducir las características del hongo
st.header('Introduce las características del hongo')

# Lista de características
features = {
    'cap-shape': st.selectbox('Cap Shape', ['b', 'c', 'x', 'f', 'k', 's']),
    'cap-surface': st.selectbox('Cap Surface', ['f', 'g', 'y', 's']),
    'cap-color': st.selectbox('Cap Color', ['n', 'b', 'c', 'g', 'r', 'p', 'u', 'e', 'w', 'y']),
    'bruises': st.selectbox('Bruises', ['t', 'f']),
    'odor': st.selectbox('Odor', ['a', 'l', 'c', 'y', 'f', 'm', 'n', 'p', 's']),
    'gill-attachment': st.selectbox('Gill Attachment', ['a', 'd', 'f', 'n']),
    'gill-spacing': st.selectbox('Gill Spacing', ['c', 'w']),
    'gill-size': st.selectbox('Gill Size', ['b', 'n']),
    'gill-color': st.selectbox('Gill Color', ['k', 'n', 'b', 'h', 'g', 'r', 'o', 'p', 'u', 'e', 'w', 'y']),
    'stalk-shape': st.selectbox('Stalk Shape', ['e', 't']),
    'stalk-root': st.selectbox('Stalk Root', ['b', 'c', 'u', 'e', 'z', 'r', '?']),
    'stalk-surface-above-ring': st.selectbox('Stalk Surface Above Ring', ['f', 'y', 'k', 's']),
    'stalk-surface-below-ring': st.selectbox('Stalk Surface Below Ring', ['f', 'y', 'k', 's']),
    'stalk-color-above-ring': st.selectbox('Stalk Color Above Ring', ['n', 'b', 'c', 'g', 'o', 'p', 'e', 'w', 'y']),
    'stalk-color-below-ring': st.selectbox('Stalk Color Below Ring', ['n', 'b', 'c', 'g', 'o', 'p', 'e', 'w', 'y']),
    'veil-type': st.selectbox('Veil Type', ['p']),
    'veil-color': st.selectbox('Veil Color', ['n', 'o', 'w', 'y']),
    'ring-number': st.selectbox('Ring Number', ['n', 'o', 't']),
    'ring-type': st.selectbox('Ring Type', ['c', 'e', 'f', 'l', 'n', 'p', 's', 'z']),
    'spore-print-color': st.selectbox('Spore Print Color', ['k', 'n', 'b', 'h', 'r', 'o', 'u', 'w', 'y']),
    'population': st.selectbox('Population', ['a', 'c', 'n', 's', 'v', 'y']),
    'habitat': st.selectbox('Habitat', ['g', 'l', 'm', 'p', 'u', 'w', 'd'])
}

# Convertir las características a un DataFrame
features_df = pd.DataFrame([features])

# Hacer predicción
if st.button('Predecir'):
    prediction = model.predict(features_df)
    if prediction == 1:
        st.write('El hongo es **venenoso**.')
    else:
        st.write('El hongo es **comestible**.')

