import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle

# Título de la aplicación
st.title('Predicción de Hongos Venenosos')

# Cargar el modelo entrenado y los label encoders
model = pickle.load(open('mushroom_random_forest.model', 'rb'))
label_encoders = pickle.load(open('label_encoders.pkl', 'rb'))

# Diccionarios para mapear descripciones a valores codificados
attribute_info = {
    'cap-shape': {
        'bell': 'b', 'conical': 'c', 'convex': 'x', 'flat': 'f', 'knobbed': 'k', 'sunken': 's'
    },
    'cap-surface': {
        'fibroso': 'f', 'ranurado': 'g', 'escamoso': 'y', 'suave': 's'
    },
    'cap-color': {
        'marrón': 'n', 'ocre': 'b', 'canela': 'c', 'gris': 'g', 'verde': 'r', 'rosa': 'p',
        'púrpura': 'u', 'rojo': 'e', 'blanco': 'w', 'amarillo': 'y'
    },
    'bruises': {
        'magulladuras': 't', 'no': 'f'
    },
    'odor': {
        'almendra': 'a', 'anís': 'l', 'creosota': 'c', 'pescado': 'y', 'hediondo': 'f', 'moho': 'm',
        'ninguno': 'n', 'picante': 'p', 'especiado': 's'
    },
    'gill-attachment': {
        'adjunto': 'a', 'libre': 'f'
    },
    'gill-spacing': {
        'cerrado': 'c', 'apretado': 'w', 'distante': 'd'
    },
    'gill-size': {
        'ancho': 'b', 'estrecho': 'n'
    },
    'gill-color': {
        'negro': 'k', 'marrón': 'n', 'ocre': 'b', 'chocolate': 'h', 'gris': 'g', 'verde': 'r',
        'naranja': 'o', 'rosa': 'p', 'púrpura': 'u', 'rojo': 'e', 'blanco': 'w', 'amarillo': 'y'
    },
    'stalk-shape': {
        'engrosamiento': 'e', 'estrechamiento': 't'
    },
    'stalk-root': {
        'bulboso': 'b', 'club': 'c', 'copa': 'u', 'igual': 'e', 'rizomorfos': 'z', 'enraizado': 'r', 'ausente': '?'
    },
    'stalk-surface-above-ring': {
        'fibroso': 'f', 'escamoso': 'y', 'sedoso': 'k', 'suave': 's'
    },
    'stalk-surface-below-ring': {
        'fibroso': 'f', 'escamoso': 'y', 'sedoso': 'k', 'suave': 's'
    },
    'stalk-color-above-ring': {
        'marrón': 'n', 'ocre': 'b', 'canela': 'c', 'gris': 'g', 'naranja': 'o', 'rosa': 'p',
        'rojo': 'e', 'blanco': 'w', 'amarillo': 'y'
    },
    'stalk-color-below-ring': {
        'marrón': 'n', 'ocre': 'b', 'canela': 'c', 'gris': 'g', 'naranja': 'o', 'rosa': 'p',
        'rojo': 'e', 'blanco': 'w', 'amarillo': 'y'
    },
    'veil-type': {
        'parcial': 'p'
    },
    'veil-color': {
        'marrón': 'n', 'naranja': 'o', 'blanco': 'w', 'amarillo': 'y'
    },
    'ring-number': {
        'ninguno': 'n', 'uno': 'o', 'dos': 't'
    },
    'ring-type': {
        'evanescente': 'e', 'alargado': 'f', 'grande': 'l', 'ninguno': 'n', 'pendiente': 'p', 'vaina': 's', 'zona': 'z'
    },
    'spore-print-color': {
        'negro': 'k', 'marrón': 'n', 'ocre': 'b', 'chocolate': 'h', 'verde': 'r', 'naranja': 'o',
        'púrpura': 'u', 'blanco': 'w', 'amarillo': 'y'
    },
    'population': {
        'abundante': 'a', 'agrupado': 'c', 'numeroso': 'n', 'disperso': 's', 'varios': 'v', 'solitario': 'y'
    },
    'habitat': {
        'césped': 'g', 'hojas': 'l', 'praderas': 'm', 'caminos': 'p', 'urbano': 'u', 'residuos': 'w', 'bosques': 'd'
    }
}

# Diccionario para traducir los nombres de las columnas al español
column_names_es = {
    'cap-shape': 'Forma del sombrero',
    'cap-surface': 'Superficie del sombrero',
    'cap-color': 'Color del sombrero',
    'bruises': 'Magulladuras',
    'odor': 'Olor',
    'gill-attachment': 'Unión de las láminas',
    'gill-spacing': 'Espaciado de las láminas',
    'gill-size': 'Tamaño de las láminas',
    'gill-color': 'Color de las láminas',
    'stalk-shape': 'Forma del tallo',
    'stalk-root': 'Raíz del tallo',
    'stalk-surface-above-ring': 'Superficie del tallo por encima del anillo',
    'stalk-surface-below-ring': 'Superficie del tallo por debajo del anillo',
    'stalk-color-above-ring': 'Color del tallo por encima del anillo',
    'stalk-color-below-ring': 'Color del tallo por debajo del anillo',
    'veil-type': 'Tipo de velo',
    'veil-color': 'Color del velo',
    'ring-number': 'Número de anillos',
    'ring-type': 'Tipo de anillo',
    'spore-print-color': 'Color de la impresión de esporas',
    'population': 'Población',
    'habitat': 'Hábitat'
}

# Crear un formulario para introducir las características del hongo
st.header('Introduce las características del hongo')

# Generar las selecciones con las descripciones completas en español
features = {}
for attribute, mapping in attribute_info.items():
    translated_attribute = column_names_es[attribute]
    selected_value = st.radio(f"**{translated_attribute}**", list(mapping.keys()))
    features[attribute] = mapping[selected_value]

# Convertir las características a un DataFrame
features_df = pd.DataFrame([features])

# Transformar las entradas del usuario usando los LabelEncoders
for column in features_df.columns:
    le = label_encoders[column]
    # Verificar si el valor está en las clases conocidas del LabelEncoder
    if not set(features_df[column]).issubset(set(le.classes_)):
        st.error(f"El valor para {column} contiene etiquetas no vistas previamente: {set(features_df[column]) - set(le.classes_)}.")
        st.stop()
    features_df[column] = le.transform(features_df[column])

# Hacer predicción
if st.button('Predecir'):
    prediction = model.predict(features_df)
    if prediction == 1:
        st.write('El hongo es **venenoso**.')
    else:
        st.write('El hongo es **comestible**.')
