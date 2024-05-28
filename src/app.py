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
        'fibrous': 'f', 'grooves': 'g', 'scaly': 'y', 'smooth': 's'
    },
    'cap-color': {
        'brown': 'n', 'buff': 'b', 'cinnamon': 'c', 'gray': 'g', 'green': 'r', 'pink': 'p',
        'purple': 'u', 'red': 'e', 'white': 'w', 'yellow': 'y'
    },
    'bruises': {
        'bruises': 't', 'no': 'f'
    },
    'odor': {
        'almond': 'a', 'anise': 'l', 'creosote': 'c', 'fishy': 'y', 'foul': 'f', 'musty': 'm',
        'none': 'n', 'pungent': 'p', 'spicy': 's'
    },
    'gill-attachment': {
        'attached': 'a', 'descending': 'd', 'free': 'f', 'notched': 'n'
    },
    'gill-spacing': {
        'close': 'c', 'crowded': 'w', 'distant': 'd'
    },
    'gill-size': {
        'broad': 'b', 'narrow': 'n'
    },
    'gill-color': {
        'black': 'k', 'brown': 'n', 'buff': 'b', 'chocolate': 'h', 'gray': 'g', 'green': 'r',
        'orange': 'o', 'pink': 'p', 'purple': 'u', 'red': 'e', 'white': 'w', 'yellow': 'y'
    },
    'stalk-shape': {
        'enlarging': 'e', 'tapering': 't'
    },
    'stalk-root': {
        'bulbous': 'b', 'club': 'c', 'cup': 'u', 'equal': 'e', 'rhizomorphs': 'z', 'rooted': 'r', 'missing': '?'
    },
    'stalk-surface-above-ring': {
        'fibrous': 'f', 'scaly': 'y', 'silky': 'k', 'smooth': 's'
    },
    'stalk-surface-below-ring': {
        'fibrous': 'f', 'scaly': 'y', 'silky': 'k', 'smooth': 's'
    },
    'stalk-color-above-ring': {
        'brown': 'n', 'buff': 'b', 'cinnamon': 'c', 'gray': 'g', 'orange': 'o', 'pink': 'p',
        'red': 'e', 'white': 'w', 'yellow': 'y'
    },
    'stalk-color-below-ring': {
        'brown': 'n', 'buff': 'b', 'cinnamon': 'c', 'gray': 'g', 'orange': 'o', 'pink': 'p',
        'red': 'e', 'white': 'w', 'yellow': 'y'
    },
    'veil-type': {
        'partial': 'p'
    },
    'veil-color': {
        'brown': 'n', 'orange': 'o', 'white': 'w', 'yellow': 'y'
    },
    'ring-number': {
        'none': 'n', 'one': 'o', 'two': 't'
    },
    'ring-type': {
        'cobwebby': 'c', 'evanescent': 'e', 'flaring': 'f', 'large': 'l', 'none': 'n', 'pendant': 'p', 'sheathing': 's', 'zone': 'z'
    },
    'spore-print-color': {
        'black': 'k', 'brown': 'n', 'buff': 'b', 'chocolate': 'h', 'green': 'r', 'orange': 'o',
        'purple': 'u', 'white': 'w', 'yellow': 'y'
    },
    'population': {
        'abundant': 'a', 'clustered': 'c', 'numerous': 'n', 'scattered': 's', 'several': 'v', 'solitary': 'y'
    },
    'habitat': {
        'grasses': 'g', 'leaves': 'l', 'meadows': 'm', 'paths': 'p', 'urban': 'u', 'waste': 'w', 'woods': 'd'
    }
}

# Inversión de los diccionarios para buscar por valor
# attribute_info_inverted = {attr: {v: k for k, v in d.items()} for attr, d in attribute_info.items()}

# Crear un formulario para introducir las características del hongo
st.header('Introduce las características del hongo')

# Generar las selecciones con las descripciones completas
features = {}
for attribute, mapping in attribute_info.items():
    selected_value = st.selectbox(attribute.replace('-', ' ').title(), list(mapping.keys()))
    features[attribute] = mapping[selected_value]

# Convertir las características a un DataFrame
features_df = pd.DataFrame([features])

# Transformar las entradas del usuario usando los LabelEncoders
for column in features_df.columns:
    le = label_encoders[column]
    features_df[column] = le.transform(features_df[column])

# Hacer predicción
if st.button('Predecir'):
    prediction = model.predict(features_df)
    if prediction == 1:
        st.write('El hongo es **venenoso**.')
    else:
        st.write('El hongo es **comestible**.')