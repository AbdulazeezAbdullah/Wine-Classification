import streamlit as st
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor

def load_model():
    with open('Wine_Classification_Model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model
        
wine_model = load_model()


def show_predict_page():
    st.header('Wine Classification')
    
    st.subheader('We need some information to analyze the type of wine you purchased')
    
    Alcohol = st.slider('Alcohol', min_value=11.0, max_value=15.0, step=0.01)
    Malic_acid = st.slider('Malic acid', min_value=0.70, max_value=6.00, step=0.01)
    Ash = st.slider('Ash', min_value=1.30, max_value=3.30, step=0.01)
    Alcalinity = st.slider('Alcalinity of ash', min_value=10.00, max_value=30.00, step=0.10)
    Magnesium = st.slider('Magnesium', min_value=60, max_value=170, step=1)
    Phenols = st.slider('Total phenols', min_value=0.80, max_value=4.00, step= 0.01)
    Flavanoids = st.slider('Flavanoids', min_value=0.30, max_value=6.00, step=0.01)
    Nonflavanoids = st.slider('Nonflavanoid phenols', min_value=0.10, max_value=0.70, step=0.01)
    Proanthocyanins = st.slider('Proanthocyanins', min_value=0.40, max_value=4.00, step=0.01)
    Color_intensity = st.slider('Color intensity', min_value=1.00, max_value=13.00, step=0.01)
    Hue =st.slider('Hue', min_value=0.40, max_value=2.00, step=0.01)
    OD280_315_of_diluted_wines = st.slider('OD280/OD315 of diluted wines', min_value=1.20, max_value=4.00, step=0.01)
    Proline = st.slider('Proline', min_value=270, max_value=1680, step=1)
    
    input = st.button("Predict Wine Category")
    
    classes = ['Variety A','Variety B','Variety C']
    
    if input:
        input = np.array([[Alcohol, Malic_acid, Ash, Alcalinity, Magnesium, Phenols,
                           Flavanoids, Nonflavanoids, Proanthocyanins, Color_intensity,
                            Hue, OD280_315_of_diluted_wines, Proline]])
        
        
        predictions = wine_model.predict(input)
        
        for category in predictions:
            wine_category = classes[category]
    
        st.subheader(f'The wine category is: {wine_category}')