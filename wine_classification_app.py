import streamlit as st
from predict_page import show_predict_page

st.markdown(""" <style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style> """, unsafe_allow_html=True)

logo = st.sidebar.image('Wine.jpg')

description = st.sidebar.markdown('Machine learning model was built from UCI wine dataset.  \nThese data are the results of a chemical analysis of wines grown in the same region in Italy but derived from three different cultivars.  \nThe analysis determined the quantities of 13 constituents found in each of the three types of wines.  \nDatasource: https://archive.ics.uci.edu/ml/datasets/wine')

show_predict_page()
