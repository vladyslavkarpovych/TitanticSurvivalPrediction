# źródło danych [https://www.kaggle.com/c/titanic/](https://www.kaggle.com/c/titanic)

import streamlit as st
import pickle
from datetime import datetime
startTime = datetime.now()
import pathlib
from pathlib import Path

import os

if os.name == "nt":  # Check if the OS is Windows
    temp = pathlib.PosixPath
    pathlib.PosixPath = pathlib.WindowsPath

filename = "model.sv"
model = pickle.load(open(filename, 'rb'))

# Mapping dictionaries
pclass_d = {0: "First Class", 1: "Second Class", 2: "Third Class"}
embarked_d = {0: "Cherbourg", 1: "Queenstown", 2: "Southampton"}
sex_d = {0: "Female", 1: "Male"}   # Added labels for "women" and "men"

def main():
    # Application configuration
    st.set_page_config(page_title="Titanic Survival Prediction")

    # Application sections
    overview = st.container()
    left, right = st.columns(2)
    prediction = st.container()

    # Updated image to a more relevant one
    st.image("titanic.webp")

    # Overview section
    with overview:
        st.title("Let's 'Try' to survive")

    # Input fields in the left column
    with left:
        sex_radio = st.radio("Gender", list(sex_d.keys()), format_func=lambda x: sex_d[x])
        embarked_radio = st.radio(
            "Port of Embarkation", 
            list(embarked_d.keys()), 
            index=2, 
            format_func=lambda x: embarked_d[x]
        )
        pclass_radio = st.radio("Class", list(pclass_d.keys()), format_func=lambda x: pclass_d[x])  # Added new variable

     # Input fields in the right column
    with right:
        age_slider = st.slider("Age", min_value=0, max_value=80, step=1)  # Adjusted min/max values
        sibsp_slider = st.slider("Number of Siblings/Spouses", min_value=0, max_value=8, step=1)
        parch_slider = st.slider("Number of Parents/Children", min_value=0, max_value=6, step=1)
        fare_slider = st.slider("Ticket Fare", min_value=0, max_value=512, step=1)  # Based on dataset max fare

    # Prediction
    data = [[pclass_radio, sex_radio, age_slider, sibsp_slider, parch_slider, fare_slider, embarked_radio]]
    survival = model.predict(data)
    s_confidence = model.predict_proba(data)

    with prediction:
        st.subheader("Would this person survive the disaster?")
        st.subheader("Yes" if survival[0] == 1 else "No")
        st.write("Prediction Confidence: {:.2f}%".format(s_confidence[0][survival[0]] * 100))

if __name__ == "__main__":
    main()
