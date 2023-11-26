from StreamLitApp.preprocessing import DataPreprocessor
import streamlit as st
import pickle
import pandas as pd

#with open('finalized_model.sav', 'rb') as file:
#trained_model = pickle.load(file)


preprocessor = DataPreprocessor()

# stremlit app
st.title("Cardio Prediction App")

# Collect user inputs
st.sidebar.header("User Input:")
age = st.sidebar.slider("Age", 0, 100, 50)
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
height = st.sidebar.slider("Height (cm)", 0, 300, 165)
weight = st.sidebar.slider("Weight (kg)", 0, 300, 70)
ap_hi = st.sidebar.slider("Systolic Blood Pressure (ap_hi)", 50, 250, 120)
ap_lo = st.sidebar.slider("Diastolic Blood Pressure (ap_lo)", 30, 200, 80)
cholesterol = st.sidebar.selectbox("Cholesterol Level", ["Normal", "Above Normal", "High"])
gluc = st.sidebar.selectbox("Glucose Level", ["Normal", "Above Normal", "High"])
smoke = st.sidebar.selectbox("Smoking", ["No", "Yes"])
alco = st.sidebar.selectbox("Alcohol Consumption", ["No", "Yes"])
active = st.sidebar.selectbox("Physical Activity", ["No", "Yes"])

# Map user inputs to data
gender_mapping = {"Male": 0, "Female": 1}
cholesterol_mapping = {"Normal": 1, "Above Normal": 2, "High": 3}
gluc_mapping = {"Normal": 1, "Above Normal": 2, "High": 3}
smoke_mapping = {"No": 0, "Yes": 1}
alco_mapping = {"No": 0, "Yes": 1}
active_mapping = {"No": 0, "Yes": 1}

user_input = {
    'age': age,
    'gender': gender_mapping[gender],
    'height': height,
    'weight': weight,
    'ap_hi': ap_hi,
    'ap_lo': ap_lo,
    'cholesterol': cholesterol_mapping[cholesterol],
    'gluc': gluc_mapping[gluc],
    'smoke': smoke_mapping[smoke],
    'alco': alco_mapping[alco],
    'active': active_mapping[active]
}

user_input_data = pd.DataFrame(user_input, index=[0])


final_df = preprocessor.preprocess_and_cluster(user_input_data)

# make predictions
predictions = trained_model.predict(final_df)

st.sidebar.header("Cardio Prediction: ")
if predictions[0] == 1:
    st.sidebar.warning("Risk of Cardiovascular Disease Detected!")
else:
    st.sidebar.success("No Cardiovascular Disease Detected")






