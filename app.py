import streamlit as st
import joblib
import os
from datetime import datetime


st.set_page_config(page_title="Astronaut Space Walk Predictor", page_icon="🚀")

MODEL_PATH = "rf_model.pkl"
ENCODER_PATH = "gender_encoder.pkl"
MODEL_FEATURES = ['Year', 'gender_encode', 'Space Flights', 'Space Walks', 'rank_encode']

# Encoding

GENDER_MAP = {"Male": 1, "Female": 0}
RANK_OPTIONS = {
    "Colonel": 1,
    "Commander": 2,
    "Lieutenant Colonel": 3,
    "Captain": 4,
    "Other": 5
}
 
@st.cache_resource
def load_model(path):
    if os.path.exists(path):
        return joblib.load(path)
    else:
        st.error(f"Model not found at: {path}")
        st.stop()
 

model = load_model(MODEL_PATH) 
st.title(" Predict Astronaut Space Walk Hours")

st.markdown("Fill in astronaut details below:")
col1, col2 = st.columns(2)

with col1:
    current_year = datetime.now().year
    year = st.number_input("🗓️ Year", min_value=1959, max_value=current_year, step=1, value=current_year)

    gender_label = st.selectbox("👤 Gender", list(GENDER_MAP.keys()))
    space_flights = st.number_input("🚀 Space Flights", min_value=0, step=1)

with col2:
    space_walks = st.number_input("🚶 Space Walks", min_value=0, step=1)
    rank_label = st.selectbox("🎖️ Military Rank", list(RANK_OPTIONS.keys()))
    rank_encoded = RANK_OPTIONS[rank_label]
 
if st.button("🔮 Predict Space Walk Hours"):
    try:
        gender_encoded = GENDER_MAP[gender_label]

        input_vector = [[
            year,
            gender_encoded,
            space_flights,
            space_walks,
            rank_encoded
        ]]

        prediction = model.predict(input_vector)[0]
        st.success(f"✅ Predicted Space Walk Hours: **{prediction:.2f}**")

    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")