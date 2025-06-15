import streamlit as st
import joblib
import os
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



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
@st.cache_data
def load_astronaut_data():
    return pd.read_csv("astronauts_clean.csv")

df_astronauts = load_astronaut_data() 


@st.cache_resource
def load_model(path):
    if os.path.exists(path):
        return joblib.load(path)
    else:
        st.error(f"Model not found at: {path}")
        st.stop()
model = load_model(MODEL_PATH)

 
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select a page:", [
    "Home",
    "Dataset",
    "Visualizations",
    "Prediction"
])
if page == "Home":
    st.title("Astronaut Analytics and Space Walk Prediction")
    st.write("""
        This application provides:
        - Insights into astronaut demographics and career statistics
        - Interactive data visualizations
        - A machine learning model to predict estimated space walk hours
    """)
    
elif page == "Dataset":
    st.title("Astronaut Dataset")
    st.dataframe(df_astronauts)
    
elif page == "Visualizations":
    st.title("Exploratory Data Visualizations")

    st.subheader("Astronauts by Gender Over Time")
    gender_trend = df_astronauts.groupby(['Year', 'Gender']).size().reset_index(name='Count')
    fig1, ax1 = plt.subplots()
    sns.lineplot(data=gender_trend, x='Year', y='Count', hue='Gender', marker='o', ax=ax1)
    ax1.set_title("Number of Astronauts by Gender Over Time")
    st.pyplot(fig1)

    st.subheader("Missing and Unique Values per Column")
    missingValues = df_astronauts.isna().sum().sort_values(ascending=False)
    unique_values = df_astronauts.nunique().sort_values(ascending=False)
    summary = pd.DataFrame({'Missing Values': missingValues, 'Unique Values': unique_values})
    fig2, ax2 = plt.subplots()
    summary.plot(kind='bar', ax=ax2)
    ax2.set_title("Missing vs Unique Value Counts")
    st.pyplot(fig2)
    st.subheader("Space Walks vs Total Walk Hours")
    fig3, ax3 = plt.subplots()
    ax3.scatter(df_astronauts['Space Walks'], df_astronauts['Space Walks (hr)'])
    ax3.set_title("Space Walk Count vs Total Space Walk Hours")
    ax3.set_xlabel("Space Walks")
    ax3.set_ylabel("Walk Hours")
    st.pyplot(fig3)

    st.subheader("Space Flights vs Total Flight Hours")
    fig4, ax4 = plt.subplots()
    ax4.scatter(df_astronauts['Space Flights'], df_astronauts['Space Flight (hr)'])
    ax4.set_title("Space Flight Count vs Total Flight Hours")
    ax4.set_xlabel("Space Flights")
    ax4.set_ylabel("Flight Hours")
    st.pyplot(fig4)
    
    st.subheader("Distribution of Age at Death")
    dates = df_astronauts.loc[df_astronauts['Death Date'].notna(), ['Birth Date', 'Death Date']]
    age = (pd.to_datetime(dates['Death Date'], format="mixed") - pd.to_datetime(dates['Birth Date'], format="mixed")).dt.days // 365
    fig5, ax5 = plt.subplots(figsize=(6, 5))
    ax5.hist(age, bins=10, color='skyblue', edgecolor='black')
    ax5.set_xlabel("Age at Death (Years)")
    ax5.set_ylabel("Number of Astronauts")
    ax5.set_title("Distribution of Age at Death")
    ax5.grid(True)
    st.pyplot(fig5)

    st.subheader("Distribution of Space Flights by Gender")
    fig6, ax6 = plt.subplots()
    sns.boxplot(x='Gender', y='Space Flights', data=df_astronauts, ax=ax6)
    ax6.set_title("Space Flights by Gender")
    st.pyplot(fig6)
    
    corr = df_astronauts.corr(numeric_only=True)

    # Plot using Streamlit
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', center=0, square=True, ax=ax)
    ax.set_title("Correlation Heatmap")
    st.pyplot(fig)


    
elif page == "Prediction":
    st.title("Predict Estimated Space Walk Hours")

    col1, col2 = st.columns(2)
    with col1:
        year = st.number_input("Year of Selection", 1959, datetime.now().year, value=datetime.now().year)
        gender_label = st.selectbox("Gender", list(GENDER_MAP.keys()))
        space_flights = st.number_input("Number of Space Flights", min_value=0)
    with col2:
        space_walks = st.number_input("Number of Space Walks", min_value=0)
        rank_label = st.selectbox("Military Rank", list(RANK_OPTIONS.keys()))
        rank_encoded = RANK_OPTIONS[rank_label]

    if st.button("Predict"):
        if model:
            input_vector = [[
                year,
                GENDER_MAP[gender_label],
                space_flights,
                space_walks,
                rank_encoded
            ]]
            prediction = model.predict(input_vector)[0]
            st.success(f"Predicted Space Walk Hours: {prediction:.2f}")
        else:
            st.error("Model not found or failed to load.")