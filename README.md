# 🧑‍🚀 NASA Astronauts – EDA & Spacewalk Prediction App

This project explores the NASA Astronauts dataset through **Exploratory Data Analysis (EDA)** and builds a **Streamlit web app** to predict whether an astronaut is likely to perform a **spacewalk**, using a **Random Forest Classifier**.

---

## 📊 Project Overview

### ✅ Objectives:
- Perform **comprehensive EDA** to understand astronaut demographics and mission trends.
- Handle **missing values**, apply **encoding**, and explore **correlations**.
- Use **grouping** and **visual trends** to identify patterns.
- Train a **Random Forest model** to predict spacewalk participation.
- Deploy a **Streamlit app** for real-time predictions.

---

## 📁 Project Structure

```
nasa-astronauts/
│
├── app.py                 # Streamlit app
├── astronauts.ipynb              # EDA notebook
├── rf_model.pkl       # Trained RandomForest model
├── 
│  astronauts.csv     # NASA Astronauts dataset
└── README.md              # Project documentation
```

---

## 📌 Features

### 🔍 Exploratory Data Analysis (EDA)
- Trend analysis by **year**
- Visualization of **spaceflight hours** and **mission counts**
- **Correlation heatmaps** to detect feature relationships
- Handling **missing values** with appropriate strategies
- **Categorical encoding** for machine learning compatibility
- **Grouping** (e.g., by gender, military service, selection year)

### 🤖 Machine Learning
- Target variable: **Spacewalk Participation**
- Features: Age, Gender, Military Background, Flight Hours, etc.
- Model: **RandomForestClassifier**
- Evaluation: Accuracy, confusion matrix, and feature importance

### 🌐 Streamlit App
- User-friendly interface to enter astronaut profile
- Predict whether they will perform a **spacewalk**
- Live model interaction with informative results

---
 

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Launch the Streamlit App
```bash
streamlit run app.py
```
 

## 📦 Requirements

- pandas  
- numpy  
- matplotlib / seaborn  
- scikit-learn  
- streamlit  

 
