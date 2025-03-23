import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load trained model
model_path = "enhanced_rf_model.pkl"
model = joblib.load(model_path)

# Define performance categories
def categorize_performance(score):
    if score >= 7.5:
        return "High Performance 🚀"
    elif score >= 5:
        return "Moderate Performance 🔄"
    else:
        return "Needs Improvement ⚠️"

# Prediction function
def predict_score(input_time):
    try:
        input_timestamp = int(pd.to_datetime(input_time).timestamp())
        input_df = pd.DataFrame([[input_timestamp]], columns=["timestamp"])
        predicted_score = model.predict(input_df)[0]
        category = categorize_performance(predicted_score)
        return {"datetime": input_time, "score": round(predicted_score, 2), "category": category}
    except Exception as e:
        return {"datetime": input_time, "score": "Error", "category": str(e)}

# Streamlit UI
st.set_page_config(page_title="Duolingo Learning Predictor", page_icon="📚", layout="centered")
st.title("📚 Duolingo Learning Performance Predictor")
st.write("Enter one or multiple date & time values to predict the learning performance score.")

# User Input
input_dates = st.text_area("Enter Date & Time (One per line, format: YYYY-MM-DD HH:MM:SS)",
                           "2022-07-01 18:00:00\n2024-03-23 14:30:00")

if st.button("🔮 Predict Scores"):
    date_list = input_dates.split("\n")
    results = [predict_score(date.strip()) for date in date_list if date.strip()]
    df_results = pd.DataFrame(results)

    if not df_results.empty:
        st.write("### 📊 Predicted Scores:")
        st.dataframe(df_results)

        # Visualization
        colors = {"High Performance 🚀": "green", "Moderate Performance 🔄": "orange", "Needs Improvement ⚠️": "red"}
        df_results["color"] = df_results["category"].map(colors)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh(df_results["datetime"], df_results["score"], color=df_results["color"])
        ax.set_xlabel("Predicted Score")
        ax.set_ylabel("Date & Time")
        ax.set_title("Predicted Scores Across Dates")
        st.pyplot(fig)

st.sidebar.header("ℹ️ About")
st.sidebar.write("This app predicts learning performance scores based on timestamps using a trained Random Forest model.")

st.success("✅ Ready for Predictions!")
#put below code in terminal to run this
#streamlit run E:\Python\SCDIC-1\duostory.py
