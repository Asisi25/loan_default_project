import streamlit as st
import pandas as pd
import joblib
import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# -------------------------------
# 1️⃣ Page Configuration
# -------------------------------
st.set_page_config(page_title="💳 Loan Default Prediction", layout="wide")
st.title("💳 Loan Default Prediction System")
st.write("Test and compare multiple machine learning models")

# -------------------------------
# 2️⃣ Define base directories
# -------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

# -------------------------------
# 3️⃣ Load model metrics
# -------------------------------
metrics_path = os.path.join(BASE_DIR, "model_metrics.csv")
try:
    metrics_df = pd.read_csv(metrics_path)
    st.subheader("Model Performance Summary")
    st.dataframe(metrics_df)
except Exception as e:
    st.warning(f"Failed to load model metrics: {e}")

# -------------------------------
# 4️⃣ Load models safely (memory-friendly)
# -------------------------------
st.subheader("Load Models")

MODEL_NAMES = ["Logistic Regression", "Linear SVM", "Neural Network (MLP)", "Random Forest"]
loaded_models = {}

for name in MODEL_NAMES:
    model_file = os.path.join(MODELS_DIR, name.lower().replace(" ", "_") + "_model.pkl")
    try:
        loaded_models[name] = joblib.load(model_file)
        st.success(f"{name} ✅ loaded successfully")
    except MemoryError:
        st.warning(f"{name} ❌ Not enough memory to load this model. Skipping it.")
    except Exception as e:
        st.error(f"{name} ❌ Failed to load. Error: {e}")

if not loaded_models:
    st.error("No models could be loaded due to memory limits or errors.")
    st.stop()

# -------------------------------
# 5️⃣ Model selection
# -------------------------------
model_name = st.selectbox("Select Model", list(loaded_models.keys()))
model = loaded_models[model_name]

# -------------------------------
# 6️⃣ Threshold tuning
# -------------------------------
threshold = st.slider(
    "Adjust prediction threshold (for recall tuning)",
    min_value=0.0,
    max_value=1.0,
    value=0.5
)

# -------------------------------
# 7️⃣ Load dataset (Google Drive CSV)
# -------------------------------
csv_url = "https://drive.google.com/uc?id=1SMbShCB72w5pglNN3Cv19cEIEOaoBGex"
try:
    df = pd.read_csv(csv_url)
except Exception as e:
    st.error(f"Failed to load dataset from Google Drive. Error: {e}")
    st.stop()

# Determine input columns (exclude target and IDs)
X_columns = df.drop(["LoanID", "Default"], axis=1).columns

# -------------------------------
# 8️⃣ Single Applicant Prediction
# -------------------------------
st.subheader("Single Applicant Prediction")

# Collect user input
user_input = {}
for col in X_columns:
    if df[col].dtype in ["int64", "float64"]:
        user_input[col] = st.number_input(col, value=float(df[col].median()))
    else:
        user_input[col] = st.selectbox(col, df[col].unique())

input_df = pd.DataFrame([user_input])

if st.button("Predict Single Applicant"):
    try:
        # Use predict_proba if available
        if hasattr(model, "predict_proba"):
            pred_prob = model.predict_proba(input_df)[:, 1][0]
        else:
            # Fallback for Linear SVM or models without predict_proba
            score = model.decision_function(input_df)
            pred_prob = MinMaxScaler().fit_transform(score.reshape(-1, 1))[0][0]

        pred_class = int(pred_prob > threshold)

        if pred_class == 1:
            st.error(f"⚠️ High Default Risk (Probability: {pred_prob:.2f})")
        else:
            st.success(f"✅ Low Default Risk (Probability: {pred_prob:.2f})")
    except Exception as e:
        st.warning(f"Prediction failed. Check your inputs. Error: {e}")

# -------------------------------
# 9️⃣ Batch Prediction (CSV Upload)
# -------------------------------
st.subheader("Batch Prediction (CSV Upload)")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:
    try:
        batch_df = pd.read_csv(uploaded_file)

        # Predict probabilities
        if hasattr(model, "predict_proba"):
            pred_probs = model.predict_proba(batch_df)[:, 1]
        else:
            scores = model.decision_function(batch_df)
            pred_probs = MinMaxScaler().fit_transform(scores.reshape(-1, 1)).ravel()

        # Apply threshold
        pred_classes = (pred_probs > threshold).astype(int)

        # Append predictions
        batch_df["Prediction"] = pred_classes
        batch_df["Default_Probability"] = pred_probs

        st.write(batch_df.head())
        st.download_button(
            "Download Predictions",
            batch_df.to_csv(index=False),
            "predictions.csv"
        )
    except Exception as e:
        st.error(f"Batch prediction failed. Check CSV format and columns. Error: {e}")
