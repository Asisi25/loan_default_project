# ============================================================
# LOAN DEFAULT PREDICTION WEB APP (STREAMLIT)
# ============================================================

import streamlit as st
import pandas as pd
import joblib
import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# ============================================================
# 1️⃣ PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="💳 Loan Default Prediction",
    layout="wide"
)

st.title("💳 Loan Default Prediction System")
st.write(
    """
    This dashboard allows you to:
    - Compare multiple machine learning models
    - Tune prediction threshold for recall-sensitive decisions
    - Predict loan default risk for single or multiple applicants
    """
)

# ============================================================
# 2️⃣ DIRECTORY SETUP
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

# ============================================================
# 3️⃣ LOAD MODEL METRICS (REAL RESULTS)
# ============================================================
st.subheader("📊 Model Performance Summary")

metrics_path = os.path.join(BASE_DIR, "model_metrics.csv")

try:
    metrics_df = pd.read_csv(metrics_path)
    st.dataframe(metrics_df)

    st.subheader("Model Comparison (Recall & ROC-AUC)")
    st.bar_chart(
        metrics_df.set_index("Model")[["Recall", "ROC_AUC"]]
    )

except Exception as e:
    st.warning(f"Could not load model metrics: {e}")

# ============================================================
# 4️⃣ LOAD TRAINED MODELS SAFELY
# ============================================================
st.subheader("📦 Load Trained Models")

MODEL_NAMES = [
    "Logistic Regression",
    "Random Forest",
    "Linear SVM",
    "Neural Network (MLP)"
]

loaded_models = {}

for name in MODEL_NAMES:
    file_name = name.lower().replace(" ", "_") + "_model.pkl"
    model_path = os.path.join(MODELS_DIR, file_name)

    try:
        loaded_models[name] = joblib.load(model_path)
        st.success(f"{name} loaded successfully ✅")
    except Exception as e:
        st.error(f"{name} failed to load: {e}")

if not loaded_models:
    st.error("No models available. Please train models first.")
    st.stop()

# ============================================================
# 5️⃣ MODEL SELECTION
# ============================================================
model_name = st.selectbox(
    "Select a model for prediction",
    list(loaded_models.keys())
)

model = loaded_models[model_name]

# ============================================================
# 6️⃣ THRESHOLD TUNING (RECALL-FOCUSED)
# ============================================================
st.subheader("🎚️ Threshold Tuning")

threshold = st.slider(
    "Adjust decision threshold (lower = higher recall)",
    min_value=0.05,
    max_value=0.95,
    value=0.5,
    step=0.01
)

st.info(
    "Lower thresholds capture more defaulters (higher recall) "
    "but may increase false positives."
)

# ============================================================
# 7️⃣ LOAD DATASET (FOR INPUT SCHEMA)
# ============================================================
@st.cache_data
def load_reference_data():
    return pd.read_csv(os.path.join(BASE_DIR, "..", "Loan_default_prediction.csv"))

df = load_reference_data()
X_columns = df.drop(columns=["LoanID", "Default"]).columns

# ============================================================
# 8️⃣ SINGLE APPLICANT PREDICTION
# ============================================================
st.subheader("🧍 Single Applicant Prediction")

user_input = {}

for col in X_columns:
    if df[col].dtype in ["int64", "float64"]:
        user_input[col] = st.number_input(
            col,
            value=float(df[col].median())
        )
    else:
        user_input[col] = st.selectbox(
            col,
            sorted(df[col].dropna().unique())
        )

input_df = pd.DataFrame([user_input])

if st.button("Predict Default Risk"):
    try:
        model_step = model.named_steps["model"]

        # Probability handling across model types
        if hasattr(model_step, "predict_proba"):
            prob = model.predict_proba(input_df)[:, 1][0]
        else:
            score = model.decision_function(input_df)
            prob = MinMaxScaler().fit_transform(
                score.reshape(-1, 1)
            )[0][0]

        prediction = int(prob > threshold)

        if prediction == 1:
            st.error(f"⚠️ High Default Risk (Probability: {prob:.2f})")
        else:
            st.success(f"✅ Low Default Risk (Probability: {prob:.2f})")

    except Exception as e:
        st.error(f"Prediction failed: {e}")

# ============================================================
# 9️⃣ BATCH PREDICTION (CSV UPLOAD)
# ============================================================
st.subheader("📂 Batch Prediction")

uploaded_file = st.file_uploader(
    "Upload CSV file (same structure as training data)",
    type=["csv"]
)

if uploaded_file:
    try:
        batch_df = pd.read_csv(uploaded_file)

        model_step = model.named_steps["model"]

        if hasattr(model_step, "predict_proba"):
            probs = model.predict_proba(batch_df)[:, 1]
        else:
            scores = model.decision_function(batch_df)
            probs = MinMaxScaler().fit_transform(
                scores.reshape(-1, 1)
            ).ravel()

        preds = (probs > threshold).astype(int)

        batch_df["Default_Probability"] = probs
        batch_df["Prediction"] = preds

        st.write(batch_df.head())

        st.download_button(
            "Download Predictions",
            batch_df.to_csv(index=False),
            "loan_default_predictions.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"Batch prediction failed: {e}")

# ============================================================
# 🔟 FEATURE IMPORTANCE (CORRECT & LABELED)
# ============================================================
st.subheader("📌 Feature Importance / Interpretability")

try:
    model_step = model.named_steps["model"]
    preprocessor = model.named_steps["preprocessor"]

    # Rebuild feature names from preprocessing pipeline
    numeric_features = preprocessor.transformers_[0][2]

    low_cat_features = (
        preprocessor
        .named_transformers_["low_cat"]
        .named_steps["encoder"]
        .get_feature_names_out(
            preprocessor.transformers_[1][2]
        )
    )

    high_cat_features = preprocessor.transformers_[2][2]

    feature_names = np.concatenate(
        [numeric_features, low_cat_features, high_cat_features]
    )

    # Tree-based models
    if hasattr(model_step, "feature_importances_"):
        importances = model_step.feature_importances_

    # Linear models (Logistic Regression, Linear SVM)
    elif hasattr(model_step, "coef_"):
        importances = np.abs(model_step.coef_[0])

    else:
        st.info("Feature importance not supported for this model.")
        st.stop()

    fi_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values("Importance", ascending=False).head(15)

    st.bar_chart(
        fi_df.set_index("Feature")
    )

except Exception as e:
    st.warning(f"Could not display feature importance: {e}")
