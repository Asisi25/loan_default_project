import streamlit as st
import pandas as pd
import joblib
import os

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(page_title="Loan Default Prediction", layout="wide")
st.title("💳 Loan Default Prediction System")
st.write("Compare models and predict loan default risk")

# =====================================================
# PATH SETUP (DO NOT CHANGE)
# =====================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))          # web_app/
PROJECT_DIR = os.path.dirname(BASE_DIR)                       # loan_default_project/

DATASET_PATH = os.path.join(PROJECT_DIR, "Loan_default_prediction.csv")
METRICS_PATH = os.path.join(BASE_DIR, "model_metrics.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")

DUMMY_LOAN_ID = "TEMP_LOAN_ID"

# =====================================================
# LOAD MODEL METRICS
# =====================================================
if not os.path.exists(METRICS_PATH):
    st.error("❌ model_metrics.csv not found in web_app/")
    st.stop()

metrics_df = pd.read_csv(METRICS_PATH)
st.subheader("📊 Model Performance Summary")
st.dataframe(metrics_df, use_container_width=True)

# =====================================================
# MODEL PATHS (LAZY LOAD TO PREVENT MEMORY ERROR)
# =====================================================
MODEL_PATHS = {
    "Logistic Regression": os.path.join(MODELS_DIR, "logistic_regression_model.pkl"),
    "Random Forest": os.path.join(MODELS_DIR, "random_forest_model.pkl"),
    "Linear SVM": os.path.join(MODELS_DIR, "linear_svm_model.pkl"),
    "Neural Network (MLP)": os.path.join(MODELS_DIR, "mlp_model.pkl"),
}

DEFAULT_MODEL = "Logistic Regression"

model_name = st.selectbox(
    "🔍 Select Model",
    list(MODEL_PATHS.keys()),
    index=list(MODEL_PATHS.keys()).index(DEFAULT_MODEL)
)

@st.cache_resource
def load_model(model_path):
    return joblib.load(model_path)

try:
    model = load_model(MODEL_PATHS[model_name])
    st.success(f"✅ {model_name} loaded successfully")
except MemoryError:
    st.error(
        "❌ Not enough memory to load this model.\n\n"
        "Try Logistic Regression or Linear SVM."
    )
    st.stop()

# =====================================================
# THRESHOLD TUNING
# =====================================================
threshold = st.slider(
    "Adjust prediction threshold (lower = higher recall)",
    0.0, 1.0, 0.5, 0.01
)

# =====================================================
# LOAD DATASET FOR INPUT SCHEMA
# =====================================================
if not os.path.exists(DATASET_PATH):
    st.error("❌ Loan_default_prediction.csv not found in project root")
    st.stop()

df = pd.read_csv(DATASET_PATH)

DROP_COLS = ["Default", "LoanID"]
FEATURE_COLUMNS = [c for c in df.columns if c not in DROP_COLS]

# =====================================================
# SINGLE APPLICANT PREDICTION
# =====================================================
st.subheader("👤 Single Applicant Prediction")

user_input = {}

for col in FEATURE_COLUMNS:
    if df[col].dtype in ["int64", "float64"]:
        user_input[col] = st.number_input(
            label=col,
            value=float(df[col].median())
        )
    else:
        user_input[col] = st.selectbox(
            label=col,
            options=sorted(df[col].dropna().unique())
        )

input_df = pd.DataFrame([user_input])

# FIX: Add LoanID back for model compatibility
input_df.insert(0, "LoanID", DUMMY_LOAN_ID)

if st.button("Predict Single Applicant"):
    try:
        prob = model.predict_proba(input_df)[:, 1][0]
        prediction = int(prob >= threshold)

        if prediction == 1:
            st.error(f"⚠️ High Default Risk\n\nProbability: **{prob:.2f}**")
        else:
            st.success(f"✅ Low Default Risk\n\nProbability: **{prob:.2f}**")

    except Exception as e:
        st.error(f"Prediction failed: {e}")

# =====================================================
# BATCH PREDICTION
# =====================================================
st.subheader("📂 Batch Prediction (CSV Upload)")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:
    batch_df = pd.read_csv(uploaded_file)

    # FIX: Ensure LoanID exists
    if "LoanID" not in batch_df.columns:
        batch_df.insert(0, "LoanID", DUMMY_LOAN_ID)

    missing_cols = set(FEATURE_COLUMNS + ["LoanID"]) - set(batch_df.columns)

    if missing_cols:
        st.error(f"❌ Missing required columns: {missing_cols}")
    else:
        try:
            probs = model.predict_proba(batch_df)[:, 1]
            preds = (probs >= threshold).astype(int)

            batch_df["Default_Probability"] = probs
            batch_df["Prediction"] = preds

            st.write("🔎 Preview Results")
            st.dataframe(batch_df.head(), use_container_width=True)

            st.download_button(
                "⬇️ Download Predictions",
                batch_df.to_csv(index=False),
                "loan_default_predictions.csv",
                "text/csv"
            )

        except Exception as e:
            st.error(f"Batch prediction failed: {e}")
