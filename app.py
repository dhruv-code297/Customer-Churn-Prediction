import pickle

import pandas as pd
import streamlit as st


MODEL_PATH = "customer_churn_model.pkl"


CATEGORY_MAPS = {
    "gender": ["Female", "Male"],
    "Partner": ["No", "Yes"],
    "Dependents": ["No", "Yes"],
    "PhoneService": ["No", "Yes"],
    "MultipleLines": ["No", "No phone service", "Yes"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    "OnlineSecurity": ["No", "No internet service", "Yes"],
    "OnlineBackup": ["No", "No internet service", "Yes"],
    "DeviceProtection": ["No", "No internet service", "Yes"],
    "TechSupport": ["No", "No internet service", "Yes"],
    "StreamingTV": ["No", "No internet service", "Yes"],
    "StreamingMovies": ["No", "No internet service", "Yes"],
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaperlessBilling": ["No", "Yes"],
    "PaymentMethod": [
        "Bank transfer (automatic)",
        "Credit card (automatic)",
        "Electronic check",
        "Mailed check",
    ],
}


def encode_value(column_name: str, value: str) -> int:
    return CATEGORY_MAPS[column_name].index(value)


@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as file:
        model_data = pickle.load(file)
    return model_data["model"], model_data["feature_names"]


def build_input_frame(user_input: dict, feature_names: list[str]) -> pd.DataFrame:
    encoded_row = {}
    for column_name, value in user_input.items():
        if column_name in CATEGORY_MAPS:
            encoded_row[column_name] = encode_value(column_name, value)
        else:
            encoded_row[column_name] = value

    input_frame = pd.DataFrame([encoded_row])
    return input_frame.reindex(columns=feature_names)


st.set_page_config(page_title="Customer Churn Prediction", page_icon="📉", layout="centered")

st.title("Customer Churn Prediction")
st.write("Enter customer details to predict churn using the trained model saved in this project.")

try:
    model, feature_names = load_model()
except Exception as error:
    st.error(f"Could not load the saved model: {error}")
    st.stop()


with st.form("churn_form"):
    left, right = st.columns(2)

    with left:
        gender = st.selectbox("Gender", CATEGORY_MAPS["gender"])
        senior_citizen = st.selectbox("Senior Citizen", [0, 1], format_func=lambda value: "Yes" if value == 1 else "No")
        partner = st.selectbox("Partner", CATEGORY_MAPS["Partner"])
        dependents = st.selectbox("Dependents", CATEGORY_MAPS["Dependents"])
        tenure = st.number_input("Tenure", min_value=0, max_value=100, value=12, step=1)
        phone_service = st.selectbox("Phone Service", CATEGORY_MAPS["PhoneService"])
        multiple_lines = st.selectbox("Multiple Lines", CATEGORY_MAPS["MultipleLines"])
        internet_service = st.selectbox("Internet Service", CATEGORY_MAPS["InternetService"])
        online_security = st.selectbox("Online Security", CATEGORY_MAPS["OnlineSecurity"])
        online_backup = st.selectbox("Online Backup", CATEGORY_MAPS["OnlineBackup"])

    with right:
        device_protection = st.selectbox("Device Protection", CATEGORY_MAPS["DeviceProtection"])
        tech_support = st.selectbox("Tech Support", CATEGORY_MAPS["TechSupport"])
        streaming_tv = st.selectbox("Streaming TV", CATEGORY_MAPS["StreamingTV"])
        streaming_movies = st.selectbox("Streaming Movies", CATEGORY_MAPS["StreamingMovies"])
        contract = st.selectbox("Contract", CATEGORY_MAPS["Contract"])
        paperless_billing = st.selectbox("Paperless Billing", CATEGORY_MAPS["PaperlessBilling"])
        payment_method = st.selectbox("Payment Method", CATEGORY_MAPS["PaymentMethod"])
        monthly_charges = st.number_input("Monthly Charges", min_value=0.0, value=70.0, step=0.05)
        total_charges = st.number_input("Total Charges", min_value=0.0, value=float(tenure) * float(monthly_charges), step=0.05)

    submitted = st.form_submit_button("Predict Churn")


if submitted:
    user_input = {
        "gender": gender,
        "SeniorCitizen": senior_citizen,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
    }

    input_frame = build_input_frame(user_input, feature_names)
    prediction = model.predict(input_frame)[0]
    probability = model.predict_proba(input_frame)[0][1]

    if prediction == 1:
        st.error(f"Prediction: Churn likely ({probability:.1%} probability)")
    else:
        st.success(f"Prediction: No churn likely ({1 - probability:.1%} confidence)")

    with st.expander("Encoded input sent to the model"):
        st.dataframe(input_frame, use_container_width=True)
