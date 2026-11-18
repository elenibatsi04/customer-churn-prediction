"""Score one customer with the saved model. Edit the dict below and run: python predict.py"""
import joblib
import pandas as pd

bundle = joblib.load("models/churn_model.joblib")

customer = {
    "gender": "Female", "SeniorCitizen": "No", "Partner": "No", "Dependents": "No",
    "tenure": 3, "PhoneService": "Yes", "InternetService": "Fiber optic",
    "OnlineSecurity": "No", "TechSupport": "No", "Contract": "Month-to-month",
    "PaperlessBilling": "Yes", "PaymentMethod": "Electronic check",
    "MonthlyCharges": 85.0, "TotalCharges": 255.0,
}
# NOTE: the real Telco dataset has more columns (StreamingTV, MultipleLines, ...).
# Add any column the model was trained on; `bundle["columns"]` lists them all.

row = pd.DataFrame([customer]).reindex(columns=bundle["columns"])
p = bundle["pipeline"].predict_proba(row)[0, 1]
flag = p >= bundle["threshold"]
print(f"Churn probability: {p:.1%}  ->  {'FLAG for retention offer' if flag else 'low risk'}")
