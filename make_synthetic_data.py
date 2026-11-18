"""Generate a synthetic dataset with the same columns as the Telco Churn data.
Only for testing the pipeline. Use the real Kaggle data for your actual project."""
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
n = 5000

tenure = rng.integers(0, 73, n)
contract = rng.choice(["Month-to-month", "One year", "Two year"], n, p=[0.55, 0.21, 0.24])
internet = rng.choice(["DSL", "Fiber optic", "No"], n, p=[0.34, 0.44, 0.22])
payment = rng.choice(["Electronic check", "Mailed check", "Bank transfer (automatic)",
                      "Credit card (automatic)"], n)
senior = rng.choice([0, 1], n, p=[0.84, 0.16])
partner = rng.choice(["Yes", "No"], n)
dependents = rng.choice(["Yes", "No"], n, p=[0.3, 0.7])
paperless = rng.choice(["Yes", "No"], n, p=[0.6, 0.4])
tech = rng.choice(["Yes", "No"], n)
security = rng.choice(["Yes", "No"], n)

base = np.where(internet == "Fiber optic", 70, np.where(internet == "DSL", 50, 22))
monthly = np.round(base + rng.normal(0, 8, n), 2).clip(18, 120)
total = np.round(monthly * tenure + rng.normal(0, 20, n), 2).clip(0)

logit = (-0.5
         + 1.3 * (contract == "Month-to-month")
         - 1.0 * (contract == "Two year")
         + 0.7 * (internet == "Fiber optic")
         + 0.6 * (payment == "Electronic check")
         + 0.4 * senior
         - 0.04 * tenure
         + 0.015 * (monthly - 65)
         - 0.4 * (tech == "Yes") - 0.4 * (security == "Yes"))
churn = rng.random(n) < 1 / (1 + np.exp(-logit))

df = pd.DataFrame({
    "customerID": [f"{i:05d}-SYN" for i in range(n)],
    "gender": rng.choice(["Male", "Female"], n),
    "SeniorCitizen": senior, "Partner": partner, "Dependents": dependents,
    "tenure": tenure, "PhoneService": rng.choice(["Yes", "No"], n, p=[0.9, 0.1]),
    "InternetService": internet, "OnlineSecurity": security, "TechSupport": tech,
    "Contract": contract, "PaperlessBilling": paperless, "PaymentMethod": payment,
    "MonthlyCharges": monthly, "TotalCharges": total,
    "Churn": np.where(churn, "Yes", "No"),
})
# mimic the real data's blank TotalCharges for brand-new customers
df["TotalCharges"] = df["TotalCharges"].astype(str)
df.loc[df.tenure == 0, "TotalCharges"] = " "
df.to_csv("data/synthetic_churn.csv", index=False)
print(f"Saved data/synthetic_churn.csv  shape={df.shape}  churn rate={churn.mean():.1%}")
