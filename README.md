# Customer Churn Prediction

Predict which telecom customers are likely to cancel, explain why, and pick a
decision threshold based on business cost rather than raw accuracy.

## Quick start
```bash
pip install -r requirements.txt

# 1. Download the real data (free Kaggle account needed):
#    https://www.kaggle.com/datasets/blastchar/telco-customer-churn
#    Put WA_Fn-UseC_-Telco-Customer-Churn.csv in data/

# 2. Run the pipeline
python churn_pipeline.py --data data/WA_Fn-UseC_-Telco-Customer-Churn.csv

# 3. Score a single customer
python predict.py
```
No data yet? `python make_synthetic_data.py` creates a fake dataset with the same
columns so you can test the pipeline.

## What the pipeline does
1. **Cleans** the data (blank `TotalCharges`, 0/1 -> categorical, drops ID).
2. **EDA** plots -> `outputs/01_eda.png`
3. **Trains** logistic regression, random forest, gradient boosting (+ XGBoost if installed)
   with 5-fold CV, then scores on a stratified 20% hold-out set.
4. **Evaluates** with ROC-AUC, PR-AUC, precision, recall, F1 (not just accuracy).
5. **Tunes the threshold** using costs: `--cost-fn` (missed churner, default 500)
   vs `--cost-fp` (wasted retention offer, default 50).
6. **Explains** the model with permutation importance (and SHAP if installed).
7. **Saves** the model and a ranked list of at-risk customers.

## Outputs
| File | Contents |
|---|---|
| `outputs/model_comparison.csv` | metrics for every model |
| `outputs/02_roc_pr_curves.png` | ROC and precision-recall curves |
| `outputs/03_threshold_cost.png` | cost vs threshold |
| `outputs/04_confusion_matrix.png` | confusion matrix at tuned threshold |
| `outputs/05_feature_importance.png` | top churn drivers |
| `outputs/test_set_risk_scores.csv` | customers ranked by churn probability |
| `models/churn_model.joblib` | pipeline + tuned threshold |

<img width="2080" height="1170" alt="01_eda" src="https://github.com/user-attachments/assets/00c0097d-2791-4d8e-9f71-cdd8dadd4e8a" />
<img width="1690" height="650" alt="02_roc_pr_curves" src="https://github.com/user-attachments/assets/723bfdf6-d2e0-45ea-b8bd-386f65c52ae2" />[test_set_risk_scores.csv](https://github.com/user-attachments/files/35088510/test_set_risk_scores.csv)
[model_comparison.csv](https://github.com/user-attachments/files/35088509/model_comparison.csv)
<img width="1040" height="975" alt="06_shap_summary" src="https://github.com/user-attachments/assets/ddddcce6-6570-463e-a2a3-12022ad6ee9e" />
<img width="1040" height="715" alt="05_feature_importance" src="https://github.com/user-attachments/assets/40d5030c-2ebc-410a-8649-594c936947e6" />
<img width="650" height="585" alt="04_confusion_matrix" src="https://github.com/user-attachments/assets/a09b77c6-844a-4cce-b953-006605f23125" />
<img width="910" height="585" alt="03_threshold_cost" src="https://github.com/user-attachments/assets/f58b7ad6-7bd0-422f-8326-57550c1c2c94" />


## Ideas to extend
- Hyperparameter search (Optuna / `RandomizedSearchCV`)
- Probability calibration (`CalibratedClassifierCV`)
- Streamlit app or FastAPI endpoint around `predict.py`
- Customer lifetime value to weight the cost of each miss
