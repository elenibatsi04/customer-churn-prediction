"""
Customer churn prediction pipeline.

Usage:
    python churn_pipeline.py --data data/WA_Fn-UseC_-Telco-Customer-Churn.csv
    python churn_pipeline.py --data data/synthetic_churn.csv      # quick test

Outputs (plots, metrics, model) go to outputs/ and models/.
"""
import argparse
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, PrecisionRecallDisplay,
                             RocCurveDisplay, average_precision_score, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")
RANDOM_STATE = 42
OUT = Path("outputs")
MODELS = Path("models")


# ----------------------------------------------------------------------------
# 1. Load and clean
# ----------------------------------------------------------------------------
def load_and_clean(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"Loaded {df.shape[0]:,} rows x {df.shape[1]} columns")

    # TotalCharges has blank strings in the real Telco data -> coerce to numeric
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    # blanks are brand-new customers (tenure 0), so 0 is a sensible fill
    df["TotalCharges"] = df["TotalCharges"].fillna(0)

    # Target: Yes/No -> 1/0
    df["Churn"] = (df["Churn"].str.strip() == "Yes").astype(int)

    # SeniorCitizen is 0/1 but really categorical
    df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})

    # ID column carries no signal (and would be leakage-like noise)
    df = df.drop(columns=["customerID"])
    df = df.drop_duplicates()
    return df


# ----------------------------------------------------------------------------
# 2. EDA plots
# ----------------------------------------------------------------------------
def run_eda(df: pd.DataFrame) -> None:
    print(f"\nChurn rate: {df['Churn'].mean():.1%}")

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    df["Churn"].value_counts().rename({0: "Stayed", 1: "Churned"}).plot.bar(
        ax=axes[0, 0], color=["#4C72B0", "#C44E52"], rot=0)
    axes[0, 0].set_title("Class balance")

    df.groupby("Contract")["Churn"].mean().plot.bar(ax=axes[0, 1], rot=15, color="#C44E52")
    axes[0, 1].set_title("Churn rate by contract")

    df.groupby("InternetService")["Churn"].mean().plot.bar(ax=axes[0, 2], rot=0, color="#C44E52")
    axes[0, 2].set_title("Churn rate by internet service")

    for label, color in [(0, "#4C72B0"), (1, "#C44E52")]:
        axes[1, 0].hist(df.loc[df.Churn == label, "tenure"], bins=30, alpha=0.6,
                        color=color, label="Churned" if label else "Stayed")
    axes[1, 0].set_title("Tenure distribution"); axes[1, 0].legend()

    for label, color in [(0, "#4C72B0"), (1, "#C44E52")]:
        axes[1, 1].hist(df.loc[df.Churn == label, "MonthlyCharges"], bins=30, alpha=0.6,
                        color=color, label="Churned" if label else "Stayed")
    axes[1, 1].set_title("Monthly charges distribution"); axes[1, 1].legend()

    df.groupby("PaymentMethod")["Churn"].mean().sort_values().plot.barh(
        ax=axes[1, 2], color="#C44E52")
    axes[1, 2].set_title("Churn rate by payment method")

    plt.tight_layout()
    plt.savefig(OUT / "01_eda.png", dpi=130)
    plt.close()


# ----------------------------------------------------------------------------
# 3. Models
# ----------------------------------------------------------------------------
def build_models(num_cols, cat_cols) -> dict:
    scaled = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
    ])
    # tree models don't need scaling, but sharing the same preprocessor keeps it simple
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=5, class_weight="balanced",
            n_jobs=-1, random_state=RANDOM_STATE),
        "Gradient Boosting": HistGradientBoostingClassifier(
            learning_rate=0.05, max_iter=300, max_depth=4,
            class_weight="balanced", random_state=RANDOM_STATE),
    }
    try:  # optional
        from xgboost import XGBClassifier
        models["XGBoost"] = XGBClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=4, subsample=0.8,
            colsample_bytree=0.8, eval_metric="logloss", random_state=RANDOM_STATE)
    except Exception as e:  # not installed, or libomp missing on macOS
        print("(XGBoost unavailable - skipping it. On Mac: brew install libomp)")

    return {name: Pipeline([("prep", scaled), ("model", m)]) for name, m in models.items()}


# ----------------------------------------------------------------------------
# 4. Evaluation helpers
# ----------------------------------------------------------------------------
def evaluate(name, pipe, X_test, y_test, threshold=0.5) -> dict:
    proba = pipe.predict_proba(X_test)[:, 1]
    pred = (proba >= threshold).astype(int)
    return {
        "Model": name,
        "ROC-AUC": roc_auc_score(y_test, proba),
        "PR-AUC": average_precision_score(y_test, proba),
        "Precision": precision_score(y_test, pred),
        "Recall": recall_score(y_test, pred),
        "F1": f1_score(y_test, pred),
    }


def best_threshold(y_true, proba, cost_fn: float, cost_fp: float):
    """Pick the threshold with the lowest total business cost.
    cost_fn: money lost when we MISS a churner
    cost_fp: money wasted when we offer a retention deal to someone who'd have stayed"""
    thresholds = np.linspace(0.05, 0.95, 91)
    costs = []
    for t in thresholds:
        pred = proba >= t
        fn = ((~pred) & (y_true == 1)).sum()
        fp = (pred & (y_true == 0)).sum()
        costs.append(fn * cost_fn + fp * cost_fp)
    i = int(np.argmin(costs))
    return thresholds[i], thresholds, np.array(costs)


# ----------------------------------------------------------------------------
# 5. Main
# ----------------------------------------------------------------------------
def main(args):
    OUT.mkdir(exist_ok=True)
    MODELS.mkdir(exist_ok=True)

    df = load_and_clean(args.data)
    run_eda(df)

    X, y = df.drop(columns="Churn"), df["Churn"]
    num_cols = X.select_dtypes(include="number").columns.tolist()
    cat_cols = X.select_dtypes(exclude="number").columns.tolist()
    print(f"Numeric: {num_cols}\nCategorical: {len(cat_cols)} columns")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)

    models = build_models(num_cols, cat_cols)

    # --- cross-validate on train set, then fit and score on held-out test set
    cv = StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)
    rows, fitted = [], {}
    for name, pipe in models.items():
        cv_auc = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc").mean()
        pipe.fit(X_train, y_train)
        fitted[name] = pipe
        row = evaluate(name, pipe, X_test, y_test)
        row["CV ROC-AUC"] = cv_auc
        rows.append(row)
        print(f"  {name:<20} CV AUC={cv_auc:.3f}  test AUC={row['ROC-AUC']:.3f}")

    results = pd.DataFrame(rows).set_index("Model").round(3)
    results.to_csv(OUT / "model_comparison.csv")
    print("\n", results.to_string())

    best_name = results["PR-AUC"].idxmax()
    best = fitted[best_name]
    print(f"\nBest model (by PR-AUC): {best_name}")

    # --- ROC and PR curves
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    for name, pipe in fitted.items():
        RocCurveDisplay.from_estimator(pipe, X_test, y_test, ax=ax[0], name=name)
        PrecisionRecallDisplay.from_estimator(pipe, X_test, y_test, ax=ax[1], name=name)
    ax[0].set_title("ROC curves"); ax[1].set_title("Precision-recall curves")
    plt.tight_layout(); plt.savefig(OUT / "02_roc_pr_curves.png", dpi=130); plt.close()

    # --- Business-cost threshold tuning
    proba = best.predict_proba(X_test)[:, 1]
    thr, ts, costs = best_threshold(y_test.values, proba, args.cost_fn, args.cost_fp)
    default_cost = costs[np.argmin(np.abs(ts - 0.5))]
    print(f"Optimal threshold: {thr:.2f}  (cost {costs.min():,.0f} vs {default_cost:,.0f} at 0.50)")

    plt.figure(figsize=(7, 4.5))
    plt.plot(ts, costs, color="#C44E52")
    plt.axvline(thr, ls="--", color="k", label=f"best = {thr:.2f}")
    plt.xlabel("Decision threshold"); plt.ylabel("Total cost")
    plt.title(f"Cost vs threshold (FN={args.cost_fn:g}, FP={args.cost_fp:g})")
    plt.legend(); plt.tight_layout(); plt.savefig(OUT / "03_threshold_cost.png", dpi=130); plt.close()

    # --- Confusion matrix at the tuned threshold
    pred = (proba >= thr).astype(int)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ConfusionMatrixDisplay.from_predictions(
        y_test, pred, display_labels=["Stayed", "Churned"], ax=ax, cmap="Blues")
    ax.set_title(f"{best_name} @ threshold {thr:.2f}")
    plt.tight_layout(); plt.savefig(OUT / "04_confusion_matrix.png", dpi=130); plt.close()
    tuned = evaluate(best_name, best, X_test, y_test, threshold=thr)
    print(f"At tuned threshold -> precision {tuned['Precision']:.2f}, "
          f"recall {tuned['Recall']:.2f}, F1 {tuned['F1']:.2f}")

    # --- Explainability: permutation importance on the ORIGINAL columns
    imp = permutation_importance(best, X_test, y_test, scoring="roc_auc",
                                 n_repeats=8, random_state=RANDOM_STATE)
    imp_df = (pd.DataFrame({"feature": X.columns, "importance": imp.importances_mean,
                            "std": imp.importances_std})
              .sort_values("importance", ascending=True).tail(12))
    plt.figure(figsize=(8, 5.5))
    plt.barh(imp_df["feature"], imp_df["importance"], xerr=imp_df["std"], color="#4C72B0")
    plt.xlabel("Drop in ROC-AUC when feature is shuffled")
    plt.title(f"Top drivers of churn ({best_name})")
    plt.tight_layout(); plt.savefig(OUT / "05_feature_importance.png", dpi=130); plt.close()

    # --- Optional SHAP (works only if installed and best model is tree-based)
    try:
        import shap
        if best_name in ("Random Forest", "Gradient Boosting", "XGBoost"):
            prep, mdl = best.named_steps["prep"], best.named_steps["model"]
            Xt = prep.transform(X_test)
            names = prep.get_feature_names_out()
            sample = pd.DataFrame(Xt, columns=names).sample(min(500, len(Xt)), random_state=0)
            sv = shap.TreeExplainer(mdl).shap_values(sample)
            if isinstance(sv, list):
                sv = sv[1]
            if getattr(sv, "ndim", 2) == 3:
                sv = sv[:, :, 1]
            shap.summary_plot(sv, sample, show=False, max_display=15)
            plt.tight_layout(); plt.savefig(OUT / "06_shap_summary.png", dpi=130); plt.close()
            print("Saved SHAP summary plot")
    except ImportError:
        print("(shap not installed - skipping SHAP plot; pip install shap)")
    except Exception as e:
        print(f"(SHAP skipped: {e})")

    # --- Risk scores for the whole test set + save model
    scored = X_test.copy()
    scored["churn_probability"] = proba
    scored["actual_churn"] = y_test.values
    scored["flag_for_retention"] = (proba >= thr).astype(int)
    scored.sort_values("churn_probability", ascending=False).to_csv(
        OUT / "test_set_risk_scores.csv", index=False)

    joblib.dump({"pipeline": best, "threshold": float(thr), "columns": list(X.columns)},
                MODELS / "churn_model.joblib")
    print(f"\nSaved model to {MODELS/'churn_model.joblib'} and plots to {OUT}/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="path to the churn CSV")
    ap.add_argument("--cost-fn", type=float, default=500,
                    help="cost of missing a churner (lost customer value)")
    ap.add_argument("--cost-fp", type=float, default=50,
                    help="cost of a wasted retention offer")
    main(ap.parse_args())
