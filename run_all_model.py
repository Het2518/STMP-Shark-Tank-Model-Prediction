# run_all_models.py
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_squared_error, r2_score, mean_absolute_error,
    confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression, BayesianRidge, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.svm import SVC
import joblib
import matplotlib.pyplot as plt

# Try to import xgboost if available
try:
    from xgboost import XGBClassifier, XGBRegressor
    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False

# --------------------------
# Load and preprocess data
# --------------------------
DATA_FILE = "sharkTankIndia.xlsx"  # change if needed

if not os.path.exists(DATA_FILE):
    raise FileNotFoundError(f"File not found: {DATA_FILE}")

df = pd.read_excel(DATA_FILE)

# Quick fixes and new features
if 'Pitchers Average Age ' in df.columns:
    df.rename(columns={'Pitchers Average Age ': 'Pitchers Average Age'}, inplace=True)

shark_present_cols = ['Namita Present', 'Vineeta Present', 'Anupam Present', 'Aman Present',
                      'Peyush Present', 'Ritesh Present', 'Amit Present', 'Guest Present']
for c in shark_present_cols:
    if c not in df.columns:
        df[c] = 0
df['Num Sharks Present'] = df[shark_present_cols].sum(axis=1)

df['Valuation_per_Presenter'] = df['Valuation Requested'] / df['No of Presenters'].replace(0, 1)
df['Equity_per_Shark'] = df['Original Offered Equity'] / df['Num Sharks Present'].replace(0, 1)

# Define features and targets
features = [
    'Season No', 'Episode No', 'Pitch No', 'No of Presenters', 'Male Presenters',
    'Female Presenters', 'Couple Presenters', 'Pitchers Average Age',
    'Pitchers City', 'Pitchers State', 'Original Ask Amount',
    'Original Offered Equity', 'Valuation Requested', 'Industry',
    'Num Sharks Present', 'Valuation_per_Presenter', 'Equity_per_Shark'
]

target_col = 'Accepted Offer'
target_valuation = 'Deal Valuation'
shark_investors = [
    'Namita Investment Amount', 'Vineeta Invested Amount', 'Anupam Investment Amount',
    'Aman Investment Amount', 'Peyush Investment Amount', 'Ritesh Investment Amount',
    'Amit Investment Amount', 'Guest Investment Amount'
]

# Ensure columns exist
for col in features:
    if col not in df.columns:
        df[col] = 0
for col in shark_investors:
    if col not in df.columns:
        df[col] = 0

# Multi-label shark matrix
df_sharks = (df[shark_investors].fillna(0) > 0).astype(int)
df_sharks.columns = [c.split(' ')[0] for c in shark_investors]

# Fill missing values
for col in features:
    if df[col].dtype == 'object':
        df[col] = df[col].fillna(df[col].mode().iloc[0])
    else:
        df[col] = df[col].fillna(df[col].median())

if target_valuation in df.columns:
    df[target_valuation] = df[target_valuation].fillna(0)

df['Valuation_per_Presenter'] = df['Valuation_per_Presenter'].fillna(df['Valuation_per_Presenter'].median())
df['Equity_per_Shark'] = df['Equity_per_Shark'].fillna(df['Equity_per_Shark'].median())

# Encode categorical features
categorical_cols = [c for c in features if df[c].dtype == 'object']
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

X_all = df[features].copy()

# --------------------------
# Utilities
# --------------------------
def report_classification(name, y_true, y_pred, y_prob=None):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    roc = roc_auc_score(y_true, y_prob) if (y_prob is not None and len(np.unique(y_true)) > 1) else None
    return {"model": name, "accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "roc_auc": roc}

def report_regression(name, y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {"model": name, "RMSE": rmse, "MAE": mae, "R2": r2}

# --------------------------
# TASK 1: Deal (classification)
# --------------------------
if target_col in df.columns:
    print("\n=== TASK 1: Deal Prediction (Classification) ===")
    y = df[target_col].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X_all, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    classifiers = {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
        "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42),
        "GradientBoosting": GradientBoostingClassifier(n_estimators=200, random_state=42),
        "SVC": SVC(probability=True, random_state=42)
    }
    if XGBOOST_AVAILABLE:
        classifiers["XGBoost"] = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)

    task1_results = []
    for name, clf in classifiers.items():
        clf.fit(X_train_s, y_train)
        y_pred = clf.predict(X_test_s)
        y_prob = clf.predict_proba(X_test_s)[:, 1] if hasattr(clf, "predict_proba") else None
        res = report_classification(name, y_test, y_pred, y_prob)
        task1_results.append(res)

        print(f"\n{name} Metrics:")
        print(f" Accuracy:  {res['accuracy']:.4f}")
        print(f" Precision: {res['precision']:.4f}")
        print(f" Recall:    {res['recall']:.4f}")
        print(f" F1-Score:  {res['f1']:.4f}")
        print(f" ROC-AUC:   {res['roc_auc']}")

        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap='Blues', values_format='d')
        plt.title(f"Confusion Matrix — {name}")
        plt.show()

        joblib.dump((clf, scaler), f"best_{name}_deal.pkl")

    t1_df = pd.DataFrame(task1_results)
    t1_df['score_for_select'] = t1_df['f1'].fillna(0) + 0.1 * t1_df['roc_auc'].fillna(0)
    best1 = t1_df.sort_values("score_for_select", ascending=False).iloc[0]
    print("\nTask 1 Summary:\n", t1_df.sort_values("f1", ascending=False).to_string(index=False))
    print(f"\n=> Best model for Deal prediction: {best1['model']}")
else:
    print("Skipping Task 1 — target column not present.")

# --------------------------
# TASK 2: Deal Valuation (regression)
# --------------------------
if target_valuation in df.columns:
    print("\n\n=== TASK 2: Deal Valuation (Regression) ===")
    df_deals = df[df[target_col] == 1] if target_col in df.columns else df.copy()
    if df_deals.shape[0] < 10:
        print("Not enough accepted deals to train a valuation model. Skipping.")
    else:
        X_reg = df_deals[features]
        y_reg = df_deals[target_valuation].astype(float)

        X_train, X_test, y_train, y_test = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

        scaler_r = StandardScaler()
        X_train_r = scaler_r.fit_transform(X_train)
        X_test_r = scaler_r.transform(X_test)

        regressors = {
            "BayesianRidge": BayesianRidge(),
            "Ridge": Ridge(),
            "Lasso": Lasso(),
            "ElasticNet": ElasticNet(),
            "RandomForestRegressor": RandomForestRegressor(n_estimators=200, random_state=42),
            "GradientBoostingRegressor": GradientBoostingRegressor(n_estimators=200, random_state=42)
        }
        if XGBOOST_AVAILABLE:
            regressors["XGBoostRegressor"] = XGBRegressor(random_state=42)

        task2_results = []
        for name, reg in regressors.items():
            reg.fit(X_train_r, y_train)
            y_pred = reg.predict(X_test_r)
            res = report_regression(name, y_test, y_pred)
            task2_results.append(res)
            print(f"\n{name} -> RMSE:{res['RMSE']:.2f}  MAE:{res['MAE']:.2f}  R2:{res['R2']:.4f}")
            joblib.dump((reg, scaler_r), f"best_{name}_valuation.pkl")

        t2_df = pd.DataFrame(task2_results)
        best2 = t2_df.sort_values("R2", ascending=False).iloc[0]
        print("\nTask 2 Summary:\n", t2_df.sort_values("R2", ascending=False).to_string(index=False))
        print(f"\n=> Best model for Valuation prediction: {best2['model']} (R2={best2['R2']:.4f})")
else:
    print("Skipping Task 2 — valuation column not present.")

# --------------------------
# TASK 3: Shark prediction (multi-label)
# --------------------------
print("\n\n=== TASK 3: Predicting Sharks (Multi-label) ===")
if 'Accepted Offer' in df.columns:
    df_shark_deals = df[df['Accepted Offer'] == 1]
else:
    df_shark_deals = df.copy()

if df_shark_deals.shape[0] < 10:
    print("Not enough deals for shark modeling. Skipping.")
else:
    X_shark = df_shark_deals[features]
    y_shark = (df_shark_deals[shark_investors].fillna(0) > 0).astype(int)
    y_shark.columns = [c.split(' ')[0] for c in shark_investors]

    X_train, X_test, y_train, y_test = train_test_split(X_shark, y_shark, test_size=0.2, random_state=42)

    scaler_s = StandardScaler()
    X_train_s = scaler_s.fit_transform(X_train)
    X_test_s = scaler_s.transform(X_test)

    multi_models = {
        "Logistic_OVR": OneVsRestClassifier(LogisticRegression(max_iter=1000, random_state=42)),
        "RandomForest_OVR": OneVsRestClassifier(RandomForestClassifier(n_estimators=200, random_state=42)),
    }
    if XGBOOST_AVAILABLE:
        multi_models["XGBoost_OVR"] = OneVsRestClassifier(XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42))

    multi_results = {}
    for name, clf in multi_models.items():
        clf.fit(X_train_s, y_train)
        y_pred = clf.predict(X_test_s)

        per_shark = []
        print(f"\n{name} - Individual Shark Metrics:")
        for idx, col in enumerate(y_test.columns):
            y_true_shark = y_test.iloc[:, idx]
            y_pred_shark = y_pred[:, idx]
            acc = accuracy_score(y_true_shark, y_pred_shark)
            prec = precision_score(y_true_shark, y_pred_shark, zero_division=0)
            rec = recall_score(y_true_shark, y_pred_shark, zero_division=0)
            f1 = f1_score(y_true_shark, y_pred_shark, zero_division=0)
            per_shark.append({"shark": col, "acc": acc, "prec": prec, "rec": rec, "f1": f1})
            print(f"  {col}: acc={acc:.4f}  prec={prec:.4f}  rec={rec:.4f}  f1={f1:.4f}")

            # Confusion Matrix per shark
            cm = confusion_matrix(y_true_shark, y_pred_shark)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm)
            disp.plot(cmap='Purples', values_format='d')
            plt.title(f"Confusion Matrix — {name} ({col})")
            plt.show()

        per_shark_df = pd.DataFrame(per_shark)
        macro_acc = per_shark_df["acc"].mean()
        macro_prec = per_shark_df["prec"].mean()
        macro_rec = per_shark_df["rec"].mean()
        macro_f1 = per_shark_df["f1"].mean()
        print(f"\n{name} Macro Averages: acc={macro_acc:.4f}, prec={macro_prec:.4f}, rec={macro_rec:.4f}, f1={macro_f1:.4f}")

        multi_results[name] = {
            "per_shark": per_shark_df,
            "macro_acc": macro_acc,
            "macro_prec": macro_prec,
            "macro_rec": macro_rec,
            "macro_f1": macro_f1
        }
        joblib.dump((clf, scaler_s), f"best_{name}_sharks.pkl")

    best_multi = sorted(multi_results.items(), key=lambda kv: kv[1]["macro_f1"], reverse=True)[0]
    print(f"\n=> Best multi-label model for Shark prediction: {best_multi[0]} (macro_f1={best_multi[1]['macro_f1']:.4f})")

print("\n\nAll done. Best models saved as files: best_* .pkl")
# Save label encoders for categorical columns
joblib.dump(label_encoders, "label_encoders.pkl")
print("\n✅ Saved label encoders as 'label_encoders.pkl'")

