# run_all_models.py
import warnings
warnings.filterwarnings("ignore")

import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, mean_absolute_error, r2_score, confusion_matrix, roc_auc_score
from sklearn.linear_model import LogisticRegression, BayesianRidge, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

# Optional XGBoost
try:
    from xgboost import XGBClassifier, XGBRegressor
    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False

# --------------------------
# Config
# --------------------------
DATA_FILE = "sharkTankIndia.xlsx"  # or "sharkTankIndia.csv"
if not os.path.exists(DATA_FILE):
    if os.path.exists("/mnt/data/sharkTankIndia.csv"):
        DATA_FILE = "/mnt/data/sharkTankIndia.csv"
    else:
        raise FileNotFoundError(f"File not found: {DATA_FILE}")

# --------------------------
# Load data
# --------------------------
if DATA_FILE.endswith(".csv"):
    df = pd.read_csv(DATA_FILE)
else:
    df = pd.read_excel(DATA_FILE)

print(f"✅ Loaded dataset: {DATA_FILE} with {len(df)} rows")

# Quick rename fix
if 'Pitchers Average Age ' in df.columns:
    df.rename(columns={'Pitchers Average Age ': 'Pitchers Average Age'}, inplace=True)

# Make sure required columns exist (create defaults if missing)
shark_investors = [
    'Namita Investment Amount', 'Vineeta Invested Amount', 'Anupam Investment Amount',
    'Aman Investment Amount', 'Peyush Investment Amount', 'Ritesh Investment Amount',
    'Amit Investment Amount', 'Guest Investment Amount'
]
shark_present_cols = ['Namita Present', 'Vineeta Present', 'Anupam Present', 'Aman Present',
                      'Peyush Present', 'Ritesh Present', 'Amit Present', 'Guest Present']

for c in shark_present_cols + shark_investors:
    if c not in df.columns:
        df[c] = 0

# Derived counts
df['Num Sharks Present'] = df[shark_present_cols].sum(axis=1)

# Ensure numeric Ask/Equity columns exist
if 'Original Ask Amount' not in df.columns and 'Ask Amount' in df.columns:
    df.rename(columns={'Ask Amount': 'Original Ask Amount'}, inplace=True)
if 'Original Offered Equity' not in df.columns and 'Offered Equity' in df.columns:
    df.rename(columns={'Offered Equity': 'Original Offered Equity'}, inplace=True)

# Fill defaults
df['No of Presenters'] = df.get('No of Presenters', pd.Series(1, index=df.index)).fillna(1)
df['Valuation Requested'] = df.get('Valuation Requested', 0)
df['Original Offered Equity'] = df.get('Original Offered Equity', 0)
df['Original Ask Amount'] = df.get('Original Ask Amount', 0)

# Derived features
df['Valuation_per_Presenter'] = df['Valuation Requested'] / df['No of Presenters'].replace(0, 1)
df['Equity_per_Shark'] = df['Original Offered Equity'] / df['Num Sharks Present'].replace(0, 1)

# Feature list (keep stable)
features = [
    'Season No', 'Episode No', 'Pitch No', 'No of Presenters', 'Male Presenters',
    'Female Presenters', 'Couple Presenters', 'Pitchers Average Age',
    'Pitchers City', 'Pitchers State', 'Original Ask Amount',
    'Original Offered Equity', 'Valuation Requested', 'Industry',
    'Num Sharks Present', 'Valuation_per_Presenter', 'Equity_per_Shark'
]

# Ensure columns exist
for col in features:
    if col not in df.columns:
        df[col] = 0

target_col = 'Accepted Offer'
target_valuation = 'Deal Valuation'

# Fill missing values
for col in features:
    if df[col].dtype == 'object':
        df[col] = df[col].fillna(df[col].mode().iloc[0] if not df[col].mode().empty else "Unknown")
    else:
        df[col] = df[col].fillna(df[col].median())

if target_valuation in df.columns:
    df[target_valuation] = df[target_valuation].fillna(0)

df['Valuation_per_Presenter'] = df['Valuation_per_Presenter'].fillna(df['Valuation_per_Presenter'].median())
df['Equity_per_Shark'] = df['Equity_per_Shark'].fillna(df['Equity_per_Shark'].median())

# --------------------------
# Encode categorical features (save encoders)
# --------------------------
categorical_cols = [c for c in features if df[c].dtype == 'object']
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = df[col].astype(str)
    le.fit(df[col].tolist())
    df[col] = le.transform(df[col])
    label_encoders[col] = le

joblib.dump(label_encoders, "label_encoders.pkl")
print("✅ Saved label_encoders.pkl")

# X matrix for all tasks
X_all = df[features].copy()

# Utility reports
def report_classification_metrics(name, y_true, y_pred, y_prob=None):
    res = {}
    res['model'] = name
    res['accuracy'] = accuracy_score(y_true, y_pred)
    res['precision'] = precision_score(y_true, y_pred, zero_division=0)
    res['recall'] = recall_score(y_true, y_pred, zero_division=0)
    res['f1'] = f1_score(y_true, y_pred, zero_division=0)
    if y_prob is not None and len(np.unique(y_true)) > 1:
        try:
            res['roc_auc'] = roc_auc_score(y_true, y_prob)
        except:
            res['roc_auc'] = None
    else:
        res['roc_auc'] = None
    return res

def report_regression_metrics(name, y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {"model": name, "RMSE": rmse, "MAE": mae, "R2": r2}

# --------------------------
# TASK 1: Deal (classification)
# --------------------------
if target_col in df.columns:
    print("\n" + "="*60)
    print("TASK 1: Deal Prediction (Classification)")
    print("="*60)
    
    y = df[target_col].astype(int).fillna(0)
    stratify = y if len(np.unique(y)) > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(X_all, y, test_size=0.2, random_state=42, stratify=stratify)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    classifiers = {
        "LogisticRegression": LogisticRegression(max_iter=2000, random_state=42),
        "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42),
        "GradientBoosting": GradientBoostingClassifier(n_estimators=200, random_state=42),
        "SVC": SVC(probability=True, random_state=42)
    }
    if XGBOOST_AVAILABLE:
        classifiers["XGBoost"] = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)

    task1_results = []
    best_model_name = None
    best_model_score = -1
    best_clf = None
    
    for name, clf in classifiers.items():
        try:
            print(f"\nTraining {name}...")
            clf.fit(X_train_s, y_train)
            y_pred = clf.predict(X_test_s)
            y_prob = clf.predict_proba(X_test_s)[:, 1] if hasattr(clf, "predict_proba") else None
            res = report_classification_metrics(name, y_test, y_pred, y_prob)
            task1_results.append(res)

            score = res['f1'] + 0.1 * (res['roc_auc'] if res['roc_auc'] is not None else 0)
            if score > best_model_score:
                best_model_score = score
                best_model_name = name
                best_clf = clf

            print(f"  Accuracy:  {res['accuracy']:.4f}")
            print(f"  Precision: {res['precision']:.4f}")
            print(f"  Recall:    {res['recall']:.4f}")
            print(f"  F1-Score:  {res['f1']:.4f}")
            print(f"  ROC-AUC:   {res['roc_auc']}")
            
            # Confusion Matrix
            cm = confusion_matrix(y_test, y_pred)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm)
            disp.plot(cmap='Blues', values_format='d')
            plt.title(f"Confusion Matrix — {name}")
            plt.tight_layout()
            plt.savefig(f"confusion_matrix_deal_{name}.png")
            plt.close()
            
        except Exception as e:
            print(f"❌ Skipping classifier {name} due to error: {e}")

    t1_df = pd.DataFrame(task1_results)
    if not t1_df.empty:
        print("\n📊 Task 1 Summary:")
        print(t1_df.sort_values("f1", ascending=False).to_string(index=False))
        print(f"\n🏆 Best model: {best_model_name} (score: {best_model_score:.4f})")

    # Save best classifier and scaler
    deal_model_tuple = (best_clf, scaler)
    joblib.dump(deal_model_tuple, "deal_model.pkl")
    print("✅ Saved deal_model.pkl")

else:
    print("⚠️ Skipping Task 1 — target column not present.")

# --------------------------
# TASK 2: Deal Valuation (regression)
# --------------------------
if target_valuation in df.columns:
    print("\n" + "="*60)
    print("TASK 2: Deal Valuation (Regression)")
    print("="*60)
    
    df_deals = df[df[target_col] == 1] if target_col in df.columns else df.copy()
    print(f"Using {len(df_deals)} accepted deals for valuation training")
    
    if df_deals.shape[0] < 10:
        print("⚠️ Not enough accepted deals to train a valuation model.")
        mean_valuation = df_deals[target_valuation].mean() if not df_deals.empty else 0
        joblib.dump(("mean", mean_valuation), "valuation_model.pkl")
        print(f"✅ Saved fallback valuation_model.pkl (mean={mean_valuation:.2f})")
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
        best_r2 = -999
        best_reg = None
        best_reg_name = None
        
        for name, reg in regressors.items():
            try:
                print(f"\nTraining {name}...")
                reg.fit(X_train_r, y_train)
                y_pred = reg.predict(X_test_r)
                res = report_regression_metrics(name, y_test, y_pred)
                task2_results.append(res)
                
                if res['R2'] > best_r2:
                    best_r2 = res['R2']
                    best_reg = reg
                    best_reg_name = name
                    
                print(f"  RMSE: {res['RMSE']:.2f}")
                print(f"  MAE:  {res['MAE']:.2f}")
                print(f"  R2:   {res['R2']:.4f}")
                
            except Exception as e:
                print(f"❌ Skipping regressor {name} due to error: {e}")

        t2_df = pd.DataFrame(task2_results)
        if not t2_df.empty:
            print("\n📊 Task 2 Summary:")
            print(t2_df.sort_values("R2", ascending=False).to_string(index=False))
            print(f"\n🏆 Best model: {best_reg_name} (R2={best_r2:.4f})")

        joblib.dump((best_reg, scaler_r), "valuation_model.pkl")
        print("✅ Saved valuation_model.pkl")
else:
    print("⚠️ Skipping Task 2 — valuation column not present.")

# --------------------------
# TASK 3: Shark prediction (multi-label)
# --------------------------
print("\n" + "="*60)
print("TASK 3: Predicting Sharks (Multi-label)")
print("="*60)

df_sharks = (df[shark_investors].fillna(0) > 0).astype(int)
df_sharks.columns = [c.split(' ')[0] for c in shark_investors]

if target_col in df.columns:
    df_shark_deals = df[df[target_col] == 1]
else:
    df_shark_deals = df.copy()

print(f"Using {len(df_shark_deals)} deals for shark prediction training")

if df_shark_deals.shape[0] < 10:
    print("⚠️ Not enough accepted deals. Training on entire dataset as fallback.")
    X_shark = df[features]
    y_shark = df_sharks
else:
    X_shark = df_shark_deals[features]
    y_shark = (df_shark_deals[shark_investors].fillna(0) > 0).astype(int)
    y_shark.columns = [c.split(' ')[0] for c in shark_investors]

X_train, X_test, y_train, y_test = train_test_split(X_shark, y_shark, test_size=0.2, random_state=42)

scaler_s = StandardScaler()
X_train_s = scaler_s.fit_transform(X_train)
X_test_s = scaler_s.transform(X_test)

multi_models = {
    "Logistic_OVR": OneVsRestClassifier(LogisticRegression(max_iter=2000, random_state=42)),
    "RandomForest_OVR": OneVsRestClassifier(RandomForestClassifier(n_estimators=200, random_state=42))
}
if XGBOOST_AVAILABLE:
    multi_models["XGBoost_OVR"] = OneVsRestClassifier(XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42))

multi_results = {}
best_name = None
best_macro_f1 = -1
best_model = None

for name, clf in multi_models.items():
    try:
        print(f"\nTraining {name}...")
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
            print(f"  {col:8s}: acc={acc:.4f}  prec={prec:.4f}  rec={rec:.4f}  f1={f1:.4f}")
            
            # Confusion Matrix per shark
            cm = confusion_matrix(y_true_shark, y_pred_shark)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm)
            disp.plot(cmap='Purples', values_format='d')
            plt.title(f"Confusion Matrix — {name} ({col})")
            plt.tight_layout()
            plt.savefig(f"confusion_matrix_shark_{name}_{col}.png")
            plt.close()
            
        per_shark_df = pd.DataFrame(per_shark)
        macro_f1 = per_shark_df["f1"].mean()
        multi_results[name] = {"per_shark": per_shark_df, "macro_f1": macro_f1}
        print(f"\n{name} Macro F1: {macro_f1:.4f}")
        
        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            best_name = name
            best_model = clf
            
    except Exception as e:
        print(f"❌ Skipping multi model {name} due to error: {e}")

if best_model is not None:
    joblib.dump((best_model, scaler_s), "shark_model.pkl")
    print(f"\n✅ Saved shark_model.pkl (best: {best_name}, macro_f1={best_macro_f1:.4f})")
else:
    print("❌ No multi-label shark model saved (training failed).")

# --------------------------
# Create industry-level shark probabilities (historical)
# --------------------------
print("\n" + "="*60)
print("Computing Industry-Level Shark Investment Probabilities")
print("="*60)

industry_col = 'Industry'
if industry_col in df.columns:
    industry_stats = {}
    sharks_short = [c.split(' ')[0] for c in shark_investors]
    
    for ind_val in sorted(df[industry_col].unique()):
        mask = (df[industry_col] == ind_val)
        if mask.sum() == 0:
            continue
        row = {}
        for idx, s in enumerate(sharks_short):
            orig_col = shark_investors[idx]
            row[s] = (df.loc[mask, orig_col].fillna(0) > 0).mean()
        industry_stats[int(ind_val)] = row
        
    industry_stats_df = pd.DataFrame.from_dict(industry_stats, orient='index').fillna(0)
    print(f"Created industry stats for {len(industry_stats_df)} industries")
else:
    sharks_short = [c.split(' ')[0] for c in shark_investors]
    global_probs = {}
    for idx, s in enumerate(sharks_short):
        orig_col = shark_investors[idx]
        global_probs[s] = (df[orig_col].fillna(0) > 0).mean()
    industry_stats_df = pd.DataFrame([global_probs])
    industry_stats_df.index = [0]
    print("Created global shark investment probabilities (fallback)")

joblib.dump(industry_stats_df, "industry_stats.pkl")
print("✅ Saved industry_stats.pkl")

# --------------------------
# Save feature columns used
# --------------------------
joblib.dump(features, "feature_columns.pkl")
print("✅ Saved feature_columns.pkl")

print("\n" + "="*60)
print("✅ ALL DONE!")
print("="*60)
print("Saved files:")
print("  - deal_model.pkl")
print("  - valuation_model.pkl")
print("  - shark_model.pkl")
print("  - label_encoders.pkl")
print("  - industry_stats.pkl")
print("  - feature_columns.pkl")
print("\nYou can now run the Streamlit app with: streamlit run app.py")