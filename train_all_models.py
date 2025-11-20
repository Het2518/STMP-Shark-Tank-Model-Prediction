# train_all_models.py
# Clean Windows-safe version (no emoji, no unicode)

import warnings, os, numpy as np, pandas as pd, joblib, matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, mean_squared_error, r2_score, mean_absolute_error,
    confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import (
    LogisticRegression, BayesianRidge, Ridge, Lasso, ElasticNet
)
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor
)
from sklearn.svm import SVC

# Optional imports
try:
    from xgboost import XGBClassifier, XGBRegressor
    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False

try:
    from catboost import CatBoostClassifier, CatBoostRegressor
    CATBOOST_AVAILABLE = True
except Exception:
    import subprocess, sys
    subprocess.run([sys.executable, "-m", "pip", "install", "catboost", "-q"])
    from catboost import CatBoostClassifier, CatBoostRegressor
    CATBOOST_AVAILABLE = True

# ===============================================================
# Load Dataset
# ===============================================================
DATA_FILE = "sharkTankIndia.xlsx"

if not os.path.exists(DATA_FILE):
    raise FileNotFoundError("Dataset not found: " + DATA_FILE)

df = pd.read_excel(DATA_FILE)
print("Loaded dataset:", df.shape[0], "rows,", df.shape[1], "columns")

# Fix column name
if "Pitchers Average Age " in df.columns:
    df.rename(columns={"Pitchers Average Age ": "Pitchers Average Age"}, inplace=True)

# Add missing shark presence columns
shark_present_cols = [
    "Namita Present","Vineeta Present","Anupam Present","Aman Present",
    "Peyush Present","Ritesh Present","Amit Present","Guest Present"
]
for c in shark_present_cols:
    if c not in df.columns:
        df[c] = 0

# Additional engineered features
df["Num Sharks Present"] = df[shark_present_cols].sum(axis=1)
df["Valuation_per_Presenter"] = df["Valuation Requested"] / df["No of Presenters"].replace(0,1)
df["Equity_per_Shark"] = df["Original Offered Equity"] / df["Num Sharks Present"].replace(0,1)

# Main features
features = [
    "Season No","Episode No","Pitch No","No of Presenters","Male Presenters",
    "Female Presenters","Couple Presenters","Pitchers Average Age",
    "Pitchers City","Pitchers State","Original Ask Amount",
    "Original Offered Equity","Valuation Requested","Industry",
    "Num Sharks Present","Valuation_per_Presenter","Equity_per_Shark"
]

target_col = "Accepted Offer"
valuation_col = "Deal Valuation"

shark_investors = [
    "Namita Investment Amount","Vineeta Invested Amount","Anupam Investment Amount",
    "Aman Investment Amount","Peyush Investment Amount","Ritesh Investment Amount",
    "Amit Investment Amount","Guest Investment Amount"
]

# Ensure missing columns exist
for c in features:
    if c not in df.columns:
        df[c] = 0
for c in shark_investors:
    if c not in df.columns:
        df[c] = 0

# Fill NaN properly
for c in features:
    if df[c].dtype == "object":
        df[c] = df[c].fillna(df[c].mode().iloc[0])
    else:
        df[c] = df[c].fillna(df[c].median())

if valuation_col in df.columns:
    df[valuation_col] = df[valuation_col].fillna(0)

# Encode categorical variables
categorical = [c for c in features if df[c].dtype == "object"]
for c in categorical:
    le = LabelEncoder()
    df[c] = le.fit_transform(df[c].astype(str))

X_all = df[features].copy()

# ===============================================================
# Helper functions
# ===============================================================
def cls_report(name, y_true, y_pred, y_prob=None):
    acc = accuracy_score(y_true,y_pred)
    prec = precision_score(y_true,y_pred,zero_division=0)
    rec = recall_score(y_true,y_pred,zero_division=0)
    f1 = f1_score(y_true,y_pred,zero_division=0)
    roc = roc_auc_score(y_true,y_prob) if (
        y_prob is not None and len(np.unique(y_true))>1
    ) else None
    return {
        "model": name,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": roc
    }

def reg_report(name, y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true,y_pred))
    mae = mean_absolute_error(y_true,y_pred)
    r2  = r2_score(y_true,y_pred)
    return {"model":name,"RMSE":rmse,"MAE":mae,"R2":r2}

# ===============================================================
# TASK 1 – Deal Classification
# ===============================================================
if target_col in df.columns:
    print("\n=== Task 1: Deal Classification ===")

    y = df[target_col].astype(int)
    Xtr, Xte, ytr, yte = train_test_split(X_all, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(Xtr)
    Xte_s = scaler.transform(Xte)

    cls_models = {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
        "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42),
        "GradientBoosting": GradientBoostingClassifier(n_estimators=200, random_state=42),
        "SVC": SVC(probability=True, random_state=42)
    }

    if XGBOOST_AVAILABLE:
        cls_models["XGBoost"] = XGBClassifier(
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42
        )
    if CATBOOST_AVAILABLE:
        cls_models["CatBoost"] = CatBoostClassifier(
            iterations=800,
            learning_rate=0.03,
            depth=6,
            random_seed=42,
            verbose=0
        )

    results = []
    for name, model in cls_models.items():
        print("Training:", name)

        if "CatBoost" in name:
            model.fit(Xtr, ytr, eval_set=(Xte,yte), use_best_model=True)
            yp = model.predict(Xte)
            yp_prob = model.predict_proba(Xte)[:,1]
        else:
            model.fit(Xtr_s, ytr)
            yp = model.predict(Xte_s)
            yp_prob = model.predict_proba(Xte_s)[:,1] if hasattr(model,"predict_proba") else None

        r = cls_report(name, yte, yp, yp_prob)
        results.append(r)

        cm = confusion_matrix(yte, yp)
        ConfusionMatrixDisplay(cm).plot(cmap="Blues")
        plt.title(name)
        plt.show()

        joblib.dump((model, scaler), "best_" + name + "_deal.pkl")

    print("\nSummary:")
    print(pd.DataFrame(results).sort_values("f1",ascending=False).to_string(index=False))

# ===============================================================
# TASK 2 – Deal Valuation Regression
# ===============================================================
if valuation_col in df.columns:
    print("\n=== Task 2: Deal Valuation Regression ===")

    deals = df[df[target_col]==1]
    if deals.shape[0] < 10:
        print("Not enough deals for training valuation model.")
    else:
        Xr = deals[features]
        yr = deals[valuation_col].astype(float)

        Xtr, Xte, ytr, yte = train_test_split(Xr, yr, test_size=0.2, random_state=42)

        scaler_r = StandardScaler()
        Xtr_s = scaler_r.fit_transform(Xtr)
        Xte_s = scaler_r.transform(Xte)

        reg_models = {
            "BayesianRidge": BayesianRidge(),
            "Ridge": Ridge(),
            "Lasso": Lasso(),
            "ElasticNet": ElasticNet(),
            "RandomForestRegressor": RandomForestRegressor(n_estimators=200, random_state=42),
            "GradientBoostingRegressor": GradientBoostingRegressor(n_estimators=200, random_state=42)
        }

        if XGBOOST_AVAILABLE:
            reg_models["XGBoostRegressor"] = XGBRegressor(random_state=42)

        if CATBOOST_AVAILABLE:
            reg_models["CatBoostRegressor"] = CatBoostRegressor(
                iterations=800,
                learning_rate=0.05,
                depth=6,
                random_seed=42,
                verbose=0
            )

        results2 = []
        for name, model in reg_models.items():
            print("Training:", name)
            if "CatBoost" in name:
                model.fit(Xtr, ytr, eval_set=(Xte,yte), use_best_model=True)
                yp = model.predict(Xte)
            else:
                model.fit(Xtr_s, ytr)
                yp = model.predict(Xte_s)

            r = reg_report(name, yte, yp)
            results2.append(r)

            joblib.dump((model, scaler_r), "best_" + name + "_valuation.pkl")

        print("\nSummary:")
        print(pd.DataFrame(results2).sort_values("R2",ascending=False).to_string(index=False))

# ===============================================================
# TASK 3 – Multi-label Shark Prediction
# ===============================================================
print("\n=== Task 3: Multi-label Shark Prediction ===")

deals = df[df[target_col]==1]
if deals.shape[0] < 10:
    print("Not enough deals for shark prediction.")
else:
    Xs = deals[features]
    ys = (deals[shark_investors].fillna(0) > 0).astype(int)
    ys.columns = [c.split(" ")[0] for c in shark_investors]  # short names

    Xtr, Xte, ytr, yte = train_test_split(Xs, ys, test_size=0.2, random_state=42)

    scaler_s = StandardScaler()
    Xtr_s = scaler_s.fit_transform(Xtr)
    Xte_s = scaler_s.transform(Xte)

    multi_models = {
        "Logistic_OVR": OneVsRestClassifier(LogisticRegression(max_iter=1000, random_state=42)),
        "RandomForest_OVR": OneVsRestClassifier(RandomForestClassifier(n_estimators=200, random_state=42))
    }

    if XGBOOST_AVAILABLE:
        from sklearn.base import BaseEstimator, ClassifierMixin

        class XGBWrapper(BaseEstimator, ClassifierMixin):
            def __init__(self):
                self.model = XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)
            def fit(self, X, y):
                self.model.fit(X, y)
                return self
            def predict(self, X):
                return self.model.predict(X)
            def predict_proba(self, X):
                return self.model.predict_proba(X)

        multi_models["XGBoost_OVR"] = OneVsRestClassifier(XGBWrapper())

    if CATBOOST_AVAILABLE:
        multi_models["CatBoost_OVR"] = OneVsRestClassifier(
            CatBoostClassifier(
                iterations=400, learning_rate=0.05, depth=6, random_seed=42, verbose=0
            )
        )

    best_scores = {}
    for name, clf in multi_models.items():
        print("Training:", name)
        clf.fit(Xtr_s, ytr)
        yp = clf.predict(Xte_s)

        per_shark = []
        for i, col in enumerate(yte.columns):
            yt = yte.iloc[:, i]
            yp_col = yp[:, i]
            acc = accuracy_score(yt, yp_col)
            prec = precision_score(yt, yp_col, zero_division=0)
            rec = recall_score(yt, yp_col, zero_division=0)
            f1  = f1_score(yt, yp_col, zero_division=0)
            per_shark.append({"shark":col,"acc":acc,"prec":prec,"rec":rec,"f1":f1})

            cm = confusion_matrix(yt, yp_col)
            ConfusionMatrixDisplay(cm).plot(cmap="Purples")
            plt.title(name + " - " + col)
            plt.show()

        dfp = pd.DataFrame(per_shark)
        macro = dfp[["acc","prec","rec","f1"]].mean()
        print("Macro averages:", macro.to_dict())

        best_scores[name] = macro["f1"]

        joblib.dump((clf, scaler_s), "best_" + name + "_sharks.pkl")

    best_model = max(best_scores, key=lambda k: best_scores[k])
    print("Best shark model:", best_model, "F1 =", best_scores[best_model])

print("All tasks completed. Models saved.")
