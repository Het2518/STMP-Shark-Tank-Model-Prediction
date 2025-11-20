
---

# **STMP — Shark Tank Model Prediction**

## **Project Links**

* **GitHub Repository:** [https://github.com/Het2518/STMP-Shark-Tank-Model-Prediction](https://github.com/Het2518/STMP-Shark-Tank-Model-Prediction)
* **Live Demo:** [https://stmp-shark-tank-model-prediction.streamlit.app/](https://stmp-shark-tank-model-prediction.streamlit.app/)

---

## Overview

A small, easy-to-run project that helps you explore pitch outcomes from Shark Tank India episodes. It includes a simple web UI and scripts to train models that predict:

* whether a pitch will get a deal,
* the likely final valuation, and
* which sharks are most likely to invest.

The UI is built with Streamlit so anyone can try ideas without deep machine-learning knowledge.

---

## Get started

1. Create a virtual environment and install dependencies (Windows PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

2. Put your dataset at the project root.
   The app and training script look for `sharkTankIndia.xlsx` (or `sharkTankIndia.csv`).

3. Train models (only if you need to re-train):

```powershell
python train_all_models.py
```

4. Start the web UI:

```powershell
streamlit run app.py
```

Open the URL Streamlit prints (usually [http://localhost:8501](http://localhost:8501)) to use the app.

---

## Requirements

* Python 3.8 or newer is recommended.
* Install with:

```powershell
pip install -r requirements.txt
```

### Main packages used:

* `streamlit`, `pandas`, `numpy`, `scikit-learn`, `joblib`, `matplotlib`, `openpyxl`
* Optional for training: `xgboost`, `catboost`
  *(the training script will try to install `catboost` if it is missing)*

---

## Data format (input)

The training script and the UI expect a dataset with columns like these (they can be computed if missing):

* Season No, Episode No, Pitch No
* No of Presenters, Male Presenters, Female Presenters, Couple Presenters
* Pitchers Average Age, Pitchers City, Pitchers State
* Original Ask Amount, Original Offered Equity, Valuation Requested, Industry
* Num Sharks Present, Valuation_per_Presenter, Equity_per_Shark

### Optional targets for training:

* `Accepted Offer` (0 or 1)
* `Deal Valuation`
* Per-shark investment columns:

  * `Namita Investment Amount`, `Vineeta Invested Amount`, etc.

### Example (CSV):

```csv
Season No,Episode No,Pitch No,No of Presenters,Male Presenters,Female Presenters,Pitchers Average Age,Pitchers City,Pitchers State,Original Ask Amount,Original Offered Equity,Valuation Requested,Industry,Num Sharks Present

1,1,1,2,1,1,32,Mumbai,Maharashtra,100,10,100,Food,5
```

---

## What the code writes (outputs)

* Trained models (saved via `joblib.dump((model, scaler), filename)`):

  * `best_<ModelName>_deal.pkl` (deal classifier + scaler)
  * `best_<ModelName>_valuation.pkl` (valuation regressor + scaler)
  * `best_<ModelName>_sharks.pkl` (multi-label shark model + scaler)

* If CatBoost is used, a folder `catboost_info/` is created with:

  * `catboost_training.json`
  * `learn_error.tsv`
  * TensorBoard event logs

* The Streamlit UI shows predictions in the browser:

  * deal probability
  * predicted valuation
  * shark investment likelihood table
  * negotiation tips

---

## Troubleshooting

* **"Model files missing!"**
  → Run `python train_all_models.py`
  or copy trained `*_deal.pkl`, `*_valuation.pkl`, `*_sharks.pkl` into the project root.

* **CatBoost/XGBoost errors during training**
  → Try:

  ```powershell
  pip install catboost xgboost
  ```

* **Streamlit port already in use**
  Streamlit will automatically use another port and display the correct URL.

---