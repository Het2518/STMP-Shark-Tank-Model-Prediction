# app.py — Full Featured Shark Tank India AI Predictor
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.base import BaseEstimator, ClassifierMixin

# ============================================================
#  XGBWrapper for compatibility
# ============================================================
try:
    from xgboost import XGBClassifier
    
    class XGBWrapper(BaseEstimator, ClassifierMixin):
        def __init__(self):
            self.model = XGBClassifier(use_label_encoder=False, eval_metric="logloss")

        def fit(self, X, y):
            self.model.fit(X, y)
            return self

        def predict(self, X):
            return self.model.predict(X)

        def predict_proba(self, X):
            try:
                return self.model.predict_proba(X)
            except:
                return np.zeros((len(X), 2))
except:
    pass


# ============================================================
#  PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Shark Tank India AI Predictor",
    page_icon="🦈",
    layout="wide"
)

st.title("🦈 Shark Tank India — AI Deal Predictor")
st.caption("Predict Deal Outcomes • Negotiate Valuations • Identify Interested Sharks — Powered by Machine Learning")

DATA_FILE = "sharkTankIndia.xlsx"

# ============================================================
#  LOAD DATASET
# ============================================================
if not os.path.exists(DATA_FILE):
    if os.path.exists("/mnt/data/sharkTankIndia.csv"):
        DATA_FILE = "/mnt/data/sharkTankIndia.csv"

if not os.path.exists(DATA_FILE):
    st.warning("⚠️ Dataset not found. Dropdown lists will be limited.")
    df = pd.DataFrame()
else:
    try:
        if DATA_FILE.endswith(".csv"):
            df = pd.read_csv(DATA_FILE)
        else:
            df = pd.read_excel(DATA_FILE)
        
        # Fix column names
        if "Pitchers Average Age " in df.columns:
            df.rename(columns={"Pitchers Average Age ": "Pitchers Average Age"}, inplace=True)
            
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        df = pd.DataFrame()

# Extract dropdown options
industries = sorted(df["Industry"].dropna().unique().tolist()) if "Industry" in df.columns else ["Other"]
cities = sorted(df["Pitchers City"].dropna().unique().tolist()) if "Pitchers City" in df.columns else ["Unknown"]
states = sorted(df["Pitchers State"].dropna().unique().tolist()) if "Pitchers State" in df.columns else ["Unknown"]

# ============================================================
#  LOAD TRAINED MODELS
# ============================================================
@st.cache_resource
def load_models():
    models = {}
    
    # Auto-detect models from directory
    for fname in os.listdir("."):
        if fname.endswith("_deal.pkl"):
            try:
                models["deal_model"] = joblib.load(fname)
            except:
                pass
        elif fname.endswith("_valuation.pkl"):
            try:
                models["valuation_model"] = joblib.load(fname)
            except:
                pass
        elif fname.endswith("_sharks.pkl"):
            try:
                models["shark_model"] = joblib.load(fname)
            except:
                pass
    
    return models

models = load_models()

# Validate models
if "deal_model" not in models or "valuation_model" not in models or "shark_model" not in models:
    st.error("❌ Model files missing! Please run train_all_models.py first.")
    st.info("Required: *_deal.pkl, *_valuation.pkl, *_sharks.pkl")
    st.stop()

deal_clf, deal_scaler = models["deal_model"]
valuation_clf, valuation_scaler = models["valuation_model"]
shark_clf, shark_scaler = models["shark_model"]

sharks = ['Namita', 'Vineeta', 'Anupam', 'Aman', 'Peyush', 'Ritesh', 'Amit', 'Guest']

# ============================================================
#  SHARK EXPERTISE MAPPING
# ============================================================
shark_expertise = {
    'Namita': {'domain': 'Healthcare & Pharma', 'style': 'Data-driven, focuses on margins & profitability'},
    'Vineeta': {'domain': 'Fashion & Lifestyle', 'style': 'Brand-focused, D2C & marketing expertise'},
    'Anupam': {'domain': 'Entertainment & Services', 'style': 'Strategic networks, mentorship approach'},
    'Aman': {'domain': 'E-commerce & Tech', 'style': 'Aggressive growth, platform scalability'},
    'Peyush': {'domain': 'Technology & SaaS', 'style': 'Product-focused, technical deep-dive'},
    'Ritesh': {'domain': 'Food & Hospitality', 'style': 'Customer experience, brand building'},
    'Amit': {'domain': 'Fashion & Retail', 'style': 'Supply chain, manufacturing networks'},
    'Guest': {'domain': 'Various', 'style': 'Depends on guest shark'}
}

# ============================================================
#  INPUT SECTION
# ============================================================
st.markdown("---")
st.markdown("### 🎯 Enter Your Pitch Details")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**👥 Team Information**")
    num_presenters = st.number_input("No. of Presenters", min_value=1, max_value=10, value=2)
    male_presenters = st.number_input("Male Presenters", min_value=0, max_value=10, value=1)
    female_presenters = st.number_input("Female Presenters", min_value=0, max_value=10, value=1)
    couple_presenters = st.number_input("Couple Presenters", min_value=0, max_value=5, value=0)

with col2:
    st.markdown("**📍 Demographics**")
    pitch_age = st.number_input("Pitchers Average Age", min_value=18, max_value=80, value=32)
    pitch_city = st.selectbox("City", options=cities)
    pitch_state = st.selectbox("State", options=states)
    industry = st.selectbox("Industry", options=industries)

with col3:
    st.markdown("**🦈 Show Details**")
    num_sharks_present = st.slider("No. of Sharks Present", min_value=1, max_value=8, value=5)
    st.info(f"💡 **Tip:** {num_sharks_present} sharks = {'High' if num_sharks_present >= 5 else 'Moderate' if num_sharks_present >= 3 else 'Low'} competition")

st.markdown("---")
st.markdown("### 💰 Financial Details")

colA, colB, colC = st.columns(3)

with colA:
    ask_amount = st.number_input(
        "Ask Amount (₹ Lakh)", 
        min_value=1.0, 
        max_value=100000.0, 
        value=100.0,
        help="How much funding are you seeking?"
    )

with colB:
    offered_equity = st.number_input(
        "Equity Offered (%)", 
        min_value=0.1, 
        max_value=100.0, 
        value=10.0,
        help="What % of your company are you offering?"
    )

with colC:
    if offered_equity > 0:
        valuation = (ask_amount / offered_equity) * 100
        valuation_cr = valuation / 100
    else:
        valuation = 0
        valuation_cr = 0
    
    st.metric(
        "🏢 Implied Valuation", 
        f"₹{valuation:,.0f} Lakh",
        f"≈ ₹{valuation_cr:.2f} Cr"
    )
    
    # Reality check
    if valuation_cr > 100:
        st.warning("⚠️ Very high valuation - Sharks may be skeptical")
    elif valuation_cr < 1:
        st.warning("⚠️ Very low valuation - Consider higher ask or lower equity")

# Derived features
valuation_per_presenter = valuation / num_presenters if num_presenters else 0
equity_per_shark = offered_equity / num_sharks_present if num_sharks_present else 0

# ============================================================
#  BUILD FEATURE VECTOR
# ============================================================
feat_dict = {
    'Season No': 1,
    'Episode No': 1,
    'Pitch No': 1,
    'No of Presenters': num_presenters,
    'Male Presenters': male_presenters,
    'Female Presenters': female_presenters,
    'Couple Presenters': couple_presenters,
    'Pitchers Average Age': pitch_age,
    'Pitchers City': pitch_city,
    'Pitchers State': pitch_state,
    'Original Ask Amount': ask_amount,
    'Original Offered Equity': offered_equity,
    'Valuation Requested': valuation,
    'Industry': industry,
    'Num Sharks Present': num_sharks_present,
    'Valuation_per_Presenter': valuation_per_presenter,
    'Equity_per_Shark': equity_per_shark
}

features = pd.DataFrame([feat_dict])

# Encode categorical features
cat_cols = ["Pitchers City", "Pitchers State", "Industry"]
for col in cat_cols:
    if col in features.columns:
        le = LabelEncoder()
        le.fit(df[col].astype(str))
        val = str(features.loc[0, col])
        try:
            features.loc[0, col] = int(le.transform([val])[0])
        except:
            features.loc[0, col] = -1

# Reorder columns
train_cols = [
    "Season No", "Episode No", "Pitch No",
    "No of Presenters", "Male Presenters", "Female Presenters", "Couple Presenters",
    "Pitchers Average Age", "Pitchers City", "Pitchers State",
    "Original Ask Amount", "Original Offered Equity", "Valuation Requested", "Industry",
    "Num Sharks Present", "Valuation_per_Presenter", "Equity_per_Shark"
]

for c in train_cols:
    if c not in features.columns:
        features[c] = 0
features = features[train_cols]

# ============================================================
#  MAKE PREDICTIONS
# ============================================================
# Deal prediction
X_deal = deal_scaler.transform(features)
deal_pred = int(deal_clf.predict(X_deal)[0])

deal_prob = None
try:
    if hasattr(deal_clf, "predict_proba"):
        deal_prob = float(deal_clf.predict_proba(X_deal)[0][1])
except:
    pass

# Valuation prediction
X_val = valuation_scaler.transform(features)
try:
    valuation_pred = float(valuation_clf.predict(X_val)[0])
except:
    valuation_pred = valuation
valuation_pred_cr = valuation_pred / 100

# Shark prediction
X_shark = shark_scaler.transform(features)
try:
    shark_pred = shark_clf.predict(X_shark)[0]
except:
    shark_pred = np.zeros(len(sharks), dtype=int)

try:
    if hasattr(shark_clf, "predict_proba"):
        shark_probs = shark_clf.predict_proba(X_shark)[0]
    else:
        shark_probs = np.zeros(len(sharks))
except:
    shark_probs = np.zeros(len(sharks))

# Logical consistency: no deal = no sharks
if deal_pred == 0:
    final_shark_pred = np.zeros(len(sharks), dtype=int)
else:
    final_shark_pred = shark_pred.astype(int)

# ============================================================
#  RESULTS SECTION
# ============================================================
st.markdown("---")
st.markdown("## 📊 Prediction Results")

# Overview Cards
overview_col1, overview_col2, overview_col3 = st.columns(3)

with overview_col1:
    if deal_pred == 1:
        st.success("✅ DEAL LIKELY")
        if deal_prob:
            st.metric("Confidence", f"{deal_prob*100:.1f}%")
            st.progress(deal_prob)
    else:
        st.error("❌ DEAL UNLIKELY")
        if deal_prob:
            st.metric("Rejection Risk", f"{(1-deal_prob)*100:.1f}%")
            st.progress(1-deal_prob)

with overview_col2:
    if deal_pred == 1:
        st.info("💰 Predicted Valuation")
        st.metric("Final Value", f"₹{valuation_pred_cr:.2f} Cr", 
                 f"vs ₹{valuation_cr:.2f} Cr asked")
    else:
        st.warning("💰 Valuation")
        st.metric("Value", "N/A (No deal predicted)")

with overview_col3:
    interested_count = sum(final_shark_pred)
    if deal_pred == 1 and interested_count > 0:
        st.success(f"🦈 {interested_count} Shark(s) Interested")
        st.metric("Investors", f"{interested_count}/{num_sharks_present}")
    elif deal_pred == 1:
        st.warning("🦈 Shark Interest")
        st.metric("Status", "Uncertain")
    else:
        st.error("🦈 No Investors")
        st.metric("Status", "No deal")

# ============================================================
#  DETAILED ANALYSIS TABS
# ============================================================
st.markdown("---")
st.markdown("## 📈 Detailed Analysis")

tab1, tab2, tab3 = st.tabs(["📊 Deal Analysis", "💰 Valuation Breakdown", "🦈 Shark Predictions"])

# --- TAB 1: DEAL ANALYSIS ---
with tab1:
    st.subheader("📊 Deal Acceptance Prediction")
    
    if deal_pred == 1:
        st.success("🎉 **Your pitch is likely to get a DEAL!**")
        
        if deal_prob is not None:
            st.progress(int(deal_prob * 100))
            st.caption(f"Model Confidence: {deal_prob*100:.2f}%")
            
            if deal_prob > 0.8:
                st.markdown("**💪 Excellent Pitch!** Very high success probability.")
            elif deal_prob > 0.6:
                st.markdown("**👍 Good Pitch!** Solid chances with potential for improvement.")
            else:
                st.markdown("**⚠️ Moderate Pitch** - Some refinements could strengthen your proposal.")
        
        st.markdown("### ✅ What's Working:")
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown(f"✓ Valuation: ₹{valuation_cr:.2f} Cr")
            st.markdown(f"✓ Equity Offer: {offered_equity:.1f}%")
        with col_b:
            st.markdown(f"✓ Ask Amount: ₹{ask_amount:.0f} Lakh")
            st.markdown(f"✓ Team Size: {num_presenters} presenters")
        
    else:
        st.error("❌ **Your pitch is predicted to be REJECTED**")
        
        if deal_prob is not None:
            rejection_conf = (1 - deal_prob) * 100
            st.progress(int(rejection_conf))
            st.caption(f"Rejection Confidence: {rejection_conf:.2f}%")
        
        st.markdown("### 💡 Suggestions to Improve:")
        
        # Calculate suggested improvements
        better_equity = offered_equity * 1.5
        better_ask = ask_amount * 0.7
        better_valuation = valuation * 0.65
        
        col_sug1, col_sug2 = st.columns(2)
        
        with col_sug1:
            st.markdown("**📈 Option 1: Increase Equity Offer**")
            st.markdown(f"• Offer **{better_equity:.1f}%** (vs {offered_equity:.1f}%)")
            st.markdown(f"• Keep ask at ₹{ask_amount:.0f} Lakh")
            new_val_1 = (ask_amount / better_equity) * 100 / 100
            st.markdown(f"• New valuation: **₹{new_val_1:.2f} Cr**")
        
        with col_sug2:
            st.markdown("**📉 Option 2: Reduce Ask Amount**")
            st.markdown(f"• Ask for **₹{better_ask:.0f} Lakh** (vs ₹{ask_amount:.0f}L)")
            st.markdown(f"• Keep equity at {offered_equity:.1f}%")
            new_val_2 = (better_ask / offered_equity) * 100 / 100
            st.markdown(f"• New valuation: **₹{new_val_2:.2f} Cr**")
        
        st.markdown("**🎯 Option 3: Balanced Approach**")
        st.markdown(f"• Target valuation: **₹{better_valuation/100:.2f} Cr**")
        suggested_ask = (better_valuation * better_equity / 100)
        st.markdown(f"• Ask: ₹{suggested_ask:.0f} Lakh for {better_equity:.1f}% equity")

# --- TAB 2: VALUATION ---
with tab2:
    st.subheader("💰 Valuation Analysis & Negotiation Strategy")
    
    if deal_pred == 1:
        st.markdown("📈 **Based on similar successful pitches:**")
        
        col_v1, col_v2, col_v3 = st.columns(3)
        
        with col_v1:
            st.metric("Your Ask", f"₹{valuation_cr:.2f} Cr")
        
        with col_v2:
            diff = valuation_pred - valuation
            st.metric("Predicted Final", f"₹{valuation_pred_cr:.2f} Cr", f"{diff:+,.0f} Lakh")
        
        with col_v3:
            neg_factor = valuation_pred / valuation if valuation > 0 else 1
            st.metric("Negotiation Factor", f"{neg_factor:.2f}x")
        
        st.markdown("---")
        st.markdown("### 🎯 Negotiation Insights")
        
        if neg_factor > 1.2:
            st.success("🎯 **You're UNDERVALUING your company!**")
            st.markdown(f"💡 **Recommendation:**")
            st.markdown(f"• Initial ask: **₹{valuation_pred * 1.2 / 100:.2f} Cr** (be ambitious)")
            st.markdown(f"• Target: **₹{valuation_pred_cr:.2f} Cr** (expected settlement)")
            st.markdown(f"• Floor: **₹{valuation_pred * 0.85 / 100:.2f} Cr** (don't go below)")
            
        elif neg_factor < 0.8:
            st.warning("📉 **Your valuation seems HIGH**")
            st.markdown(f"💡 **Recommendation:**")
            st.markdown(f"• Sharks will counter at **₹{valuation_pred_cr:.2f} Cr**")
            st.markdown(f"• That's {abs((1-neg_factor)*100):.1f}% below your ask")
            st.markdown(f"• **Action:** Justify valuation with strong metrics or accept lower offer")
            
        else:
            st.info("✅ **Your valuation is REALISTIC**")
            st.markdown(f"💡 **Recommendation:**")
            min_range = valuation_pred * 0.90 / 100
            max_range = valuation_pred * 1.10 / 100
            st.markdown(f"• Expected range: **₹{min_range:.2f} - ₹{max_range:.2f} Cr**")
            st.markdown(f"• Minor negotiations expected")
        
        st.markdown("---")
        st.markdown("### 🏆 Optimal Negotiation Strategy")
        
        optimal_ask = valuation_pred * 1.15 / 100
        optimal_target = valuation_pred_cr
        optimal_floor = valuation_pred * 0.85 / 100
        
        col_opt1, col_opt2, col_opt3 = st.columns(3)
        
        with col_opt1:
            st.metric("🎯 Initial Ask", f"₹{optimal_ask:.2f} Cr", "Be ambitious")
        with col_opt2:
            st.metric("💼 Target Settlement", f"₹{optimal_target:.2f} Cr", "Realistic goal")
        with col_opt3:
            st.metric("🛑 Bottom Line", f"₹{optimal_floor:.2f} Cr", "Walk away point")
        
    else:
        st.warning("⚠️ **Valuation analysis not applicable**")
        st.markdown("Since deal rejection is predicted, focus on improving deal acceptance first.")
        st.markdown("")
        st.markdown("**To attract sharks:**")
        st.markdown(f"1. Reduce valuation to **₹{(valuation * 0.6 / 100):.2f} Cr**")
        st.markdown(f"2. Increase equity offer to **{offered_equity * 1.5:.1f}%**")
        st.markdown("3. Show market traction and customer validation")

# --- TAB 3: SHARK PREDICTIONS ---
with tab3:
    st.subheader("🦈 Shark Investment Predictions")
    
    if deal_pred == 1:
        st.markdown("📌 **Based on your pitch profile and historical patterns:**")
        
        # Create shark table
        shark_data = []
        for i, shark_name in enumerate(sharks):
            investment_status = "✅ Likely" if final_shark_pred[i] == 1 else "❌ Unlikely"
            confidence = shark_probs[i] * 100 if len(shark_probs) > i else 0
            expertise = shark_expertise.get(shark_name, {})
            
            shark_data.append({
                'Shark': shark_name,
                'Domain': expertise.get('domain', 'General'),
                'Prediction': investment_status,
                'Confidence': f"{confidence:.1f}%"
            })
        
        df_sharks = pd.DataFrame(shark_data)
        st.dataframe(df_sharks, use_container_width=True, hide_index=True)
        
        # Interested sharks section
        likely_investors = [sharks[i] for i, pred in enumerate(final_shark_pred) if pred == 1]
        
        if likely_investors:
            st.success(f"✅ **Likely Investors:** {', '.join(likely_investors)}")
            st.markdown(f"🟢 *{len(likely_investors)} out of {num_sharks_present} sharks interested*")
            
            if len(likely_investors) > 1:
                st.info("💡 **Multiple sharks interested!** This creates competition - leverage it for better terms.")
            
        else:
            st.warning("⚠️ **Uncertain Shark Interest**")
            st.markdown("While deal is likely, specific shark interest is unclear. This could mean:")
            st.markdown("• 🤝 Joint offer from multiple sharks")
            st.markdown("• 🎯 Appeal is broad but not strongly specialized")
            st.markdown("• 💡 Emphasize industry-specific strengths")
        
        st.markdown("---")
        st.markdown("### 🏢 Shark Expertise & Strategy")
        
        for shark_name in likely_investors if likely_investors else sharks[:3]:
            if shark_name in shark_expertise:
                info = shark_expertise[shark_name]
                col_info1, col_info2 = st.columns(2)
                
                with col_info1:
                    st.markdown(f"**{shark_name}**")
                    st.markdown(f"🎯 Domain: {info['domain']}")
                
                with col_info2:
                    st.markdown(f"")
                    st.markdown(f"💼 Style: {info['style']}")
                
                st.markdown("---")
    
    else:
        st.error("🚫 **No Shark Investment Expected**")
        st.markdown("Deal rejection predicted → No sharks will make offers")
        st.markdown("")
        st.markdown("### ✅ Steps to Attract Sharks:")
        st.markdown("1. **Fix Deal Structure** - Go to Deal Analysis tab")
        st.markdown("2. **Optimize Valuation** - Lower by 30-40%")
        st.markdown(f"3. **Industry Focus** - Your sector: **{industry}**")
        st.markdown("4. **Show Traction** - Revenue, growth, customers")

# ============================================================
#  FOOTER
# ============================================================
st.markdown("---")
st.markdown("### 📚 About This Predictor")

with st.expander("ℹ️ How It Works"):
    st.markdown("""
    **Machine Learning Models Used:**
    
    1. **Deal Classification** - Predicts if pitch will result in deal
    2. **Valuation Regression** - Predicts final negotiated valuation
    3. **Multi-label Shark Classifier** - Predicts individual shark interest
    
    **Key Logic:**
    - ✅ If **DEAL** predicted → Sharks may invest
    - ❌ If **NO DEAL** predicted → No shark investments
    
    **Training Data:** Real Shark Tank India episodes
    
    **Important:** Predictions are estimates based on historical patterns. Actual results depend on pitch quality, timing, and market conditions.
    """)

with st.expander("⚙️ Model Information"):
    st.markdown(f"""
    **Features Used:** {len(train_cols)} parameters
    
    **Model Components:**
    - Deal Model: Classification (Logistic/RF/XGBoost)
    - Valuation Model: Regression (Ridge/Lasso/RF)
    - Shark Model: Multi-label OVR Classification
    
    **Data Points:** Historical Shark Tank India episodes
    """)

st.markdown("---")
st.caption("🦈 Shark Tank India AI Predictor • Powered by Streamlit + Scikit-Learn • Made with ❤️")
st.caption("⚠️ Disclaimer: Predictions are estimates. Actual results may vary based on pitch execution and market conditions.")