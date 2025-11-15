# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ==========================
# PAGE CONFIG
# ==========================
st.set_page_config(
    page_title="🦈 Shark Tank India AI Predictor",
    page_icon="🦈",
    layout="wide"
)

st.title("🦈 Shark Tank India AI Predictor")
st.caption("Predict Deal Outcomes, Negotiation Valuation, and Shark Investments — powered by Machine Learning")

# ==========================
# LOAD DATASET
# ==========================
DATA_FILE = "sharkTankIndia.xlsx"
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
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        df = pd.DataFrame()

# Extract dropdown options
industries = sorted(df["Industry"].dropna().unique().tolist()) if "Industry" in df.columns else ["Other"]
cities = sorted(df["Pitchers City"].dropna().unique().tolist()) if "Pitchers City" in df.columns else ["Unknown"]
states = sorted(df["Pitchers State"].dropna().unique().tolist()) if "Pitchers State" in df.columns else ["Unknown"]

# ==========================
# LOAD MODELS
# ==========================
@st.cache_resource
def load_models():
    models = {}
    required_files = {
        "deal_model.pkl": None,
        "valuation_model.pkl": None,
        "shark_model.pkl": None,
        "label_encoders.pkl": {},
        "industry_stats.pkl": None,
        "feature_columns.pkl": []
    }
    
    for fname in required_files.keys():
        if os.path.exists(fname):
            models[fname] = joblib.load(fname)
        else:
            models[fname] = required_files[fname]
    
    return models

models = load_models()

# Validate required models
if models["deal_model.pkl"] is None or models["valuation_model.pkl"] is None or models["shark_model.pkl"] is None:
    st.error("❌ Model files missing! Please run `run_all_models.py` first to train and save models.")
    st.info("Required files: deal_model.pkl, valuation_model.pkl, shark_model.pkl")
    st.stop()

# Extract models
deal_clf, deal_scaler = models["deal_model.pkl"]
valuation_model_obj = models["valuation_model.pkl"]
shark_clf, shark_scaler = models["shark_model.pkl"]
label_encoders = models["label_encoders.pkl"]
industry_stats_df = models["industry_stats.pkl"]
feature_columns = models["feature_columns.pkl"]

sharks = ['Namita', 'Vineeta', 'Anupam', 'Aman', 'Peyush', 'Ritesh', 'Amit', 'Guest']

# ==========================
# INPUT SECTION
# ==========================
st.markdown("---")
st.markdown("### 🎯 Enter Pitch Details")

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
    st.info(f"💡 **Tip:** More sharks = more competition for your deal!")

st.markdown("---")
st.markdown("### 💰 Financial Details")

colA, colB, colC = st.columns(3)

with colA:
    ask_amount = st.number_input(
        "Ask Amount (₹ Lakh)", 
        min_value=1.0, 
        max_value=100000.0, 
        value=100.0,
        help="How much money are you asking for?"
    )

with colB:
    offered_equity = st.number_input(
        "Equity Offered (%)", 
        min_value=0.1, 
        max_value=100.0, 
        value=10.0,
        help="What percentage of your company are you offering?"
    )

with colC:
    valuation = (ask_amount / offered_equity) * 100 if offered_equity else 0
    valuation_cr = valuation / 100
    st.metric(
        "🏢 Implied Company Valuation", 
        f"₹{valuation:,.0f} Lakh",
        f"≈ ₹{valuation_cr:.2f} Cr"
    )
    
    # Reality check
    if valuation_cr > 100:
        st.warning("⚠️ Very high valuation! Sharks might be skeptical.")
    elif valuation_cr < 1:
        st.warning("⚠️ Very low valuation. Consider asking for more equity or higher amount.")

# Derived features
valuation_per_presenter = valuation / num_presenters if num_presenters else 0
equity_per_shark = offered_equity / num_sharks_present if num_sharks_present else 0

# ==========================
# BUILD FEATURE DATAFRAME
# ==========================
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
for col, le in label_encoders.items():
    if col in features.columns:
        val = str(features.loc[0, col])
        if val in le.classes_:
            features.loc[0, col] = int(le.transform([val])[0])
        else:
            features.loc[0, col] = -1

# Align with training columns
expected_cols = feature_columns
for c in expected_cols:
    if c not in features.columns:
        features[c] = 0
features = features[expected_cols]

# ==========================
# MAKE PREDICTIONS
# ==========================
# Deal prediction
X_scaled = deal_scaler.transform(features)
deal_pred = int(deal_clf.predict(X_scaled)[0])
deal_prob = None
try:
    deal_prob = float(deal_clf.predict_proba(X_scaled)[0][1]) if hasattr(deal_clf, "predict_proba") else None
except:
    deal_prob = None

# Valuation prediction
if isinstance(valuation_model_obj, tuple):
    reg_model, reg_scaler = valuation_model_obj
    Xv_scaled = reg_scaler.transform(features)
    valuation_pred = float(reg_model.predict(Xv_scaled)[0])
else:
    valuation_pred = float(valuation_model_obj[1]) if valuation_model_obj[0] == "mean" else 0

valuation_pred_cr = valuation_pred / 100

# Shark prediction
Xs_scaled = shark_scaler.transform(features)
try:
    shark_pred = shark_clf.predict(Xs_scaled)[0]
    shark_probs_model = shark_clf.predict_proba(Xs_scaled)[0] if hasattr(shark_clf, "predict_proba") else np.zeros(len(sharks))
except:
    shark_pred = np.zeros(len(sharks), dtype=int)
    shark_probs_model = np.zeros(len(sharks))

# ==========================
# LOGICAL SHARK PREDICTION FIX
# ==========================
if deal_pred == 0:
    # NO DEAL → NO SHARKS
    final_shark_pred = np.zeros(len(sharks), dtype=int)
else:
    # DEAL → Use model + fallback logic
    final_shark_pred = shark_pred.astype(int).copy()
    
    # If model predicted no sharks despite deal (edge case)
    if final_shark_pred.sum() == 0:
        ind_val = features['Industry'].iloc[0]
        if ind_val in industry_stats_df.index:
            industry_probs = industry_stats_df.loc[ind_val].reindex(sharks).fillna(0).values
        else:
            industry_probs = industry_stats_df.mean(axis=0).reindex(sharks).fillna(0).values
        
        # Combine model + historical
        combined = 0.6 * np.array(shark_probs_model) + 0.4 * np.array(industry_probs)
        
        # Pick top K sharks
        k = min(int(num_sharks_present), max(1, int(np.ceil(num_sharks_present / 2))))
        top_idx = np.argsort(combined)[::-1][:k]
        final_shark_pred = np.zeros(len(sharks), dtype=int)
        final_shark_pred[top_idx] = 1
    
    # Trim if too many sharks predicted
    elif final_shark_pred.sum() > num_sharks_present:
        idxs = np.where(final_shark_pred == 1)[0]
        probs_selected = np.array(shark_probs_model)[idxs]
        keep = idxs[np.argsort(probs_selected)[::-1][:num_sharks_present]]
        new_pred = np.zeros(len(sharks), dtype=int)
        new_pred[keep] = 1
        final_shark_pred = new_pred

# ==========================
# OVERVIEW SECTION
# ==========================
st.markdown("---")
st.markdown("## 📋 Prediction Summary")

overview_col1, overview_col2, overview_col3 = st.columns(3)

with overview_col1:
    if deal_pred == 1:
        st.success("✅ **DEAL LIKELY**")
        if deal_prob:
            st.metric("Confidence", f"{deal_prob*100:.1f}%")
        st.markdown("🎉 Your pitch has strong potential!")
    else:
        st.error("❌ **NO DEAL LIKELY**")
        if deal_prob:
            st.metric("Rejection Confidence", f"{(1-deal_prob)*100:.1f}%")
        st.markdown("💡 Consider adjusting your ask/equity")

with overview_col2:
    if deal_pred == 1:
        st.info("💰 **Expected Final Deal**")
        st.metric("Valuation", f"₹{valuation_pred_cr:.2f} Cr")
        diff_pct = ((valuation_pred - valuation) / valuation * 100) if valuation > 0 else 0
        if abs(diff_pct) > 20:
            st.caption(f"{'📈' if diff_pct > 0 else '📉'} {abs(diff_pct):.1f}% from your ask")
    else:
        st.warning("💰 **No Valuation**")
        st.caption("Deal unlikely to proceed")

with overview_col3:
    if deal_pred == 1:
        likely_investors = [sharks[i] for i, pred in enumerate(final_shark_pred) if pred == 1]
        if likely_investors:
            st.success(f"🦈 **{len(likely_investors)} Shark(s)**")
            st.caption(", ".join(likely_investors))
        else:
            st.warning("🦈 **Uncertain**")
            st.caption("Deal possible but unclear who")
    else:
        st.error("🦈 **No Investors**")
        st.caption("No deal = no investment")

# ==========================
# DETAILED TABS
# ==========================
st.markdown("---")
st.markdown("## 📊 Detailed Analysis")

tab1, tab2, tab3 = st.tabs(["📊 Deal Analysis", "💰 Valuation Breakdown", "🦈 Shark Predictions"])

# --- TAB 1: DEAL ---
with tab1:
    st.subheader("📊 Deal Acceptance Prediction")
    
    if deal_pred == 1:
        st.success("🎉 **Your pitch is likely to get a deal!**")
        
        if deal_prob is not None:
            st.progress(int(deal_prob * 100))
            st.caption(f"Model Confidence: {deal_prob*100:.2f}%")
            
            if deal_prob > 0.8:
                st.markdown("**💪 Excellent Pitch!** Very high success probability.")
            elif deal_prob > 0.6:
                st.markdown("**👍 Good Pitch!** Solid chances with minor improvements possible.")
            else:
                st.markdown("**⚠️ Moderate Pitch** - Consider optimizing your ask/equity ratio.")
        
        st.markdown("### ✅ What's Working:")
        st.markdown(f"- Your valuation of ₹{valuation_cr:.2f} Cr seems reasonable")
        st.markdown(f"- {offered_equity:.1f}% equity offering is in acceptable range")
        st.markdown(f"- {num_sharks_present} sharks present increases competition")
        
    else:
        st.error("❌ **Your pitch is unlikely to get a deal.**")
        
        if deal_prob is not None:
            rejection_conf = (1 - deal_prob) * 100
            st.progress(int(rejection_conf))
            st.caption(f"Rejection Confidence: {rejection_conf:.2f}%")
        
        st.markdown("### 💡 Suggestions to Improve:")
        
        # Calculate better values
        better_equity = offered_equity * 1.3
        better_ask = ask_amount * 0.8
        better_valuation = valuation * 0.7
        
        col_sug1, col_sug2 = st.columns(2)
        
        with col_sug1:
            st.markdown("**🎯 Option 1: Increase Equity**")
            st.markdown(f"- Offer **{better_equity:.1f}%** instead of {offered_equity:.1f}%")
            st.markdown(f"- Keep ask at ₹{ask_amount:.0f} Lakh")
            st.markdown(f"- New valuation: ₹{(ask_amount/better_equity*100/100):.2f} Cr")
        
        with col_sug2:
            st.markdown("**💸 Option 2: Reduce Ask**")
            st.markdown(f"- Ask for **₹{better_ask:.0f} Lakh** instead")
            st.markdown(f"- Keep equity at {offered_equity:.1f}%")
            st.markdown(f"- New valuation: ₹{(better_ask/offered_equity*100/100):.2f} Cr")
        
        st.markdown("**🎯 Option 3: Combined Approach**")
        st.markdown(f"- Target valuation: **₹{better_valuation/100:.2f} Cr**")
        st.markdown(f"- Suggested ask: ₹{better_valuation * offered_equity / 100:.0f} Lakh for {offered_equity:.1f}% equity")

# --- TAB 2: VALUATION ---
with tab2:
    st.subheader("💰 Valuation Analysis")
    
    if deal_pred == 1:
        st.markdown("📈 **Based on similar successful pitches:**")
        
        col_v1, col_v2, col_v3 = st.columns(3)
        
        with col_v1:
            st.metric("Your Ask", f"₹{valuation_cr:.2f} Cr")
        
        with col_v2:
            diff = valuation_pred - valuation
            st.metric("Predicted Final", f"₹{valuation_pred_cr:.2f} Cr", f"{diff:+,.0f} Lakh")
        
        with col_v3:
            negotiation_factor = valuation_pred / valuation if valuation > 0 else 1
            st.metric("Negotiation Factor", f"{negotiation_factor:.2f}x")
        
        st.markdown("---")
        st.markdown("### 📊 Negotiation Insights")
        
        if negotiation_factor > 1.2:
            st.success("🎯 **You're undervaluing your company!**")
            st.markdown(f"💡 Sharks might offer **MORE** than you're asking. Consider:")
            st.markdown(f"- Initial ask: **₹{valuation_pred * 1.2:,.0f} Lakh** (₹{valuation_pred_cr * 1.2:.2f} Cr)")
            st.markdown(f"- Leave room to negotiate down to ₹{valuation_pred_cr:.2f} Cr")
            
        elif negotiation_factor < 0.8:
            st.warning("📉 **Your valuation seems high**")
            st.markdown(f"💡 Expect sharks to negotiate **DOWN** significantly:")
            st.markdown(f"- They'll likely counter at **₹{valuation_pred_cr:.2f} Cr**")
            st.markdown(f"- That's {abs((1-negotiation_factor)*100):.1f}% below your ask")
            st.markdown("**Strategy:** Either justify your valuation with strong numbers, or be prepared to compromise")
            
        else:
            st.info("✅ **Your valuation is realistic!**")
            st.markdown("💡 Minor negotiations expected:")
            min_range = valuation_pred * 0.90
            max_range = valuation_pred * 1.10
            st.markdown(f"- Expect final deal between **₹{min_range/100:.2f} - ₹{max_range/100:.2f} Cr**")
        
        st.markdown("---")
        st.markdown("### 🎯 Optimal Strategy")
        
        optimal_min = valuation_pred * 0.85
        optimal_max = valuation_pred * 1.15
        
        st.markdown(f"**Negotiation Window:** ₹{optimal_min/100:.2f} - ₹{optimal_max/100:.2f} Cr")
        st.markdown(f"- **Start at:** ₹{optimal_max/100:.2f} Cr (leave room)")
        st.markdown(f"- **Target:** ₹{valuation_pred_cr:.2f} Cr (model prediction)")
        st.markdown(f"- **Bottom line:** ₹{optimal_min/100:.2f} Cr (don't go below)")
        
    else:
        st.warning("⚠️ **Valuation analysis not applicable**")
        st.markdown("Since the deal is predicted to be **rejected**, negotiation won't happen.")
        
        st.markdown("---")
        st.markdown("### 💡 To Make Your Deal Attractive:")
        
        target_val = valuation_pred if valuation_pred > 0 else valuation * 0.6
        target_val_cr = target_val / 100
        
        st.markdown(f"**Model suggests a valuation around ₹{target_val_cr:.2f} Cr**")
        st.markdown(f"- Current: ₹{valuation_cr:.2f} Cr")
        st.markdown(f"- Target: ₹{target_val_cr:.2f} Cr")
        st.markdown(f"- **Adjust equity to:** {(ask_amount / (target_val/100)):.1f}% for better chances")

# --- TAB 3: SHARKS ---
with tab3:
    st.subheader("🦈 Shark Investment Predictions")
    
    if deal_pred == 1:
        st.markdown("💡 **Based on your pitch profile, industry, and historical patterns:**")
        
        # Create prediction table
        shark_data = []
        for i, shark_name in enumerate(sharks):
            prediction_text = "✅ Likely to Invest" if final_shark_pred[i] == 1 else "❌ Likely to Pass"
            confidence = shark_probs_model[i] * 100 if len(shark_probs_model) > i else 0
            
            shark_data.append({
                'Shark': shark_name,
                'Prediction': prediction_text,
                'Confidence': f"{confidence:.1f}%" if confidence > 0 else "N/A"
            })
        
        df_sharks = pd.DataFrame(shark_data)
        
        # Highlight investors
        likely_investors = [sharks[i] for i, pred in enumerate(final_shark_pred) if pred == 1]
        
        if likely_investors:
            st.success(f"**🎯 Most Likely Investors:** {', '.join(likely_investors)}")
            st.markdown(f"🟢 *{len(likely_investors)} out of {num_sharks_present} sharks present are predicted to invest*")
            
            if len(likely_investors) > 1:
                st.info("💡 **Multiple sharks interested** - This could lead to a bidding war! Use it to your advantage.")
            
        else:
            st.warning("**⚠️ Uncertain Shark Interest**")
            st.markdown("While a deal is likely, the model is uncertain about specific sharks. This could mean:")
            st.markdown("- 🤝 Multiple sharks might make a joint offer")
            st.markdown("- 🎯 Your pitch appeals broadly but not strongly to anyone")
            st.markdown("- 💡 Consider emphasizing industry-specific strengths")
        
        # Display table
        st.dataframe(df_sharks, use_container_width=True, hide_index=True)
        
        st.markdown("---")
        st.markdown("### 💼 Shark Industry Preferences")
        
        # Shark expertise (customizable based on actual data)
        shark_expertise = {
            'Namita': {'industries': 'Pharma, Healthcare, Beauty & Wellness', 'style': 'Data-driven, focuses on margins'},
            'Vineeta': {'industries': 'Fashion, Lifestyle, D2C Brands', 'style': 'Brand-focused, marketing expertise'},
            'Anupam': {'industries': 'Entertainment, Education, Services', 'style': 'Strategic networks, mentor approach'},
            'Aman': {'industries': 'E-commerce, Logistics, Tech Platforms', 'style': 'Aggressive growth, scalability'},
            'Peyush': {'industries': 'Tech, Electronics, Innovation', 'style': 'Product-focused, technical depth'},
            'Ritesh': {'industries': 'Hospitality, Food & Beverage, Consumer', 'style': 'Customer experience, brand building'},
            'Amit': {'industries': 'Fashion, Retail, Textiles', 'style': 'Manufacturing & distribution networks'}
        }
        
        st.markdown(f"**Your Industry:** {industry}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**🎯 Best Matches for Your Industry:**")
            # You can enhance this with actual data patterns
            for shark_name in likely_investors[:3]:
                if shark_name in shark_expertise:
                    info = shark_expertise[shark_name]
                    st.markdown(f"**{shark_name}:** {info['industries']}")
        
        with col2:
            st.markdown("**💡 Pitch Strategy:**")
            for shark_name in likely_investors[:3]:
                if shark_name in shark_expertise:
                    info = shark_expertise[shark_name]
                    st.markdown(f"**{shark_name}:** {info['style']}")
        
    else:
        st.error("🚫 **No Shark Investment Predicted**")
        
        st.markdown("---")
        st.markdown("### ❌ Why No Sharks Will Invest:")
        st.markdown("The fundamental issue is that **the deal itself is predicted to be rejected**.")
        st.markdown("")
        st.markdown("Without a deal acceptance, no sharks will make offers.")
        
        st.markdown("---")
        st.markdown("### 💡 Steps to Attract Sharks:")
        
        st.markdown("**1. Fix the Deal Structure First** 🎯")
        st.markdown("   - Go to the 'Deal Analysis' tab for specific suggestions")
        st.markdown("   - Improve your deal acceptance probability above 60%")
        
        st.markdown("**2. Optimize Your Valuation** 💰")
        st.markdown(f"   - Current: ₹{valuation_cr:.2f} Cr (may be too high)")
        st.markdown(f"   - Consider: ₹{valuation_pred_cr:.2f} Cr (model suggestion)")
        
        st.markdown("**3. Industry-Specific Appeal** 🏭")
        st.markdown(f"   - Your industry: **{industry}**")
        st.markdown("   - Research which sharks invest in this sector")
        st.markdown("   - Tailor your pitch to their expertise")
        
        st.markdown("**4. Show Traction** 📈")
        st.markdown("   - Strong revenue numbers")
        st.markdown("   - Customer testimonials")
        st.markdown("   - Clear growth trajectory")
        
        st.markdown("---")
        st.info("**💡 Remember:** Sharks look for realistic valuations, strong teams, and clear ROI potential!")

# ==========================
# FOOTER
# ==========================
st.markdown("---")
st.markdown("### 📚 About This Tool")

with st.expander("ℹ️ How This Works"):
    st.markdown("""
    This AI predictor uses **Machine Learning models** trained on historical Shark Tank India data to predict:
    
    1. **Deal Acceptance** - Classification model (Logistic Regression, Random Forest, XGBoost)
    2. **Final Valuation** - Regression model (trained on accepted deals only)
    3. **Shark Investment** - Multi-label classification (predicts each shark independently)
    
    **Key Logic:**
    - If **NO DEAL** is predicted → **NO SHARKS** will invest (logical consistency)
    - If **DEAL** is predicted → Model uses both ML predictions and historical industry patterns
    - Shark predictions consider: industry fit, number of sharks present, and past investment patterns
    
    **Accuracy:** Models are trained on real data but predictions are estimates. Use as guidance, not guarantees!
    """)

with st.expander("⚙️ Model Details"):
    st.markdown(f"""
    **Deal Model:** Classification (Accuracy varies by model selected)
    **Valuation Model:** Regression (R² varies by model selected)
    **Shark Model:** Multi-label One-vs-Rest Classification
    
    **Features Used:** {len(feature_columns)} features including demographics, financial metrics, and derived features
    """)

st.markdown("---")
st.caption("Made with ❤️ by STMP Developers • Powered by Streamlit + Scikit-Learn • Shark Tank India AI Predictor")
st.caption("⚠️ **Disclaimer:** Predictions are based on historical data and should be used as guidance only. Actual results may vary.")