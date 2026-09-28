"""InsuraPredict — Streamlit frontend.  Run: streamlit run app.py"""
import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from src.train import IQRCapper  # noqa: F401  (needed to unpickle the model)

ROOT = Path(__file__).parent
st.set_page_config(page_title="InsuraPredict", page_icon="🏠", layout="wide")


@st.cache_resource
def load():
    model = joblib.load(ROOT / "models" / "model.joblib")
    metrics = json.loads((ROOT / "models" / "metrics.json").read_text())
    return model, metrics


model, metrics = load()

st.title("🏠 InsuraPredict")
st.caption("Estimate the probability that a building will file an insurance claim "
           "during its coverage period.")
tab_pred, tab_perf = st.tabs(["Predict", "Model performance"])

with tab_pred:
    c1, c2 = st.columns(2)
    with c1:
        btype = st.selectbox("Building type",
                             ["Fire-resistive", "Non-combustible", "Ordinary", "Wood-framed"])
        dim = st.number_input("Building dimension (m²)", 1, 25000, 1000, step=50)
        windows = st.slider("Number of windows (10 = 10 or more)", 0, 10, 3)
        year = st.selectbox("Year of observation", [2012, 2013, 2014, 2015, 2016], index=1)
        period = st.slider("Insured period (fraction of year)", 0.0, 1.0, 1.0, 0.05)
    with c2:
        residential = st.radio("Residential?", ["Yes", "No"], horizontal=True)
        painted = st.radio("Painted?", ["Yes", "No"], horizontal=True)
        fenced = st.radio("Fenced?", ["Yes", "No"], horizontal=True)
        garden = st.radio("Garden?", ["Yes", "No"], horizontal=True)
        settlement = st.radio("Settlement", ["Urban", "Rural"], horizontal=True)

    # The dataset's original codes: V = yes, N = no; Garden uses V = yes, O = no; U/R = urban/rural
    row = pd.DataFrame([{
        "Year_Of_Observation": year,
        "Insured_Period": period,
        "Residential": int(residential == "Yes"),
        "Building_Dimension": dim,
        "Number_Of_Windows": windows,
        "Building_Painted": "V" if painted == "Yes" else "N",
        "Building_Fenced": "V" if fenced == "Yes" else "N",
        "Garden": "V" if garden == "Yes" else "O",
        "Settlement": "U" if settlement == "Urban" else "R",
        "Building_Type": btype,
    }])

    if st.button("Predict claim risk", type="primary"):
        p = float(model.predict_proba(row)[0, 1])
        base = metrics["test"]["baseline_claim_rate"]
        level = "High" if p >= 0.6 else "Medium" if p >= 0.35 else "Low"
        st.metric("Claim probability", f"{p:.0%}",
                  f"{p - base:+.0%} vs. average building ({base:.0%})", delta_color="inverse")
        st.progress(p)
        st.markdown(f"**Risk level: {level}**")

with tab_perf:
    st.subheader(f"Selected model: {metrics['best_model']}")
    t = metrics["test"]
    a, b, c = st.columns(3)
    a.metric("ROC-AUC (test)", f"{t['roc_auc']:.3f}")
    b.metric("PR-AUC (test)", f"{t['pr_auc']:.3f}",
             f"random baseline {t['baseline_claim_rate']:.2f}", delta_color="off")
    c.metric("F1 — claim class", f"{t['f1_claim']:.3f}")
    st.markdown("**5-fold cross-validation (training set)**")
    st.dataframe(pd.DataFrame(metrics["cv"]).T, use_container_width=True)
    st.info("All preprocessing and SMOTE are fit on training folds only, so these "
            "scores reflect performance on unseen buildings.")
