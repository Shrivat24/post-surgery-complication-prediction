import streamlit as st
import joblib
from pathlib import Path
import pandas as pd

# -----------------------
# Load model artifact
# -----------------------
@st.cache_resource
def load_model():
    base_dir = Path(__file__).resolve().parent.parent
    model_path = base_dir / "models" / "post_op_lr_pipeline.pkl"
    artifact = joblib.load(model_path)
    return artifact

artifact = load_model()
pipeline = artifact["pipeline"]
THRESHOLD = artifact["threshold"]

# -----------------------
# Streamlit UI
# -----------------------
st.set_page_config(page_title="Post-Surgery Complication Risk", page_icon="🩺", layout="centered")

st.title("🩺 Post-Surgery Complication Risk Predictor")
st.write(
    """
    This tool estimates whether a post-operative patient may be **safe for discharge (0)**  
    or **needs closer monitoring (1)** based on simple clinical features.
    
    > ⚠️ This is a demo ML model trained on a small dataset.  
    It is **not** a substitute for real medical judgment.
    """
)

st.markdown("---")

st.header("📋 Enter Post-Operative Patient Details")

col1, col2 = st.columns(2)

with col1:
    l_core = st.selectbox(
        "Core temperature level (L-CORE)",
        options=["low", "mid", "high"],
        help="Internal body temperature category"
    )

    l_surf = st.selectbox(
        "Surface temperature level (L-SURF)",
        options=["low", "mid", "high"],
        help="Skin surface temperature"
    )

    l_o2 = st.selectbox(
        "Oxygen saturation (L-O2)",
        options=["excellent", "good", "poor"],
        help="Pulse oximeter oxygen saturation category"
    )

    l_bp = st.selectbox(
        "Blood pressure level (L-BP)",
        options=["low", "mid", "high"],
        help="Overall blood pressure status"
    )

with col2:
    surf_stbl = st.selectbox(
        "Surface stability (SURF-STBL)",
        options=["stable", "mod-stable", "unstable"],
        help="Stability of skin condition / perfusion"
    )

    core_stbl = st.selectbox(
        "Core stability (CORE-STBL)",
        options=["stable", "mod-stable", "unstable"],
        help="Stability of internal status"
    )

    bp_stbl = st.selectbox(
        "Blood pressure stability (BP-STBL)",
        options=["stable", "mod-stable", "unstable"],
        help="How stable blood pressure readings are"
    )

    comfort = st.slider(
        "Patient comfort score (COMFORT)",
        min_value=0,
        max_value=20,
        value=10,
        help="Subjective comfort level (0 = very uncomfortable, 20 = very comfortable)"
    )

st.markdown("---")

if st.button("🔍 Predict Risk"):
    # Build input DataFrame with EXACT column names used in training
    input_data = {
        "L-CORE": [l_core],
        "L-SURF": [l_surf],
        "L-O2": [l_o2],
        "L-BP": [l_bp],
        "SURF-STBL": [surf_stbl],
        "CORE-STBL": [core_stbl],
        "BP-STBL": [bp_stbl],
        "COMFORT": [comfort],
    }

    input_df = pd.DataFrame(input_data)

    # Predict probability of class 1 (needs monitoring)
    proba = pipeline.predict_proba(input_df)[0, 1]
    predicted_label = int(proba >= THRESHOLD)

    st.subheader("📊 Prediction Result")

    risk_percent = proba * 100
    st.write(f"**Estimated complication / monitoring risk:** `{risk_percent:.1f}%`")
    st.write(f"Model decision threshold: `{THRESHOLD}`")

    if predicted_label == 1:
        st.error(
            "⚠️ **Model Suggestion: Needs monitoring / possible complication.**\n\n"
            "The model considers this patient at higher risk. In a real setting, "
            "this would suggest closer observation, additional tests, or ICU consideration."
        )
    else:
        st.success(
            "✅ **Model Suggestion: Likely safe for discharge / low risk.**\n\n"
            "The model estimates a lower risk of complication. However, final decisions must "
            "always be made by qualified clinicians."
        )

    st.caption(
        "This prediction is based on a logistic regression model trained on a small dataset. "
        "It is for educational/demo purposes only."
    )
