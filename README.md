## Project Overview

This project predicts whether a post-operative patient is:
- **0 – safe for discharge**
- **1 – requires further monitoring / at risk of complication**

using simple clinical parameters like:
- Core temperature level (L-CORE)
- Surface temperature (L-SURF)
- Oxygen saturation (L-O2)
- Blood pressure level (L-BP)
- Stability indicators (SURF-STBL, CORE-STBL, BP-STBL)
- Patient comfort score (COMFORT)

Model:
- Logistic Regression with One-Hot Encoding
- Class imbalance handled via `class_weight="balanced"`
- Tuned probability threshold (0.3) to increase recall for high-risk patients

App:
- Built using **Streamlit**
- Takes user input via dropdowns and slider
- Outputs:
  - Estimated complication risk percentage
  - Final model suggestion: **Safe** / **Needs monitoring**


pip install -r requirements.txt
python -m src.train_model
streamlit run app/app.py


# 🩺 Post-Surgery Complication Prediction

This project predicts whether a post-operative patient is at **low risk (safe for discharge)** or **high risk (requires monitoring)** using clinical parameters.

---

### 🚀 Live Demo

Try the deployed app here:

👉 https://post-surgery-complication-prediction-y8vsdffekr3yvpzxzx23bt.streamlit.app/

---

### 🧠 Model Info

- Logistic Regression classifier
- Class imbalance addressed with class weights
- Threshold optimized to increase recall for high-risk patients

---

### 📂 Project Structure

### 🛠 Tech Stack

- Python  
- Pandas, NumPy  
- Scikit-Learn  
- Streamlit  

---

⚠️ *This project is for research & learning purposes — not for real clinical use.*