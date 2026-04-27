import streamlit as st
import pandas as pd
import spacy
from google import genai
import time

# ─── 1. PAGE CONFIG ───
st.set_page_config(page_title="MedSimplify", layout="wide")

# ─── 2. LOAD DATASETS & MODELS ───
@st.cache_data
def load_project_data():
    # This pulls from the files you showed in your GitHub screenshot
    try:
        train_df = pd.read_csv("train.csv")
        return train_df
    except:
        return None

@st.cache_resource
def load_nlp(): return spacy.load("en_core_web_sm")

@st.cache_resource
def get_client():
    key = st.secrets.get("GEMINI_API_KEY")
    return genai.Client(api_key=key.strip()) if key else None

train_data = load_project_data()
nlp, client = load_nlp(), get_client()

# ─── 3. THE SMART SIMPLIFIER ───
def simplify_with_fallback(text):
    # Try Stage 1: Gemini AI
    if client:
        try:
            # Using 1.5 Flash as it has slightly better free limits than 2.0 right now
            res = client.models.generate_content(
                model="gemini-1.5-flash", 
                contents=f"Simplify this medical text: {text}"
            )
            return res.text.strip(), "Neural AI (Gemini)"
        except Exception:
            pass # Move to Stage 2 if AI fails

    # Stage 2: Data-Driven Fallback (Using your CSV)
    if train_data is not None:
        # We look for the most similar example in your dataset
        # For a simple version, we'll just check if keywords match
        for index, row in train_data.head(100).iterrows():
            # If a word from the input is in the training data
            if any(word in row['source_text'].lower() for word in text.lower().split()[:3]):
                return row['simplified_text'], "Dataset Search (Local CSV)"
    
    return text, "Linguistic Pass-through (No Match Found)"

# ─── 4. UI STYLE ───
st.markdown("""
<style>
    body { background: #0d1117; color: #e6edf3; }
    .result-card { background: #161b22; border: 1px solid #30363d; padding: 20px; border-radius: 12px; }
    .source-tag { font-size: 0.8rem; color: #8b949e; margin-top: 10px; }
</style>
""", unsafe_allow_html=True)

# ─── 5. INTERFACE ───
st.title("🏥 MedSimplify AI")

with st.sidebar:
    page = st.radio("Navigation", ["🧪 Demo", "📊 View Dataset"])

if page == "🧪 Demo":
    input_text = st.text_area("Enter Medical Text:", height=150)
    if st.button("Simplify", type="primary"):
        with st.spinner("Pipeline Running..."):
            output, source = simplify_with_fallback(input_text)
            st.markdown(f'<div class="result-card">{output}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="source-tag">Engine: {source}</div>', unsafe_allow_html=True)

elif page == "📊 View Dataset":
    st.header("Project Training Data")
    if train_data is not None:
        st.dataframe(train_data.head(50))
    else:
        st.error("CSV files not found. Ensure train.csv is in your GitHub main folder.")
