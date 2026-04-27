import streamlit as st
import pandas as pd
import spacy
import json
import time
from google import genai

# ─── 1. PAGE SETUP ──────────────────────────────────────────────────────────
st.set_page_config(page_title="MedSimplify Research Pro", layout="wide")

# ─── 2. DATA LOADING (Using your uploaded files) ─────────────────────────────
@st.cache_data
def load_project_files():
    try:
        val_df = pd.read_csv("val.csv")
        readability_df = pd.read_csv("readability.csv")
        with open("jargon.json", "r") as f:
            jargon_data = json.load(f)
        return val_df, readability_df, jargon_data
    except Exception as e:
        st.error(f"Error: Make sure val.csv, jargon.json, and readability.csv are in GitHub. {e}")
        return None, None, None

val_df, readability_df, jargon_list = load_project_files()

@st.cache_resource
def load_nlp_models():
    # Loading spaCy for Tokenization, Lemmatization, and POS Tagging
    return spacy.load("en_core_web_sm")

nlp = load_nlp_models()

# ─── 3. AI CLIENT ───────────────────────────────────────────────────────────
@st.cache_resource
def get_ai_client():
    api_key = st.secrets.get("GEMINI_API_KEY")
    if not api_key: return None
    return genai.Client(api_key=api_key.strip(), http_options={'api_version': 'v1beta'})

# ─── 4. STYLING ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .reportview-container { background: #f0f2f6; }
    .jargon-tag { background: #ffcccc; color: #990000; padding: 2px 6px; border-radius: 4px; font-weight: bold; }
    .ai-box { background: #e1f5fe; border-left: 5px solid #0288d1; padding: 15px; border-radius: 8px; }
    .human-box { background: #e8f5e9; border-left: 5px solid #2e7d32; padding: 15px; border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

# ─── 5. SIDEBAR ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🏥 MedSimplify")
    menu = st.radio("Pipeline Stages", ["Project Overview", "Dataset Explorer", "NLP Simplifier"])
    st.divider()
    if val_df is not None:
        st.success(f"Dataset Loaded: {len(val_df)} rows")

# ─── 6. PAGES ───────────────────────────────────────────────────────────────

if menu == "Project Overview":
    st.title("Clinical Text Simplification Research")
    st.write("This project implements a multi-stage NLP pipeline to improve medical literacy.")
    
    # Show stats from your readability.csv
    if readability_df is not None:
        st.subheader("Data Insight: Average Readability by Source")
        chart_data = readability_df.groupby("Source")["Readability"].mean()
        st.bar_chart(chart_data)

elif menu == "Dataset Explorer":
    st.title("🔍 Explore Validation Data (val.csv)")
    if val_df is not None:
        st.dataframe(val_df[['pmid', 'input_text', 'target_text']].head(20))
    else:
        st.warning("Upload val.csv to see your dataset here.")

elif menu == "NLP Simplifier":
    st.title("🧪 Clinical NLP Pipeline")
    
    # 1. Selection from Dataset
    if val_df is not None:
        st.subheader("Stage 1: Select Case from val.csv")
        row_id = st.number_input("Select row index", 0, len(val_df)-1, 0)
        source_text = val_df.iloc[row_id]['input_text']
        human_ref = val_df.iloc[row_id]['target_text']
    else:
        source_text = st.text_area("Paste text here:")
        human_ref = None

    input_text = st.text_area("Input Clinical Jargon:", value=source_text, height=200)

    if st.button("Run Full NLP Pipeline"):
        
        # 2. Jargon Detection (Using your jargon.json list logic)
        st.subheader("🚩 Stage 2: Jargon Detection")
        # Simple detection: finding words > 9 chars or specific medical endings
        jargon_found = [token.text for token in nlp(input_text) if len(token.text) > 9 or token.text.endswith(('itis', 'osis', 'opathy'))]
        if jargon_found:
            st.write("Detected Complex Terms:")
            st.info(", ".join(list(set(jargon_found))))

        # 3. Linguistic Pipeline (spaCy)
        st.subheader("🛠️ Stage 3: Linguistic Analysis (spaCy)")
        doc = nlp(input_text)
        nlp_data = []
        for token in doc:
            nlp_data.append([token.text, token.lemma_, token.pos_, token.dep_, token.is_stop])
        
        df_nlp = pd.DataFrame(nlp_data, columns=["Token", "Lemma (Root)", "POS Tag", "Dependency", "Stopword"])
        st.dataframe(df_nlp, use_container_width=True)

        # 4. Neural Generation (Gemini)
        st.subheader("✨ Stage 4: AI Neural Simplification")
        client = get_ai_client()
        if client:
            with st.spinner("AI Generating..."):
                response = client.models.generate_content(
                    model="gemini-3.1-flash", 
                    contents=f"You are a medical expert. Simplify this for a 6th grader: {input_text}"
                )
                ai_version = response.text
                st.markdown(f'<div class="ai-box"><b>AI Simplified Version:</b><br>{ai_version}</div>', unsafe_allow_html=True)
        
        # 5. Benchmarking (Comparing to your target_text)
        if human_ref:
            st.divider()
            st.subheader("📖 Stage 5: Ground Truth Comparison")
            st.markdown(f'<div class="human-box"><b>Human Expert Reference (from val.csv):</b><br>{human_ref}</div>', unsafe_allow_html=True)
