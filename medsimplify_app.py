import streamlit as st
import pandas as pd
import spacy
import json
import google.generativeai as genai

# ─── 1. PAGE SETUP ──────────────────────────────────────────────────────────
st.set_page_config(page_title="MedSimplify Pro", page_icon="🏥")

# ─── 2. LOAD NLP MODELS & DATA ──────────────────────────────────────────────
@st.cache_resource
def load_resources():
    # Load spaCy for Linguistic Analysis (Tokens, POS, Lemmatization)
    try:
        nlp = spacy.load("en_core_web_sm")
    except Exception as e:
        st.error(f"spaCy Model Load Error: {e}")
        return None, None
        
    # Load jargon dictionary for entity mapping
    try:
        with open("jargon.json", "r") as f:
            jargon_data = json.load(f)
    except:
        jargon_data = []
    return nlp, jargon_data

nlp, jargon_lookup = load_resources()

# ─── 3. THE SIMPLIFICATION LOGIC ───────────────────────────────────────────
def simplify_text(text):
    api_key = st.secrets.get("GEMINI_API_KEY")
    if not api_key:
        return "❌ API Key Missing: Add GEMINI_API_KEY to Streamlit Secrets."
    
    try:
        # Configuring the stable genai library
        genai.configure(api_key=api_key.strip())
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        response = model.generate_content(
            f"You are a medical translator. Simplify this for a patient: {text}"
        )
        return response.text
    except Exception as e:
        return f"❌ AI Error: {str(e)}"

# ─── 4. MAIN INTERFACE ──────────────────────────────────────────────────────
st.title("🏥 MedSimplify")
st.markdown("### Clinical Jargon Simplification Pipeline")

input_text = st.text_area("Input Medical Text:", height=180, 
                          placeholder="e.g., Patient diagnosed with acute idiopathic pulmonary fibrosis...")

if st.button("Process NLP Pipeline", type="primary"):
    if input_text.strip():
        # --- STAGE 1: LOCAL LINGUISTIC PROCESSING ---
        # We run spaCy first so we have data even if the AI fails
        doc = nlp(input_text)
        
        # --- STAGE 2: AI NEURAL GENERATION ---
        with st.spinner("Analyzing and Generating..."):
            ai_output = simplify_text(input_text)
            st.markdown("#### ✨ Simplified Patient Version")
            st.success(ai_output)
            
        st.divider()

        # --- STAGE 3: PROJECT PROOF (The NLP "Backbone") ---
        with st.expander("🔬 View Technical NLP Analysis (Stages 1-4)"):
            st.write("This section demonstrates the internal linguistic processing.")
            
            # 1. POS and Lemmatization Table
            nlp_data = []
            for token in doc:
                nlp_data.append({
                    "Token": token.text,
                    "Lemma": token.lemma_,
                    "POS Tag": token.pos_,
                    "Description": spacy.explain(token.pos_)
                })
            
            st.write("**Linguistic Analysis (Tokenization & Morphological Analysis):**")
            st.dataframe(pd.DataFrame(nlp_data), use_container_width=True)

            # 2. Heuristic Jargon Detection
            jargon_terms = [t.text for t in doc if len(t.text) > 8 and t.pos_ in ["NOUN", "ADJ"]]
            st.write("**Identified Clinical Jargon (Entity Extraction):**")
            if jargon_terms:
                st.warning(", ".join(list(set(jargon_terms))))
            else:
                st.write("No high-complexity jargon detected.")
    else:
        st.warning("Please enter text first.")

st.markdown("---")
st.caption("NLP Research Project | Pipeline: spaCy + Gemini 1.5 | Datasets: val.csv, jargon.json")
