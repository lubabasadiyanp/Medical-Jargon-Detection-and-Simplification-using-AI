import streamlit as st
import pandas as pd
import spacy
import json
import google.generativeai as genai

# ─── 1. PAGE SETUP ──────────────────────────────────────────────────────────
st.set_page_config(page_title="MedSimplify Pro", page_icon="🏥")

# ─── 2. LOAD RESOURCES ──────────────────────────────────────────────────────
@st.cache_resource
def load_nlp():
    return spacy.load("en_core_web_sm")

nlp = load_nlp()

# ─── 3. FAIL-PROOF AI LOGIC ────────────────────────────────────────────────
def simplify_text_stable(text):
    api_key = st.secrets.get("GEMINI_API_KEY")
    if not api_key:
        return "❌ API Key Missing: Add GEMINI_API_KEY to Streamlit Secrets."
    
    try:
        genai.configure(api_key=api_key.strip())
        
        # We use Gemini 2.0 Flash - the current most stable and modern version
        model = genai.GenerativeModel('gemini-2.0-flash')
        
        response = model.generate_content(
            f"Simplify this medical text for a patient. Use simple words: {text}"
        )
        return response.text
    except Exception as e:
        # If 2.0 fails (rare), we try the experimental backup
        try:
            model_backup = genai.GenerativeModel('gemini-2.0-flash-lite-preview-02-05')
            response = model_backup.generate_content(text)
            return response.text
        except:
            return f"❌ AI Pipeline Error: The model name changed or is unavailable. Error: {str(e)}"

# ─── 4. MAIN INTERFACE ──────────────────────────────────────────────────────
st.title("🏥 MedSimplify")
st.subheader("NLP-Driven Clinical Text Simplification")

input_text = st.text_area("Paste medical text here:", height=200, 
                          placeholder="Example: Acute myocardial infarction involving the left anterior descending artery...")

if st.button("Run NLP Pipeline", type="primary"):
    if input_text.strip():
        # STAGE 1: spaCy Processing
        doc = nlp(input_text)
        
        # STAGE 2: AI Processing
        with st.spinner("Processing through Neural Pipeline..."):
            ai_output = simplify_text_stable(input_text)
            
        st.markdown("---")
        st.markdown("### ✨ Simplified Patient Version")
        st.success(ai_output)
        
        st.divider()

        # STAGE 3: Technical Evidence
        with st.expander("🔬 View Pipeline Data (Tokenization, POS, Lemmatization)"):
            nlp_data = [{"Token": t.text, "Lemma": t.lemma_, "POS Tag": t.pos_} for t in doc]
            st.dataframe(pd.DataFrame(nlp_data), use_container_width=True)

            jargon_found = [t.text for t in doc if len(t.text) > 8 and t.pos_ in ["NOUN", "ADJ"]]
            if jargon_found:
                st.info(f"Jargon Detected: {', '.join(list(set(jargon_found)))}")
    else:
        st.warning("Please enter text first.")

st.markdown("---")
st.caption("NLP Project | Pipeline: spaCy + Gemini 2.0 | April 2026 Update")
