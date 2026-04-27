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

    genai.configure(api_key=api_key.strip())

    models_to_try = [
        "gemini-1.5-flash",
        "gemini-1.5-flash-8b",
    ]

    prompt = (
        "You are a medical assistant. Simplify the following medical text so a "
        "non-medical patient can easily understand it. Use simple words, avoid "
        "jargon, and keep the meaning the same.\n\n"
        f"Medical text: {text}\n\nSimplified version:"
    )

    for model_name in models_to_try:
        try:
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            if "429" in str(e) or "quota" in str(e).lower():
                continue
            return f"❌ Error: {str(e)}"

    return "❌ Rate limit reached. Please wait a few minutes and try again."

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
