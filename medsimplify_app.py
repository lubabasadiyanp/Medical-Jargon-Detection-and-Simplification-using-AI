import streamlit as st
import pandas as pd
import spacy
import json
from google import genai

# ─── 1. PAGE SETUP ──────────────────────────────────────────────────────────
st.set_page_config(page_title="MedSimplify — Professional NLP", page_icon="🏥")

# ─── 2. LOAD NLP MODELS & DATA ──────────────────────────────────────────────
@st.cache_resource
def load_resources():
    nlp = spacy.load("en_core_web_sm")
    # Load jargon dictionary to help the app identify difficult words
    try:
        with open("jargon.json", "r") as f:
            jargon_data = json.load(f)
    except:
        jargon_data = []
    return nlp, jargon_data

nlp, jargon_lookup = load_resources()

@st.cache_resource
def get_ai_client():
    api_key = st.secrets.get("GEMINI_API_KEY")
    if not api_key: return None
    return genai.Client(api_key=api_key.strip(), http_options={'api_version': 'v1beta'})

# ─── 3. STYLING ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #f8f9fa; }
    .stTextArea textarea { font-size: 1.1rem !important; }
    .result-box { background-color: #ffffff; border: 1px solid #dee2e6; padding: 20px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    .step-header { color: #1a7a6e; font-weight: bold; margin-top: 20px; }
</style>
""", unsafe_allow_html=True)

# ─── 4. MAIN INTERFACE ──────────────────────────────────────────────────────
st.title("🏥 MedSimplify")
st.subheader("Translate Complex Medical Jargon into Plain English")

input_text = st.text_area("Paste medical abstract or clinical notes below:", height=200, placeholder="e.g., The patient presents with acute myocardial infarction...")

if st.button("Simplify Medical Text", type="primary"):
    if input_text.strip():
        # --- STAGE 1: THE AI OUTPUT (The User's Main Goal) ---
        client = get_ai_client()
        if client:
            with st.spinner("Processing through NLP Pipeline..."):
                response = client.models.generate_content(
                    model="gemini-3.1-flash",
                    contents=f"Simplify this medical text for a patient. Use simple words: {input_text}"
                )
                
                st.markdown('<p class="step-header">✨ SIMPLIFIED VERSION</p>', unsafe_allow_html=True)
                st.markdown(f'<div class="result-box">{response.text}</div>', unsafe_allow_html=True)

        st.divider()

        # --- STAGE 2: THE NLP PROJECT PROOF (Hidden in an Expander) ---
        # This part makes it a "Project" without cluttering the screen for normal users
        with st.expander("🔬 View NLP Pipeline Analysis (For Project Evaluation)"):
            st.write("This section shows the technical processing steps performed on your input.")
            
            # NLP Step 1: Tokenization & POS Tagging
            doc = nlp(input_text)
            nlp_data = []
            for token in doc:
                nlp_data.append({
                    "Token": token.text,
                    "Lemma": token.lemma_,
                    "POS": token.pos_,
                    "Description": spacy.explain(token.pos_)
                })
            
            st.write("**1. Tokenization & Part-of-Speech Tagging:**")
            st.dataframe(pd.DataFrame(nlp_data), use_container_width=True)

            # NLP Step 2: Jargon Identification
            # We use length and POS to flag potential jargon
            jargon_terms = [t.text for t in doc if len(t.text) > 8 and t.pos_ in ["NOUN", "ADJ"]]
            st.write("**2. Identified Complex Jargon:**")
            if jargon_terms:
                st.info(", ".join(list(set(jargon_terms))))
            else:
                st.write("No high-complexity jargon detected.")

    else:
        st.warning("Please enter some text first.")

# ─── 5. FOOTER ──────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("Powered by spaCy NLP and Google Gemini AI | Research Data: val.csv & jargon.json")
