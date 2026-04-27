import streamlit as st
import pandas as pd
import spacy
import json
from google import genai
from google.genai import errors

# ─── 1. PAGE SETUP ──────────────────────────────────────────────────────────
st.set_page_config(page_title="MedSimplify — Professional NLP", page_icon="🏥")

# ─── 2. LOAD NLP MODELS & DATA ──────────────────────────────────────────────
@st.cache_resource
def load_resources():
    # This handles the Tokenization, POS Tagging, and Lemmatization
    nlp = spacy.load("en_core_web_sm")
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
    # Using v1beta version for better compatibility with free tier keys
    return genai.Client(api_key=api_key.strip(), http_options={'api_version': 'v1beta'})

# ─── 3. THE SIMPLIFICATION FUNCTION ──────────────────────────────────────────
def simplify_text(text):
    client = get_ai_client()
    if not client:
        return "❌ API Key Missing: Please add GEMINI_API_KEY to Streamlit Secrets."
    
    try:
        response = client.models.generate_content(
            model="gemini-3.1-flash",
            contents=f"Simplify this medical text for a patient. Use simple words: {text}"
        )
        return response.text
    except Exception as e:
        # This catches the ClientError and explains it simply
        if "403" in str(e):
            return "❌ API Error: Permission Denied. Your API key might be restricted or inactive."
        elif "429" in str(e):
            return "⚠️ Quota full: Please wait 60 seconds and try again."
        else:
            return f"❌ AI Error: {str(e)}"

# ─── 4. STYLING ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .stTextArea textarea { font-size: 1.1rem !important; }
    .result-box { background-color: #ffffff; border: 1px solid #dee2e6; padding: 20px; border-radius: 10px; color: #1a5a50; }
    .step-header { color: #1a7a6e; font-weight: bold; margin-top: 20px; }
</style>
""", unsafe_allow_html=True)

# ─── 5. MAIN INTERFACE ──────────────────────────────────────────────────────
st.title("🏥 MedSimplify")
st.subheader("Professional NLP Medical Simplifier")

input_text = st.text_area("Paste medical text below:", height=200, 
                          placeholder="Example: The patient exhibits acute myocardial infarction...")

if st.button("Simplify Medical Text", type="primary"):
    if input_text.strip():
        # RUN THE PIPELINE
        with st.spinner("Running NLP Pipeline..."):
            
            # --- STAGE 1: AI OUTPUT ---
            output = simplify_text(input_text)
            st.markdown('<p class="step-header">✨ SIMPLIFIED VERSION</p>', unsafe_allow_html=True)
            st.markdown(f'<div class="result-box">{output}</div>', unsafe_allow_html=True)
            
            st.divider()

            # --- STAGE 2: NLP PROJECT PROOF (The Pipeline) ---
            with st.expander("🔬 View Technical NLP Analysis (Pipeline Stages)"):
                doc = nlp(input_text)
                
                # 1. Tokenization & POS Tagging Table
                nlp_data = []
                for token in doc:
                    nlp_data.append({
                        "Token": token.text,
                        "Lemma (Root)": token.lemma_,
                        "POS Tag": token.pos_,
                        "Description": spacy.explain(token.pos_)
                    })
                
                st.write("**Stage 1-3: Tokenization, Lemmatization, & POS Tagging**")
                st.dataframe(pd.DataFrame(nlp_data), use_container_width=True)

                # 2. Jargon Detection
                jargon_terms = [t.text for t in doc if len(t.text) > 8 and t.pos_ in ["NOUN", "ADJ"]]
                st.write("**Stage 4: Jargon Entity Identification**")
                if jargon_terms:
                    st.info(", ".join(list(set(jargon_terms))))
                else:
                    st.write("No complex jargon detected.")
    else:
        st.warning("Please enter text first.")

st.markdown("---")
st.caption("NLP Project Pipeline: spaCy + Gemini 3.1 | Data: val.csv, jargon.json")
