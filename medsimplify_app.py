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
def simplify_text(text):
    client = get_ai_client()
    if not client:
        return "❌ API Key Missing: Please add GEMINI_API_KEY to Streamlit Secrets."
    
    try:
        # Update: Added 'models/' prefix which is often required by the new SDK
        response = client.models.generate_content(
            model="models/gemini-1.5-flash", 
            contents=f"Simplify this medical text for a patient. Use simple words: {text}"
        )
        return response.text
    except Exception as e:
        # If models/gemini-1.5-flash fails, try the short name as a backup
        try:
            response = client.models.generate_content(
                model="gemini-1.5-flash",
                contents=f"Simplify this medical text for a patient. Use simple words: {text}"
            )
            return response.text
        except Exception as second_error:
            if "404" in str(second_error):
                return "❌ Model Error: Please ensure your API Key has access to Gemini 1.5 Flash in AI Studio."
            return f"❌ AI Error: {str(second_error)}"
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
