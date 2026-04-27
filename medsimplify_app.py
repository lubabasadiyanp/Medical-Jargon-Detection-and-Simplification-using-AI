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
    # Load spaCy for Linguistic Analysis
    try:
        nlp = spacy.load("en_core_web_sm")
    except:
        st.error("spaCy model not found. Check requirements.txt")
        return None, None
        
    # Load jargon dictionary (hidden from user)
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
        genai.configure(api_key=api_key.strip())
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        response = model.generate_content(
            f"Simplify this medical text for a patient. Use clear, simple language: {text}"
        )
        return response.text
    except Exception as e:
        return f"❌ AI Error: {str(e)}"

# ─── 4. MAIN UI ─────────────────────────────────────────────────────────────
st.title("🏥 MedSimplify")
st.subheader("Clinical Jargon Simplification Pipeline")

input_text = st.text_area("Enter complex medical text:", height=200, 
                          placeholder="Example: The patient presents with acute myocardial infarction...")

if st.button("Analyze & Simplify", type="primary"):
    if input_text.strip():
        with st.spinner("Processing through NLP Pipeline..."):
            
            # --- STAGE 1: AI OUTPUT (USER FACING) ---
            output = simplify_text(input_text)
            st.markdown("### ✨ Simplified Result")
            st.info(output)
            
            st.divider()

            # --- STAGE 2: TECHNICAL NLP ANALYSIS (PROJECT PROOF) ---
            with st.expander("🔬 View Pipeline Analysis (Tokenization & POS Tagging)"):
                doc = nlp(input_text)
                
                # 1. Linguistic Features Table
                nlp_data = []
                for token in doc:
                    nlp_data.append({
                        "Token": token.text,
                        "Lemma (Root)": token.lemma_,
                        "POS Tag": token.pos_,
                        "Feature": spacy.explain(token.pos_)
                    })
                
                st.write("**Linguistic Analysis Dataframe:**")
                st.dataframe(pd.DataFrame(nlp_data), use_container_width=True)

                # 2. Complex Entity Extraction
                jargon_terms = [t.text for t in doc if len(t.text) > 8 and t.pos_ in ["NOUN", "ADJ"]]
                st.write("**Identified Complex Clinical Entities:**")
                if jargon_terms:
                    st.warning(", ".join(list(set(jargon_terms))))
                else:
                    st.write("No high-complexity jargon detected.")
    else:
        st.warning("Please enter text first.")

st.markdown("---")
st.caption("NLP Project Pipeline | Powered by spaCy + Gemini 1.5 | Ground Truth: val.csv, jargon.json")
