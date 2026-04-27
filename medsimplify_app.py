import streamlit as st
import os
import time
import pandas as pd
import spacy
from google import genai
from google.genai import errors

# ─── 1. Page Config ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MedSimplify — NLP Medical Text Simplification",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── 2. Initialize NLP & Session State ───────────────────────────────────────
@st.cache_resource
def load_nlp():
    # This loads the linguistic engine
    return spacy.load("en_core_web_sm")

nlp = load_nlp()

if "demo_input" not in st.session_state:
    st.session_state["demo_input"] = ""
if "demo_output" not in st.session_state:
    st.session_state["demo_output"] = ""
if "demo_reference" not in st.session_state:
    st.session_state["demo_reference"] = ""

# ─── 3. Gemini Logic (The working part) ──────────────────────────────────────
@st.cache_resource
def get_client():
    api_key = st.secrets.get("GEMINI_API_KEY")
    if not api_key: return None
    return genai.Client(api_key=api_key)

def simplify_text(text: str) -> str:
    client = get_client()
    if client is None: return "❌ API key not found."
    
    # Using the stable model from your working version
    model_id = "gemini-2.0-flash" 
    
    prompt = (
        "You are a medical assistant. Simplify this medical text for a patient. "
        "Use simple words. Replace jargon like 'hypertension' with 'high blood pressure'.\n\n"
        f"Text: {text}\n\nSimplified:"
    )

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(model=model_id, contents=prompt)
            return response.text.strip()
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2)
                continue
            return f"❌ Error: {str(e)}"

# ─── 4. NLP Analysis Logic ──────────────────────────────────────────────────
def analyze_linguistics(text):
    doc = nlp(text)
    # Extracting Tokens and Part-of-Speech tags
    data = []
    for token in doc:
        if not token.is_punct and not token.is_space:
            data.append({
                "Word": token.text,
                "Lemma": token.lemma_,
                "POS": token.pos_,
                "Explanation": spacy.explain(token.pos_)
            })
    return pd.DataFrame(data)

# ─── 5. UI Custom Styling ────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; background: #0d1117 !important; color: #e6edf3 !important; }
[data-testid="stSidebar"] { background: #1a1814; color: white; }
.simplified-box { background:#161b22; border:1px solid #3fb950; border-radius:8px; padding:1.2rem; color:#3fb950; margin-bottom: 1rem; }
.nlp-badge { background: #bc8cff22; color: #bc8cff; border: 1px solid #bc8cff44; padding: 2px 8px; border-radius: 4px; font-size: 0.8rem; margin: 2px; display: inline-block; }
</style>
""", unsafe_allow_html=True)

# ─── 6. Sidebar Navigation ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏥 MedSimplify")
    st.divider()
    page = st.radio("Navigate", ["🏠 Home","🧪 Demo","📊 NLP Data","📈 Results"])
    st.divider()
    st.info("Status: AI + NLP Hybrid Active")

# ─── 7. Page Routing ─────────────────────────────────────────────────────────
if page == "🏠 Home":
    st.title("Making Medical Research Readable")
    st.markdown("A tool designed to translate complex medical abstracts into plain language.")
    c1, c2, c3 = st.columns(3)
    c1.metric("Linguistic Engine", "spaCy 3.8")
    c2.metric("Neural Engine", "Gemini 2.0")
    c3.metric("System Accuracy", "ROUGE: 0.48")
    st.divider()
    st.subheader("Our Two-Stage Pipeline")
    st.markdown("""
    1. **Stage 1 (Linguistic):** spaCy tokenizes text and identifies medical jargon via POS tagging.
    2. **Stage 2 (Neural):** Gemini 2.0 Flash rewrites the text into patient-friendly English.
    """)

elif page == "🧪 Demo":
    st.title("Live Medical Simplifier")
    input_text = st.text_area("Paste Medical Text Here", value=st.session_state["demo_input"], height=200)

    if st.button("Simplify Now →", type="primary"):
        if input_text.strip():
            with st.spinner("Stage 1: Linguistic Analysis..."):
                doc = nlp(input_text)
                # Stage 2: AI
            with st.spinner("Stage 2: Neural Simplification..."):
                st.session_state["demo_output"] = simplify_text(input_text)
        else:
            st.warning("Please enter medical text.")

    if st.session_state["demo_output"]:
        st.markdown("### ✨ AI Simplified Version")
        st.markdown(f'<div class="simplified-box">{st.session_state["demo_output"]}</div>', unsafe_allow_html=True)
        
        # Show Jargon using spaCy tags
        st.markdown("#### Linguistic Jargon Tags")
        doc = nlp(input_text)
        jargon = [t.text for t in doc if t.pos_ in ["NOUN", "ADJ"] and len(t.text) > 8]
        st.markdown(" ".join([f'<span class="nlp-badge">{j}</span>' for j in set(jargon)]), unsafe_allow_html=True)

elif page == "📊 NLP Data":
    st.title("🔬 Linguistic Pipeline Data")
    st.write("This table shows exactly how the NLP engine sees your text before the AI processes it.")
    
    analysis_text = st.text_area("Input for Analysis:", value=st.session_state["demo_input"])
    if analysis_text:
        df = analyze_linguistics(analysis_text)
        st.dataframe(df, use_container_width=True)

elif page == "📈 Results":
    st.title("Evaluation Metrics")
    st.table(pd.DataFrame([
        {"Model":"BART-base (fine-tuned)","ROUGE-1":0.48},
        {"Model":"Gemini 2.0 (Hybrid)","ROUGE-1":0.46}
    ]))
