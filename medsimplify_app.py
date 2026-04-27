import streamlit as st
import pandas as pd
import spacy
from google import genai
import os

# ─── 1. PAGE CONFIG ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MedSimplify — AI Medical Text Simplification",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── 2. LOAD NLP & CLIENT ───────────────────────────────────────────────────
@st.cache_resource
def load_nlp():
    return spacy.load("en_core_web_sm")

@st.cache_resource
def get_gemini_client():
    api_key = st.secrets.get("GEMINI_API_KEY")
    if not api_key: return None
    return genai.Client(api_key=api_key.strip())

nlp = load_nlp()
client = get_gemini_client()

# ─── 3. FAIL-SAFE SIMPLIFY LOGIC ─────────────────────────────────────────────
def simplify_text(text: str) -> str:
    if not client: return "❌ API Key Missing in Streamlit Secrets."
    
    # Priority list to bypass 404 and Rate Limits
    models = ["gemini-2.0-flash", "gemini-2.0-flash-lite", "gemini-1.5-flash"]
    
    prompt = f"Rewrite this medical text into simple English for a patient. Return ONLY the simplified text:\n\n{text}"

    for model_name in models:
        try:
            response = client.models.generate_content(model=model_name, contents=prompt)
            return response.text.strip()
        except Exception:
            continue # Try next model
            
    # FINAL FALLBACK (If API is totally down)
    doc = nlp(text)
    fallback = " ".join([t.lemma_ if len(t.text) > 10 else t.text for t in doc])
    return f"⚠️ [NLP Pipeline Mode] {fallback} (Note: AI is currently busy; showing local linguistic analysis.)"

# ─── 4. CSS THEME (SAME AS YOUR ORIGINAL) ────────────────────────────────────
st.markdown("""
<style>
    :root { --bg: #0d1117; --surface: #161b22; --accent: #58a6ff; --text: #e6edf3; --purple: #bc8cff; }
    html, body, [class*="css"] { background: var(--bg) !important; color: var(--text) !important; font-family: 'DM Sans', sans-serif; }
    [data-testid="stSidebar"] { background: var(--surface) !important; border-right: 1px solid #30363d !important; }
    .result-card-success { background: rgba(63,185,80,0.05); border: 1px solid rgba(63,185,80,0.3); padding: 1.5rem; border-radius: 12px; }
    .jargon-chip { display: inline-block; background: rgba(188,140,255,0.1); border: 1px solid rgba(188,140,255,0.3); color: var(--purple); padding: 2px 8px; border-radius: 4px; margin: 2px; font-size: 0.8rem; font-family: monospace; }
    .stMetric { background: var(--surface) !important; border: 1px solid #30363d !important; border-radius: 10px; padding: 10px; }
</style>
""", unsafe_allow_html=True)

# ─── 5. SIDEBAR NAVIGATION ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🏥 MedSimplify")
    page = st.radio("", ["🏠 Home", "🧪 Simplifier", "🔬 NLP Analysis", "📊 Dataset", "📈 Results", "ℹ️ About"], label_visibility="collapsed")
    st.divider()
    st.caption("🤖 Model: Gemini 2.0 Flash")
    st.caption("🧠 NLP: spaCy Pipeline")

# ─── 6. PAGE LOGIC ──────────────────────────────────────────────────────────

# --- HOME ---
if page == "🏠 Home":
    st.title("Making medical research readable")
    st.write("A neural NLP pipeline that rewrites complex medical abstracts into plain language.")
    c1, c2, c3 = st.columns(3)
    c1.metric("Pairs", "921")
    c2.metric("Topics", "75")
    c3.metric("ROUGE-1", "0.48")
    st.image("https://img.freepik.com/free-vector/medical-technology-science-background-vector-blue-with-blank-space_53876-117739.jpg?w=1000", caption="NLP Architecture Overview")

# --- SIMPLIFIER ---
elif page == "🧪 Simplifier":
    st.title("🧪 Medical Simplifier")
    input_text = st.text_area("Input Medical Text:", height=200, placeholder="Enter clinical notes...")
    
    if st.button("⚡ Simplify", type="primary"):
        if input_text:
            with st.spinner("Running AI Pipeline..."):
                output = simplify_text(input_text)
                st.markdown("### Simplified Result")
                st.markdown(f'<div class="result-card-success">{output}</div>', unsafe_allow_html=True)
                
                # Show Jargon
                doc = nlp(input_text)
                jargon = [t.text for t in doc if len(t.text) > 9 and t.pos_ in ["NOUN", "ADJ"]]
                if jargon:
                    st.write("Detected Jargon:")
                    st.markdown("".join([f'<span class="jargon-chip">{j}</span>' for j in set(jargon)]), unsafe_allow_html=True)
        else: st.warning("Please enter text.")

# --- NLP ANALYSIS ---
elif page == "🔬 NLP Analysis":
    st.title("🔬 Deep NLP Analysis")
    raw_text = st.text_area("Analyze Text:", height=100)
    if raw_text:
        doc = nlp(raw_text)
        data = [{"Token": t.text, "Lemma": t.lemma_, "POS": t.pos_, "Label": spacy.explain(t.pos_)} for t in doc if not t.is_space]
        st.dataframe(pd.DataFrame(data), use_container_width=True)

# --- DATASET ---
elif page == "📊 Dataset":
    st.title("📊 MedSimp Corpus")
    st.write("Total Pairs: 921 | Training: 635 | Validation: 138 | Test: 148")
    st.info("Dataset sourced from PubMed and Cochrane reviews.")

# --- RESULTS ---
elif page == "📈 Results":
    st.title("📈 Model Performance")
    results = pd.DataFrame({
        "Model": ["T5-small", "T5-base", "BART-base", "Flan-T5", "Gemini"],
        "ROUGE-1": [0.38, 0.44, 0.48, 0.46, 0.43]
    })
    st.bar_chart(results.set_index("Model"))

# --- ABOUT ---
elif page == "ℹ️ About":
    st.title("ℹ️ About MedSimplify")
    st.write("Developed for NLP Research 2026. This app uses spaCy for linguistic analysis and Gemini for neural text-to-text simplification.")

st.markdown("---")
st.caption("MedSimplify · 2026 Submission Version")
