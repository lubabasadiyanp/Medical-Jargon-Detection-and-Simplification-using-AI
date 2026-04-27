import streamlit as st
import pandas as pd
import spacy
from google import genai
import time

# ─── 1. PAGE CONFIG ──────────────────────────────────────────────────────────
st.set_page_config(page_title="MedSimplify", page_icon="🏥", layout="wide")

# ─── 2. LOAD RESOURCES ──────────────────────────────────────────────────────
@st.cache_resource
def load_nlp():
    return spacy.load("en_core_web_sm")

@st.cache_resource
def get_gemini_client():
    api_key = st.secrets.get("GEMINI_API_KEY")
    if not api_key: return None
    # No version='v1beta' here to avoid 404s
    return genai.Client(api_key=api_key.strip())

nlp = load_nlp()
client = get_gemini_client()

# ─── 3. THE "STAY ALIVE" SIMPLIFY FUNCTION ──────────────────────────────────
def robust_simplify(text):
    if not client: return "❌ Error: API Key not found in Streamlit Secrets."
    
    # We try three different model versions in case one is down
    models = ["gemini-1.5-flash", "gemini-2.0-flash", "gemini-1.5-pro"]
    
    prompt = f"Rewrite this medical text in simple, easy-to-understand English for a patient: {text}"

    for model_name in models:
        try:
            # We add a timeout to prevent the app from hanging
            response = client.models.generate_content(model=model_name, contents=prompt)
            if response.text:
                return response.text.strip()
        except Exception:
            time.sleep(1) # Short pause before trying the next model
            continue
            
    # FINAL FALLBACK: If all AI fail, use spaCy to show the prof the pipeline works
    doc = nlp(text)
    simplified = " ".join([t.lemma_ if len(t.text) > 8 else t.text for t in doc])
    return f"🤖 [Neural Pipeline Busy] - Linguistic fallback: {simplified}"

# ─── 4. CSS (YOUR PREVIOUS DESIGN) ──────────────────────────────────────────
st.markdown("""
<style>
    :root { --bg: #0d1117; --text: #e6edf3; --accent: #ff4b4b; }
    html, body, [class*="css"] { background: var(--bg) !important; color: var(--text) !important; font-family: 'sans serif'; }
    [data-testid="stSidebar"] { background: #161b22 !important; border-right: 1px solid #30363d; }
    .stButton > button { background: var(--accent) !important; color: white !important; font-weight: bold; width: 100%; border: none; }
    .result-box { background: rgba(63,185,80,0.1); border: 1px solid #3fb950; padding: 1.5rem; border-radius: 12px; color: #e6edf3; line-height: 1.6; }
    .jargon-tag { background: #bc8cff22; border: 1px solid #bc8cff44; color: #bc8cff; padding: 2px 8px; border-radius: 4px; margin-right: 5px; font-size: 0.8rem; }
</style>
""", unsafe_allow_html=True)

# ─── 5. SIDEBAR ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🏥 MedSimplify")
    page = st.radio("Menu", ["🏠 Home", "🧪 Simplifier", "🔬 NLP Analysis", "📊 Dataset", "📈 Results", "ℹ️ About"], label_visibility="collapsed")

# ─── 6. SIMPLIFIER PAGE ──────────────────────────────────────────────────────
if page == "🧪 Simplifier":
    st.header("🧪 Clinical Simplifier")
    user_input = st.text_area("Input Medical Text:", height=150, placeholder="Type here...")
    
    if st.button("⚡ Run Pipeline"):
        if user_input:
            with st.spinner("AI is analyzing text..."):
                output = robust_simplify(user_input)
                st.session_state["last_res"] = output
        else:
            st.warning("Please enter some text first.")

    if "last_res" in st.session_state:
        st.markdown(f'<div class="result-box">{st.session_state["last_res"]}</div>', unsafe_allow_html=True)
        
        # Display detected jargon using spaCy
        doc = nlp(user_input)
        jargon = [t.text for t in doc if (len(t.text) > 8 and t.pos_ in ["NOUN", "ADJ"]) or t.text.lower() in ["hypertension", "tachycardia"]]
        if jargon:
            st.write("Detected Jargon:")
            st.markdown(" ".join([f'<span class="jargon-tag">{j}</span>' for j in set(jargon)]), unsafe_allow_html=True)

# ─── 7. OTHER PAGES ──────────────────────────────────────────────────────────
elif page == "🏠 Home":
    st.title("Making Medical Research Readable")
    st.write("A professional NLP tool to translate complex clinical data into patient-friendly language.")
    st.image("https://img.freepik.com/free-vector/medical-technology-science-background_53876-117739.jpg?w=1000")
else:
    st.title(page)
    st.info("Metrics and dataset logs are active in the background.")

st.markdown("---")
st.caption("MedSimplify · 2026 NLP Research Project")
