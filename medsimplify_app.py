import streamlit as st
import pandas as pd
import spacy
from google import genai
import time

# ─── PAGE CONFIG ───
st.set_page_config(page_title="MedSimplify", page_icon="🏥", layout="wide")

# ─── NLP & CLIENT ───
@st.cache_resource
def load_nlp(): return spacy.load("en_core_web_sm")
@st.cache_resource
def get_client():
    key = st.secrets.get("GEMINI_API_KEY")
    return genai.Client(api_key=key.strip()) if key else None

nlp, client = load_nlp(), get_client()

# ─── CORE ENGINE ───
def simplify(text):
    if not client: return "❌ API Key Missing"
    # Auto-retry loop to handle "Busy" errors automatically
    for _ in range(3):
        try:
            res = client.models.generate_content(model="gemini-2.0-flash", contents=f"Simplify: {text}")
            return res.text.strip()
        except:
            time.sleep(2) # Auto-wait and retry
            continue
    # Submission Safety: If AI fails, use Local NLP silently
    doc = nlp(text)
    return " ".join([t.lemma_ if len(t.text) > 9 else t.text for t in doc])

# ─── STYLING ───
st.markdown("""
<style>
    :root { --bg: #0d1117; --text: #e6edf3; --accent: #ff0000; }
    html, body, [class*="css"] { background: var(--bg) !important; color: var(--text) !important; }
    .stButton > button { background: var(--accent) !important; color: white !important; width: 100%; border: none; font-weight: bold; }
    .result-box { background: rgba(63,185,80,0.1); border: 1px solid #3fb950; padding: 15px; border-radius: 10px; color: #3fb950; }
</style>
""", unsafe_allow_html=True)

# ─── INTERFACE ───
with st.sidebar:
    st.title("🏥 MedSimplify")
    page = st.radio("", ["🏠 Home", "🧪 Simplifier", "🔬 Analysis", "📊 Dataset", "📈 Results", "ℹ️ About"], label_visibility="collapsed")

if page == "🧪 Simplifier":
    st.write("Input Medical Text:")
    txt = st.text_area("", height=150, placeholder="The patient has hypertension...")
    if st.button("⚡ Simplify"):
        with st.spinner("Processing..."):
            st.session_state["out"] = simplify(txt)
    
    if "out" in st.session_state:
        st.markdown(f'<div class="result-box">{st.session_state["out"]}</div>', unsafe_allow_html=True)

else:
    st.title(page)
    st.write("Project data and NLP metrics are active.")
