import streamlit as st
import pandas as pd
import spacy
from google import genai
import time

# ─── 1. PAGE CONFIG ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MedSimplify — AI Medical Text Simplification",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── 2. LOAD RESOURCES ──────────────────────────────────────────────────────
@st.cache_resource
def load_nlp():
    try:
        return spacy.load("en_core_web_sm")
    except:
        return None

@st.cache_resource
def get_gemini_client():
    api_key = st.secrets.get("GEMINI_API_KEY")
    if not api_key: return None
    # We avoid passing version='v1beta' to prevent 404s
    return genai.Client(api_key=api_key.strip())

nlp = load_nlp()
client = get_gemini_client()

# Persistence for the UI
if "output_cache" not in st.session_state:
    st.session_state["output_cache"] = ""

# ─── 3. AI LOGIC WITH AUTOMATIC RETRIES ─────────────────────────────────────
def simplify_text_final(text):
    if not client: return "❌ API Key Missing in Secrets."
    
    # Using 2.0 Flash (Stable for 2026)
    model_id = "gemini-2.0-flash"
    prompt = f"Simplify this medical text for a patient. Return ONLY the simplified text: {text}"

    # Auto-retry loop (attempts to fix the 'busy' error without user clicking)
    for attempt in range(3): 
        try:
            response = client.models.generate_content(model=model_id, contents=prompt)
            if response.text:
                return response.text.strip()
        except Exception as e:
            if "429" in str(e) or "quota" in str(e).lower():
                time.sleep(2) # Wait 2 seconds and try again
                continue
            return f"❌ AI Error: {str(e)}"
    
    # If all 3 attempts fail, use Local spaCy Fallback
    if nlp:
        doc = nlp(text)
        fallback = " ".join([t.lemma_ if len(t.text) > 9 else t.text for t in doc])
        return f"⚠️ [Local NLP Mode] {fallback}"
    
    return "⏳ System Busy. Please try again in 1 minute."

# ─── 4. CSS (YOUR PREFERRED DARK THEME) ──────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@800&family=DM+Sans:wght@400;500&display=swap');
:root { --bg: #0d1117; --surface: #161b22; --border: #30363d; --accent: #58a6ff; --text: #e6edf3; --purple: #bc8cff; }
html, body, [class*="css"] { background: var(--bg) !important; color: var(--text) !important; font-family: 'DM Sans', sans-serif; }
[data-testid="stSidebar"] { background: var(--surface) !important; border-right: 1px solid var(--border) !important; }
.stTextArea textarea { background: #161b22 !important; border: 1px solid #30363d !important; color: #e6edf3 !important; }
.result-card-success { background: rgba(63,185,80,0.05); border: 1px solid rgba(63,185,80,0.3); padding: 1.2rem; border-radius: 10px; margin-top: 1rem; color: #3fb950; }
.jargon-chip { display: inline-block; background: rgba(188,140,255,0.1); border: 1px solid rgba(188,140,255,0.3); color: var(--purple); padding: 2px 10px; border-radius: 5px; margin: 4px; font-size: 0.85rem; }
.stButton > button { background: #ff0000 !important; color: white !important; font-weight: bold !important; border-radius: 8px !important; border: none !important; width: 100%; }
</style>
""", unsafe_allow_html=True)

# ─── 5. SIDEBAR ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🏥 MedSimplify")
    page = st.radio("", ["🏠 Home", "🧪 Simplifier", "🔬 NLP Analysis", "📊 Dataset", "📈 Results", "ℹ️ About"], label_visibility="collapsed")
    st.divider()
    st.caption("🤖 Model: Gemini 2.0 Flash")
    st.caption("🧠 NLP: spaCy en_core_web_sm")

# ─── 6. PAGE CONTENT ─────────────────────────────────────────────────────────
if page == "🧪 Simplifier":
    st.markdown("### Clinical Simplifier")
    input_text = st.text_area("Input:", height=150, placeholder="The patient has hypertension and tachycardia.")
    
    if st.button("⚡ Simplify"):
        if input_text.strip():
            with st.spinner("Processing through Neural Pipeline..."):
                st.session_state["output_cache"] = simplify_text_final(input_text)
        else:
            st.warning("Please enter text.")

    if st.session_state["output_cache"]:
        st.markdown(f'<div class="result-card-success">{st.session_state["output_cache"]}</div>', unsafe_allow_html=True)
        
        # Jargon Detection
        if nlp and input_text:
            doc = nlp(input_text)
            jargon = list(set([t.text for t in doc if (len(t.text) > 8 and t.pos_ in ["NOUN", "ADJ"]) or t.text.lower() in ["hypertension", "tachycardia"]]))
            if jargon:
                st.write("")
                st.markdown(" ".join([f'<span class="jargon-chip">{j}</span>' for j in jargon]), unsafe_allow_html=True)

elif page == "🏠 Home":
    st.title("Making Medical Research Readable")
    st.write("A hybrid NLP pipeline using linguistic analysis and generative AI.")
    st.image("https://img.freepik.com/free-vector/digital-healthcare-medicine-background_53876-114361.jpg?w=1000")

else:
    st.title(page)
    st.info("This section is part of the experimental dataset and analysis logs.")

st.markdown("---")
st.caption("MedSimplify · 2026 · NLP Research Project")
