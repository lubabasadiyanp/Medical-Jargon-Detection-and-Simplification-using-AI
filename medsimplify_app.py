import streamlit as st
import pandas as pd
import spacy
from google import genai
import time

# ─── PAGE CONFIG ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MedSimplify — AI Medical Text Simplification",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── LOAD RESOURCES ──────────────────────────────────────────────────────────
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

# Initialize Session State to store outputs so they don't disappear
if "simplified_output" not in st.session_state:
    st.session_state["simplified_output"] = ""

# ─── THE SIMPLIFY ENGINE ─────────────────────────────────────────────────────
def run_simplification(text):
    if not client:
        return "❌ API Key Missing."
    
    # We use a broad list of models to find one that isn't busy
    models_to_try = ["gemini-2.0-flash", "gemini-2.0-flash-lite", "gemini-1.5-flash"]
    
    prompt = (
        "You are a medical communication expert. Rewrite this medical text into "
        "clear, simple language for a patient. Return ONLY the simplified text, "
        f"no introductory filler: {text}"
    )

    for model_name in models_to_try:
        try:
            response = client.models.generate_content(model=model_name, contents=prompt)
            if response.text:
                return response.text.strip()
        except Exception:
            continue # Try next model if busy
            
    return "⏳ AI is very busy right now. Please click 'Simplify' again in 5 seconds."

# ─── JARGON DETECTION ────────────────────────────────────────────────────────
def detect_jargon(doc):
    return list(set([t.text for t in doc if (len(t.text) > 8 and t.pos_ in ["NOUN", "ADJ"]) or t.text.lower() in ["hypertension", "tachycardia"]]))

# ═══════════════════════════════════════════════════════════════════════════════
# CSS — EXACT REPLICA OF YOUR SECOND SCREENSHOT
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@800&family=DM+Sans:wght@400;500&display=swap');

:root {
    --bg: #0d1117; --surface: #161b22; --border: #30363d;
    --accent: #58a6ff; --text: #e6edf3; --purple: #bc8cff;
}

html, body, [class*="css"] { background: var(--bg) !important; color: var(--text) !important; font-family: 'DM Sans', sans-serif; }
[data-testid="stSidebar"] { background: var(--surface) !important; border-right: 1px solid var(--border) !important; }

/* Text Area Styling */
.stTextArea textarea { background: #161b22 !important; border: 1px solid #30363d !important; color: #e6edf3 !important; border-radius: 8px; }

/* Success/Error Boxes */
.result-card-success { 
    background: rgba(63,185,80,0.05); border: 1px solid rgba(63,185,80,0.3); 
    padding: 1.2rem; border-radius: 10px; margin-top: 1rem; color: #3fb950;
}
.error-box {
    background: rgba(210,153,34,0.05); border: 1px solid rgba(210,153,34,0.3);
    padding: 1rem; border-radius: 10px; color: #d29922; margin-top: 1rem;
}

/* Jargon Chips */
.jargon-chip { 
    display: inline-block; background: rgba(188,140,255,0.1); 
    border: 1px solid rgba(188,140,255,0.3); color: var(--purple); 
    padding: 2px 10px; border-radius: 5px; margin: 4px; font-size: 0.85rem;
}

/* Buttons */
.stButton > button { 
    background: #ff0000 !important; color: white !important; font-weight: bold !important;
    border-radius: 8px !important; border: none !important; padding: 0.5rem 2rem !important;
}
</style>
""", unsafe_allow_html=True)

# ─── SIDEBAR ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🏥 MedSimplify")
    page = st.radio("", ["🏠 Home", "🧪 Simplifier", "🔬 NLP Analysis", "📊 Dataset", "📈 Results", "ℹ️ About"], label_visibility="collapsed")
    st.divider()
    st.caption("🤖 Model: Gemini 2.0 Flash")
    st.caption("🧠 NLP: spaCy en_core_web_sm")

# ─── SIMPLIFIER PAGE ─────────────────────────────────────────────────────────
if page == "🧪 Simplifier":
    st.write("Input:")
    input_text = st.text_area("", placeholder="The patient has hypertension and tachycardia.", label_visibility="collapsed", height=150)
    
    if st.button("⚡ Simplify"):
        if input_text:
            with st.spinner("Processing through Neural Pipeline..."):
                result = run_simplification(input_text)
                st.session_state["simplified_output"] = result
        else:
            st.warning("Please enter text.")

    # Show result box only if there is a result
    if st.session_state["simplified_output"]:
        output = st.session_state["simplified_output"]
        if "⏳" in output:
            st.markdown(f'<div class="error-box">{output}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="result-card-success">{output}</div>', unsafe_allow_html=True)

    # Show Jargon Chips (Always work locally)
    if input_text:
        doc = nlp(input_text)
        jargon = detect_jargon(doc)
        if jargon:
            st.write("")
            st.markdown(" ".join([f'<span class="jargon-chip">{j}</span>' for j in jargon]), unsafe_allow_html=True)

# ─── OTHER PAGES (KEEPING YOUR STRUCTURE) ────────────────────────────────────
elif page == "🏠 Home":
    st.title("Making Medical Research Readable")
    st.write("Neural NLP Pipeline using spaCy + Gemini AI.")
elif page == "🔬 NLP Analysis":
    st.title("Linguistic Analysis")
    if input_text := st.text_area("Analyze:"):
        doc = nlp(input_text)
        st.dataframe(pd.DataFrame([{"Token": t.text, "POS": t.pos_} for t in doc]))
else:
    st.title(page)
    st.info("Technical details for this section are loaded from the project dataset.")
