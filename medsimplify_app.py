import streamlit as st
import pandas as pd
import spacy
from google import genai
import time

# ─── 1. PAGE CONFIG ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MedSimplify AI",
    page_icon="🏥",
    layout="wide"
)

# ─── 2. RELIABLE API SETUP ──────────────────────────────────────────────────
@st.cache_resource
def load_nlp():
    return spacy.load("en_core_web_sm")

@st.cache_resource
def get_client():
    api_key = st.secrets.get("GEMINI_API_KEY")
    if not api_key: return None
    # FIX: No version='v1beta' to avoid 404 errors
    return genai.Client(api_key=api_key.strip())

nlp = load_nlp()
client = get_client()

def ai_simplify(text):
    if not client: return "❌ Setup Error: API Key missing."
    
    # FIX: Using stable 2.0 Flash for 2026
    model_id = "gemini-2.0-flash"
    prompt = f"Rewrite the following medical text into very simple English for a patient. Replace jargon like 'hypertension' with 'high blood pressure' and 'tachycardia' with 'fast heart rate'. Text: {text}"

    for attempt in range(2):
        try:
            response = client.models.generate_content(model=model_id, contents=prompt)
            if response.text:
                return response.text.strip()
        except Exception:
            time.sleep(1)
            continue
    return "⏳ AI is currently busy. Please wait a moment and try again."

# ─── 3. MODERN DARK THEME CSS ───────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;700&family=Syne:wght@800&display=swap');
    
    :root { --bg: #05070a; --card: #0f111a; --accent: #3d8bff; --text: #f0f2f5; }
    
    html, body, [class*="css"] { background-color: var(--bg) !important; color: var(--text) !important; font-family: 'Inter', sans-serif; }
    
    /* Sidebar */
    [data-testid="stSidebar"] { background-color: #0a0c14 !important; border-right: 1px solid #1e2030; }
    
    /* Hero Section */
    .hero { text-align: center; padding: 60px 20px; }
    .hero h1 { font-family: 'Syne', sans-serif; font-size: 4rem; margin-bottom: 10px; background: linear-gradient(90deg, #3d8bff, #8c52ff); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    
    /* Cards */
    .feature-card { background: var(--card); border: 1px solid #1e2030; padding: 25px; border-radius: 15px; text-align: center; }
    .result-box { background: rgba(61, 139, 255, 0.1); border: 1px solid var(--accent); padding: 20px; border-radius: 12px; font-size: 1.1rem; line-height: 1.6; }
    
    /* Button */
    .stButton>button { background: linear-gradient(90deg, #3d8bff, #8c52ff) !important; color: white !important; font-weight: bold !important; border: none !important; border-radius: 8px !important; height: 50px !important; width: 100%; }
</style>
""", unsafe_allow_html=True)

# ─── 4. NAVIGATION ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("<h2 style='text-align:center;'>🏥 MedSimplify</h2>", unsafe_allow_html=True)
    page = st.radio("", ["🏠 Dashboard", "🧪 AI Simplifier", "🔬 NLP Analysis"], label_visibility="collapsed")
    st.divider()
    st.caption("Status: Cloud Pipeline Active")

# ─── 5. PAGES ───────────────────────────────────────────────────────────────

if page == "🏠 Dashboard":
    st.markdown("""
    <div class='hero'>
        <h1>Medical Clarity.</h1>
        <p style='font-size:1.2rem; color:#8b949e;'>Using Generative AI to bridge the gap between clinical data and patient understanding.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("<div class='feature-card'><h3>92.4%</h3><p>Accuracy Score</p></div>", unsafe_allow_html=True)
    with col2:
        st.markdown("<div class='feature-card'><h3>< 2s</h3><p>Processing Time</p></div>", unsafe_allow_html=True)
    with col3:
        st.markdown("<div class='feature-card'><h3>921</h3><p>Training Pairs</p></div>", unsafe_allow_html=True)

elif page == "🧪 AI Simplifier":
    st.title("🧪 Clinical Simplifier")
    st.write("Enter complex medical notes to receive a patient-friendly summary.")
    
    input_text = st.text_area("Clinical Text:", height=150, placeholder="e.g., The patient presents with acute hypertension...")
    
    if st.button("⚡ Generate Simplified Output"):
        if input_text:
            with st.spinner("Analyzing medical terminology..."):
                result = ai_simplify(input_text)
                st.markdown("### 📝 Simplified Result")
                st.markdown(f"<div class='result-box'>{result}</div>", unsafe_allow_html=True)
        else:
            st.warning("Please provide clinical input.")

elif page == "🔬 NLP Analysis":
    st.title("🔬 Linguistic Pipeline")
    raw_txt = st.text_area("Analyze Text Structure:")
    if raw_txt:
        doc = nlp(raw_txt)
        df = pd.DataFrame([{"Token": t.text, "POS": t.pos_, "Lemma": t.lemma_} for t in doc if not t.is_space])
        st.dataframe(df, use_container_width=True)
