import streamlit as st
import os
import time
import pandas as pd
from google import genai
from google.genai import errors

# ─── 1. Page Config ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MedSimplify — NLP Medical Text Simplification",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── 2. Initialize Session State (Prevents Crashes) ──────────────────────────
if "demo_input" not in st.session_state:
    st.session_state["demo_input"] = ""
if "demo_output" not in st.session_state:
    st.session_state["demo_output"] = ""
if "demo_reference" not in st.session_state:
    st.session_state["demo_reference"] = ""

# ─── 3. Data Constants ───────────────────────────────────────────────────────
TOPICS = {
    "1":"Muscle cramps","2":"Duloxetine & lower urinary tract","3":"Hyperkalemia",
    "4":"Diabetes mellitus classification","5":"Popliteal cyst treatment",
    "6":"Hyperthyroidism diagnosis","7":"Group A streptococcal tonsillopharyngitis",
    "8":"Clozapine vs perphenazine","9":"Upper GI foreign bodies","10":"Finger pain",
    "75":"BCL11A & sickle cell disease",
}

READABILITY_SCORES = {
    "MSD Manual":4.08,"Cochrane":4.23,"eLife":4.56,
    "PLOS Biology":4.63,"Wikipedia":4.15,
}

EXAMPLES = [
    {
        "label":"Example 1 — Muscle Cramps",
        "input":"Exercise-Associated Muscle Cramps (EAMC) are a common painful condition of muscle spasms. Despite scientists tried to understand the physiological mechanism that underlies these common phenomena, the etiology is still unclear.",
        "reference":"Muscle cramps from exercise are painful spasms. Scientists don't fully know why they happen yet, but new research suggests it's about how nerves talk to muscles rather than just being dehydrated."
    },
    {
        "label":"Example 2 — Urinary Incontinence",
        "input":"Urinary incontinence is the inability to willingly control bladder voiding. Stress urinary incontinence (SUI) is the most frequently occurring type of incontinence in women.",
        "reference":"Urinary incontinence is when you cannot control when you pee. The most common type for women is 'stress' incontinence, which happens during physical movement."
    }
]

MODEL_RESULTS = [
    {"Model":"BART-base (fine-tuned) ⭐ Best","ROUGE-1":0.48,"ROUGE-2":0.24,"ROUGE-L":0.44},
    {"Model":"Gemini 1.5 Flash (free)","ROUGE-1":0.43,"ROUGE-2":0.21,"ROUGE-L":0.40},
]

# ─── 4. Gemini Client & Logic ────────────────────────────────────────────────
@st.cache_resource
def get_client():
    # Looks for key in Streamlit Secrets dashboard
    api_key = st.secrets.get("GEMINI_API_KEY")
    if not api_key:
        return None
    return genai.Client(api_key=api_key)

def simplify_text(text: str) -> str:
    client = get_client()
    if client is None:
        return "❌ API key not found. Please add `GEMINI_API_KEY` to your Streamlit Secrets dashboard."
    
    model_id = "gemini-1.5-flash"
    prompt = (
        "You are a medical assistant. Simplify the following medical text for a patient. "
        "Use simple everyday words, avoid jargon, and be concise.\n\n"
        f"Medical text: {text}\n\nSimplified version:"
    )

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(model=model_id, contents=prompt)
            return response.text.strip()
        except errors.ClientError as e:
            if "429" in str(e):
                if attempt < max_retries - 1:
                    time.sleep(2)
                    continue
                return "⚠️ Quota exceeded. Please wait a minute and try again."
            return f"❌ API Error: {str(e)}"
        except Exception as e:
            return f"❌ Error: {str(e)}"

# ─── 5. UI Styles ────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
[data-testid="stSidebar"] { background: #1a1814; color: white; }
.simplified-box { background:#e4f5f2; border-left:4px solid #1a7a6e; border-radius:8px; padding:1.2rem; color:#1a5a50; margin-bottom: 1rem; }
.reference-box { background:#fffbf5; border-left:4px solid #b87c30; border-radius:8px; padding:1.2rem; color:#6b5030; }
</style>
""", unsafe_allow_html=True)

# ─── 6. Sidebar Navigation ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏥 MedSimplify")
    st.divider()
    page = st.radio("Navigate", ["🏠 Home","🧪 Demo","📈 Results","ℹ️ About"])
    st.divider()
    st.info("Powered by Gemini 1.5 Flash")

# ─── 7. Page Content ─────────────────────────────────────────────────────────
if page == "🏠 Home":
    st.title("Making Medical Research Readable")
    st.markdown("This tool uses AI to bridge the gap between complex clinical text and patient understanding.")
    c1, c2, c3 = st.columns(3)
    c1.metric("Parallel Pairs","921")
    c2.metric("Topics","75")
    c3.metric("Best ROUGE-1","0.48")
    st.divider()
    st.subheader("Try an Example")
    st.write(EXAMPLES[0]["input"])
    st.markdown(f'<div class="simplified-box"><b>Simplified:</b> {EXAMPLES[0]["reference"]}</div>', unsafe_allow_html=True)

elif page == "🧪 Demo":
    st.title("Live Demo")
    st.write("Select an example or paste your own medical text below.")
    
    ex_cols = st.columns(len(EXAMPLES))
    for i, ex in enumerate(EXAMPLES):
        if ex_cols[i].button(ex["label"]):
            st.session_state["demo_input"] = ex["input"]
            st.session_state["demo_reference"] = ex["reference"]
            st.session_state["demo_output"] = ""

    input_text = st.text_area("Medical Text Input", value=st.session_state["demo_input"], height=200)

    if st.button("Simplify →", type="primary"):
        if input_text.strip():
            with st.spinner("AI is working..."):
                st.session_state["demo_output"] = simplify_text(input_text)
        else:
            st.warning("Please enter some text first.")

    if st.session_state["demo_output"]:
        st.markdown("### ✨ Simplified Output")
        st.markdown(f'<div class="simplified-box">{st.session_state["demo_output"]}</div>', unsafe_allow_html=True)
        
    if st.session_state["demo_reference"]:
        st.divider()
        st.markdown("### 📖 Expert Human Reference")
        st.markdown(f'<div class="reference-box">{st.session_state["demo_reference"]}</div>', unsafe_allow_html=True)

elif page == "📈 Results":
    st.title("Model Evaluation")
    st.table(pd.DataFrame(MODEL_RESULTS))

elif page == "ℹ️ About":
    st.title("About MedSimplify")
    st.markdown("""
    This project focuses on **Lexical Simplification** and **Neural Text Generation**. 
    Medical abstracts are often too dense for the general public. We use Gemini 1.5 Flash 
    to translate that complexity into clear, actionable language.
    """)
