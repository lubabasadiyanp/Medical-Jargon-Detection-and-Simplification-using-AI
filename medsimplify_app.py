import streamlit as st
import pandas as pd
import spacy
from google import genai
import time

# ─── 1. PAGE CONFIG ───
st.set_page_config(page_title="MedSimplify AI", page_icon="🏥", layout="wide")

# ─── 2. RESOURCE LOADING ───
@st.cache_resource
def load_nlp():
    return spacy.load("en_core_web_sm")

@st.cache_resource
def get_client():
    api_key = st.secrets.get("GEMINI_API_KEY")
    if not api_key: return None
    return genai.Client(api_key=api_key.strip())

@st.cache_data
def load_corpus():
    try:
        # Loading your dataset from GitHub
        df = pd.read_csv("train.csv")
        return df
    except:
        return None

nlp = load_nlp()
client = get_client()
train_df = load_corpus()

# ─── 3. HYBRID SIMPLIFICATION LOGIC ───
def simplify_logic(text):
    # STEP 1: Try Gemini AI
    if client:
        try:
            # Using 2.0 Flash (Stable for 2026)
            res = client.models.generate_content(
                model="gemini-2.0-flash", 
                contents=f"Simplify this medical text for a patient: {text}"
            )
            return res.text.strip(), "🤖 Neural Engine (Gemini)"
        except Exception:
            pass # AI Failed/Quota Exhausted, move to Dataset

    # STEP 2: Dataset Lookup (Using your train.csv)
    if train_df is not None:
        # Look for a match in your data
        # Note: Change 'source' and 'target' to match your CSV column names
        matches = train_df[train_df['source'].str.contains(text[:15], case=False, na=False)]
        if not matches.empty:
            return matches.iloc[0]['target'], "📊 Dataset Retrieval (train.csv)"

    # STEP 3: Basic NLP Fallback
    doc = nlp(text)
    tokens = [t.lemma_ if len(t.text) > 10 else t.text for t in doc]
    return " ".join(tokens), "🔬 Linguistic Fallback"

# ─── 4. PREMIUM INTERFACE (DARK THEME) ───
st.markdown("""
<style>
    body { background-color: #0d1117; color: #e6edf3; }
    [data-testid="stSidebar"] { background-color: #161b22; border-right: 1px solid #30363d; }
    .stButton>button { background: #ff4b4b; color: white; width: 100%; border-radius: 8px; font-weight: bold; }
    .output-card { background: #161b22; border: 1px solid #3fb950; padding: 20px; border-radius: 12px; color: #3fb950; }
    .chip { background: #bc8cff22; color: #bc8cff; border: 1px solid #bc8cff44; padding: 2px 10px; border-radius: 15px; font-size: 0.8rem; margin-right: 5px; }
</style>
""", unsafe_allow_html=True)

# ─── 5. APP STRUCTURE ───
with st.sidebar:
    st.title("🏥 MedSimplify")
    page = st.radio("Navigation", ["Home", "Simplifier", "Dataset Analysis"])
    st.divider()
    st.caption("Status: Hybrid Pipeline Active")

if page == "Home":
    st.title("Making Medical Research Readable")
    st.write("A professional NLP pipeline that combines Neural Generation with your project's Research Dataset.")
    st.image("https://img.freepik.com/free-vector/medical-technology-science-background_53876-117739.jpg?w=1000")

elif page == "Simplifier":
    st.header("🧪 AI + Dataset Simplifier")
    user_input = st.text_area("Clinical Input:", placeholder="Enter medical jargon...")
    
    if st.button("⚡ Run Pipeline"):
        if user_input:
            with st.spinner("Processing through NLP & AI layers..."):
                output, engine = simplify_logic(user_input)
                st.markdown(f'<div class="output-card">{output}</div>', unsafe_allow_html=True)
                st.caption(f"Engine Used: {engine}")
                
                # Show Jargon Chips via spaCy
                doc = nlp(user_input)
                jargon = [t.text for t in doc if t.pos_ in ["NOUN", "ADJ"] and len(t.text) > 8]
                if jargon:
                    st.write("Linguistic Tags:")
                    st.markdown(" ".join([f'<span class="chip">{j}</span>' for j in set(jargon)]), unsafe_allow_html=True)

elif page == "Dataset Analysis":
    st.header("📊 Project Dataset (`train.csv`)")
    if train_df is not None:
        st.dataframe(train_df.head(100), use_container_width=True)
    else:
        st.error("Dataset not detected. Ensure 'train.csv' is in your main GitHub directory.")
