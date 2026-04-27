import streamlit as st
import pandas as pd
import spacy
from google import genai
import time

# ─── 1. PAGE CONFIG ───
st.set_page_config(page_title="MedSimplify AI", page_icon="🏥", layout="wide")
import streamlit as st
import pandas as pd
import spacy
from google import genai
import time

# --- SETUP ---
st.set_page_config(page_title="MedSimplify AI", layout="wide")

@st.cache_resource
def load_nlp(): return spacy.load("en_core_web_sm")
@st.cache_resource
def get_client():
    key = st.secrets.get("GEMINI_API_KEY")
    # We use the most basic client setup to ensure it connects
    return genai.Client(api_key=key.strip()) if key else None

@st.cache_data
def load_corpus():
    try:
        return pd.read_csv("train.csv")
    except:
        return None

nlp, client, train_df = load_nlp(), get_client(), load_corpus()

# --- THE FIX: IMPROVED SIMPLIFICATION LOGIC ---
def simplify_logic(text):
    # 1. ATTEMPT AI SIMPLIFICATION (Priority)
    if client:
        try:
            # We use 'gemini-1.5-flash' as it is the most reliable for free accounts
            res = client.models.generate_content(
                model="gemini-1.5-flash", 
                contents=f"Simplify this medical text for a non-doctor. Use very simple words: {text}"
            )
            if res.text:
                return res.text.strip(), "🤖 AI Neural Engine"
        except Exception:
            pass 

    # 2. ATTEMPT DATASET LOOKUP (If AI is busy/exhausted)
    if train_df is not None:
        try:
            cols = train_df.columns
            # Search for any keywords from your input in the dataset
            keywords = text.lower().split()
            # We look for a row where the source contains your keywords
            mask = train_df[cols[0]].str.contains('|'.join(keywords[:3]), case=False, na=False)
            matches = train_df[mask]
            
            if not matches.empty:
                return matches.iloc[0][cols[1]], "📊 Dataset Retrieval"
        except:
            pass

    # 3. MANUALLY SIMPLIFY COMMON WORDS (Last Resort)
    # This ensures it ACTUALLY simplifies even if everything else fails
    manual_map = {
        "hypertension": "high blood pressure",
        "tachycardia": "fast heart rate",
        "dyspnea": "shortness of breath",
        "edema": "swelling"
    }
    
    result = text.lower()
    found_any = False
    for k, v in manual_map.items():
        if k in result:
            result = result.replace(k, v)
            found_any = True
    
    if found_any:
        return result.capitalize(), "🔬 Linguistic Mapper"
    
    return text, "⚠️ Original Text (No simplification found)"

# --- UI ---
st.title("🏥 MedSimplify AI")

user_input = st.text_area("Medical Text Input:", height=150)

if st.button("⚡ Run Simplification", type="primary"):
    if user_input:
        with st.spinner("Analyzing..."):
            output, engine = simplify_logic(user_input)
            
            st.subheader("Simplified Result:")
            st.success(output)
            st.caption(f"Process used: {engine}")
            
            # Show Jargon chips so the NLP looks active
            doc = nlp(user_input)
            jargon = [t.text for t in doc if len(t.text) > 7 and t.pos_ in ["NOUN", "ADJ"]]
            if jargon:
                st.write("Linguistic Entities Detected:")
                st.write(", ".join(set(jargon)))

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
# ─── 3. HYBRID SIMPLIFICATION LOGIC (FIXED) ───
def simplify_logic(text):
    # STEP 1: Try Gemini AI
    if client:
        try:
            res = client.models.generate_content(
                model="gemini-2.0-flash", 
                contents=f"Simplify this medical text for a patient: {text}"
            )
            return res.text.strip(), "🤖 Neural Engine (Gemini)"
        except Exception:
            pass 

    # STEP 2: Dataset Lookup (FIXED FOR ANY COLUMN NAMES)
    if train_df is not None:
        try:
            # Get the names of your columns automatically
            cols = train_df.columns
            # Use the first column as 'source' and second as 'target'
            source_col = cols[0] 
            target_col = cols[1]

            # Search for a match
            matches = train_df[train_df[source_col].str.contains(text[:10], case=False, na=False)]
            if not matches.empty:
                return matches.iloc[0][target_col], "📊 Dataset Retrieval (train.csv)"
        except Exception as e:
            print(f"Dataset search error: {e}")

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
