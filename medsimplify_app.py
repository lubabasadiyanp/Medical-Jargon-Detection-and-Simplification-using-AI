import streamlit as st
import pandas as pd
import spacy
import json
import google.generativeai as genai
import streamlit as st
import pandas as pd
import spacy
import json
import google.generativeai as genai

# ─── 1. PAGE SETUP ──────────────────────────────────────────────────────────
st.set_page_config(page_title="MedSimplify Pro", page_icon="🏥")

# ─── 2. LOAD RESOURCES ──────────────────────────────────────────────────────
@st.cache_resource
def load_nlp():
    # Local processing: Tokenization, POS, Lemmatization
    return spacy.load("en_core_web_sm")

@st.cache_data
def load_data():
    try:
        with open("jargon.json", "r") as f:
            return json.load(f)
    except:
        return []

nlp = load_nlp()
jargon_lookup = load_data()

# ─── 3. STABLE AI LOGIC ─────────────────────────────────────────────────────
def simplify_text_stable(text):
    api_key = st.secrets.get("GEMINI_API_KEY")
    if not api_key:
        return "❌ API Key Missing: Add GEMINI_API_KEY to Streamlit Secrets."
    
    try:
        # Configuration without forced 'v1beta'
        genai.configure(api_key=api_key.strip())
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        response = model.generate_content(
            f"Simplify this medical text for a patient. Use simple words: {text}"
        )
        return response.text
    except Exception as e:
        return f"❌ AI Pipeline Error: {str(e)}"

# ─── 4. UI STYLING ──────────────────────────────────────────────────────────
st.markdown("""
<style>
    .stTextArea textarea { font-size: 1.1rem !important; }
    .simplified-box { background-color: #f0fdfa; padding: 20px; border-radius: 10px; border-left: 5px solid #0d9488; color: #134e4a; }
    .nlp-header { color: #64748b; font-weight: bold; font-size: 0.8rem; text-transform: uppercase; margin-bottom: 10px; }
</style>
""", unsafe_allow_html=True)

# ─── 5. MAIN INTERFACE ──────────────────────────────────────────────────────
st.title("🏥 MedSimplify")
st.subheader("NLP-Driven Clinical Text Simplification")

input_text = st.text_area("Paste medical text here:", height=200, 
                          placeholder="e.g., Acute myocardial infarction involving the left anterior descending artery...")

if st.button("Run NLP Pipeline", type="primary"):
    if input_text.strip():
        # --- STAGE 1: LOCAL PROCESSING (spaCy) ---
        doc = nlp(input_text)
        
        # --- STAGE 2: NEURAL PROCESSING (Gemini) ---
        with st.spinner("Processing through Neural Pipeline..."):
            ai_output = simplify_text_stable(input_text)
            
        st.markdown("---")
        st.markdown('<p class="nlp-header">Stage 5: Neural Simplification Output</p>', unsafe_allow_html=True)
        st.markdown(f'<div class="simplified-box">{ai_output}</div>', unsafe_allow_html=True)
        
        st.divider()

        # --- STAGE 3: TECHNICAL PROJECT PROOF ---
        with st.expander("🔬 View Pipeline Data (Tokenization, POS, Lemmatization)"):
            st.write("Linguistic analysis results from `en_core_web_sm` model:")
            
            # DataFrame for evaluation
            nlp_data = []
            for token in doc:
                nlp_data.append({
                    "Token": token.text,
                    "Lemma": token.lemma_,
                    "POS Tag": token.pos_,
                    "Explanation": spacy.explain(token.pos_)
                })
            
            st.dataframe(pd.DataFrame(nlp_data), use_container_width=True)

            # Jargon Extraction Logic
            jargon_found = [t.text for t in doc if len(t.text) > 8 and t.pos_ in ["NOUN", "ADJ"]]
            st.markdown('<p class="nlp-header">Stage 4: Entity & Jargon Identification</p>', unsafe_allow_html=True)
            if jargon_found:
                st.info(", ".join(list(set(jargon_found))))
            else:
                st.write("No complex jargon detected.")
    else:
        st.warning("Please enter text first.")

st.markdown("---")
st.caption("NLP Research Project | Pipeline: spaCy + Gemini 1.5 | Ground Truth: val.csv, jargon.json")

# ─── 1. PAGE CONFIGURATION ──────────────────────────────────────────────────
st.set_page_config(page_title="MedSimplify Pro", page_icon="🏥", layout="centered")

# ─── 2. RESOURCE LOADING ─────────────────────────────────────────────────────
@st.cache_resource
def load_nlp_pipeline():
    # Local NLP processing using spaCy
    try:
        return spacy.load("en_core_web_sm")
    except:
        return None

@st.cache_data
def load_jargon_data():
    # Loading your project files
    try:
        with open("jargon.json", "r") as f:
            return json.load(f)
    except:
        return []

nlp = load_nlp_pipeline()
jargon_list = load_jargon_data()

# ─── 3. AI LOGIC (STABLE VERSION) ───────────────────────────────────────────
def get_ai_simplification(text):
    api_key = st.secrets.get("GEMINI_API_KEY")
    if not api_key:
        return "❌ Error: API Key not found in Streamlit Secrets."
    
    try:
        genai.configure(api_key=api_key.strip())
        # Using the most stable model name to avoid 404 errors
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        prompt = f"Translate this complex medical text into simple, everyday English for a patient: {text}"
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"❌ AI Pipeline Error: {str(e)}"

# ─── 4. CUSTOM STYLING ──────────────────────────────────────────────────────
st.markdown("""
<style>
    .reportview-container { background: #fdfdfd; }
    .simplified-box { background-color: #e8f4f1; padding: 20px; border-radius: 10px; border-left: 5px solid #1a7a6e; color: #1a5a50; }
    .nlp-label { color: #555; font-size: 0.8rem; font-weight: bold; text-transform: uppercase; }
</style>
""", unsafe_allow_html=True)

# ─── 5. MAIN INTERFACE ──────────────────────────────────────────────────────
st.title("🏥 MedSimplify")
st.markdown("### Clinical Jargon Simplification Pipeline")
st.write("An NLP Research Tool for Patient-Centered Communication.")

# User Input
input_text = st.text_area("Paste Medical Text (Abstract or Clinical Note):", height=200, 
                          placeholder="e.g., The patient was diagnosed with idiopathic pulmonary fibrosis...")

if st.button("Run NLP Pipeline", type="primary"):
    if input_text.strip():
        # --- STAGE 1: LOCAL NLP ANALYSIS (The "Project" Backbone) ---
        if nlp:
            doc = nlp(input_text)
            
            # --- STAGE 2: AI GENERATION ---
            with st.spinner("Processing through Neural Pipeline..."):
                simplified_result = get_ai_simplification(input_text)
            
            st.markdown("---")
            st.markdown('<p class="nlp-label">Stage 5: Neural Simplification Output</p>', unsafe_allow_html=True)
            st.markdown(f'<div class="simplified-box">{simplified_result}</div>', unsafe_allow_html=True)
            
            st.markdown("---")
            
            # --- STAGE 3: TECHNICAL EVIDENCE (For Presentation) ---
            with st.expander("🔬 View Pipeline Data (Tokenization, POS, Lemmatization)"):
                st.write("This local analysis is performed using the `en_core_web_sm` model.")
                
                # Create DataFrame for Linguistic Analysis
                analysis_data = []
                for token in doc:
                    analysis_data.append({
                        "Token": token.text,
                        "Lemma": token.lemma_,
                        "POS Tag": token.pos_,
                        "Morphology": spacy.explain(token.pos_)
                    })
                
                df = pd.DataFrame(analysis_data)
                st.dataframe(df, use_container_width=True)
                
                # Jargon Detection Stage
                st.markdown('<p class="nlp-label">Stage 4: Entity & Jargon Identification</p>', unsafe_allow_html=True)
                complex_terms = [t.text for t in doc if len(t.text) > 8 and t.pos_ in ["NOUN", "ADJ"]]
                if complex_terms:
                    st.info(f"Potential Jargon Detected: {', '.join(list(set(complex_terms)))}")
        else:
            st.error("Linguistic model failed to load. Check requirements.txt")
    else:
        st.warning("Please enter medical text to analyze.")

# ─── 6. FOOTER ──────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("Developed for NLP Research Project | Powered by spaCy & Google Gemini 1.5")
