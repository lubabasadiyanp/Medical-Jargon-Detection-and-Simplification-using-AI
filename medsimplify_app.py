import streamlit as st
import pandas as pd
import spacy
from google import genai

# --- 1. SETUP ---
st.set_page_config(page_title="MedSimplify AI", layout="wide")

@st.cache_resource
def load_nlp():
    # We use the medium model for better "training" accuracy if available, 
    # otherwise small is fine.
    return spacy.load("en_core_web_sm")

@st.cache_data
def load_data():
    try:
        df = pd.read_csv("train.csv")
        # Ensure we know which columns are which
        return df.iloc[:, [0, 1]] # Takes first two columns
    except:
        return None

nlp = load_nlp()
train_df = load_data()

# --- 2. THE SEARCH/TRAINING ENGINE ---
def get_best_match(user_query):
    if train_df is None:
        return None
    
    # NLP Process: Vectorize the user's search
    query_doc = nlp(user_query.lower())
    
    best_score = 0
    best_result = ""
    
    # We "train" the search by comparing similarity across the dataset
    # We check the first 200 rows for speed
    for _, row in train_df.head(200).iterrows():
        sample_doc = nlp(str(row[0]).lower())
        
        # This is the core NLP Similarity process
        score = query_doc.similarity(sample_doc)
        
        if score > best_score:
            best_score = score
            best_result = row[1] # The simplified version
            
    # Only return if the "training" match is strong enough (>60% match)
    return best_result if best_score > 0.6 else None

# --- 3. MAIN LOGIC ---
def simplify_logic(text):
    # STEP A: Check the "Trained" Dataset first
    match = get_best_match(text)
    if match:
        return match, "📊 Dataset Match (Semantic Search)"
    
    # STEP B: If no data match, use AI as the backup brain
    try:
        api_key = st.secrets.get("GEMINI_API_KEY")
        client = genai.Client(api_key=api_key.strip())
        res = client.models.generate_content(
            model="gemini-1.5-flash", 
            contents=f"Simplify this medical term: {text}"
        )
        return res.text.strip(), "🤖 Neural AI Engine"
    except:
        return text, "⚠️ No training match found."

# --- 4. UI ---
st.title("🏥 MedSimplify: Data-Driven NLP")

user_input = st.text_input("Search Medical Term (e.g., ductal carcinoma):")

if user_input:
    with st.spinner("Searching knowledge base..."):
        output, source = simplify_logic(user_input)
        
        st.markdown("### Output:")
        st.success(output)
        st.info(f"Source: {source}")

        # Show the NLP process in action
        doc = nlp(user_input)
        st.write("**NLP Breakdown:**")
        cols = st.columns(len(doc))
        for i, token in enumerate(doc):
            cols[i].caption(f"{token.text}\n({token.pos_})")
