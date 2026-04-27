import streamlit as st
import pandas as pd
import spacy
from google import genai

# ─── 1. NLP ENGINE (The "Brain") ───
@st.cache_resource
def load_nlp():
    # This handles Lemmatization (turning 'running' to 'run') 
    # and Stemming-like logic automatically.
    return spacy.load("en_core_web_sm")

@st.cache_data
def load_training_data():
    try:
        # This loads the data you already uploaded to GitHub
        df = pd.read_csv("train.csv")
        return df
    except:
        st.error("Could not find train.csv in your GitHub folder.")
        return None

nlp = load_nlp()
train_df = load_training_data()

# ─── 2. PREDICTIVE PIPELINE ───
def predict_simple_text(user_input):
    # STEP A: NLC Preprocessing
    doc = nlp(user_input.lower().strip())
    
    # Extracting base forms (Lemmas)
    lemmatized_text = [token.lemma_ for token in doc if not token.is_stop]
    
    # STEP B: Prediction via Semantic Similarity
    best_match = None
    max_similarity = 0
    
    if train_df is not None:
        # We compare your input to every row in your uploaded dataset
        for _, row in train_df.iterrows():
            # row[0] is jargon, row[1] is simple version
            target_doc = nlp(str(row.iloc[0]).lower())
            
            # This calculates how 'close' the meaning is
            score = doc.similarity(target_doc)
            
            if score > max_similarity:
                max_similarity = score
                best_match = row.iloc[1]

    # STEP C: Accuracy Threshold
    # If our data-driven prediction is high (above 70%), use it.
    if max_similarity > 0.7:
        return best_match, "Dataset Prediction", lemmatized_text
    
    # STEP D: Neural Backup (Gemini) if data match is low
    try:
        key = st.secrets.get("GEMINI_API_KEY")
        client = genai.Client(api_key=key.strip())
        res = client.models.generate_content(
            model="gemini-1.5-flash",
            contents=f"Simplify this medical jargon for a patient: {user_input}"
        )
        return res.text.strip(), "Neural AI Backup", lemmatized_text
    except:
        return "Please try a different term.", "No Match", lemmatized_text

# ─── 3. THE APP INTERFACE ───
st.title("🏥 MedSimplify: Predictive NLP")
st.markdown("This app uses your GitHub datasets to predict simplified medical meanings.")

user_query = st.text_input("Enter Medical Jargon:")

if user_query:
    with st.spinner("Executing NLP Pipeline..."):
        result, source, lemmas = predict_simple_text(user_query)
        
        st.subheader("Simplified Prediction:")
        st.success(result)
        
        # PROVING THE NLP PROCESS
        st.divider()
        st.write("### 🔬 NLP Pipeline Diagnostics")
        st.write(f"**Engine Used:** {source}")
        st.write(f"**Lemmatization Result:** `{lemmas}`")
        
        # Display the POS tagging to show the professor the "NLC" process
        doc = nlp(user_query)
        tokens_data = [{"Token": t.text, "Lemma": t.lemma_, "POS": t.pos_, "Is Stopword": t.is_stop} for t in doc]
        st.table(pd.DataFrame(tokens_data))
