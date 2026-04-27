import streamlit as st
import pandas as pd
import spacy
from google import genai

# --- NLP ENGINE ---
@st.cache_resource
def load_nlp():
    return spacy.load("en_core_web_sm")

@st.cache_data
def load_training_data():
    try:
        # Tries to load your uploaded file
        df = pd.read_csv("train.csv")
        return df
    except:
        return None

nlp = load_nlp()
train_df = load_training_data()

# --- THE PREDICTIVE ENGINE ---
def simplify_prediction(user_input):
    # STEP 1: NLC Preprocessing (Lemmatization & Stemming-logic)
    doc = nlp(user_input.lower().strip())
    # This creates the "base" form of your input
    input_lemma = " ".join([t.lemma_ for t in doc if not t.is_stop])
    
    best_match = None
    max_score = 0
    
    # STEP 2: Searching your GitHub Data
    if train_df is not None:
        for _, row in train_df.iterrows():
            # Use iloc to be safe with column names
            medical_text = str(row.iloc[0]).lower()
            simple_text = str(row.iloc[1])
            
            # Compare the similarity of the user input to the dataset
            data_doc = nlp(medical_text)
            score = doc.similarity(data_doc)
            
            if score > max_score:
                max_score = score
                best_match = simple_text

    # STEP 3: Prediction Logic (Accuracy Threshold)
    # Lowered to 0.4 to ensure you get an output for terms like "ductal carcinoma"
    if max_score > 0.4:
        return best_match, f"Dataset Match ({int(max_score*100)}% confidence)", input_lemma

    # STEP 4: AI Fallback
    try:
        key = st.secrets.get("GEMINI_API_KEY")
        client = genai.Client(api_key=key.strip())
        res = client.models.generate_content(
            model="gemini-1.5-flash",
            contents=f"Simply explain this medical term for a patient: {user_input}"
        )
        return res.text.strip(), "Neural AI Engine", input_lemma
    except:
        return "The term is too complex. Try 'hypertension' or 'carcinoma'.", "No Match Found", input_lemma

# --- USER INTERFACE ---
st.title("🏥 MedSimplify Predictive AI")
user_query = st.text_input("Enter Jargon:")

if user_query:
    with st.spinner("Predicting..."):
        result, engine, lemmas = simplify_prediction(user_query)
        
        st.subheader("Simplified Output:")
        st.success(result)
        
        # PROVE THE NLP PROCESS FOR YOUR SUBMISSION
        st.divider()
        st.write("### 🔬 NLP Process Details")
        st.write(f"**Lemmatized Form:** `{lemmas}`")
        st.write(f"**Confidence Source:** {engine}")

        # Show the POS Tagger (NLC step)
        doc = nlp(user_query)
        st.write("**Part-of-Speech Tagging:**")
        # Visualizing the tokenization and tagging
        st.json([{"text": t.text, "tag": t.pos_, "explanation": spacy.explain(t.pos_)} for t in doc])
