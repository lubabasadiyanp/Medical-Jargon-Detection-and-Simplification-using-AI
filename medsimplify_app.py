
import streamlit as st
import pandas as pd
import json
import spacy
import re
from nltk.stem 
import WordNetLemmatizer
import nltk


# These lines ensure the data is downloaded on the Streamlit server
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')
    nltk.download('omw-1.4')
# Download necessary NLTK data
nltk.download('wordnet')
nltk.download('omw-1.4')

# Load NLP model
try:
    nlp = spacy.load("en_core_web_sm")
except:
    import os
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

lemmatizer = WordNetLemmatizer()

# --- STEP 1: LOAD AND PREPROCESS DATASET ---
@st.cache_data
def load_dictionary():
    # We combine jargon mapping from your uploaded files
    medical_dict = {}
    
    # Example loading from jargon.json (based on your file structure)
    with open('jargon.json', 'r') as f:
        jargon_data = json.load(f)
        for item in jargon_data:
            if 'entities' in item:
                for entity in item['entities']:
                    # mapping complex token to a simpler placeholder or the "simple" version
                    # This logic assumes the entity list contains [start, end, label, [text]]
                    complex_term = entity[3][0].lower()
                    # In a real scenario, you'd map this to a target_text from your CSVs
                    medical_dict[complex_term] = "simplified version" 

    # Adding manual high-impact mappings from your data.json and CSVs
    # Examples based on the files you provided:
    manual_mappings = {
        "nephrotic syndrome": "kidney damage causing protein leak",
        "chylothorax": "lymph fluid buildup around lungs",
        "myeloid cells": "bone marrow immune cells",
        "vaso-occlusive crises": "painful blood vessel blockages",
        "cerebral vasculopathy": "brain blood vessel disease",
        "soma": "cell body",
        "edema": "swelling",
        "dyspnea": "shortness of breath"
    }
    medical_dict.update(manual_mappings)
    return medical_dict

medical_lookup = load_dictionary()

# --- STEP 2: SIMPLIFICATION LOGIC ---
def simplify_text(text):
    # NLC Preprocessing: Lemmatization
    doc = nlp(text)
    simplified_words = []
    
    for token in doc:
        # Get the base form of the word (Lemmatization)
        lemma = token.lemma_.lower()
        
        # Check if the word or its lemma is in our medical dictionary
        if lemma in medical_lookup:
            simplified_words.append(f"**{medical_lookup[lemma]}** ({token.text})")
        elif token.text.lower() in medical_lookup:
            simplified_words.append(f"**{medical_lookup[token.text.lower()]}** ({token.text})")
        else:
            simplified_words.append(token.text)
            
    return " ".join(simplified_words)

# --- STEP 3: STREAMLIT UI ---
st.set_page_config(page_title="MedSimplify", page_icon="🏥")

st.title("🏥 MedSimplify AI")
st.markdown("Transform complex medical jargon into plain English using NLP.")

input_text = st.text_area("Enter medical notes or terms here:", 
                         placeholder="Patient presents with acute dyspnea and severe edema...")

if st.button("Simplify Now"):
    if input_text:
        with st.spinner('Processing medical terms...'):
            result = simplify_text(input_text)
            st.subheader("Simplified Interpretation:")
            st.write(result)
            
            st.info("💡 Note: This tool uses lemmatization to identify base medical terms.")
    else:
        st.warning("Please enter some text first!")

# --- STEP 4: DATA INSIGHTS (Optional) ---
if st.checkbox("Show dataset coverage"):
    st.write(f"Total unique medical terms in library: {len(medical_lookup)}")
    st.dataframe(pd.DataFrame(list(medical_lookup.items()), columns=["Complex Term", "Simple Term"]))
