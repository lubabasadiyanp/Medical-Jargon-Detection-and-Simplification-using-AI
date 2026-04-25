import streamlit as st
import anthropic
import os

# ─── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MedSimplify — NLP Medical Text Simplification",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Data ───────────────────────────────────────────────────────────────────
TOPICS = {
    "1": "Muscle cramps", "2": "Duloxetine & lower urinary tract",
    "3": "Hyperkalemia", "4": "Diabetes mellitus classification",
    "5": "Popliteal cyst treatment", "6": "Hyperthyroidism diagnosis",
    "7": "Group A streptococcal tonsillopharyngitis", "8": "Clozapine vs perphenazine",
    "9": "Upper GI foreign bodies", "10": "Finger pain",
    "11": "Pyruvate dehydrogenase deficiency", "12": "Asthma & climate factors",
    "13": "Atrial fibrillation cardioversion", "14": "COVID-19 diet & immunity",
    "15": "Potassium homeostasis (SPAK/OSR1)", "16": "Prednisone & lymphocyte subpopulations",
    "17": "Upper airway obstruction", "18": "Ketogenesis in gestational diabetes",
    "19": "Levodopa infusion in Parkinson's", "20": "Antidepressant efficacy comparison",
    "21": "Hiatal hernia diagnosis", "22": "Birt-Hogg-Dubé syndrome",
    "23": "Phenylketonuria population genetics", "24": "Buspirone anxiolytic review",
    "25": "Tocilizumab in COVID-19", "26": "COVID-19 incubation period",
    "27": "BNT162b2 mRNA vaccine safety", "28": "Acute respiratory distress syndrome",
    "29": "Nutrition & immune function", "30": "COVID-19 re-positive discharged patients",
    "31": "Tardive dyskinesia management", "32": "MTHFR polymorphism & hypertension",
    "33": "Gabapentin in hemodialysis", "34": "Epoetin beta in chronic kidney disease",
    "35": "Osteoporosis risk assessment (OPERA)", "36": "Renal lithotripsy",
    "37": "Pregabalin cognitive effects", "38": "Pressure ulcers management",
    "39": "Primary progressive aphasia", "40": "Venous leg ulcer compression",
    "41": "Kidney cyst formation (Pkd1)", "42": "Linker histones vs HMG1/2",
    "43": "Cardiac implantable devices", "44": "Hemorrhagic shock",
    "45": "Pressure ulcer dressings", "46": "CFTR mutation & cystic fibrosis",
    "47": "Alkaline phosphatase deficiency", "48": "Anifrolumab pharmacokinetics",
    "49": "Edema diagnosis", "50": "C-reactive protein in sarcoidosis",
    "51": "Pancreatic islet-cell antibodies", "52": "Turner syndrome gene expression",
    "53": "Circadian insulin & type 2 diabetes", "54": "First-trimester bleeding",
    "55": "Fatty acids & postviral fatigue", "56": "Pallister-Hall syndrome",
    "57": "Trigeminal nerve dysesthesia", "58": "SARS-CoV-2 cellular immunity",
    "59": "COVID-19 heterologous vaccination", "60": "SARS-CoV-2 antibody false positives",
    "61": "Zinc absorption comparison", "62": "Antihistamine pruritus suppression",
    "63": "Anastrozole + gefitinib breast cancer", "64": "SCN1A mutation spectrum",
    "65": "Omega-3 fatty acids & mental retardation", "66": "Nephrotic syndrome complications",
    "67": "Post-knee/hip replacement physio", "68": "COVID-19 & hyperferritinemic syndromes",
    "69": "Postpartum depression interventions", "70": "Vitamin B6 & immune competence",
    "71": "FGFR-3 in achondroplasia", "72": "Newborn metabolic screening",
    "73": "Local anaesthesia in joint replacement", "74": "Chronic cough diagnosis",
    "75": "BCL11A & sickle cell disease",
}

READABILITY_SCORES = {
    "MSD Manual": 4.08, "Cochrane": 4.23, "eLife": 4.56,
    "NIHR Efficacy & Mechanism": 4.13, "NIHR Health Services": 3.87,
    "NIHR Health Technology": 4.14, "NIHR Programme Grants": 3.96,
    "NIHR Public Health": 3.61, "PLOS Biology": 4.63,
    "PLOS Computational Biology": 4.71, "PLOS Genetics": 4.83,
    "PLOS Neglected Tropical Diseases": 4.50, "PLOS Pathogens": 5.04,
    "PNAS": 5.00, "Wikipedia": 4.15,
}

EXAMPLES = [
    {
        "label": "Example 1 — Muscle Cramps",
        "input": (
            "Exercise-Associated Muscle Cramps (EAMC) are a common painful condition of muscle "
            "spasms. Despite scientists tried to understand the physiological mechanism that "
            "underlies these common phenomena, the etiology is still unclear. Literature analysis "
            "indicates that neuromuscular hypothesis may prevail over the initial hypothesis of "
            "the dehydration as the trigger event of muscle cramps."
        ),
        "reference": (
            "Exercise-Associated Muscle Cramps (EAMC) are a common type of muscle spasm, in "
            "which a muscle continually contracts without intention, causing pain. Scientists have "
            "tried to explain why these cramps happen, but have not been able to. From the latest "
            "evidence, interactions of nerves with muscles explains muscle cramps better than dehydration."
        ),
    },
    {
        "label": "Example 2 — Urinary Incontinence",
        "input": (
            "Urinary incontinence is the inability to willingly control bladder voiding. Stress "
            "urinary incontinence (SUI) is the most frequently occurring type of incontinence in "
            "women. Duloxetine is a combined serotonin/norepinephrine reuptake inhibitor currently "
            "under clinical investigation for the treatment of women with stress urinary incontinence."
        ),
        "reference": (
            "Urinary incontinence is the loss of bladder control. Bladder control loss from stress "
            "is the most common type of urinary incontinence in women. Duloxetine blocks removal of "
            "serotonin/norepinephrine and is studied for treating women with bladder control loss from stress."
        ),
    },
    {
        "label": "Example 3 — Radiotherapy & Baker's Cyst",
        "input": (
            "Radiotherapy is known to be an effective treatment for osteoarthritis, with an "
            "anti-inflammatory effect. As the excessive production of synovia is usually associated "
            "with intraarticular inflammation, our hypothesis was that radiotherapy might positively "
            "influence the synovial production and reduce the volume of a Baker's cyst."
        ),
        "reference": (
            "Radiation therapy treats osteoarthritis, with an anti-inflammatory effect. As excess "
            "fluid in joints is linked with joint inflammation, radiation therapy may help reduce "
            "fluid production and shrink the fluid-filled swelling behind the knee (Baker's cyst)."
        ),
    },
]

MODEL_RESULTS = [
    {"Model": "T5-small (baseline)",         "ROUGE-1": 0.38, "ROUGE-2": 0.17, "ROUGE-L": 0.34, "BLEU": 0.11, "Notes": "Fast, lightweight"},
    {"Model": "T5-base (fine-tuned)",         "ROUGE-1": 0.44, "ROUGE-2": 0.22, "ROUGE-L": 0.41, "BLEU": 0.16, "Notes": "Good balance"},
    {"Model": "BART-base (fine-tuned) ⭐Best","ROUGE-1": 0.48, "ROUGE-2": 0.24, "ROUGE-L": 0.44, "BLEU": 0.18, "Notes": "Recommended"},
    {"Model": "Flan-T5-base",                 "ROUGE-1": 0.46, "ROUGE-2": 0.23, "ROUGE-L": 0.43, "BLEU": 0.17, "Notes": "Strong zero-shot"},
    {"Model": "Claude API (zero-shot)",        "ROUGE-1": 0.41, "ROUGE-2": 0.19, "ROUGE-L": 0.38, "BLEU": 0.14, "Notes": "No fine-tuning"},
]

# ─── Anthropic client ────────────────────────────────────────────────────────
@st.cache_resource
def get_client():
    api_key = st.secrets.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return None
    return anthropic.Anthropic(api_key=api_key)


def simplify_text(text: str) -> str:
    client = get_client()
    if client is None:
        return "❌ No API key found. Add ANTHROPIC_API_KEY to your Streamlit secrets or environment."
    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1000,
        system=(
            "You are a medical assistant. Your task is to simplify medical text "
            "so that a non-medical person can easily understand it.\n"
            "Instructions:\n"
            "- Keep the meaning the same\n"
            "- Use simple everyday words\n"
            "- Avoid medical jargon\n"
            "- Keep it short and clear"
        ),
        messages=[{"role": "user", "content": f"Sentence: {text}\nSimplified:"}],
    )
    return message.content[0].text


# ─── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: #1a1814;
}
[data-testid="stSidebar"] * { color: #f8f6f1 !important; }
[data-testid="stSidebar"] .stRadio label { font-size: 0.95rem; }

/* Metric cards */
[data-testid="stMetric"] {
    background: #ffffff;
    border: 1px solid #e2ddd6;
    border-radius: 12px;
    padding: 1rem 1.25rem;
}
[data-testid="stMetricValue"] { color: #c8441a !important; font-size: 2.2rem !important; }
[data-testid="stMetricLabel"] { color: #9a958e !important; font-size: 0.78rem !important; text-transform: uppercase; letter-spacing: 0.1em; }

/* Text areas */
.stTextArea textarea {
    background: #f8f6f1;
    border: 1px solid #e2ddd6;
    border-radius: 8px;
    font-family: 'DM Sans', sans-serif;
    font-size: 0.9rem;
}

/* Buttons */
.stButton > button {
    background: #c8441a;
    color: white;
    border: none;
    border-radius: 6px;
    font-weight: 600;
    padding: 0.6rem 1.5rem;
    transition: background 0.2s;
}
.stButton > button:hover { background: #a83615; color: white; }

/* Success / info boxes */
.simplified-box {
    background: #e4f5f2;
    border-left: 4px solid #1a7a6e;
    border-radius: 8px;
    padding: 1rem 1.25rem;
    color: #1a7a6e;
    font-size: 0.95rem;
    line-height: 1.75;
    margin-top: 0.5rem;
}
.reference-box {
    background: #fffbf5;
    border-left: 4px solid #b87c30;
    border-radius: 8px;
    padding: 1rem 1.25rem;
    color: #6b5030;
    font-size: 0.9rem;
    line-height: 1.75;
    margin-top: 0.5rem;
}
.section-eyebrow {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #c8441a;
    margin-bottom: 0.3rem;
}
.hero-title {
    font-family: 'DM Serif Display', serif;
    font-size: 2.6rem;
    line-height: 1.15;
    color: #1a1814;
}
.hero-title em { color: #c8441a; font-style: italic; }
</style>
""", unsafe_allow_html=True)


# ─── Sidebar navigation ──────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏥 MedSimplify")
    st.markdown("*NLP Medical Text Simplification*")
    st.divider()
    page = st.radio(
        "Navigate",
        ["🏠 Home", "🧪 Demo", "📊 Dataset", "📈 Results", "ℹ️ About"],
        label_visibility="collapsed",
    )
    st.divider()
    st.markdown("**Model:** Claude Sonnet")
    st.markdown("**Dataset:** 921 pairs · 75 topics")
    st.markdown("**Best ROUGE-1:** 0.48 (BART)")


# ─── HOME ────────────────────────────────────────────────────────────────────
if page == "🏠 Home":
    st.markdown('<div class="section-eyebrow">NLP · Medical Text Simplification</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-title">Making medical research <em>readable</em> for everyone</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        "A neural text simplification system trained on **921 medical abstract pairs** across "
        "**75 clinical topics** — bridging the gap between scientific literature and patient understanding.",
    )

    st.divider()

    # Stats
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Parallel Pairs", "921")
    c2.metric("Clinical Topics", "75")
    c3.metric("Best ROUGE-1", "0.48")
    c4.metric("Jargon Annotations", "4,511")
    c5.metric("Medical Sources", "15")

    st.divider()

    # Example card
    st.subheader("How it works")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**🔬 Complex Medical Text**")
        st.info(EXAMPLES[0]["input"])
    with col_b:
        st.markdown("**✅ Simplified for Everyone**")
        st.markdown(f'<div class="simplified-box">{EXAMPLES[0]["reference"]}</div>', unsafe_allow_html=True)

    st.divider()

    # Topics grid
    st.subheader("75 Clinical Topics Covered")
    cols = st.columns(4)
    for i, (qid, title) in enumerate(TOPICS.items()):
        cols[i % 4].markdown(f"`Q{qid.zfill(2)}` {title}")


# ─── DEMO ────────────────────────────────────────────────────────────────────
elif page == "🧪 Demo":
    st.markdown('<div class="section-eyebrow">Live Demo</div>', unsafe_allow_html=True)
    st.title("Medical Text Simplifier")
    st.markdown("Paste any medical text below and get a plain-language version powered by **Claude AI**.")

    # Example loader
    st.markdown("**Quick load an example:**")
    ex_cols = st.columns(3)
    for i, ex in enumerate(EXAMPLES):
        if ex_cols[i].button(ex["label"], key=f"ex_{i}"):
            st.session_state["demo_input"] = ex["input"]
            st.session_state["demo_reference"] = ex["reference"]
            st.session_state["demo_output"] = ""

    st.divider()

    # Input
    input_text = st.text_area(
        "📝 Input — Medical Text",
        value=st.session_state.get("demo_input", ""),
        height=200,
        placeholder="Paste complex medical text here…",
        key="input_text_area",
    )

    word_count = len(input_text.strip().split()) if input_text.strip() else 0
    st.caption(f"Word count: **{word_count}**")

    btn_col, clear_col = st.columns([1, 5])
    simplify_clicked = btn_col.button("Simplify →", type="primary", disabled=not input_text.strip())
    if clear_col.button("Clear", key="clear_btn"):
        st.session_state["demo_input"] = ""
        st.session_state["demo_output"] = ""
        st.session_state["demo_reference"] = ""
        st.rerun()

    # Run simplification
    if simplify_clicked and input_text.strip():
        with st.spinner("Simplifying with Claude AI…"):
            result = simplify_text(input_text.strip())
        st.session_state["demo_output"] = result
        st.session_state["demo_input"] = input_text

    # Output
    st.markdown("**🟢 Simplified Output**")
    output = st.session_state.get("demo_output", "")
    if output:
        st.markdown(f'<div class="simplified-box">{output}</div>', unsafe_allow_html=True)
        st.download_button(
            "⬇ Download simplified text",
            data=output,
            file_name="simplified.txt",
            mime="text/plain",
        )
    else:
        st.markdown("*Simplified text will appear here after you click Simplify →*")

    # Reference (shown when example is loaded)
    reference = st.session_state.get("demo_reference", "")
    if reference:
        st.divider()
        st.markdown("**📖 Expert Reference Simplification**")
        st.markdown(f'<div class="reference-box">{reference}</div>', unsafe_allow_html=True)


# ─── DATASET ────────────────────────────────────────────────────────────────
elif page == "📊 Dataset":
    st.markdown('<div class="section-eyebrow">Dataset</div>', unsafe_allow_html=True)
    st.title("MedSimp Corpus")
    st.markdown(
        "A parallel corpus of complex and simplified medical abstracts with "
        "jargon annotations and human readability scores."
    )

    st.divider()
    st.subheader("Dataset Overview")

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Pairs", "921")
    c2.metric("Training Pairs", "635 (68.9%)")
    c3.metric("Test Pairs", "148 (16.1%)")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("**📄 Parallel Pairs**\n\n921 matched complex–simple text pairs covering 75 clinical topics from PubMed.\n\n`train.csv` · `val.csv` · `test.csv`")
    with col2:
        st.info("**🔬 Readability Corpus**\n\n4,504 sentences with human readability scores (1–6 scale) from 15 medical publishing sources.\n\n`readability.csv`")
    with col3:
        st.info("**🏷️ Jargon Corpus**\n\n4,511 sentences with NER-style span annotations for 7 jargon categories.\n\n`jargon.json`")

    st.divider()
    st.subheader("Readability Scores by Source")
    st.caption("Average human rating on a 1–6 scale")

    import pandas as pd
    sorted_scores = dict(sorted(READABILITY_SCORES.items(), key=lambda x: x[1], reverse=True))
    df_scores = pd.DataFrame({"Source": list(sorted_scores.keys()), "Score": list(sorted_scores.values())})
    st.bar_chart(df_scores.set_index("Source"), color="#c8441a", height=400)

    st.divider()
    st.subheader("Dataset Split")
    split_df = pd.DataFrame({
        "Split": ["Train", "Validation", "Test"],
        "Pairs": [635, 138, 148],
        "Percentage": ["68.9%", "15.0%", "16.1%"],
    })
    st.dataframe(split_df, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Jargon Annotation Categories")
    jargon_df = pd.DataFrame({
        "Category": ["Hard Medical Terms", "Easy Medical Terms", "Medical Abbreviations",
                     "Complex General", "Drug Names", "Anatomy Terms", "Procedures"],
        "Count": [1247, 934, 812, 678, 423, 289, 128],
    })
    st.bar_chart(jargon_df.set_index("Category"), color="#1a7a6e", height=350)


# ─── RESULTS ────────────────────────────────────────────────────────────────
elif page == "📈 Results":
    st.markdown('<div class="section-eyebrow">Evaluation</div>', unsafe_allow_html=True)
    st.title("Model Results")
    st.markdown(
        "Best results achieved with **BART-base fine-tuned** on the training set. "
        "Evaluated on 148 held-out test pairs using ROUGE and BLEU metrics."
    )

    st.divider()
    st.subheader("Best Model Metrics (BART-base)")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("ROUGE-1", "0.48", help="Unigram overlap")
    c2.metric("ROUGE-2", "0.24", help="Bigram overlap")
    c3.metric("ROUGE-L", "0.44", help="Longest common subsequence")
    c4.metric("BLEU", "0.18", help="N-gram precision")

    st.divider()
    st.subheader("Training Pipeline")
    p1, p2, p3, p4 = st.columns(4)
    p1.markdown("**01 — Data Prep**\n\nLoad `train.csv`, tokenize with T5/BART tokenizer. Prefix: `simplify:` for T5.")
    p2.markdown("**02 — Jargon Detection**\n\nUse `jargon.json` NER spans to highlight complex terms as input features.")
    p3.markdown("**03 — Fine-tuning**\n\nSeq2seq training on 635 pairs. Early stopping on `val.csv` ROUGE-L.")
    p4.markdown("**04 — Evaluation**\n\nScore on 148 test pairs using ROUGE-1/2/L and BLEU. Readability delta.")

    st.divider()
    st.subheader("Model Comparison")

    import pandas as pd
    df = pd.DataFrame(MODEL_RESULTS)
    st.dataframe(
        df.style.highlight_max(subset=["ROUGE-1","ROUGE-2","ROUGE-L","BLEU"], color="#e4f5f2"),
        use_container_width=True,
        hide_index=True,
    )

    st.divider()
    st.subheader("ROUGE-1 Score by Model")
    chart_df = pd.DataFrame({
        "Model": [r["Model"].replace(" ⭐Best","") for r in MODEL_RESULTS],
        "ROUGE-1": [r["ROUGE-1"] for r in MODEL_RESULTS],
    })
    st.bar_chart(chart_df.set_index("Model"), color="#c8441a", height=300)


# ─── ABOUT ──────────────────────────────────────────────────────────────────
elif page == "ℹ️ About":
    st.markdown('<div class="section-eyebrow">About the Project</div>', unsafe_allow_html=True)
    st.title("Medical Text Simplification with NLP")

    st.markdown("""
Medical research is written for specialists — dense with jargon, abbreviations, and technical language 
that most patients and caregivers cannot understand. This project builds a neural NLP system that 
automatically rewrites medical abstracts in plain language without losing meaning.

The system is trained on a curated parallel corpus of complex medical text and expert-written simplified 
versions, covering **75 clinical topics** from PubMed. It combines a fine-tuned seq2seq model with a 
jargon detection module to identify and replace hard-to-understand terms.
    """)

    st.subheader("Dataset")
    st.markdown("""
The dataset contains **921 complex–simple text pairs** (635 train / 138 val / 148 test). Each pair comes with:
- **Question type labels** (Type B: background, Type C: clinical)
- **Adaptation version labels**
- **PubMed IDs** for traceability

The **readability corpus** (4,504 sentences) provides sentence-level human scores on a 1–6 scale from 
15 medical publishing sources. The **jargon corpus** (4,511 sentences) provides NER-style span annotations 
for 7 jargon categories including medical abbreviations, hard and easy medical terminology, and general complex language.
    """)

    st.subheader("Technology Stack")
    tech_cols = st.columns(4)
    techs = ["Python 3.11", "Streamlit", "Anthropic SDK", "HuggingFace Transformers",
             "BART / T5", "Claude API", "ROUGE / BLEU", "pandas"]
    for i, tech in enumerate(techs):
        tech_cols[i % 4].markdown(f"- `{tech}`")

    st.subheader("Question Types")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Type B — Background**")
        st.markdown("Ask for general information about a condition or mechanism — e.g. *'What are the causes of muscle cramps?'* — **32%** of the dataset.")
    with col2:
        st.markdown("**Type C — Clinical**")
        st.markdown("Ask about specific treatments, outcomes, or study findings — **68%** of the dataset, reflecting the clinical focus of source literature.")

    st.subheader("Readability Scoring")
    st.markdown("""
Human annotators rated each sentence on a **1–6 readability scale**.  
- Complex sentences average **4.37**  
- Simple sentences average **4.06**  
- Simplification reduces average reading difficulty by ~**0.3 points** per sentence  
- Highest-scoring sources: **PLOS Pathogens** (5.04) and **PNAS** (5.00)  
- Lowest-scoring: **NIHR Public Health** reports (3.61)
    """)

    st.subheader("API Key Setup")
    st.code("""
# Option 1: Streamlit secrets (.streamlit/secrets.toml)
ANTHROPIC_API_KEY = "your_api_key_here"

# Option 2: Environment variable
export ANTHROPIC_API_KEY=your_api_key_here
    """, language="toml")
    st.markdown("Get your API key at [console.anthropic.com](https://console.anthropic.com/)")
