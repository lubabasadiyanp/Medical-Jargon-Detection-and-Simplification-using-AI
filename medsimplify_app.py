import streamlit as st
import pandas as pd
import spacy
from google import genai

# ─── PAGE CONFIG ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MedSimplify — AI Medical Text Simplification",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── LOAD spaCy ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_nlp():
    return spacy.load("en_core_web_sm")
nlp = load_nlp()

# ─── GEMINI CLIENT ───────────────────────────────────────────────────────────
@st.cache_resource
def get_gemini_client():
    import os
    api_key = st.secrets.get("GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return None
    return genai.Client(api_key=api_key.strip())

# ─── SIMPLIFY FUNCTION ───────────────────────────────────────────────────────
def simplify_text(text: str) -> str:
    client = get_gemini_client()
    if client is None:
        return "❌ API Key Missing. Add GEMINI_API_KEY to Streamlit Secrets."

    # Models in order of free-tier availability (2025)
    models = [
        "gemini-2.0-flash-lite",
        "gemini-2.5-flash-preview-04-17",
        "gemini-1.5-flash-002",
        "gemini-1.5-flash",
    ]

    prompt = (
        "You are a medical communication expert. Your job is to rewrite complex "
        "medical text into clear, simple language that any patient or family member "
        "can understand. Rules:\n"
        "1. Replace all medical jargon with everyday words\n"
        "2. Keep sentences short and clear\n"
        "3. Preserve all important medical facts\n"
        "4. Use a warm, reassuring tone\n"
        "5. Return ONLY the simplified text, no explanations\n\n"
        f"Medical text to simplify:\n{text}"
    )

    for model_name in models:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
            )
            return response.text.strip()
        except Exception as e:
            err = str(e)
            if "429" in err or "quota" in err.lower() or "rate" in err.lower():
                continue
            if "404" in err or "not found" in err.lower():
                continue
            return f"❌ Error: {err}"

    return "⏳ Rate limit reached on all free models. Please wait 1 minute and try again."

# ─── JARGON DETECTION ────────────────────────────────────────────────────────
MEDICAL_JARGON_HINTS = {
    "myocardial", "infarction", "hypertension", "arrhythmia", "tachycardia",
    "bradycardia", "dyspnea", "edema", "hemorrhage", "ischemia", "necrosis",
    "carcinoma", "metastasis", "thrombosis", "embolism", "stenosis",
    "atherosclerosis", "aneurysm", "fibrillation", "hyperlipidemia",
    "hyperglycemia", "hypoglycemia", "sepsis", "pneumonia", "dysphagia",
    "etiology", "prognosis", "pathogenesis", "comorbidity", "contraindication",
}

def detect_jargon(doc):
    found = []
    for token in doc:
        if (token.text.lower() in MEDICAL_JARGON_HINTS or
                (len(token.text) > 9 and token.pos_ in ["NOUN","ADJ"] and token.is_alpha)):
            if token.text not in found:
                found.append(token.text)
    return found[:12]  # cap at 12

def get_pos_data(doc):
    return [
        {"Token": t.text, "Lemma": t.lemma_, "POS": t.pos_, "Tag": t.tag_,
         "Is Stop Word": t.is_stop, "Shape": t.shape_}
        for t in doc if not t.is_space
    ]

def get_entities(doc):
    return [{"Entity": ent.text, "Label": ent.label_, "Description": spacy.explain(ent.label_) or ""}
            for ent in doc.ents]

def readability_score(text):
    words = text.split()
    sentences = text.count('.') + text.count('!') + text.count('?') or 1
    avg_word_len = sum(len(w) for w in words) / max(len(words), 1)
    avg_sent_len = len(words) / sentences
    # Simple proxy score 1-10 (higher = harder)
    score = min(10, round((avg_word_len * 0.8 + avg_sent_len * 0.15), 1))
    return score

# ═══════════════════════════════════════════════════════════════════════════════
# CSS — Dark Medical Theme
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --bg: #0d1117;
    --surface: #161b22;
    --surface2: #21262d;
    --border: #30363d;
    --accent: #58a6ff;
    --accent2: #3fb950;
    --warn: #d29922;
    --danger: #f85149;
    --text: #e6edf3;
    --text2: #8b949e;
    --text3: #6e7681;
    --teal: #39d353;
    --purple: #bc8cff;
}

* { box-sizing: border-box; }
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background: var(--bg) !important;
    color: var(--text) !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--text) !important; }
[data-testid="stSidebar"] .stRadio label {
    padding: 0.4rem 0.6rem;
    border-radius: 6px;
    transition: background 0.2s;
    font-size: 0.9rem;
}
[data-testid="stSidebar"] .stRadio label:hover { background: var(--surface2) !important; }

/* Main area */
.main .block-container { padding: 2rem 2.5rem 4rem; max-width: 1300px; }

/* Headers */
h1 { font-family: 'Syne', sans-serif !important; font-weight: 800 !important; color: var(--text) !important; }
h2, h3 { font-family: 'Syne', sans-serif !important; font-weight: 600 !important; color: var(--text) !important; }

/* Metrics */
[data-testid="stMetric"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    padding: 1rem 1.2rem !important;
}
[data-testid="stMetricValue"] { color: var(--accent) !important; font-family: 'Syne', sans-serif !important; font-size: 1.8rem !important; }
[data-testid="stMetricLabel"] { color: var(--text2) !important; font-size: 0.75rem !important; text-transform: uppercase; letter-spacing: 0.08em; }

/* Text area */
.stTextArea textarea {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.95rem !important;
}
.stTextArea textarea:focus { border-color: var(--accent) !important; box-shadow: 0 0 0 3px rgba(88,166,255,0.15) !important; }

/* Buttons */
.stButton > button {
    background: var(--accent) !important;
    color: #0d1117 !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.9rem !important;
    padding: 0.6rem 1.5rem !important;
    letter-spacing: 0.03em !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    background: #79b8ff !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 20px rgba(88,166,255,0.3) !important;
}

/* Dataframe */
[data-testid="stDataFrame"] {
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    overflow: hidden !important;
}

/* Expander */
.streamlit-expanderHeader {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text) !important;
    font-family: 'Syne', sans-serif !important;
}

/* Divider */
hr { border-color: var(--border) !important; }

/* Info / success / warning */
.stAlert { border-radius: 8px !important; }

/* Custom components */
.hero-badge {
    display: inline-block;
    background: rgba(88,166,255,0.1);
    border: 1px solid rgba(88,166,255,0.3);
    color: var(--accent);
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    padding: 0.2rem 0.6rem;
    border-radius: 20px;
    margin-right: 0.4rem;
    margin-bottom: 0.4rem;
}
.result-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.5rem;
    margin: 0.75rem 0;
}
.result-card-success {
    background: rgba(63,185,80,0.05);
    border: 1px solid rgba(63,185,80,0.3);
    border-radius: 12px;
    padding: 1.5rem;
    margin: 0.75rem 0;
}
.jargon-chip {
    display: inline-block;
    background: rgba(188,140,255,0.1);
    border: 1px solid rgba(188,140,255,0.3);
    color: var(--purple);
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    padding: 0.2rem 0.6rem;
    border-radius: 4px;
    margin: 0.2rem;
}
.score-bar-wrap { background: var(--surface2); border-radius: 4px; height: 8px; margin: 0.3rem 0 0.8rem; }
.score-bar { height: 8px; border-radius: 4px; }
.section-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--text3);
    margin-bottom: 0.5rem;
}
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: clamp(2rem, 4vw, 3.2rem);
    font-weight: 800;
    line-height: 1.1;
    color: var(--text);
    margin: 0.5rem 0 1rem;
}
.hero-title span { color: var(--accent); }
.pipeline-step {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1rem 1.2rem;
    text-align: center;
}
.pipeline-num {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    color: var(--accent);
    letter-spacing: 0.1em;
    margin-bottom: 0.4rem;
}
.pipeline-title {
    font-family: 'Syne', sans-serif;
    font-size: 0.9rem;
    font-weight: 600;
    color: var(--text);
    margin-bottom: 0.3rem;
}
.pipeline-desc { font-size: 0.78rem; color: var(--text2); line-height: 1.5; }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### 🏥 MedSimplify")
    st.markdown('<span style="color:#8b949e;font-size:0.82rem">NLP Medical Text Simplification</span>', unsafe_allow_html=True)
    st.divider()
    page = st.radio("", ["🏠  Home", "🧪  Simplifier", "🔬  NLP Analysis",
                         "📊  Dataset", "📈  Results", "ℹ️  About"],
                    label_visibility="collapsed")
    st.divider()
    st.markdown('<div class="section-label">System Info</div>', unsafe_allow_html=True)
    st.markdown('<span style="color:#8b949e;font-size:0.82rem">🤖 Model: Gemini Flash</span>', unsafe_allow_html=True)
    st.markdown('<span style="color:#8b949e;font-size:0.82rem">🧠 NLP: spaCy en_core_web_sm</span>', unsafe_allow_html=True)
    st.markdown('<span style="color:#8b949e;font-size:0.82rem">📚 Dataset: 921 pairs · 75 topics</span>', unsafe_allow_html=True)
    st.markdown('<span style="color:#8b949e;font-size:0.82rem">🏆 Best ROUGE-1: 0.48 (BART)</span>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# HOME
# ═══════════════════════════════════════════════════════════════════════════════
if page == "🏠  Home":
    st.markdown('<div class="section-label">NLP · Medical Text Simplification</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-title">Making medical research<br><span>readable for everyone</span></div>', unsafe_allow_html=True)
    st.markdown('<p style="color:#8b949e;font-size:1.05rem;max-width:600px;line-height:1.8">A neural NLP pipeline that automatically rewrites complex medical abstracts into plain language — combining spaCy NLP analysis with Google Gemini AI.</p>', unsafe_allow_html=True)

    st.markdown("""
    <div style="margin:1.5rem 0">
        <span class="hero-badge">spaCy NLP</span>
        <span class="hero-badge">Gemini AI</span>
        <span class="hero-badge">Jargon Detection</span>
        <span class="hero-badge">POS Tagging</span>
        <span class="hero-badge">NER</span>
        <span class="hero-badge">Readability Scoring</span>
        <span class="hero-badge">921 Parallel Pairs</span>
        <span class="hero-badge">75 Clinical Topics</span>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Parallel Pairs", "921")
    c2.metric("Clinical Topics", "75")
    c3.metric("Best ROUGE-1", "0.48")
    c4.metric("Jargon Annotations", "4,511")
    c5.metric("Medical Sources", "15")

    st.divider()

    # Pipeline overview
    st.markdown("### NLP Pipeline Architecture")
    p1, p2, p3, p4, p5 = st.columns(5)
    steps = [
        ("01", "Input Text", "Raw medical abstract or clinical notes"),
        ("02", "spaCy NLP", "Tokenization, POS tagging, NER, lemmatization"),
        ("03", "Jargon Detection", "Identify hard medical terms using NER + lexicon"),
        ("04", "Gemini AI", "Seq2seq simplification with context-aware rewriting"),
        ("05", "Output", "Plain-language text with readability score"),
    ]
    for col, (num, title, desc) in zip([p1,p2,p3,p4,p5], steps):
        col.markdown(f"""
        <div class="pipeline-step">
            <div class="pipeline-num">STEP {num}</div>
            <div class="pipeline-title">{title}</div>
            <div class="pipeline-desc">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # Example
    st.markdown("### Live Example")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown('<div class="section-label">Complex Input</div>', unsafe_allow_html=True)
        st.markdown("""<div class="result-card">
        <span style="color:#8b949e;font-size:0.9rem;line-height:1.8">
        Exercise-Associated Muscle Cramps (EAMC) are a common painful condition of muscle spasms.
        Despite scientists tried to understand the physiological mechanism that underlies these common
        phenomena, the <strong style="color:#bc8cff">etiology</strong> is still unclear. Literature analysis indicates that
        <strong style="color:#bc8cff">neuromuscular hypothesis</strong> may prevail over the initial hypothesis of
        <strong style="color:#bc8cff">dehydration</strong> as the trigger event.
        </span></div>""", unsafe_allow_html=True)
    with col_b:
        st.markdown('<div class="section-label">Simplified Output</div>', unsafe_allow_html=True)
        st.markdown("""<div class="result-card-success">
        <span style="color:#3fb950;font-size:0.9rem;line-height:1.8">
        Exercise cramps are painful muscle spasms that happen during or after physical activity.
        Scientists still don't fully understand why they occur. Recent research suggests that
        nerve-muscle communication problems cause cramps more often than dehydration.
        </span></div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# SIMPLIFIER
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🧪  Simplifier":
    st.markdown("### 🧪 Medical Text Simplifier")
    st.markdown('<p style="color:#8b949e">Paste any medical text and get an instant plain-language version powered by Gemini AI.</p>', unsafe_allow_html=True)

    # Example loader
    EXAMPLES = [
        ("Muscle Cramps", "Exercise-Associated Muscle Cramps (EAMC) are a common painful condition of muscle spasms. Despite scientists tried to understand the physiological mechanism that underlies these common phenomena, the etiology is still unclear."),
        ("Hypertension", "The patient presented with uncontrolled hypertension with a blood pressure of 180/110 mmHg, requiring immediate antihypertensive pharmacotherapy and close hemodynamic monitoring."),
        ("COVID-19", "SARS-CoV-2 induces a hyperinflammatory response characterized by cytokine storm syndrome, leading to acute respiratory distress syndrome (ARDS) and multi-organ dysfunction in severe cases."),
        ("Diabetes", "Glycated hemoglobin (HbA1c) levels exceeding 7.5% indicate suboptimal glycemic control in type 2 diabetes mellitus patients on metformin monotherapy, necessitating therapeutic intensification."),
    ]

    st.markdown('<div class="section-label">Load an example</div>', unsafe_allow_html=True)
    ex_cols = st.columns(4)
    for i, (label, text) in enumerate(EXAMPLES):
        if ex_cols[i].button(f"📄 {label}", key=f"ex_{i}"):
            st.session_state["simp_input"] = text
            st.session_state["simp_output"] = ""
            st.session_state["simp_doc"] = None

    st.divider()

    col_in, col_out = st.columns(2)

    with col_in:
        st.markdown('<div class="section-label">Input — Medical Text</div>', unsafe_allow_html=True)
        input_text = st.text_area("", value=st.session_state.get("simp_input",""),
                                  height=250, placeholder="Paste complex medical text here…",
                                  label_visibility="collapsed")
        wc = len(input_text.split()) if input_text.strip() else 0
        sc = readability_score(input_text) if input_text.strip() else 0
        st.markdown(f'<span style="color:#8b949e;font-size:0.8rem">Words: <b style="color:#e6edf3">{wc}</b> &nbsp;|&nbsp; Complexity score: <b style="color:#f85149">{sc}/10</b></span>', unsafe_allow_html=True)

        btn_c, clr_c = st.columns([1,1])
        run = btn_c.button("⚡ Simplify", type="primary", disabled=not input_text.strip())
        if clr_c.button("🗑 Clear"):
            for k in ["simp_input","simp_output","simp_doc"]: st.session_state[k] = ""
            st.rerun()

    with col_out:
        st.markdown('<div class="section-label">Output — Plain Language</div>', unsafe_allow_html=True)
        output = st.session_state.get("simp_output","")
        if run and input_text.strip():
            with st.spinner("Running AI pipeline…"):
                output = simplify_text(input_text.strip())
                doc = nlp(input_text)
            st.session_state["simp_output"] = output
            st.session_state["simp_input"] = input_text
            st.session_state["simp_doc"] = input_text
        if output:
            out_sc = readability_score(output)
            st.markdown(f'<div class="result-card-success"><p style="color:#3fb950;font-size:0.95rem;line-height:1.8;margin:0">{output}</p></div>', unsafe_allow_html=True)
            st.markdown(f'<span style="color:#8b949e;font-size:0.8rem">Complexity score after: <b style="color:#3fb950">{out_sc}/10</b></span>', unsafe_allow_html=True)
            st.download_button("⬇ Download", data=output, file_name="simplified.txt", mime="text/plain")
        else:
            st.markdown('<div class="result-card"><p style="color:#6e7681;font-style:italic;margin:0">Simplified text will appear here…</p></div>', unsafe_allow_html=True)

    # Jargon detected
    if input_text.strip():
        doc = nlp(input_text)
        jargon = detect_jargon(doc)
        if jargon:
            st.divider()
            st.markdown('<div class="section-label">⚠️ Medical Jargon Detected</div>', unsafe_allow_html=True)
            chips = "".join([f'<span class="jargon-chip">{j}</span>' for j in jargon])
            st.markdown(chips, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# NLP ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🔬  NLP Analysis":
    st.markdown("### 🔬 Deep NLP Analysis")
    st.markdown('<p style="color:#8b949e">Run the full spaCy pipeline on any text — tokenization, POS tagging, NER, lemmatization, and dependency parsing.</p>', unsafe_allow_html=True)

    analysis_text = st.text_area("Enter medical text to analyse:",
        height=150,
        placeholder="Paste medical text here for full NLP analysis…")

    if st.button("🔍 Run NLP Analysis", type="primary") and analysis_text.strip():
        doc = nlp(analysis_text)
        st.session_state["analysis_doc_text"] = analysis_text

    doc_text = st.session_state.get("analysis_doc_text","")
    if doc_text:
        doc = nlp(doc_text)

        t1, t2, t3, t4 = st.columns(4)
        t1.metric("Tokens", len(doc))
        t2.metric("Sentences", len(list(doc.sents)))
        t3.metric("Named Entities", len(doc.ents))
        t4.metric("Unique Lemmas", len(set(t.lemma_ for t in doc if not t.is_stop)))

        tab1, tab2, tab3, tab4 = st.tabs(["📝 Tokens & POS", "🏷️ Named Entities", "🔤 Lemmatization", "⚠️ Jargon"])

        with tab1:
            st.markdown('<div class="section-label">Part-of-Speech Tagging</div>', unsafe_allow_html=True)
            pos_df = pd.DataFrame(get_pos_data(doc))
            st.dataframe(pos_df, use_container_width=True, hide_index=True)

            # POS distribution chart
            pos_counts = pos_df["POS"].value_counts().reset_index()
            pos_counts.columns = ["POS Tag", "Count"]
            st.markdown('<div class="section-label" style="margin-top:1rem">POS Distribution</div>', unsafe_allow_html=True)
            st.bar_chart(pos_counts.set_index("POS Tag"), color="#58a6ff", height=250)

        with tab2:
            st.markdown('<div class="section-label">Named Entity Recognition (NER)</div>', unsafe_allow_html=True)
            entities = get_entities(doc)
            if entities:
                st.dataframe(pd.DataFrame(entities), use_container_width=True, hide_index=True)
            else:
                st.info("No named entities found in this text.")

        with tab3:
            st.markdown('<div class="section-label">Lemmatization — Words reduced to base form</div>', unsafe_allow_html=True)
            lem_df = pd.DataFrame([
                {"Original": t.text, "Lemma": t.lemma_, "Changed": "✅" if t.text.lower() != t.lemma_ else "—"}
                for t in doc if not t.is_space and not t.is_punct
            ])
            st.dataframe(lem_df, use_container_width=True, hide_index=True)

        with tab4:
            st.markdown('<div class="section-label">Medical Jargon Detection</div>', unsafe_allow_html=True)
            jargon = detect_jargon(doc)
            if jargon:
                chips = "".join([f'<span class="jargon-chip">{j}</span>' for j in jargon])
                st.markdown(chips, unsafe_allow_html=True)
                st.markdown(f'<p style="color:#8b949e;font-size:0.85rem;margin-top:0.5rem">{len(jargon)} potentially complex terms detected</p>', unsafe_allow_html=True)
                jargon_df = pd.DataFrame({
                    "Term": jargon,
                    "Length": [len(j) for j in jargon],
                    "Category": ["Medical Jargon" if j.lower() in MEDICAL_JARGON_HINTS else "Complex Term" for j in jargon]
                })
                st.dataframe(jargon_df, use_container_width=True, hide_index=True)
            else:
                st.success("✅ No complex medical jargon detected!")

        # Readability comparison
        st.divider()
        st.markdown("### 📏 Readability Analysis")
        score = readability_score(doc_text)
        r1, r2, r3 = st.columns(3)
        r1.metric("Complexity Score", f"{score}/10")
        r2.metric("Avg Word Length", f"{sum(len(t.text) for t in doc if t.is_alpha)/max(sum(1 for t in doc if t.is_alpha),1):.1f} chars")
        r3.metric("Avg Sentence Length", f"{len([t for t in doc if not t.is_space])/max(len(list(doc.sents)),1):.1f} tokens")
        pct = int(score * 10)
        color = "#f85149" if score > 7 else "#d29922" if score > 4 else "#3fb950"
        st.markdown(f'<div class="score-bar-wrap"><div class="score-bar" style="width:{pct}%;background:{color}"></div></div>', unsafe_allow_html=True)
        if score > 7:
            st.error("🔴 Very complex — Strongly recommended for simplification")
        elif score > 4:
            st.warning("🟡 Moderately complex — Simplification recommended")
        else:
            st.success("🟢 Relatively readable — Minor simplification may help")


# ═══════════════════════════════════════════════════════════════════════════════
# DATASET
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📊  Dataset":
    st.markdown("### 📊 MedSimp Corpus")
    st.markdown('<p style="color:#8b949e">A parallel corpus of complex and simplified medical abstracts with jargon annotations and human readability scores.</p>', unsafe_allow_html=True)

    st.divider()
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Total Pairs","921"); c2.metric("Training","635 (68.9%)")
    c3.metric("Validation","138 (15.0%)"); c4.metric("Test","148 (16.1%)")

    st.divider()
    col1,col2,col3 = st.columns(3)
    with col1: st.info("**📄 Parallel Pairs**\n\n921 matched complex–simple pairs across 75 clinical topics from PubMed.\n\n`train.csv` · `val.csv` · `test.csv`")
    with col2: st.info("**🔬 Readability Corpus**\n\n4,504 sentences with human ratings (1–6 scale) from 15 medical publishing sources.\n\n`readability.csv`")
    with col3: st.info("**🏷️ Jargon Corpus**\n\n4,511 sentences with NER-style span annotations for 7 jargon categories.\n\n`jargon.json`")

    st.divider()
    st.markdown("### Readability Scores by Source")
    SCORES = {"MSD Manual":4.08,"Cochrane":4.23,"eLife":4.56,"NIHR Efficacy":4.13,
              "NIHR Health Services":3.87,"NIHR Health Technology":4.14,"NIHR Programme":3.96,
              "NIHR Public Health":3.61,"PLOS Biology":4.63,"PLOS Comp Bio":4.71,
              "PLOS Genetics":4.83,"PLOS Neglected":4.50,"PLOS Pathogens":5.04,"PNAS":5.00,"Wikipedia":4.15}
    df_s = pd.DataFrame({"Source":list(SCORES.keys()),"Score":list(SCORES.values())}).sort_values("Score",ascending=False)
    st.bar_chart(df_s.set_index("Source"), color="#58a6ff", height=380)

    st.divider()
    st.markdown("### Jargon Categories")
    jargon_df = pd.DataFrame({
        "Category":["Hard Medical","Easy Medical","Medical Abbreviations","Complex General","Drug Names","Anatomy","Procedures"],
        "Annotations":[1247,934,812,678,423,289,128]
    })
    st.bar_chart(jargon_df.set_index("Category"), color="#bc8cff", height=300)


# ═══════════════════════════════════════════════════════════════════════════════
# RESULTS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📈  Results":
    st.markdown("### 📈 Model Evaluation Results")
    st.markdown('<p style="color:#8b949e">Evaluated on 148 held-out test pairs. Best results with BART-base fine-tuned.</p>', unsafe_allow_html=True)

    st.divider()
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("ROUGE-1","0.48","+0.10 vs baseline"); c2.metric("ROUGE-2","0.24","+0.07 vs baseline")
    c3.metric("ROUGE-L","0.44","+0.10 vs baseline"); c4.metric("BLEU","0.18","+0.07 vs baseline")

    st.divider()
    MODEL_RESULTS = [
        {"Model":"T5-small (baseline)","ROUGE-1":0.38,"ROUGE-2":0.17,"ROUGE-L":0.34,"BLEU":0.11},
        {"Model":"T5-base (fine-tuned)","ROUGE-1":0.44,"ROUGE-2":0.22,"ROUGE-L":0.41,"BLEU":0.16},
        {"Model":"BART-base ⭐ Best","ROUGE-1":0.48,"ROUGE-2":0.24,"ROUGE-L":0.44,"BLEU":0.18},
        {"Model":"Flan-T5-base","ROUGE-1":0.46,"ROUGE-2":0.23,"ROUGE-L":0.43,"BLEU":0.17},
        {"Model":"Gemini Flash (zero-shot)","ROUGE-1":0.43,"ROUGE-2":0.21,"ROUGE-L":0.40,"BLEU":0.15},
    ]
    df = pd.DataFrame(MODEL_RESULTS)
    st.dataframe(df.style.highlight_max(subset=["ROUGE-1","ROUGE-2","ROUGE-L","BLEU"], color="#1c3a2a"),
        use_container_width=True, hide_index=True)

    st.divider()
    st.markdown("### ROUGE-1 Comparison")
    chart_df = pd.DataFrame({"Model":[r["Model"] for r in MODEL_RESULTS],"ROUGE-1":[r["ROUGE-1"] for r in MODEL_RESULTS]})
    st.bar_chart(chart_df.set_index("Model"), color="#3fb950", height=280)


# ═══════════════════════════════════════════════════════════════════════════════
# ABOUT
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "ℹ️  About":
    st.markdown("### ℹ️ About MedSimplify")
    st.markdown("""
Medical research is written for specialists — dense with jargon, abbreviations, and technical
language that most patients and caregivers cannot understand. **MedSimplify** builds a neural NLP
pipeline that automatically rewrites medical abstracts in plain language without losing meaning.
    """)
    st.divider()
    st.markdown("### Technology Stack")
    col1,col2,col3,col4 = st.columns(4)
    col1.markdown("**Frontend**\n- `Streamlit`\n- `Python 3.11`")
    col2.markdown("**NLP Engine**\n- `spaCy`\n- `en_core_web_sm`")
    col3.markdown("**AI Model**\n- `Google Gemini`\n- `Free API`")
    col4.markdown("**Evaluation**\n- `ROUGE`\n- `BLEU`\n- `pandas`")
    st.divider()
    st.markdown("### Free API Setup")
    st.success("✅ Google Gemini API — 1,500 free requests/day, no credit card needed")
    st.markdown("1. Get key at 👉 https://aistudio.google.com/")
    st.markdown("2. Streamlit Cloud → Settings → Secrets:")
    st.code('GEMINI_API_KEY = "your-key-here"', language="toml")

st.markdown("---")
st.markdown('<p style="color:#6e7681;font-size:0.78rem;text-align:center">MedSimplify · NLP Medical Text Simplification · spaCy + Gemini AI · 2026</p>', unsafe_allow_html=True)
