# Medical-Jargon-Detection-and-Simplification-using-AI
# 🩺 Medical Jargon Detection and Simplification

A Streamlit web app for the full NLP research pipeline.  
**Free deployment · 3 collaborators · HuggingFace Spaces**

---

## 🚀 Quick Start (3 steps)

### Step 1 — Clone & run locally
```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd YOUR_REPO
pip install -r requirements.txt
streamlit run app.py
```

### Step 2 — Add your data files
Place these in the repo root:
- `train.csv`
- `val.csv`
- `test.csv`
- `readability.csv`
- `data.json`

### Step 3 — Deploy free on HuggingFace Spaces
1. Go to https://huggingface.co → New Space
2. Choose **Streamlit** as the SDK
3. Link your GitHub repo
4. It auto-deploys! Your URL: `https://huggingface.co/spaces/YOUR_NAME/medical-nlp`

---

## 📋 App Pages

| Page | Description | Owner |
|------|-------------|-------|
| 🏠 Home | Project overview + quick demo | All |
| 📊 Data Explorer | Load and visualize all CSV/JSON data | Person 1 |
| 🔍 Jargon Detection | Binary/3-class/7-class detection + FKGL | Person 1 |
| 🔄 Cross-Dataset Eval | MedReadMe ↔ PLABA transfer analysis | Person 2 |
| 🤖 LLM Simplification | Jargon-aware vs baseline prompting | Person 2 |
| 📈 Evaluation Dashboard | SARI, BLEU, FKGL, BERTScore | Person 3 |
| 👥 Human Annotation | 1–5 rating interface + CSV export | Person 3 |
| 📖 About | Tech stack and references | All |

---

## 🔌 Connecting a Real LLM (Optional, Free)

### Option A: Groq API (fastest, free tier)
```bash
pip install groq
```
Add to HuggingFace Spaces secrets: `GROQ_API_KEY=your_key`

```python
from groq import Groq
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
response = client.chat.completions.create(
    model="llama3-8b-8192",
    messages=[{"role": "user", "content": prompt}]
)
```

### Option B: HuggingFace Inference API (free)
Add secret: `HF_TOKEN=your_token`
```python
import requests
API_URL = "https://api-inference.huggingface.co/models/meta-llama/Meta-Llama-3-8B-Instruct"
response = requests.post(API_URL, headers={"Authorization": f"Bearer {HF_TOKEN}"},
                         json={"inputs": prompt})
```

---

## 🤝 Team Workflow

### Person 1 — Data + Jargon Detection
- Upload and verify all CSV/JSON data
- Implement BERT/BioBERT NER fine-tuning (Colab)
- Test binary/3-class/7-class taxonomy pages

### Person 2 — Cross-dataset + LLM
- Run cross-dataset evaluation experiments
- Implement jargon-aware prompting via Groq/HF API
- Compare prompt strategies

### Person 3 — Evaluation + Deployment
- Add BERTScore and SARI implementations
- Set up human annotation workflow
- Deploy and manage HuggingFace Space

---

## 📁 Expected Data Format

**train.csv / val.csv / test.csv**
```
sentence,label,jargon_spans,dataset_source
"Patient had myocardial infarction...",1,"myocardial infarction",MedReadMe
```

**readability.csv**
```
sentence,fkgl,avg_token_length,dataset
"...",14.2,3.4,MedReadMe
```

---

## 🏗 Tech Stack (all free)

- **Streamlit** — web framework
- **HuggingFace Spaces** — free deployment
- **Groq API** — free LLM inference (Llama3)
- **GitHub** — version control and CI
- **Google Colab** — free GPU for model training
