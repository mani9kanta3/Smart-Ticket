# 🎫 SmartTicket — Intelligent Customer Support Ticket Classifier & Priority Router

---

## 🌐 Live Demo

| Component | Link |
|:----------|:-----|
| **🖥️ Dashboard** | [smart-ticket.streamlit.app](https://smart-ticket.streamlit.app) |
| **⚡ API Docs** | [smartticket-363391410596.asia-south1.run.app/docs](https://smartticket-363391410596.asia-south1.run.app/docs) |
| **💻 GitHub** | [github.com/mani9kanta3/Smart-Ticket](https://github.com/mani9kanta3/Smart-Ticket) |

> The Streamlit dashboard connects to the FastAPI backend deployed on **Google Cloud Run (Mumbai, asia-south1)** for real-time predictions powered by **DistilBERT + ONNX Runtime**.

---

## The Problem

Every day, customer support teams across banking, fintech, and SaaS companies receive **thousands of support tickets**. Each one needs to be read by a human agent who must:

1. **Categorize it** — Is this a payment issue? A card problem? An account access request?
2. **Assign priority** — Is the customer locked out of their account (urgent)? Or just asking about exchange rates (low priority)?
3. **Route it** — Send it to the right team so the right expert handles it.

This manual process is slow, inconsistent, and expensive. Tickets get **mis-routed** (bouncing between teams for days), **mis-prioritized** (a fraud report sitting in the low-priority queue), and **inconsistently categorized** (one agent calls it "Payment Issues," another calls it "Card Services"). The result? Frustrated customers, burnt-out agents, and rising support costs.

**What if AI could do all three — instantly, consistently, and accurately?**

---

## The Solution: SmartTicket

SmartTicket is an **end-to-end AI-powered ticket intelligence system** that automatically classifies incoming support tickets into the correct category, assigns business-appropriate priority, and routes to the right team — all in under 30 milliseconds.

But SmartTicket isn't just a model — it's a **complete production system**:

- **3 progressively advanced ML/DL models** showing a clear engineering journey
- **Multi-task learning** — one model predicts both category AND priority simultaneously
- **ONNX-optimized inference** for production-grade speed
- **FastAPI backend** deployed on **Google Cloud Run** serving real-time and batch predictions
- **3-tab Streamlit dashboard** deployed on **Streamlit Community Cloud**, connected to the API
- **MLflow experiment tracking** with 30+ logged experiments
- **Evidently AI drift monitoring** to catch model degradation early
- **Docker containerization** and **CI/CD** via GitHub Actions

---

## The Journey: From Baseline to Production

### 📊 Starting Point — Understanding the Data

I started with two real-world datasets from HuggingFace:

- **Banking77** — 13,083 real customer banking queries across 77 intent categories
- **CLINC150** — 23,700 queries including 1,350 out-of-scope queries (things like "what movies are playing?" that aren't banking-related)

But 77 categories was too granular. Real support teams have ~10 departments, not 77. So I **consolidated 77 intents into 10 business-meaningful categories** — grouping labels by which team would actually handle them. For example, "card_payment_not_recognised," "card_payment_wrong_amount," and "declined_card_payment" all go to the same **Payment Issues** team.

I also created a **4-level priority system (P0–P3)** using both category-based rules and keyword detection. A query containing "stolen" or "fraud" gets P0-Critical regardless of category. A query about exchange rates gets P3-Low. This mimics real-world SLA systems used by Freshdesk, Zendesk, and ServiceNow.

### 🔬 Model 1: TF-IDF + Logistic Regression (Baseline)

**The question:** How well can we classify tickets using just word frequencies?

TF-IDF converts each query into a 10,000-dimensional vector where words that are informative for classification (like "stolen" or "refund") get high scores, while common words (like "the" or "is") get low scores. Logistic Regression then draws decision boundaries in this space.

**Result: 87.5% accuracy, 0.875 F1**

This became our **floor** — any deep learning model must beat this to justify its complexity. The baseline was surprisingly strong, but its main weakness was clear: it treats "payment not received" and "received payment confirmation" as nearly identical because it ignores word order.

### 🧠 Model 2: BiLSTM — Learning Word Order

**The question:** Does understanding word ORDER improve classification?

I built a Bidirectional LSTM from scratch in PyTorch — custom Dataset class, embedding layer, 2-layer BiLSTM with 256 hidden units, and multi-task classification heads. Unlike TF-IDF, the BiLSTM reads each query word by word (both forward and backward), maintaining a memory of what it's read.

**Result: 87.8% accuracy, 0.878 F1**

Only a marginal improvement. Why? With short queries (average 11 words) and only 11,546 training samples, the BiLSTM didn't have enough data to learn meaningful language patterns from scratch. This is like teaching a baby to understand English AND classify tickets simultaneously — it's asking too much with too little data.

**This motivated the key insight: we need transfer learning.**

### 🚀 Model 3: Fine-tuned DistilBERT — The Power of Pre-training

**The question:** What if the model already understands English?

DistilBERT was pre-trained on **billions of words** of English text. It already knows grammar, word meanings, context, and nuance. Instead of learning language from scratch, I just taught it our specific classification task — like hiring an expert who already speaks English fluently and training them on your company's ticket categories.

I chose DistilBERT over full BERT because it's **40% smaller and 60% faster** while retaining **97% of BERT's accuracy**. This was critical — it allowed me to fine-tune on my laptop's GTX 1650 Ti (4GB VRAM) with batch_size=8 and gradient accumulation.

I froze the first 4 of 6 transformer layers (preserving general language knowledge) and trained only the last 2 layers plus custom classification heads.

**Result: 90.6% accuracy, 0.906 F1**

A significant jump — **+3% over the baseline**. Every single category improved. The model correctly distinguishes "payment not received" (Payment Issues) from "fee was charged" (Fees & Charges) because it understands the semantic difference, not just the shared vocabulary.

### ⚡ Model 4: ONNX Optimization — Production Speed

The final step was exporting the fine-tuned DistilBERT to ONNX format. ONNX Runtime applies graph optimizations that make inference **~2x faster on CPU** compared to PyTorch — critical for production deployments without GPU.

**Result: 90.6% accuracy (zero loss), ~25ms inference on CPU**

---

## Results at a Glance

| Model | Category Accuracy | Category F1 | Priority F1 | Inference | Size |
|:------|:-:|:-:|:-:|:-:|:-:|
| TF-IDF + Logistic Regression | 87.5% | 0.875 | 0.893 | 0.39 ms | 5 MB |
| BiLSTM (PyTorch) | 87.8% | 0.878 | 0.905 | 14.15 ms | 50 MB |
| DistilBERT Fine-tuned | **90.6%** | **0.906** | **0.916** | 8.15 ms | 250 MB |
| DistilBERT + ONNX | **90.6%** | **0.906** | **0.916** | 25.28 ms | 254 MB |

---

## System Architecture

```
                    ┌─────────────────────────────────────────────────────┐
                    │                  DATA LAYER                         │
                    │   Banking77 (13K queries) + CLINC150 (24K queries)  │
                    │   77 intents → 10 categories + P0-P3 priorities     │
                    └──────────────────────┬──────────────────────────────┘
                                           │
                    ┌──────────────────────▼──────────────────────────────┐
                    │              PREPROCESSING LAYER                    │
                    │   Text cleaning → Tokenization → Encoding           │
                    │   TF-IDF / Word Embeddings / BERT Subword Tokens    │
                    └──────────────────────┬──────────────────────────────┘
                                           │
                    ┌──────────────────────▼──────────────────────────────┐
                    │                ML / DL LAYER                        │
                    │   Logistic Regression → BiLSTM → DistilBERT         │
                    │   Multi-task heads: Category (10) + Priority (4)    │
                    └──────────────────────┬──────────────────────────────┘
                                           │
                    ┌──────────────────────▼──────────────────────────────┐
                    │             OPTIMIZATION LAYER                      │
                    │   ONNX Runtime export → 2x CPU speedup              │
                    │   Zero accuracy loss after conversion               │
                    └──────────────────────┬──────────────────────────────┘
                                           │
    ┌────────────────────────────────────────────────────────────────────────────┐
    │                          SERVING LAYER                                     │
    │                                                                            │
    │   FastAPI Backend (Google Cloud Run)    Streamlit Dashboard (Cloud)        │
    │   ├── POST /classify                    ├── Tab 1: Live Demo               │
    │   ├── POST /batch_classify      ◄────►  ├── Tab 2: Batch Analysis          │
    │   └── GET  /health              (API)   └── Tab 3: Model Comparison        │
    │                                                                            │
    │   🌏 Mumbai (asia-south1)               🌏 Streamlit Community Cloud      │
    └────────────────────────────────────────────────────────────────────────────┘
                                           │
              ┌────────────────────────────▼───────────────────────────────┐
              │                     MLOps LAYER                            │
              │   MLflow (experiment tracking) + Evidently AI (drift)      │
              │   Docker (containerization) + GitHub Actions (CI/CD)       │
              └────────────────────────────────────────────────────────────┘
```

---

## Tech Stack

| Category | Tools & Libraries |
|:---------|:------------------|
| **Language** | Python 3.12 |
| **ML Baseline** | scikit-learn (TF-IDF, Logistic Regression) |
| **Deep Learning** | PyTorch (BiLSTM from scratch) |
| **Transformers** | HuggingFace Transformers (DistilBERT fine-tuning) |
| **Inference Optimization** | ONNX Runtime |
| **API Backend** | FastAPI, Pydantic, Uvicorn |
| **Dashboard** | Streamlit, Plotly |
| **Experiment Tracking** | MLflow |
| **Model Monitoring** | Evidently AI |
| **Cloud Deployment** | Google Cloud Run (Mumbai), Streamlit Community Cloud |
| **Containerization** | Docker, Docker Compose |
| **CI/CD** | GitHub Actions |
| **Datasets** | Banking77, CLINC150 (HuggingFace) |

---

## Project Structure

```
SmartTicket/
│
├── src/
│   ├── data/                  # Data download, EDA, preparation
│   │   ├── download_data.py   # Download Banking77 + CLINC150
│   │   ├── explore_data.py    # Exploratory data analysis
│   │   └── prepare_data.py    # Category consolidation, priority labels, split
│   │
│   ├── models/                # Model training and export
│   │   ├── baseline_logreg.py # TF-IDF + Logistic Regression
│   │   ├── bilstm_model.py    # PyTorch BiLSTM (from scratch)
│   │   ├── bert_train.py      # DistilBERT fine-tuning
│   │   └── export_onnx.py     # ONNX export + model comparison
│   │
│   ├── evaluation/            # Monitoring
│   │   └── drift_monitor.py   # Evidently AI drift detection
│   │
│   ├── api/                   # FastAPI backend
│   │   ├── schemas.py         # Request/response models
│   │   ├── predict.py         # ONNX inference engine
│   │   ├── predict_baseline.py# Baseline predictor (cloud fallback)
│   │   └── app.py             # API endpoints
│   │
│   └── dashboard/             # Streamlit frontend
│       └── streamlit_app.py   # 3-tab dashboard (connected to API)
│
├── configs/config.yaml        # All hyperparameters in one place
├── models/                    # Saved model files (.onnx, .pth, .joblib)
├── data/
│   ├── raw/                   # Downloaded datasets
│   └── processed/             # Train/val/test splits + mappings
├── reports/
│   ├── figures/               # EDA plots, confusion matrices, comparisons
│   └── drift_reports/         # Evidently AI HTML reports
├── tests/                     # Unit tests (9/9 passing)
├── Dockerfile                 # Container build for Cloud Run
├── docker-compose.yml         # Multi-service orchestration
└── .github/workflows/         # CI/CD pipeline
```

---

## Getting Started

### Prerequisites

- Python 3.10+
- NVIDIA GPU with CUDA support (for training; inference runs on CPU)
- Git

### 1. Clone & Setup

```bash
git clone https://github.com/mani9kanta3/Smart-Ticket.git
cd Smart-Ticket
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux/Mac
pip install -r requirements.txt
```

### 2. Download & Prepare Data

```bash
python src/data/download_data.py     # Downloads Banking77 + CLINC150
python src/data/explore_data.py      # Generates EDA charts
python src/data/prepare_data.py      # Consolidates categories, creates splits
```

### 3. Train Models (Progressive)

```bash
python src/models/baseline_logreg.py   # Baseline: ~87.5% accuracy
python src/models/bilstm_model.py      # BiLSTM:   ~87.8% accuracy
python src/models/bert_train.py        # BERT:     ~90.6% accuracy
python src/models/export_onnx.py       # ONNX export + benchmarks
```

### 4. Run the API Locally

```bash
uvicorn src.api.app:app --port 8000
# Swagger docs: http://localhost:8000/docs
```

### 5. Run the Dashboard Locally

```bash
streamlit run src/dashboard/streamlit_app.py
# Dashboard: http://localhost:8501
```

### 6. Run with Docker

```bash
docker-compose up --build
# API:       http://localhost:8000
# Dashboard: http://localhost:8501
```

---

## Cloud Deployment

### FastAPI Backend → Google Cloud Run

The API is deployed on Google Cloud Run in the **Mumbai (asia-south1)** region for low-latency access from India. Cloud Run automatically scales to zero when idle (no cost) and handles traffic spikes.

```bash
gcloud run deploy smartticket --source . --region asia-south1 --allow-unauthenticated --port 8080 --memory 1Gi
```

**Live API:** [smartticket-363391410596.asia-south1.run.app/docs](https://smartticket-363391410596.asia-south1.run.app/docs)

### Streamlit Dashboard → Streamlit Community Cloud

The dashboard is deployed on Streamlit Community Cloud and **connects to the FastAPI backend** for all predictions. This is a production-like microservices architecture where the frontend and backend are separate services communicating via REST API.

**Live Dashboard:** [smart-ticket.streamlit.app](https://smart-ticket.streamlit.app)

---

## API Usage

### Live API Endpoint

```bash
curl -X POST https://smartticket-363391410596.asia-south1.run.app/classify \
  -H "Content-Type: application/json" \
  -d '{"text": "Someone stole my card and made unauthorized purchases"}'
```

**Response:**

```json
{
  "text": "Someone stole my card and made unauthorized purchases",
  "category": "Account Access",
  "category_confidence": 0.91,
  "priority": "P0-Critical",
  "routing_team": "Security & Authentication Team",
  "inference_time_ms": 24.7
}
```

### Batch Classification

Upload a CSV with a `text` column to the `/batch_classify` endpoint — returns classified CSV with category, priority, confidence, and routing team added.

### Health Check

```bash
curl https://smartticket-363391410596.asia-south1.run.app/health
```

---

## Categories & Priority System

### 10 Business Categories

| Category | Description | Routing Team |
|:---------|:------------|:-------------|
| Account Access | Login, password, verification issues | Security & Authentication |
| Account Management | Account lifecycle, top-ups, settings | Account Services |
| Balance & Statement | Balance checks, transaction history | Account Services |
| Card Services | Card delivery, activation, replacement | Card Operations |
| Fees & Charges | Fee inquiries, exchange rates | Billing & Fees |
| General Inquiry | General questions, out-of-scope queries | General Support |
| Payment Issues | Payment failures, declined transactions | Payments |
| Refund & Dispute | Refund requests, charge disputes | Disputes & Resolution |
| Technical Issues | App errors, system problems | Technical Support |
| Transfer & Transaction | Money transfers, pending transactions | Transfers |

### Priority Levels & SLAs

| Priority | Target Response | Trigger Examples |
|:---------|:----------------|:-----------------|
| 🔴 P0-Critical | 15 minutes | "Someone stole my card," "unauthorized transaction," "account hacked" |
| 🟠 P1-High | 1 hour | "Payment failed," "money not received," "charged twice" |
| 🟡 P2-Medium | 4 hours | "When will I get my refund?", "Card hasn't arrived" |
| 🟢 P3-Low | 24 hours | "What's the exchange rate?", "How do I close my account?" |

---

## Key Design Decisions

**Why 3 models instead of just using BERT?**
The progression tells a story: TF-IDF shows what's possible with simple features, BiLSTM demonstrates the limits of learning from scratch with limited data, and BERT proves the power of transfer learning. Each model's limitations motivate the next.

**Why consolidate 77 → 10 categories?**
Real support teams have ~10 departments. Granular intents like "card_payment_not_recognised" and "card_payment_wrong_amount" go to the same Payments team. Consolidation creates actionable, business-meaningful groups.

**Why multi-task learning?**
A single model predicts both category AND priority. The shared representation acts as implicit regularization, and it halves inference cost compared to running two separate models.

**Why DistilBERT over full BERT?**
40% smaller, 60% faster, retains 97% accuracy. Critically, it fits in 4GB GPU VRAM for fine-tuning — making it accessible without expensive cloud GPU instances.

**Why ONNX?**
Production deployments often run on CPU (cheaper than GPU instances). ONNX Runtime provides ~2x CPU speedup over PyTorch with zero accuracy loss. It also removes the PyTorch dependency from the deployment stack.

**Why separate frontend and backend?**
The Streamlit dashboard calls the FastAPI backend via REST API — a production-like microservices architecture. The API can serve mobile apps, Slack bots, or any other client. The dashboard is just one consumer of the API.

---

## MLOps Features

- **MLflow Experiment Tracking** — 30+ experiments logged with parameters, metrics, and model artifacts. Compare models side-by-side in the MLflow UI.
- **Evidently AI Drift Monitoring** — Automated detection of data distribution changes. Generates HTML reports comparing training vs production data. Catches model degradation before it impacts customers.
- **Docker Containerization** — Reproducible deployment with Docker Compose orchestrating FastAPI + Streamlit services. Deployed to Google Cloud Run.
- **CI/CD Pipeline** — GitHub Actions automatically runs 9 unit tests on every push. Failed tests block deployment.
- **Model Registry** — Best model versioned and tagged in MLflow for reproducibility.

---

## Running Tests

```bash
pytest tests/ -v
# 9 tests covering: label mappings, config, model files, data integrity
```

---

## What I Learned

1. **TF-IDF is a strong baseline** — Don't underestimate simple models. With well-engineered features (bigrams, sublinear TF), Logistic Regression achieved 87.5% on short text classification.

2. **Small data limits deep learning** — BiLSTM barely beat the baseline because 11K samples isn't enough to learn language from scratch. Transfer learning is essential for real-world NLP.

3. **Freezing layers matters** — Freezing DistilBERT's first 4 layers preserved general language knowledge while allowing task-specific adaptation. Without freezing, the model overfit.

4. **Multi-task learning is free regularization** — Training category + priority jointly improved both tasks compared to training them separately.

5. **Production optimization is not optional** — ONNX export provided 2x CPU speedup with zero accuracy loss. In production, inference speed directly impacts user experience and infrastructure costs.

6. **Microservices architecture scales** — Separating the API (Cloud Run) from the dashboard (Streamlit Cloud) means each can scale independently. The API can serve multiple frontends without code changes.

---

## Future Enhancements

- **Confidence thresholding** — Flag predictions below 60% confidence for human review
- **SHAP/LIME explanations** — Show which words drove each prediction
- **Active learning** — Use low-confidence predictions to find the most valuable examples for retraining
- **Multilingual support** — Fine-tune multilingual BERT for non-English tickets
- **Real-time retraining pipeline** — Automated retraining when Evidently detects significant drift
- **PostgreSQL logging** — Store all predictions for analytics and audit trail

---

## Author

**Manikanta Pudi**

- 🌐 [manikantapudi.com](https://manikantapudi.com)
- 💻 [github.com/Mani9kanta3](https://github.com/Mani9kanta3)

---