"""
SmartTicket - 3-Tab Streamlit Business Dashboard
Connected to FastAPI backend for predictions.
Tab 1: Live Demo | Tab 2: Batch Analysis | Tab 3: Model Performance
"""

import json
import time
import io
import requests
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# ============================================================
# Configuration
# ============================================================

# FastAPI backend URL (Google Cloud Run)
API_URL = "https://smartticket-363391410596.asia-south1.run.app"

st.set_page_config(
    page_title="SmartTicket Dashboard",
    page_icon="🎫",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# API Helper Functions
# ============================================================

def classify_ticket(text: str) -> dict:
    """Call FastAPI backend to classify a single ticket."""
    try:
        response = requests.post(
            f"{API_URL}/classify",
            json={"text": text},
            timeout=30,
        )
        if response.status_code == 200:
            result = response.json()
            # Add all_cat_probs for the chart (not in API response, so we simulate)
            return result
        else:
            st.error(f"API Error: {response.status_code} - {response.text}")
            return None
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to API. The backend may be starting up (cold start). Please try again in 10 seconds.")
        return None
    except requests.exceptions.Timeout:
        st.error("API request timed out. Please try again.")
        return None


def check_health() -> dict:
    """Check API health status."""
    try:
        response = requests.get(f"{API_URL}/health", timeout=10)
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return None


# ============================================================
# Sidebar
# ============================================================

st.sidebar.title("🎫 SmartTicket")
st.sidebar.markdown("**Intelligent Ticket Classifier**")
st.sidebar.markdown("---")

# Check API health
health = check_health()
if health and health.get("status") == "healthy":
    st.sidebar.success("🟢 API Connected")
    st.sidebar.markdown(f"**Model:** {health.get('model_type', 'N/A')}")
else:
    st.sidebar.warning("🟡 API Starting...")
    st.sidebar.markdown("Cloud Run cold start ~10s")

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Models:** TF-IDF, BiLSTM, DistilBERT\n\n"
    "**Best Model:** DistilBERT + ONNX\n\n"
    "**Accuracy:** 90.6%\n\n"
    "**Categories:** 10\n\n"
    "**Priorities:** P0-P3"
)
st.sidebar.markdown("---")
st.sidebar.markdown(f"**API:** [Swagger Docs]({API_URL}/docs)")
st.sidebar.markdown("Built by **Manikanta Pudi**")


# ============================================================
# Tabs
# ============================================================

tab1, tab2, tab3 = st.tabs([
    "🔍 Live Demo",
    "📊 Batch Analysis",
    "📈 Model Performance",
])


# ============================================================
# TAB 1: Live Demo
# ============================================================

with tab1:
    st.header("Live Ticket Classification")
    st.markdown("Enter a customer support query below. Predictions are served by the **FastAPI backend** on Google Cloud Run.")

    sample_queries = [
        "My card payment was declined and I need help urgently",
        "I forgot my password and can't log in",
        "When will I receive my new card?",
        "I was charged twice for the same transaction",
        "What is the exchange rate for USD to EUR?",
        "Someone stole my card and made unauthorized purchases",
        "How do I close my account?",
        "The app keeps crashing when I try to transfer money",
        "Can I get a refund for the wrong item?",
        "What movies are playing near me?",
    ]

    col_input, col_samples = st.columns([3, 1])

    with col_input:
        user_text = st.text_area(
            "Enter ticket text:",
            height=100,
            placeholder="Type a customer support query here...",
        )

    with col_samples:
        st.markdown("**Quick Samples:**")
        for i, sample in enumerate(sample_queries[:5]):
            if st.button(f"📝 {sample[:40]}...", key=f"sample_{i}"):
                user_text = sample

    if st.button("🚀 Classify Ticket", type="primary") or user_text:
        if user_text and user_text.strip():
            with st.spinner("Calling API..."):
                result = classify_ticket(user_text.strip())

            if result:
                st.markdown("---")

                col1, col2, col3, col4 = st.columns(4)

                pri_colors = {
                    "P0-Critical": "🔴",
                    "P1-High": "🟠",
                    "P2-Medium": "🟡",
                    "P3-Low": "🟢",
                }

                with col1:
                    st.metric("Category", result["category"])
                with col2:
                    pri_icon = pri_colors.get(result["priority"], "⚪")
                    st.metric("Priority", f"{pri_icon} {result['priority']}")
                with col3:
                    st.metric("Routing Team", result["routing_team"])
                with col4:
                    st.metric("Inference Time", f"{result['inference_time_ms']:.1f} ms")

                st.markdown("---")
                col_cat, col_pri = st.columns(2)

                with col_cat:
                    st.subheader("Category Confidence")
                    st.progress(result["category_confidence"],
                                text=f"{result['category']}: {result['category_confidence']:.1%}")

                with col_pri:
                    st.subheader("Priority Confidence")
                    st.progress(result["priority_confidence"],
                                text=f"{result['priority']}: {result['priority_confidence']:.1%}")

                    pri_sla = {
                        "P0-Critical": ("⚡ Immediate Response", "15 minutes"),
                        "P1-High": ("🔥 Urgent", "1 hour"),
                        "P2-Medium": ("📋 Standard", "4 hours"),
                        "P3-Low": ("📝 Low Priority", "24 hours"),
                    }
                    sla_info = pri_sla.get(result["priority"], ("", ""))
                    st.markdown(f"**{sla_info[0]}**")
                    st.markdown(f"Target Response Time: **{sla_info[1]}**")

        elif not user_text or not user_text.strip():
            st.warning("Please enter some text to classify.")


# ============================================================
# TAB 2: Batch Analysis
# ============================================================

with tab2:
    st.header("Batch Ticket Analysis")
    st.markdown("Upload a CSV file with a **'text'** column to classify multiple tickets at once.")

    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        if "text" not in df.columns:
            st.error("CSV must contain a 'text' column!")
        else:
            st.markdown(f"**Uploaded:** {len(df)} tickets")

            with st.spinner(f"Classifying {len(df)} tickets via API..."):
                results = []
                progress_bar = st.progress(0)

                for i, text in enumerate(df["text"]):
                    result = classify_ticket(str(text))
                    if result:
                        results.append(result)
                    else:
                        results.append({
                            "category": "Error", "priority": "Unknown",
                            "category_confidence": 0, "routing_team": "N/A",
                        })
                    progress_bar.progress((i + 1) / len(df))

                progress_bar.empty()

            df["category"] = [r["category"] for r in results]
            df["priority"] = [r["priority"] for r in results]
            df["confidence"] = [r["category_confidence"] for r in results]
            df["routing_team"] = [r["routing_team"] for r in results]

            st.success(f"Classified {len(df)} tickets!")

            st.subheader("Preview (first 10 rows)")
            st.dataframe(df.head(10), use_container_width=True)

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Category Distribution")
                cat_counts = df["category"].value_counts()
                fig = px.pie(
                    values=cat_counts.values,
                    names=cat_counts.index,
                    title="Tickets by Category",
                    hole=0.4,
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.subheader("Priority Distribution")
                pri_counts = df["priority"].value_counts().sort_index()
                colors_pri = ["#d32f2f", "#f57c00", "#fbc02d", "#388e3c"]
                fig = px.bar(
                    x=pri_counts.index,
                    y=pri_counts.values,
                    title="Tickets by Priority",
                    color=pri_counts.index,
                    color_discrete_sequence=colors_pri,
                )
                fig.update_layout(height=400, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

            st.subheader("Routing Team Breakdown")
            team_counts = df["routing_team"].value_counts()
            fig = px.bar(
                x=team_counts.values,
                y=team_counts.index,
                orientation="h",
                title="Tickets per Routing Team",
                color=team_counts.values,
                color_continuous_scale="Viridis",
            )
            fig.update_layout(height=400, yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig, use_container_width=True)

            csv_output = df.to_csv(index=False)
            st.download_button(
                label="📥 Download Classified CSV",
                data=csv_output,
                file_name="classified_tickets.csv",
                mime="text/csv",
            )


# ============================================================
# TAB 3: Model Performance
# ============================================================

with tab3:
    st.header("Model Performance Comparison")
    st.markdown("Comparing all 4 models trained during the SmartTicket project.")

    # Hardcoded comparison data (from our training results)
    comparison = {
        "TF-IDF + LogReg": {
            "cat_accuracy": 0.8753, "cat_f1": 0.8752,
            "pri_accuracy": 0.8927, "pri_f1": 0.8928,
            "inference_ms": 0.39, "model_size_mb": 5,
        },
        "BiLSTM (PyTorch)": {
            "cat_accuracy": 0.8781, "cat_f1": 0.8780,
            "pri_accuracy": 0.9051, "pri_f1": 0.9050,
            "inference_ms": 14.15, "model_size_mb": 50,
        },
        "DistilBERT (PyTorch)": {
            "cat_accuracy": 0.9058, "cat_f1": 0.9058,
            "pri_accuracy": 0.9155, "pri_f1": 0.9155,
            "inference_ms": 8.15, "model_size_mb": 250,
        },
        "DistilBERT (ONNX)": {
            "cat_accuracy": 0.9058, "cat_f1": 0.9058,
            "pri_accuracy": 0.9155, "pri_f1": 0.9155,
            "inference_ms": 25.28, "model_size_mb": 254,
        },
    }

    comp_df = pd.DataFrame([
        {
            "Model": name,
            "Category Accuracy": data["cat_accuracy"],
            "Category F1": data["cat_f1"],
            "Priority Accuracy": data["pri_accuracy"],
            "Priority F1": data["pri_f1"],
            "Inference (ms)": data["inference_ms"],
            "Size (MB)": data["model_size_mb"],
        }
        for name, data in comparison.items()
    ])

    best_model = comp_df.loc[comp_df["Category F1"].idxmax()]
    fastest_model = comp_df.loc[comp_df["Inference (ms)"].idxmin()]

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Best Accuracy", f"{best_model['Category Accuracy']:.1%}",
                   delta=f"{best_model['Model']}")
    with col2:
        st.metric("Best F1 Score", f"{best_model['Category F1']:.4f}",
                   delta=f"{best_model['Model']}")
    with col3:
        st.metric("Fastest Model", f"{fastest_model['Inference (ms)']:.2f} ms",
                   delta=f"{fastest_model['Model']}")

    st.markdown("---")

    st.subheader("Full Comparison Table")
    st.dataframe(
        comp_df.style.highlight_max(
            subset=["Category Accuracy", "Category F1", "Priority Accuracy", "Priority F1"],
            color="#c8e6c9",
        ).highlight_min(
            subset=["Inference (ms)", "Size (MB)"],
            color="#c8e6c9",
        ),
        use_container_width=True,
    )

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Accuracy Progression")
        fig = go.Figure()
        fig.add_trace(go.Bar(
            name="Category Accuracy",
            x=comp_df["Model"],
            y=comp_df["Category Accuracy"],
            marker_color="#4CAF50",
        ))
        fig.add_trace(go.Bar(
            name="Priority Accuracy",
            x=comp_df["Model"],
            y=comp_df["Priority Accuracy"],
            marker_color="#2196F3",
        ))
        fig.update_layout(barmode="group", yaxis_range=[0.8, 0.95], height=400)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Inference Speed")
        fig = px.bar(
            comp_df, x="Model", y="Inference (ms)",
            color="Model", title="Inference Time (lower is better)",
        )
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Accuracy vs Speed Tradeoff")
    fig = px.scatter(
        comp_df, x="Inference (ms)", y="Category F1",
        size="Size (MB)", color="Model",
        title="F1 Score vs Inference Speed (bubble size = model size)",
        size_max=50,
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Model Progression Story")
    st.markdown("""
    | Stage | Model | Accuracy | Key Insight |
    |-------|-------|----------|-------------|
    | 1. Baseline | TF-IDF + Logistic Regression | 87.5% | Strong baseline using word frequency features |
    | 2. Deep Learning | BiLSTM (PyTorch) | 87.8% | Word order helps, but limited by small training data |
    | 3. Transfer Learning | DistilBERT Fine-tuned | 90.6% | Pre-trained language understanding is the key |
    | 4. Optimized | DistilBERT + ONNX | 90.6% | Same accuracy, faster CPU inference for production |
    """)