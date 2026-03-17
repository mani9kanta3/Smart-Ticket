"""
SmartTicket - 3-Tab Streamlit Business Dashboard
Tab 1: Live Demo (classify single tickets)
Tab 2: Batch Analysis (upload CSV, analyze results)
Tab 3: Model Performance (compare all 4 models)
"""

import os
import json
import time
import io
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import onnxruntime as ort
from transformers import DistilBertTokenizer


# ============================================================
# Page Configuration
# ============================================================

st.set_page_config(
    page_title="SmartTicket Dashboard",
    page_icon="🎫",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# Load Model (cached so it loads only once)
# ============================================================

@st.cache_resource
def load_model():
    """Load ONNX model, tokenizer, and mappings."""
    model = ort.InferenceSession(
        "models/smartticket_bert.onnx",
        providers=["CPUExecutionProvider"],
    )
    tokenizer = DistilBertTokenizer.from_pretrained("models/bert_finetuned")
    
    with open("data/processed/label_mappings.json", "r") as f:
        mappings = json.load(f)
    
    return model, tokenizer, mappings


@st.cache_data
def load_comparison_data():
    """Load model comparison metrics."""
    with open("models/model_comparison.json", "r") as f:
        return json.load(f)


def predict(text, model, tokenizer, mappings):
    """Run prediction on a single text."""
    start = time.time()
    
    encoding = tokenizer(
        text, max_length=64, padding="max_length",
        truncation=True, return_tensors="np",
    )
    
    outputs = model.run(None, {
        "input_ids": encoding["input_ids"].astype(np.int64),
        "attention_mask": encoding["attention_mask"].astype(np.int64),
    })
    
    cat_probs = _softmax(outputs[0][0])
    pri_probs = _softmax(outputs[1][0])
    
    cat_id = int(np.argmax(cat_probs))
    pri_id = int(np.argmax(pri_probs))
    
    inference_ms = (time.time() - start) * 1000
    
    routing_map = {
        "Account Access": "Security & Authentication Team",
        "Account Management": "Account Services Team",
        "Balance & Statement": "Account Services Team",
        "Card Services": "Card Operations Team",
        "Fees & Charges": "Billing & Fees Team",
        "General Inquiry": "General Support Team",
        "Payment Issues": "Payments Team",
        "Refund & Dispute": "Disputes & Resolution Team",
        "Technical Issues": "Technical Support Team",
        "Transfer & Transaction": "Transfers Team",
    }
    
    cat_name = mappings["id_to_category"][str(cat_id)]
    pri_name = mappings["id_to_priority"][str(pri_id)]
    
    return {
        "category": cat_name,
        "category_id": cat_id,
        "category_confidence": float(cat_probs[cat_id]),
        "priority": pri_name,
        "priority_id": pri_id,
        "priority_confidence": float(pri_probs[pri_id]),
        "routing_team": routing_map.get(cat_name, "General Support"),
        "inference_time_ms": round(inference_ms, 2),
        "all_cat_probs": {mappings["id_to_category"][str(i)]: float(cat_probs[i]) for i in range(10)},
    }


def _softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


# ============================================================
# Sidebar
# ============================================================

st.sidebar.title("🎫 SmartTicket")
st.sidebar.markdown("**Intelligent Ticket Classifier**")
st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Models:** TF-IDF, BiLSTM, DistilBERT\n\n"
    "**Best Model:** DistilBERT + ONNX\n\n"
    "**Accuracy:** 90.6%\n\n"
    "**Categories:** 10\n\n"
    "**Priorities:** P0-P3"
)
st.sidebar.markdown("---")
st.sidebar.markdown("Built by **Manikanta Pudi**")


# ============================================================
# Load model
# ============================================================

try:
    model, tokenizer, mappings = load_model()
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.error(f"Failed to load model: {e}")


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
    st.markdown("Enter a customer support query below and see real-time classification.")
    
    # Sample queries for quick testing
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
        if user_text and user_text.strip() and model_loaded:
            result = predict(user_text.strip(), model, tokenizer, mappings)
            
            st.markdown("---")
            
            # Results in columns
            col1, col2, col3, col4 = st.columns(4)
            
            # Priority color
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
            
            # Confidence bars
            st.markdown("---")
            col_cat, col_pri = st.columns(2)
            
            with col_cat:
                st.subheader("Category Confidence")
                st.progress(result["category_confidence"],
                            text=f"{result['category']}: {result['category_confidence']:.1%}")
                
                # Top 5 categories chart
                probs_sorted = sorted(result["all_cat_probs"].items(),
                                      key=lambda x: x[1], reverse=True)[:5]
                fig = go.Figure(go.Bar(
                    x=[p[1] for p in probs_sorted],
                    y=[p[0] for p in probs_sorted],
                    orientation="h",
                    marker_color=["#4CAF50" if i == 0 else "#90CAF9" for i in range(5)],
                ))
                fig.update_layout(
                    title="Top 5 Category Probabilities",
                    xaxis_title="Confidence",
                    height=300,
                    margin=dict(l=0, r=0, t=40, b=0),
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col_pri:
                st.subheader("Priority Confidence")
                st.progress(result["priority_confidence"],
                            text=f"{result['priority']}: {result['priority_confidence']:.1%}")
                
                # Priority explanation
                pri_sla = {
                    "P0-Critical": ("⚡ Immediate Response", "15 minutes", "#d32f2f"),
                    "P1-High": ("🔥 Urgent", "1 hour", "#f57c00"),
                    "P2-Medium": ("📋 Standard", "4 hours", "#fbc02d"),
                    "P3-Low": ("📝 Low Priority", "24 hours", "#388e3c"),
                }
                sla_info = pri_sla.get(result["priority"], ("", "", ""))
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
    
    if uploaded_file is not None and model_loaded:
        df = pd.read_csv(uploaded_file)
        
        if "text" not in df.columns:
            st.error("CSV must contain a 'text' column!")
        else:
            st.markdown(f"**Uploaded:** {len(df)} tickets")
            
            # Classify all tickets
            with st.spinner(f"Classifying {len(df)} tickets..."):
                results = []
                progress_bar = st.progress(0)
                
                for i, text in enumerate(df["text"]):
                    result = predict(str(text), model, tokenizer, mappings)
                    results.append(result)
                    progress_bar.progress((i + 1) / len(df))
                
                progress_bar.empty()
            
            # Add results to dataframe
            df["category"] = [r["category"] for r in results]
            df["priority"] = [r["priority"] for r in results]
            df["confidence"] = [r["category_confidence"] for r in results]
            df["routing_team"] = [r["routing_team"] for r in results]
            
            st.success(f"Classified {len(df)} tickets!")
            
            # Show first 10 rows
            st.subheader("Preview (first 10 rows)")
            st.dataframe(df.head(10), use_container_width=True)
            
            # Charts
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
            
            # Routing team breakdown
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
            
            # Download button
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
    
    try:
        comparison = load_comparison_data()
    except Exception:
        comparison = None
        st.warning("Model comparison data not found. Run export_onnx.py first.")
    
    if comparison:
        # Build comparison dataframe
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
        
        # Summary metrics at top
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
        
        # Full comparison table
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
        
        # Charts
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
            fig.update_layout(
                barmode="group",
                yaxis_range=[0.8, 0.95],
                height=400,
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Inference Speed")
            fig = px.bar(
                comp_df,
                x="Model",
                y="Inference (ms)",
                color="Model",
                title="Inference Time (lower is better)",
            )
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        # Accuracy vs Speed tradeoff
        st.subheader("Accuracy vs Speed Tradeoff")
        fig = px.scatter(
            comp_df,
            x="Inference (ms)",
            y="Category F1",
            size="Size (MB)",
            color="Model",
            title="F1 Score vs Inference Speed (bubble size = model size)",
            size_max=50,
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        # Model progression story
        st.subheader("Model Progression Story")
        st.markdown("""
        | Stage | Model | Accuracy | Key Insight |
        |-------|-------|----------|-------------|
        | 1. Baseline | TF-IDF + Logistic Regression | 87.5% | Strong baseline using word frequency features |
        | 2. Deep Learning | BiLSTM (PyTorch) | 87.8% | Word order helps, but limited by small training data |
        | 3. Transfer Learning | DistilBERT Fine-tuned | 90.6% | Pre-trained language understanding is the key |
        | 4. Optimized | DistilBERT + ONNX | 90.6% | Same accuracy, faster CPU inference for production |
        """)