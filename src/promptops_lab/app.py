import streamlit as st
import pandas as pd
from pathlib import Path

# Business logic and evaluation pipeline imports
from promptops_lab.agents.support_agent import SupportAgent
from promptops_lab.evaluation.evaluator_pipeline import PromptEvaluatorPipeline

# Shared UI styling from llmops-common
try:
    from llmops_common.ui.streamlit_helpers import inject_theme
except ImportError:
    inject_theme = None

# 1. Page Configurations
st.set_page_config(
    page_title="PromptOps Lab",
    page_icon="🧪",
    layout="wide"
)

# Apply central theme (Purple for PromptOps)
if inject_theme:
    inject_theme("promptops")
else:
    # Fallback inline CSS if common package theme module is pending
    st.markdown("<style>:root {--accent: #8B5CF6;}</style>", unsafe_allow_html=True)

# Initialize Core Services
agent = SupportAgent()
pipeline = PromptEvaluatorPipeline()

# 2. Header Block
st.title("🧪 PromptOps Lab")
st.markdown("### Prompt Versioning, Regression Testing & Evaluation Gateway")
st.info("Central platform connection established. Monitoring prompt drift and accuracy gates.")

# 3. Sidebar Configuration (Fixed Layout DNA)
with st.sidebar:
    st.header("⚙️ Configuration")
    st.markdown("---")
    
    selected_version = st.selectbox(
        "Active Prompt Version",
        options=["v1", "v2"],
        index=0,
        help="Select the prompt template version for single-turn testing."
    )
    
    st.markdown("### Global Evaluation Gates")
    run_batch_eval = st.button(
        "⚡ Execute Batch Regression Test",
        use_container_width=True,
        help="Runs the golden dataset against v1 and v2, logging aggregate metrics to MLflow."
    )
    
    st.markdown("---")
    st.markdown("#### Platform Links")
    st.markdown("[📊 Open MLflow Dashboard](http://localhost:5000)")

# 4. Main Panel - Split Layout (Tabs for Interactive & Analytical Views)
tab1, tab2 = st.tabs(["🎮 Interactive Sandbox", "📊 Version Control & Leaderboard"])

with tab1:
    st.subheader(f"Live Sandbox Testing — Current Active: `{selected_version}`")
    
    # Load and display active template preview
    try:
        config = agent.load_prompt_config(selected_version)
        with st.expander(f"🔍 Preview {selected_version} System Instructions", expanded=False):
            st.code(config["template"], language="yaml")
    except Exception as e:
        st.error(f"Error loading prompt config: {e}")

    # Layout: Left for input, Right for output
    col_in, col_out = st.columns(2)
    
    with col_in:
        st.markdown("#### Input Context & Question")
        sample_context = st.text_area(
            "Context (Ground Truth Source)",
            value="XYZ Corp offers a 14-day full refund policy for all SaaS subscriptions if the user has not exceeded 500 API calls.",
            height=120
        )
        sample_question = st.text_input(
            "User Question",
            value="Can I get my money back if I cancel on day 10 with 100 API calls?"
        )
        generate_btn = st.button("Run Inference", use_container_width=True)

    with col_out:
        st.markdown("#### LLM Output Response")
        if generate_btn:
            with st.spinner("Invoking agent pipeline via llmops-common..."):
                try:
                    response = agent.execute(
                        version=selected_version,
                        context=sample_context,
                        question=sample_question
                    )
                    st.markdown("---")
                    st.write(response)
                except Exception as e:
                    st.error(f"Inference failed: {e}")
        else:
            st.caption("Awaiting user execution input on the left panel...")

with tab2:
    st.subheader("📝 Prompt Blueprint Diff Viewer")
    
    # Side-by-Side Prompt Template Viewer
    col_v1, col_v2 = st.columns(2)
    try:
        v1_tmpl = agent.load_prompt_config("v1")["template"]
        v2_tmpl = agent.load_prompt_config("v2")["template"]
        
        with col_v1:
            st.markdown("### Version 1 (Baseline)")
            st.text_area("v1 Template File", value=v1_tmpl, height=250, disabled=True)
            
        with col_v2:
            st.markdown("### Version 2 (Candidate)")
            st.text_area("v2 Template File", value=v2_tmpl, height=250, disabled=True)
    except Exception as e:
        st.error(f"Failed to load side-by-side prompt configs: {e}")

    st.markdown("---")
    st.subheader("🏆 Automated Regression Leaderboard")

    # Handle Batch Evaluation State
    if run_batch_eval:
        with st.spinner("Orchestrating test suites across golden dataset. Computing Ragas/DeepEval metrics..."):
            try:
                metrics_v1 = pipeline.evaluate_version("v1")
                metrics_v2 = pipeline.evaluate_version("v2")
                
                # Format into dataframes for a beautiful leaderboard presentation
                leaderboard_data = {
                    "Prompt Version": ["v1 (Baseline)", "v2 (Candidate)"],
                    "Avg Faithfulness (↑)": [metrics_v1["avg_faithfulness"], metrics_v2["avg_faithfulness"]],
                    "Avg Relevance (↑)": [metrics_v1["avg_relevance"], metrics_v2["avg_relevance"]],
                    "Avg Hallucination (↓)": [metrics_v1["avg_hallucination"], metrics_v2["avg_hallucination"]],
                    "Status": ["PASS", "PASS" if metrics_v2["avg_hallucination"] <= metrics_v1["avg_hallucination"] else "FAIL"]
                }
                df = pd.DataFrame(leaderboard_data)
                
                # Highlight and display summary metrics
                st.dataframe(df.style.highlight_max(subset=["Avg Faithfulness (↑)", "Avg Relevance (↑)"], color="#E8F5E9"), use_container_width=True)
                st.success("Batch regression metrics calculated and successfully logged to MLflow Server!")
                
            except Exception as e:
                st.error(f"Batch evaluation run critical failure: {e}")
    else:
        st.info("Click 'Execute Batch Regression Test' in the sidebar to populate the evaluation leaderboard matrix.")

# 5. Footer Experiment Link
st.markdown("---")
st.caption("PromptOps Lab • Built with modular LLMOps standards. Execution metrics and parameters automatically sync to MLflow metadata stores.")