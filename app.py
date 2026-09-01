import streamlit as st
import time
import os

# Import the compiled LangGraph app and state from your existing agent module
from agent import app as agent_app
from agent import AgentState

# Import your mock tool and noise injector
from mock_tools import get_cached_ocr_text
from utils import inject_synthetic_noise

# ---------------------------------------------------------
# UI Configuration & Styling
# ---------------------------------------------------------
st.set_page_config(page_title="Math OCR Agent", layout="wide")

# Custom CSS to force RTL (Right-to-Left) for Persian text and LaTeX adjustments
st.markdown("""
    <style>
    .persian-text {
        direction: rtl;
        text-align: right;
        font-family: 'Tahoma', 'Arial', sans-serif;
        background-color: #1e1e1e;
        padding: 20px;
        border-radius: 8px;
        border: 1px solid #444;
        margin-bottom: 10px;
        font-size: 16px;
        line-height: 2;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🧠 LangGraph OCR Agent - Visual Debugger")
st.write("This dashboard visualizes every step of the agent's decision-making process.")

# ---------------------------------------------------------
# Sidebar Controls
# ---------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Configuration")
    image_path = st.text_input("Image Path:", value="data/q113.png")
    noise_level = st.slider("Synthetic Noise Level (Characters):", min_value=0, max_value=10, value=3)
    start_btn = st.button("🚀 Run Agentic Pipeline", type="primary", use_container_width=True)

# ---------------------------------------------------------
# Main Execution Logic
# ---------------------------------------------------------
if start_btn:
    if not os.path.exists(image_path) and not image_path.endswith(".png"):
        st.error(f"Please ensure the image path is correct: {image_path}")
    else:
        col1, col2 = st.columns(2)
        
        # Step 1: Mock OCR
        with col1:
            st.subheader("1️⃣ Initial OCR (Cached)")
            raw_ocr_text = get_cached_ocr_text(image_path)
            st.markdown(f'<div class="persian-text">{raw_ocr_text}</div>', unsafe_allow_html=True)

        # Step 2: Add Noise
        with col2:
            st.subheader("2️⃣ Corrupted Text (Noise Injected)")
            noisy_text = inject_synthetic_noise(raw_ocr_text, num_changes=noise_level)
            st.markdown(f'<div class="persian-text">{noisy_text}</div>', unsafe_allow_html=True)

        st.divider()
        st.header("🔄 Agent Execution Graph")

        # Initialize the state memory for LangGraph
        initial_state: AgentState = {
            "image_path": image_path,
            "original_ocr_text": noisy_text,
            "current_text": noisy_text,
            "attempt_count": 0,
            "max_attempts": 3,
            "final_answer": None,
            "is_solved": False
        }

        # We use st.status to show a running workflow that updates dynamically
        with st.status("Executing LangGraph Nodes...", expanded=True) as status:
            
            # agent_app.stream() yields the output of each node exactly as it finishes!
            for output in agent_app.stream(initial_state):
                
                # output is a dictionary where the key is the node name (e.g., 'solver' or 'refiner')
                for node_name, node_state in output.items():
                    st.markdown(f"### ⚙️ Node Executed: `{node_name.upper()}`")
                    
                    if node_name == "solver":
                        is_solved = node_state.get("is_solved")
                        attempt = node_state.get("attempt_count")
                        
                        st.info(f"**Attempt:** {attempt} | **Status:** {'✅ Solved' if is_solved else '❌ Failed (Needs Refinement)'}")
                        
                        with st.expander("View Solver LLM Output", expanded=True):
                            # Streamlit automatically renders Markdown and LaTeX formulas here
                            st.write(node_state.get("final_answer"))
                            
                    elif node_name == "refiner":
                        st.warning("**Action:** Cleaned up OCR text for the next solver attempt.")
                        
                        with st.expander("View Refined Text", expanded=True):
                            refined_text = node_state.get("current_text", "")
                            st.markdown(f'<div class="persian-text">{refined_text}</div>', unsafe_allow_html=True)
                    
                    st.divider()
                    time.sleep(0.5)  # Small delay for UI visualization effect

            status.update(label="Workflow Completed!", state="complete", expanded=True)

        # Final Results Display
        st.success("🎉 Agent Graph Execution Finished!")