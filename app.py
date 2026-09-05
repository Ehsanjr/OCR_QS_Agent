import json
import os
import time

import streamlit as st

# Import the compiled LangGraph app, state, and output builder from agent.py
from agent import AgentState, build_output
from agent import app as agent_app

# Import your mock tool and noise injector
from mock_tools import get_cached_ocr_text, get_mock_options
from ocr_tools import extract_text_with_datalab
from utils import inject_synthetic_noise

# ---------------------------------------------------------
# UI Configuration & Styling
# ---------------------------------------------------------
st.set_page_config(page_title="Math OCR Agent", layout="wide")

# Custom CSS to force RTL (Right-to-Left) for Persian text and LaTeX adjustments
st.markdown(
    """
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
""",
    unsafe_allow_html=True,
)

st.title("🧠 LangGraph OCR Agent - Visual Debugger")
st.write("This dashboard visualizes every step of the agent's decision-making process.")

# ---------------------------------------------------------
# Sidebar Controls
# ---------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Configuration")
    image_path = st.text_input("Image Path:", value="data/q115.png")
    noise_level = st.slider("Synthetic Noise Level (Characters/Words):", min_value=0, max_value=10, value=3)

    default_options_json = json.dumps(get_mock_options(image_path), ensure_ascii=False, indent=2)
    options_text = st.text_area("Options (JSON) — ⚠️ replace with real block options:", value=default_options_json, height=150)

    start_btn = st.button("🚀 Run Agentic Pipeline", type="primary", use_container_width=True)

# ---------------------------------------------------------
# Main Execution Logic
# ---------------------------------------------------------
if start_btn:
    if not os.path.exists(image_path) and not image_path.endswith(".png"):
        st.error(f"Please ensure the image path is correct: {image_path}")
    else:
        try:
            options = json.loads(options_text)
        except json.JSONDecodeError as e:
            st.error(f"Options JSON is invalid: {e}")
            st.stop()

        col1, col2 = st.columns(2)

        # Step 1: Mock OCR
        with col1:
            st.subheader("1️⃣ Initial OCR (Cached)")
            #raw_ocr_text = get_cached_ocr_text(image_path)
            raw_ocr_text = extract_text_with_datalab(image_path)
            st.markdown(f'<div class="persian-text">{raw_ocr_text}</div>', unsafe_allow_html=True)

        # Step 2: Add Noise
        with col2:
            st.subheader("2️⃣ Corrupted Text (Noise Injected)")
            noisy_text, num_changes = inject_synthetic_noise(raw_ocr_text, num_changes=noise_level)
            st.caption(f"{num_changes} character/word perturbation(s) applied")
            st.markdown(f'<div class="persian-text">{noisy_text}</div>', unsafe_allow_html=True)

        st.divider()
        st.header("🔄 Agent Execution Graph")

        # Initialize the state memory for LangGraph
        initial_state: AgentState = {
            "image_path": image_path,
            "options": options,
            "original_ocr_text": noisy_text,
            "current_text": noisy_text,
            "attempt_count": 0,
            "max_attempts": 3,
            "is_solved": False,
            "unresolved": False,
            "final_answer": None,
            "attempts_log": [],
        }

        # Track the merged state across streamed node updates so we can build
        # the final JSON output once the graph finishes.
        current_state: dict = dict(initial_state)

        with st.status("Executing LangGraph Nodes...", expanded=True) as status:
            # agent_app.stream() yields the output of each node exactly as it finishes!
            for output in agent_app.stream(initial_state):
                # output is a dict where the key is the node name (e.g. 'solver', 'refiner', 'finalize')
                for node_name, node_state in output.items():
                    current_state.update(node_state)
                    st.markdown(f"### ⚙️ Node Executed: `{node_name.upper()}`")

                    if node_name == "solver":
                        is_solved = node_state.get("is_solved")
                        attempt = node_state.get("attempt_count")

                        st.info(
                            f"**Attempt:** {attempt} | "
                            f"**Status:** {'✅ Matched an option' if is_solved else '❌ No option matched (needs refinement)'}"
                        )

                        with st.expander("View Solver Attempt Log", expanded=True):
                            log = node_state.get("attempts_log", [])
                            if log:
                                st.code(log[-1].get("raw_output", ""), language=None)
                                st.write(f"Parsed answer: `{log[-1].get('parsed_answer')}`")

                    elif node_name == "refiner":
                        st.warning("**Action:** Re-read the image and revised the OCR text for the next attempt.")

                        with st.expander("View Refined Text", expanded=True):
                            refined_text = node_state.get("current_text", "")
                            st.markdown(f'<div class="persian-text">{refined_text}</div>', unsafe_allow_html=True)

                    elif node_name == "finalize":
                        st.error(
                            f"**Retry cap reached.** Best-guess answer: `{node_state.get('final_answer')}` "
                            "(flagged as unresolved)"
                        )

                    st.divider()
                    time.sleep(0.5)  # Small delay for UI visualization effect

            status.update(label="Workflow Completed!", state="complete", expanded=True)

        # Final Results Display
        st.success("🎉 Agent Graph Execution Finished!")
        st.subheader("📦 Final JSON Output")
        st.json(build_output(current_state))