```markdown
# OCR-Aware Question Solving Agent

An agentic pipeline built with **LangGraph** and **LangChain** that automatically solves multiple-choice mathematical questions extracted from scanned exam documents. 

The core feature of this architecture is its **self-correction / refinement loop**: when the reasoning model fails to solve a question or cannot match the output to the provided options, it treats the failure as a sign of OCR corruption. It then routes the context to a dedicated Refiner node to fix noise, look-alike Persian character swaps, and garbled LaTeX math formulas before re-solving.

---

## 🛠️ Tech Stack & Dependencies

- **Orchestration Engine**: [LangGraph](https://github.com/langchain-ai/langgraph) (Graph-based state machine for agentic workflows)
- **Language Model Infrastructure**: [Hugging Face Inference API](https://huggingface.co/docs/api-inference/index) using `Qwen/Qwen2.5-7B-Instruct`
- **OCR Engine**: [Datalab OCR API](https://www.datalab.to/)
- **Visual Debugger / UI**: [Streamlit](https://streamlit.io/)
- **Environment & Logic**: Python 3.10+

---

## 🚀 Getting Started

### 1. Prerequisites & Installation

Clone the repository and set up a virtual environment:

```bash
git clone <YOUR_REPOSITORY_URL>
cd OCR_QS_Agent

# Create and activate virtual environment
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install requirements
pip install -r requirements.txt

```

### 2. Environment Setup

Create a `.env` file in the root directory and add your API credentials:

```env
HUGGINGFACEHUB_API_TOKEN=your_huggingface_token_here
DATALAB_API_KEY=your_datalab_api_key_here

```

> **Note on OCR Choice:** This project utilizes the **Datalab OCR API** for parsing scanned document crops and rendering LaTeX equations.

---

## 🎮 Running the Application

### Option A: Streamlit Visual Dashboard (Recommended)

To run the interactive UI that visualizes real-time node executions, graph streaming, Persian text rendering, and step-by-step agent decisions:

```bash
streamlit run app.py

```

### Option B: Terminal Pipeline

To run the agent directly from the command line:

```bash
python agent.py

```

---

## 🏗️ Architecture & Agentic Workflow

The architecture is built on a cyclic graph using `LangGraph`:

```
       +------------------+
       |   Input Text     |
       +--------+---------+
                |
                v
       +------------------+
  +--->|   Solver Node    |
  |    +--------+---------+
  |             |
  |             v
  |    +------------------+
  |    |   Router Node    |
  |    +--------+---------+
  |        /          \
  |  (Failed)      (Solved / Max Cap)
  |      /              \
  |     v                v
  |  +------+        +-------+
  +--|Refine|        |  END  |
     +------+        +-------+

```

1. **Solver Node**: Reads the current OCR text. It attempts to evaluate math expressions and solve the question. If noise or corrupted LaTeX is detected, it returns a explicit `FAILED` flag.
2. **Router**: Inspects the state. If solved or if the maximum retry count is reached, it terminates. Otherwise, it routes control to the `Refiner Node`.
3. **Refiner Node**: Analyzes corrupted Persian text and garbled math syntax, applying targeted corrections based on mathematical domain context, then loops back to the Solver.

---

## 📄 Output Format

Each solved block produces a structured JSON output conforming to the system specification:

```json
{
  "answer": "C",
  "question_text": "۱۱۳- تابع f(x)=mx -nx-k در هر بازه، هم صعودی و هم نزولی است...",
  "changed": true,
  "original_ocr_text": "۱۱۳- ت۱بع f(x)=mx -nx-k در هر ب۱زه..."
}

```

```

```