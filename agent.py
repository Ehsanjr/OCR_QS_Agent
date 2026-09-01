import os
import time
from typing import TypedDict, Optional
from dotenv import load_dotenv

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END

# Import custom mock/utility modules
from mock_tools import get_cached_ocr_text
from utils import inject_synthetic_noise

# ---------------------------------------------------------
# Step 1: Load Environment Variables & Define Agent State
# ---------------------------------------------------------
load_dotenv()

HF_TOKEN = os.getenv("HF_API_KEY")
if not HF_TOKEN:
    raise ValueError("❌ Error: HUGGINGFACEHUB_API_TOKEN is not set in the .env file.")


class AgentState(TypedDict):
    """
    Shared memory schema for the agentic workflow.
    """
    image_path: str
    original_ocr_text: str
    current_text: str
    attempt_count: int
    max_attempts: int
    final_answer: Optional[str]
    is_solved: bool


# ---------------------------------------------------------
# Step 2: Initialize HuggingFace LLM (Qwen 2.5 7B Instruct)
# ---------------------------------------------------------
llm_endpoint = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-120b",
    max_new_tokens=512,
    temperature=0.1,  # Low temperature for precise logic
    task="text-generation",
    huggingfacehub_api_token=HF_TOKEN
)

chat_model = ChatHuggingFace(llm=llm_endpoint)
print("[Info] HuggingFace LLM (gpt-oss-120b) successfully initialized.")


# ---------------------------------------------------------
# Step 3: Define Node 1 - Solver Node (Persian Prompt)
# ---------------------------------------------------------
def solve_problem_node(state: AgentState) -> dict:
    """
    Node 1: Evaluates OCR quality and solves the Persian math problem.
    """
    current_attempt = state.get("attempt_count", 0) + 1
    max_attempts = state.get("max_attempts", 3)
    current_text = state.get("current_text", "")
    
    print(f"\n[Info] Executing Solver Node - Attempt {current_attempt}/{max_attempts}")
    
    # Strictly engineered Persian system prompt
    system_prompt = r"""شما یک ارزیاب و حل‌کننده بسیار دقیق مسائل ریاضی هستید.
قبل از شروع به حل مسئله، باید متن OCR ارسال‌شده را از نظر نویز و خرابی بررسی کنید.

قانون حیاتی: اگر هر یک از موارد زیر را در متن مشاهده کردید، باید بلافاصله متوقف شده و دقیقاً و فقط کلمه 'FAILED' را خروجی دهید:
۱. غلط‌های املایی ناشی از نویز OCR در کلمات فارسی (مانند وجود اعداد لابلای حروف مثل 'ت۱بع' یا حروف اشتباه مثل 'تاپو بُشد').
۲. نمادهای بی‌معنی، نامربوط یا خراب در LaTeX (مانند dagger\، circ\ یا پرانتزهای به هم ریخته).
۳. معادلات ریاضی ناقص، به هم ریخته یا نامفهوم.

نکات مهم:
- هیچ مقدمه، سلام یا توضیحی ننویسید. دلیل رد شدن را شرح ندهید.
- اگر متن خراب است، کل خروجی شما باید فقط و فقط کلمه FAILED باشد.
- تنها و تنها زمانی که متن کاملاً سالم، خوانا و از نظر ریاضی منطقی است، مسئله را گام به گام به زبان فارسی حل کرده و پاسخ نهایی را ارائه دهید."""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"متن مسئله:\n{current_text}")
    ]
    
    try:
        response = chat_model.invoke(messages)
        output_text = response.content.strip()
        print(f"[Debug] Solver LLM Output:\n{output_text}\n" + "-"*30)
        
        if "FAILED" in output_text.upper():
            return {
                "is_solved": False,
                "final_answer": None,
                "attempt_count": current_attempt
            }
        else:
            return {
                "is_solved": True,
                "final_answer": output_text,
                "attempt_count": current_attempt
            }
            
    except Exception as e:
        print(f"[Error] LLM request failed: {e}")
        return {
            "is_solved": False,
            "final_answer": None,
            "attempt_count": current_attempt
        }


# ---------------------------------------------------------
# Step 4: Define Node 2 - Refiner Node (Persian Prompt)
# ---------------------------------------------------------
def refine_ocr_node(state: AgentState) -> dict:
    """
    Node 2: Refines and reconstructs corrupted Persian mathematical OCR text.
    """
    current_text = state.get("current_text", "")
    print(f"\n[Info] Executing Refiner Node - Reconstructing mathematical logic...")
    
    # Persian refinement prompt focused on math terms and LaTeX structure
    refinement_prompt = r"""شما یک متخصص بازسازی و تصحیح متن‌های OCR مسائل ریاضی فارسی هستید.
متن زیر حاصل خروجی OCR است و دارای نویز، کاراکترهای اشتباه یا کلمات به هم ریخته می‌باشد.

وظایف شما:
۱. ساختار منطقی و ریاضی مسئله را بازسازی کنید.
۲. املای کلمات فارسی را با استفاده از ادبیات استاندارد ریاضی تصحیح کنید (مثلاً 'ت۱بع' -> 'تابع'، 'ب۱زه' -> 'بازه'، 'پاشد' -> 'باشد').
۳. فرمول‌ها و عبارت‌های LaTeX را اصلاح کنید تا از نظر ریاضی کاملاً معتبر و مفهوم باشند.
۴. فقط و فقط متن اصلاح‌شده مسئله را خروجی دهید. از نوشتن هرگونه مقدمه، توضیحات اضافه یا سلام و احوال‌پرسی خودداری کنید.

متن خراب اولیه:
"""

    messages = [
        SystemMessage(content="شما متون OCR خراب مسائل ریاضی فارسی را به صورت دقیق تصحیح می‌کنید."),
        HumanMessage(content=refinement_prompt + current_text)
    ]
    
    try:
        response = chat_model.invoke(messages)
        refined_text = response.content.strip()
        print(f"[Debug] Refined Text Result:\n{refined_text}\n" + "-"*30)
        
        return {
            "current_text": refined_text
        }
        
    except Exception as e:
        print(f"[Error] Refinement failed: {e}")
        return {
            "current_text": current_text
        }


# ---------------------------------------------------------
# Step 5: Router Function & StateGraph Compilation
# ---------------------------------------------------------
def router(state: AgentState) -> str:
    """
    Directs workflow execution based on solution state and retry limits.
    """
    is_solved = state.get("is_solved", False)
    attempts = state.get("attempt_count", 0)
    max_attempts = state.get("max_attempts", 3)
    
    if is_solved:
        print("[Router] Problem solved successfully. Ending workflow.")
        return "end"
    
    elif attempts >= max_attempts:
        print(f"[Router] Max attempts ({max_attempts}) reached without success. Ending workflow.")
        return "end"
    
    else:
        print("[Router] Problem not solved. Routing to Refiner Node.")
        return "refine"


# Build the graph workflow
workflow = StateGraph(AgentState)

workflow.add_node("solver", solve_problem_node)
workflow.add_node("refiner", refine_ocr_node)

workflow.set_entry_point("solver")

workflow.add_conditional_edges(
    "solver",
    router,
    {
        "end": END,
        "refine": "refiner"
    }
)

workflow.add_edge("refiner", "solver")

# Compile into executable agent application
app = workflow.compile()
print("[Info] Agent Graph successfully compiled!")


# ---------------------------------------------------------
# Step 6: Terminal Test Entry Point
# ---------------------------------------------------------
if __name__ == "__main__":
    test_image_path = "data/q113.png"
    
    print(f"--- Step 1: Loading Cached OCR Text for '{test_image_path}' ---")
    raw_ocr_text = get_cached_ocr_text(test_image_path)
    print(f"[OCR Result Raw]:\n{raw_ocr_text}\n")
    
    print("--- Step 2: Injecting Synthetic Noise ---")
    noisy_text = inject_synthetic_noise(raw_ocr_text, num_changes=3)
    print(f"[Noisy Text]:\n{noisy_text}\n")
    
    initial_state: AgentState = {
        "image_path": test_image_path,
        "original_ocr_text": noisy_text,
        "current_text": noisy_text,
        "attempt_count": 0,
        "max_attempts": 3,
        "final_answer": None,
        "is_solved": False
    }
    
    print("--- Step 3: Starting LangGraph Agentic Pipeline ---")
    start_time = time.time()
    final_output = app.invoke(initial_state)
    end_time = time.time()
    
    print("\n" + "="*50)
    print("🏁 WORKFLOW COMPLETED")
    print("="*50)
    print(f"Execution Time: {end_time - start_time:.2f} seconds")
    print(f"Total Attempts Made: {final_output.get('attempt_count')}")
    print(f"Is Problem Solved: {final_output.get('is_solved')}")
    print("\n[Final Refined Text]:")
    print(final_output.get('current_text'))
    print("\n[Final Agent Solution]:")
    print(final_output.get('final_answer'))
    print("="*50)