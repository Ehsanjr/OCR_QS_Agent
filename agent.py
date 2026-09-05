import base64
import mimetypes
import os
import re
import time
from collections import Counter
from typing import Dict, List, Optional, Tuple, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langgraph.graph import END, StateGraph

# Import custom mock/utility modules
from mock_tools import get_cached_ocr_text, get_mock_options
from ocr_tools import extract_text_with_datalab
from utils import inject_synthetic_noise

# ---------------------------------------------------------
# Step 1: Load Environment Variables & Define Agent State
# ---------------------------------------------------------
load_dotenv()

HF_TOKEN = os.getenv("HF_API_KEY")
if not HF_TOKEN:
    raise ValueError("❌ Error: HUGGINGFACEHUB_API_TOKEN is not set in the .env file.")

# The model MUST be able to see the question image (both to solve the
# question and to re-read it while correcting OCR errors), so this needs to
# be a vision-capable ("image-text-to-text") model on Hugging Face's free
# inference tier. Use check_models.py to find one your token can access, and
# set it via HF_MODEL_ID in your .env file. If the multimodal call fails
# (e.g. the model/endpoint doesn't actually support images), the agent
# automatically falls back to a text-only call using just the OCR text and
# logs a warning — see _invoke_with_optional_image() below.
HF_MODEL_ID = os.getenv("HF_MODEL_ID", "openai/gpt-oss-120b")
VISION_ENABLED = os.getenv("VISION_ENABLED", "true").lower() != "false"


class AttemptLogEntry(TypedDict):
    attempt: int
    raw_output: str
    parsed_answer: Optional[str]


class AgentState(TypedDict):
    """
    Shared memory schema for the agentic workflow.
    """
    image_path: str
    options: Dict[str, str]
    original_ocr_text: str
    current_text: str
    attempt_count: int
    max_attempts: int
    is_solved: bool
    unresolved: bool
    final_answer: Optional[str]
    attempts_log: List[AttemptLogEntry]


# ---------------------------------------------------------
# Step 2: Initialize HuggingFace LLM
# ---------------------------------------------------------
llm_endpoint = HuggingFaceEndpoint(
    repo_id=HF_MODEL_ID,
    max_new_tokens=768,
    temperature=0.1,  # Low temperature for precise logic
    task="text-generation",
    huggingfacehub_api_token=HF_TOKEN,
)

chat_model = ChatHuggingFace(llm=llm_endpoint)
print(f"[Info] HuggingFace LLM ({HF_MODEL_ID}) successfully initialized.")
if VISION_ENABLED:
    print(
        "[Warn] VISION_ENABLED is true — make sure HF_MODEL_ID actually supports "
        "image inputs on your Hugging Face Inference access; otherwise every "
        "call will silently fall back to text-only mode. Run check_models.py "
        "to list vision-capable candidates."
    )


# ---------------------------------------------------------
# Step 3: Image + multimodal-call helpers
# ---------------------------------------------------------
def encode_image_base64(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _guess_mime(image_path: str) -> str:
    mime, _ = mimetypes.guess_type(image_path)
    return mime or "image/png"


def _invoke_with_optional_image(system_prompt: str, user_text: str, image_path: Optional[str]) -> str:
    """
    Calls the chat model with the given system + user text. If an image is
    available and vision is enabled, attaches it as a multimodal content
    block (LangChain's standard image_url format). Falls back to a
    text-only call if that fails for any reason.
    """
    messages = [SystemMessage(content=system_prompt)]

    if image_path and VISION_ENABLED and os.path.exists(image_path):
        try:
            b64 = encode_image_base64(image_path)
            mime = _guess_mime(image_path)
            messages.append(
                HumanMessage(
                    content=[
                        {"type": "text", "text": user_text},
                        {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
                    ]
                )
            )
            response = chat_model.invoke(messages)
            return response.content.strip()
        except Exception as e:
            print(f"[Warn] Multimodal call failed ({e}); falling back to text-only.")
            messages = [SystemMessage(content=system_prompt)]

    messages.append(HumanMessage(content=user_text))
    response = chat_model.invoke(messages)
    return response.content.strip()


def _format_options(options: Dict[str, str]) -> str:
    return "\n".join(f"{key}) {value}" for key, value in options.items())


# ---------------------------------------------------------
# Step 4: Answer parsing (strict tag + loose fallback)
# ---------------------------------------------------------
FINAL_ANSWER_PATTERN = re.compile(r"FINAL_ANSWER\s*:\s*([^\n\r]+)")


def _parse_final_answer(raw_output: str, options: Dict[str, str]) -> Optional[str]:
    """
    Looks for a strict `FINAL_ANSWER: <label>` line and matches it against
    the given option keys. Returns None if the model said NONE, or if no
    valid option key could be matched.
    """
    match = FINAL_ANSWER_PATTERN.search(raw_output)
    if not match:
        return None

    candidate = match.group(1).strip().strip(".").strip()

    for key in options.keys():
        if candidate == key or candidate.upper() == str(key).upper():
            return key

    if candidate.upper() == "NONE":
        return None

    # Loose fallback: candidate might contain the key plus extra characters
    for key in options.keys():
        if key and key in candidate:
            return key

    return None


# ---------------------------------------------------------
# Step 5: Node 1 - Solver Node
# ---------------------------------------------------------
def solve_problem_node(state: AgentState) -> dict:
    """
    Node 1: Solves the question using the image + current OCR text, and
    checks the answer against the given options.
    """
    current_attempt = state.get("attempt_count", 0) + 1
    max_attempts = state.get("max_attempts", 3)
    current_text = state.get("current_text", "")
    options = state.get("options", {})
    image_path = state.get("image_path")

    print(f"\n[Info] Executing Solver Node - Attempt {current_attempt}/{max_attempts}")

    system_prompt = r"""شما یک حل‌کننده بسیار دقیق مسائل ریاضی چندگزینه‌ای هستید.
به شما تصویر اسکن‌شده سؤال و متن آن (که با OCR استخراج شده و ممکن است دارای خطا باشد) داده می‌شود.

وظایف شما:
۱. با استفاده از تصویر و متن، مسئله را گام‌به‌گام حل کنید.
۲. پاسخ محاسبه‌شده را با گزینه‌های داده‌شده مقایسه کنید.
۳. اگر پاسخ شما دقیقاً با یکی از گزینه‌ها مطابقت داشت، حرف/برچسب همان گزینه را به‌عنوان پاسخ نهایی انتخاب کنید.
۴. اگر پاسخ شما با هیچ‌کدام از گزینه‌ها مطابقت نداشت، آن را به‌عنوان نشانه‌ای از خطای احتمالی در متن OCR در نظر بگیرید و به‌جای حدس زدن، NONE را گزارش کنید.

فرمت خروجی (اجباری):
می‌توانید مراحل حل را به‌طور خلاصه بنویسید، اما در پایان، دقیقاً و در یک خط جداگانه، این عبارت را بنویسید:
FINAL_ANSWER: <برچسب گزینه یا NONE>

هیچ متن دیگری بعد از این خط ننویسید."""

    user_text = (
        f"متن سؤال (خروجی OCR):\n{current_text}\n\n"
        f"گزینه‌ها:\n{_format_options(options)}\n\n"
        "مسئله را با استفاده از تصویر و متن بالا حل کن."
    )

    try:
        raw_output = _invoke_with_optional_image(system_prompt, user_text, image_path)
    except Exception as e:
        print(f"[Error] LLM request failed: {e}")
        raw_output = ""

    parsed = _parse_final_answer(raw_output, options)
    print(f"[Debug] Solver raw output:\n{raw_output}\n[Debug] Parsed answer: {parsed}\n" + "-" * 30)

    attempts_log: List[AttemptLogEntry] = list(state.get("attempts_log", []))
    attempts_log.append({"attempt": current_attempt, "raw_output": raw_output, "parsed_answer": parsed})

    return {
        "is_solved": parsed is not None,
        "final_answer": parsed,
        "attempt_count": current_attempt,
        "attempts_log": attempts_log,
    }


# ---------------------------------------------------------
# Step 6: Node 2 - Refiner Node
# ---------------------------------------------------------
def refine_ocr_node(state: AgentState) -> dict:
    """
    Node 2: Re-reads the image and produces a minimally-corrected version of
    the question text, since the previous attempt's answer matched none of
    the options.
    """
    current_text = state.get("current_text", "")
    options = state.get("options", {})
    image_path = state.get("image_path")

    print("\n[Info] Executing Refiner Node - re-reading image for likely OCR misreads...")

    refinement_prompt = r"""شما یک متخصص بازسازی متن‌های OCR مسائل ریاضی فارسی هستید.
پاسخ محاسبه‌شده از روی متن فعلی، با هیچ‌یک از گزینه‌های داده‌شده مطابقت نداشت.
این می‌تواند به این معنا باشد که بخشی از متن (یک عدد، نماد یا کلمه) به‌اشتباه توسط OCR خوانده شده است.

با نگاه دقیق به تصویر پیوست‌شده:
۱. فقط بخش‌هایی از متن را که واقعاً با تصویر مغایرت دارند اصلاح کنید (اعداد، نمادهای LaTeX، کلمات فارسی، برچسب گزینه‌ها).
۲. تغییرات باید حداقلی و مبتنی بر شواهد تصویر باشند؛ مسئله را از نو ننویسید و ساختار آن را تغییر ندهید.
۳. اگر بعد از بررسی دقیق، متن فعلی همان چیزی است که در تصویر دیده می‌شود، همان متن را بدون تغییر برگردانید.
۴. فقط و فقط متن نهایی (اصلاح‌شده یا بدون تغییر) مسئله را خروجی دهید؛ هیچ توضیح، مقدمه یا سلام و احوال‌پرسی ننویسید."""

    user_text = (
        f"متن فعلی (خروجی OCR):\n{current_text}\n\n"
        f"گزینه‌های داده‌شده (برای کمک به تشخیص خطای احتمالی):\n{_format_options(options)}\n\n"
        "متن را بر اساس تصویر بازبینی و در صورت نیاز اصلاح کن."
    )

    try:
        refined_text = _invoke_with_optional_image(refinement_prompt, user_text, image_path)
    except Exception as e:
        print(f"[Error] Refinement failed: {e}")
        refined_text = current_text

    print(f"[Debug] Refined Text Result:\n{refined_text}\n" + "-" * 30)

    return {"current_text": refined_text or current_text}


# ---------------------------------------------------------
# Step 7: Node 3 - Finalize Node (best-guess / unresolved)
# ---------------------------------------------------------
def finalize_node(state: AgentState) -> dict:
    """
    Node 3: Runs only when the retry cap is hit without a match. Picks a
    best-guess answer and flags the result as unresolved.

    Best-guess policy (see WRITEUP.md for rationale):
    1. Majority vote among all attempts that produced a valid, strictly-
       parsed option letter.
    2. If none did, loosely scan each attempt's raw output (most recent
       first) for any mention of an option label.
    3. If still nothing, fall back to the first option key alphabetically —
       an explicit, documented last resort.
    """
    attempts_log = state.get("attempts_log", [])
    options = state.get("options", {})

    parsed_answers = [a["parsed_answer"] for a in attempts_log if a.get("parsed_answer")]
    best_guess: Optional[str] = None

    if parsed_answers:
        best_guess = Counter(parsed_answers).most_common(1)[0][0]
    else:
        for attempt in reversed(attempts_log):
            for key in options.keys():
                if key and key in attempt.get("raw_output", ""):
                    best_guess = key
                    break
            if best_guess:
                break

    if best_guess is None and options:
        best_guess = sorted(options.keys())[0]

    print(f"[Info] Finalize Node - retry cap reached. Best-guess answer: {best_guess}")

    return {"final_answer": best_guess, "unresolved": True}


# ---------------------------------------------------------
# Step 8: Router & StateGraph Compilation
# ---------------------------------------------------------
def router(state: AgentState) -> str:
    """
    Directs workflow execution based on match state and retry limits.
    """
    is_solved = state.get("is_solved", False)
    attempts = state.get("attempt_count", 0)
    max_attempts = state.get("max_attempts", 3)

    if is_solved:
        print("[Router] Answer matched an option. Ending workflow.")
        return "end"
    elif attempts >= max_attempts:
        print(f"[Router] Max attempts ({max_attempts}) reached without a match. Finalizing with best guess.")
        return "finalize"
    else:
        print("[Router] No option matched. Routing to Refiner Node.")
        return "refine"


workflow = StateGraph(AgentState)

workflow.add_node("solver", solve_problem_node)
workflow.add_node("refiner", refine_ocr_node)
workflow.add_node("finalize", finalize_node)

workflow.set_entry_point("solver")

workflow.add_conditional_edges(
    "solver",
    router,
    {
        "end": END,
        "refine": "refiner",
        "finalize": "finalize",
    },
)

workflow.add_edge("refiner", "solver")
workflow.add_edge("finalize", END)

# Compile into executable agent application
app = workflow.compile()
print("[Info] Agent Graph successfully compiled!")


# ---------------------------------------------------------
# Step 9: Output builder (required JSON schema)
# ---------------------------------------------------------
def build_output(state: AgentState) -> dict:
    """
    Builds the final result dict per the brief's required schema, plus an
    extra `unresolved` flag for cases where the retry cap was hit without a
    confident match.
    """
    return {
        "answer": state.get("final_answer"),
        "question_text": state.get("current_text"),
        "changed": state.get("current_text") != state.get("original_ocr_text"),
        "original_ocr_text": state.get("original_ocr_text"),
        "unresolved": state.get("unresolved", False),
    }


# ---------------------------------------------------------
# Step 10: Terminal Test Entry Point
# ---------------------------------------------------------
if __name__ == "__main__":
    test_image_path = "data/q113.png"

    print(f"--- Step 1: Loading Cached OCR Text for '{test_image_path}' ---")
    # raw_ocr_text = extract_text_with_datalab(test_image_path)
    raw_ocr_text = get_cached_ocr_text(test_image_path)
    options = get_mock_options(test_image_path)
    print(f"[OCR Result Raw]:\n{raw_ocr_text}\n")
    print(f"[Options]:\n{_format_options(options)}\n")

    print("--- Step 2: Injecting Synthetic Noise ---")
    noisy_text, num_changes = inject_synthetic_noise(raw_ocr_text, num_changes=3)
    print(f"[Noisy Text] ({num_changes} perturbations):\n{noisy_text}\n")

    initial_state: AgentState = {
        "image_path": test_image_path,
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

    print("--- Step 3: Starting LangGraph Agentic Pipeline ---")
    start_time = time.time()
    final_output = app.invoke(initial_state)
    end_time = time.time()

    print("\n" + "=" * 50)
    print("🏁 WORKFLOW COMPLETED")
    print("=" * 50)
    print(f"Execution Time: {end_time - start_time:.2f} seconds")
    print(f"Total Attempts Made: {final_output.get('attempt_count')}")
    print(f"Unresolved: {final_output.get('unresolved')}")
    print("\n[Final JSON Output]:")
    import json

    print(json.dumps(build_output(final_output), ensure_ascii=False, indent=2))
    print("=" * 50)