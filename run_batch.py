"""
Batch runner: executes the OCR-aware solving agent over a set of question
blocks and writes one JSON object per block, matching the schema required
by the take-home brief:

    {"answer": ..., "question_text": ..., "changed": ..., "original_ocr_text": ...}

⚠️ BLOCKS below is a placeholder. The real sample question blocks (images +
their multiple-choice options) referenced in the task brief were not
included among the files reviewed, so you need to fill in the real data
here (e.g. load it from a `data/` folder with per-block metadata Datic
provided).
"""
import json
import os
from typing import Tuple

from agent import AgentState, app as agent_app, build_output
from mock_tools import get_cached_ocr_text, get_mock_options
from ocr_tools import extract_text_with_datalab
from utils import inject_synthetic_noise

# Set to True to use the real Datalab OCR API instead of the cached/mock text.
USE_REAL_OCR = True

# Set to True to also run every block through synthetic noise injection,
# producing a second, deliberately-corrupted copy of each block for testing
# (per the brief: test set should include both genuinely noisy OCR and OCR
# you've deliberately corrupted).
ALSO_TEST_WITH_SYNTHETIC_NOISE = True
SYNTHETIC_NOISE_CHANGES = 3

MAX_ATTEMPTS = 3

BLOCKS = [
    {
        "image_path": "data/q113.png",
        "options": {
            "1": "-1",
            "2": "-√5",
            "3": "1",
            "4": "√5",
        },
    },
    {
        "image_path": "data/q115.png",
        "options": {
            "1": "1",
            "2": "2",
            "3": "3",
            "4": "4",
        },
    },
    {
        "image_path": "data/q118.png",
        "options": {
            "1": "صفر",
            "2": "1",
            "3": "2",
            "4": "3",
        },
    },
    {
        "image_path": "data/q121.png",
        "options": {
            "1": "6√6",
            "2": "3√6",
            "3": "2√2",
            "4": "√2",
        },
    },
]


def _load_block_ocr_and_options(block: dict) -> Tuple[str, dict]:
    image_path = block["image_path"]
    if USE_REAL_OCR:
        ocr_text = extract_text_with_datalab(image_path)
    else:
        ocr_text = get_cached_ocr_text(image_path)
    options = block.get("options") or get_mock_options(image_path)
    return ocr_text, options


def run_block(image_path: str, ocr_text: str, options: dict) -> dict:
    initial_state: AgentState = {
        "image_path": image_path,
        "options": options,
        "original_ocr_text": ocr_text,
        "current_text": ocr_text,
        "attempt_count": 0,
        "max_attempts": MAX_ATTEMPTS,
        "is_solved": False,
        "unresolved": False,
        "final_answer": None,
        "attempts_log": [],
    }
    final_state = agent_app.invoke(initial_state)
    return build_output(final_state)


def main():
    results = []

    for block in BLOCKS:
        image_path = block["image_path"]
        ocr_text, options = _load_block_ocr_and_options(block)

        print(f"\n=== Block: {image_path} (original OCR) ===")
        result = run_block(image_path, ocr_text, options)
        result["block"] = os.path.basename(image_path)
        result["noise_perturbations"] = 0
        results.append(result)
        print(json.dumps(result, ensure_ascii=False, indent=2))

        if ALSO_TEST_WITH_SYNTHETIC_NOISE:
            noisy_text, num_changes = inject_synthetic_noise(ocr_text, num_changes=SYNTHETIC_NOISE_CHANGES)
            print(f"\n=== Block: {image_path} (synthetically noised, {num_changes} perturbation(s)) ===")
            noisy_result = run_block(image_path, noisy_text, options)
            noisy_result["block"] = f"{os.path.basename(image_path)} (noisy)"
            noisy_result["noise_perturbations"] = num_changes
            results.append(noisy_result)
            print(json.dumps(noisy_result, ensure_ascii=False, indent=2))

    out_path = "outputs.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n✅ Wrote {len(results)} result(s) to {out_path}")


if __name__ == "__main__":
    main()