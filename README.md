# OCR-Aware Question Solving Agent

An agent built with LangGraph that solves multiple-choice question blocks
extracted from scanned exam documents, using both the block's image and its
OCR text. If the produced answer doesn't match any of the given options, the
agent treats this as a sign the OCR text is corrupted, re-reads the image,
corrects the text, and retries — up to a configurable retry cap.

## Files

- `agent.py` — LangGraph agent: solver node, refiner node, finalize
  (best-guess) node, and `build_output()` for the required JSON schema.
- `utils.py` — synthetic OCR-noise injector for testing (character + word
  level), returns the exact number of perturbations applied.
- `mock_tools.py` — cached/mock OCR text + placeholder options for local
  testing without spending OCR API credits.
- `ocr_tools.py` — real OCR via the Datalab API.
- `run_batch.py` — runs the agent over a set of question blocks and writes
  one JSON object per block (`outputs.json`), matching the required schema.
- `check_models.py` — lists available free-tier Hugging Face models,
  including a vision-capable ("image-text-to-text") search to help pick a
  model that can actually see the question images.
- `app.py` — optional Streamlit dashboard for visually stepping through the
  agent's decisions (not part of the required deliverable — added for local
  debugging).

## OCR provider

**Datalab OCR API** (`ocr_tools.py`), using the $15 free-credit tier.

## Setup

1. Create a `.env` file with:
   ```
   HF_API_KEY=your_huggingface_token
   DATALAB_API_KEY=your_datalab_key
   HF_MODEL_ID=your_chosen_vision_model_id   # see warning below
   ```
2. Install dependencies in your existing virtual environment — no new
   packages were introduced beyond what the project already used
   (`streamlit`, `python-dotenv`, `langchain-core`, `langchain-huggingface`,
   `langgraph`, `huggingface_hub`, `requests`).
3. Run the agent on a single block:
   ```
   python agent.py
   ```
4. Run over a batch of blocks and produce the required JSON output:
   ```
   python run_batch.py
   ```
5. (Optional) Visual dashboard:
   ```
   streamlit run app.py
   ```

## ⚠️ Important: vision-model requirement

The brief requires the agent to look at the question **image**, not just the
OCR text — both to solve the question and to spot likely misreads while
correcting it. The model used in the original version (`openai/gpt-oss-120b`)
is text-only.

`agent.py` now sends the image alongside the text prompt to the configured
model, using LangChain's standard multimodal message format (`image_url`
content blocks). **You need to point `HF_MODEL_ID` at a vision-capable
("image-text-to-text") model that your Hugging Face token has free access
to** — run `check_models.py` to list candidates. If the multimodal call
fails for any reason (model/endpoint doesn't actually support it, etc.), the
agent automatically falls back to a text-only call using just the OCR text
and logs a warning, so the pipeline keeps running — but OCR-error correction
quality will suffer without a real vision model. I wasn't able to verify a
specific free model end-to-end myself (no test data or live network access
during this review) — please confirm one against `check_models.py`'s output
before relying on it for your submission.

## Sample data note

The task brief refers to sample question blocks (with real options and
ground-truth answers) provided for evaluation. Those weren't included in the
files reviewed here, so `run_batch.py` and `mock_tools.py` currently use one
placeholder block/options set for local testing. Replace `BLOCKS` in
`run_batch.py` (and the `options` for each block) with the real dataset
before generating your submission's output file.

See `WRITEUP.md` for the retry cap, best-guess policy, and noise-injection
details required by the brief.