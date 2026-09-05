import os
from dotenv import load_dotenv
from huggingface_hub import HfApi

# Load environment variables
load_dotenv()

# Get the Hugging Face API Token
HF_TOKEN = os.getenv("HF_API_KEY")

if not HF_TOKEN:
    print("❌ Error: HUGGINGFACEHUB_API_TOKEN is missing from .env file.")
    exit()

api = HfApi(token=HF_TOKEN)

print("🔍 Querying Hugging Face Serverless API for free/available text-generation models...\n")

try:
    # Fetch models configured for text-generation that support inference
    models = api.list_models(
        filter="text-generation",
        inference="warm",  # Fetch models currently active/warm on servers
        limit=50
    )

    available_models = []

    for model in models:
        # Filter models that explicitly support the free Hugging Face infrastructure
        if model.pipeline_tag == "text-generation":
            available_models.append(model.id)

    print(f"✅ Found {len(available_models)} available text-only models for your token:\n")
    print("-" * 50)
    for model_id in available_models:
        print(f"• {model_id}")
    print("-" * 50)

except Exception as e:
    print(f"❌ Failed to fetch text-generation models: {e}")

# ---------------------------------------------------------
# The agent needs to SEE the question images (to solve them and to re-read
# them while correcting OCR errors), so it needs a vision-capable model.
# This searches for "image-text-to-text" models your token can use for free.
# Pick one from this list, confirm it actually accepts image inputs on your
# access tier, and set it as HF_MODEL_ID in your .env file.
# ---------------------------------------------------------
print("\n🔍 Querying Hugging Face Serverless API for free/available vision (image-text-to-text) models...\n")

try:
    vision_models = api.list_models(
        filter="image-text-to-text",
        inference="warm",
        limit=50
    )

    vision_model_ids = [m.id for m in vision_models if m.pipeline_tag == "image-text-to-text"]

    print(f"✅ Found {len(vision_model_ids)} available vision-capable models for your token:\n")
    print("-" * 50)
    for model_id in vision_model_ids:
        print(f"• {model_id}")
    print("-" * 50)
    print(
        "\n⚠️ 'Warm'/listed here does not guarantee full compatibility with "
        "LangChain's multimodal message format through HuggingFaceEndpoint — "
        "test your chosen model against agent.py's solve_problem_node before "
        "relying on it for your submission."
    )

except Exception as e:
    print(f"❌ Failed to fetch vision models: {e}")