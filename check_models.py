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
            
    print(f"✅ Found {len(available_models)} available models for your token:\n")
    print("-" * 50)
    for model_id in available_models:  # Show top 15 models
        print(f"• {model_id}")
    print("-" * 50)

except Exception as e:
        print(f"❌ Failed to fetch models: {e}")