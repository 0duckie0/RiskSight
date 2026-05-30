import os
from dotenv import load_dotenv
from google import genai

# Load your exact API Key
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(BASE_DIR, ".env"))

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

print("🕵️ Interrogating Google Servers...")
print("Here are the exact models unlocked for your API Key:\n")

# Brutally simple: just print the name of every model found
for model in client.models.list():
    print(f"✅ {model.name}")