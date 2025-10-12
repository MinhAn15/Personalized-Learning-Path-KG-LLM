import os
import warnings

# Suppress specific warnings coming from dependencies (field attribute / deprecation)
warnings.filterwarnings("ignore", category=UserWarning, message=".*validate_default.*")
warnings.filterwarnings("ignore", category=DeprecationWarning)

from dotenv import load_dotenv
import google.generativeai as genai

# Load local .env (same behavior as backend config)
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
model_name = "models/gemini-pro"

if not api_key:
    print("No GEMINI_API_KEY or GOOGLE_API_KEY found in environment. Set one and retry.")
else:
    # Ensure GOOGLE_API_KEY is set for downstream libraries that look for it
    if os.getenv("GEMINI_API_KEY") and not os.getenv("GOOGLE_API_KEY"):
        os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY")

    # Configure the Google GenAI SDK
    genai.configure(api_key=api_key)

    try:
        prompt = "Xin chào Gemini! Hãy trả lời: 1+1 bằng mấy?"

        # Prefer a stable model name discovered from list_models
        preferred_model = 'models/gemini-pro-latest'

        try:
            # Use GenerativeModel helper which is available in this SDK
            GM = genai.GenerativeModel(preferred_model)
            print('Constructed GenerativeModel for', preferred_model)
            # generate_content expects the prompt as the first positional argument (contents)
            result = GM.generate_content(prompt)
            print('Generation result:', result)
        except Exception as e:
            print('GenerativeModel.generate_content failed:', repr(e))

    except Exception as e:
        print("Lỗi khi gọi Gemini (genai):", repr(e))
