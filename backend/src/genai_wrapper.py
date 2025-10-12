"""Small compatibility wrapper around google.generativeai to provide
`complete(prompt)` and `embed(text)` used by the codebase.
This avoids importing deprecated llama-index Gemini adapters and centralizes
GenAI usage.
"""
from typing import Any

class GenAIResponse:
    def __init__(self, text: str):
        self.text = text

class GenAIWrapper:
    def __init__(self, model_name: str, api_key: str):
        try:
            import google.generativeai as genai
        except Exception as e:
            raise RuntimeError(f"google.generativeai not available: {e}")

        if not api_key:
            raise ValueError("API key is required for GenAIWrapper")

        genai.configure(api_key=api_key)
        self._genai = genai
        # construct a GenerativeModel helper
        try:
            self._model = genai.GenerativeModel(model_name)
            self.model_name = model_name
        except Exception as e:
            # surface clear error upward
            raise RuntimeError(f"Failed to initialize GenerativeModel {model_name}: {e}")

    def complete(self, prompt: str) -> GenAIResponse:
        # generate_content accepts many input shapes; pass prompt as contents
        resp = self._model.generate_content(prompt)
        # Try to extract text from response in a robust way
        try:
            # resp.result.candidates[0].content.parts is commonly present
            parts = resp.result.candidates[0].content.parts
            if parts and isinstance(parts, list):
                text = "".join([p.text if hasattr(p, 'text') else str(p) for p in parts])
            else:
                text = str(parts)
        except Exception:
            # Fallback: try to stringify object
            try:
                text = getattr(resp, 'text', str(resp))
            except Exception:
                text = str(resp)
        return GenAIResponse(text)

    def embed(self, texts: Any, model: str = None):
        # Use top-level embed_content if available
        genai = self._genai
        try:
            if model:
                return genai.embed_content(model=model, content=texts)
            return genai.embed_content(content=texts)
        except Exception as e:
            raise RuntimeError(f"Embedding failed: {e}")
