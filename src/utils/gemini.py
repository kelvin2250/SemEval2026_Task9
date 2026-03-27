import json
import time
import re
import os
from typing import Any, Optional
import importlib

_GENAI_CONTEXT = None


def _configure_genai_context():
    global _GENAI_CONTEXT
    if _GENAI_CONTEXT is not None:
        return _GENAI_CONTEXT

    try:
        dotenv_module = importlib.import_module("dotenv")
        dotenv_module.load_dotenv()
    except Exception:
        pass

    genai = importlib.import_module("google.generativeai")
    types_module = importlib.import_module("google.generativeai.types")
    harm_block_threshold = types_module.HarmBlockThreshold
    harm_category = types_module.HarmCategory

    genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))

    safety_settings = {
        harm_category.HARM_CATEGORY_HATE_SPEECH: harm_block_threshold.BLOCK_NONE,
        harm_category.HARM_CATEGORY_HARASSMENT: harm_block_threshold.BLOCK_NONE,
        harm_category.HARM_CATEGORY_SEXUALLY_EXPLICIT: harm_block_threshold.BLOCK_NONE,
        harm_category.HARM_CATEGORY_DANGEROUS_CONTENT: harm_block_threshold.BLOCK_NONE,
    }

    _GENAI_CONTEXT = (genai, safety_settings)
    return _GENAI_CONTEXT


def extract_json(text: Optional[str]) -> Optional[Any]:
    if not text:
        return None

    text = text.strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE)
    match = re.search(r"(\{.*\}|\[.*\])", text, flags=re.DOTALL)
    if not match:
        return None

    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def call_gemini(
    model_name: str,
    prompt: str,
    system_instruction: Optional[str] = None,
    temperature: float = 0.7,
    max_retries: int = 3,
    retry_sleep_seconds: int = 10,
    response_mime_type: Optional[str] = None,
):
    last_error = None
    genai, safety_settings = _configure_genai_context()
    for attempt in range(max_retries):
        try:
            model = genai.GenerativeModel(
                model_name=model_name,
                system_instruction=system_instruction,
            )

            generation_config = genai.types.GenerationConfig(
                temperature=temperature,
                response_mime_type=response_mime_type,
            )

            response = model.generate_content(
                prompt,
                safety_settings=safety_settings,
                generation_config=generation_config,
            )
            return response.text
        except Exception as error:  # noqa: BLE001
            last_error = error
            if attempt < max_retries - 1:
                time.sleep(retry_sleep_seconds)

    print(f"Error calling {model_name}: {last_error}")
    return None
