import json
import os
import time
from typing import Any, Optional

from dotenv import load_dotenv
from google import genai
from google.genai import types

# Load local .env; Render environment variables still take priority.
load_dotenv()


def _first_env(*names: str, default: Optional[str] = None) -> Optional[str]:
    for name in names:
        value = os.getenv(name)
        if value is not None and value.strip() != "":
            return value.strip()
    return default


def _env_int(*names: str, default: int) -> int:
    value = _first_env(*names)
    return int(value) if value is not None else default


def _env_float(*names: str, default: float) -> float:
    value = _first_env(*names)
    return float(value) if value is not None else default


def _dedupe_keep_order(items: list[Optional[str]]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            result.append(item)
    return result


def _strip_json_fences(text: str) -> str:
    cleaned = text.strip()
    if "```" in cleaned:
        cleaned = cleaned.replace("```json", "").replace("```", "").strip()
    return cleaned


class AIAgent:
    def __init__(self):
        self.api_key = _first_env("GEMINI_API_KEY", "GOOGLE_API_KEY")

        self.max_retries = _env_int("GEMINI_MAX_RETRIES", default=2)
        self.retry_delay = _env_float("GEMINI_RETRY_DELAY", default=1.0)

        self.temperature = _env_float("GEMINI_TEMPERATURE", default=0.4)
        self.top_p = _env_float("GEMINI_TOP_P", default=0.9)
        self.top_k = _env_int("GEMINI_TOP_K", default=40)

        preferred_model = _first_env("GEMINI_MODEL")
        self.model_candidates = _dedupe_keep_order(
            [
                preferred_model,
                "gemini-2.5-flash",
                "gemini-2.5-pro",
            ]
        )

        if not self.api_key:
            raise ValueError("GEMINI_API_KEY (or GOOGLE_API_KEY) is not set")

        # Current SDK client pattern.
        self.client = genai.Client(api_key=self.api_key)

    def _generate(
        self,
        prompt: str,
        *,
        system_instruction: Optional[str] = None,
        response_json_schema: Optional[dict[str, Any]] = None,
    ) -> str:
        last_error: Optional[Exception] = None

        for model_name in self.model_candidates:
            for attempt in range(self.max_retries):
                try:
                    config_kwargs: dict[str, Any] = {
                        "temperature": self.temperature,
                        "top_p": self.top_p,
                        "top_k": self.top_k,
                    }

                    if system_instruction:
                        config_kwargs["system_instruction"] = system_instruction

                    if response_json_schema is not None:
                        config_kwargs["response_mime_type"] = "application/json"
                        config_kwargs["response_json_schema"] = response_json_schema

                    config = types.GenerateContentConfig(**config_kwargs)

                    response = self.client.models.generate_content(
                        model=model_name,
                        contents=prompt,
                        config=config,
                    )

                    text = getattr(response, "text", None)
                    if text and text.strip():
                        return text.strip()

                    return "Empty response"

                except Exception as e:
                    last_error = e
                    msg = str(e).lower()

                    # If this model is not available for the account/API version, try the next one.
                    if (
                        "not found" in msg
                        or "unsupported" in msg
                        or "404" in msg
                        or "is not found for api version" in msg
                    ):
                        break

                    if attempt < self.max_retries - 1:
                        time.sleep(self.retry_delay)

        if last_error:
            return f"Error: Gemini failed - {type(last_error).__name__}: {str(last_error)}"
        return "Error: Gemini failed"

    def ask_question(self, query: str, context: str) -> str:
        prompt = f"""
You are a strict document-based AI assistant.

RULES:
- Use ONLY the provided context
- If not found, say: Not found in document
- Be concise

CONTEXT:
{context}

QUESTION:
{query}

ANSWER:
"""
        return self._generate(prompt)

    def generate_quiz(self, context: str, num_questions: int = 5) -> str:
        schema = {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "question": {"type": "string"},
                    "options": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 4,
                        "maxItems": 4,
                    },
                    "answer_index": {"type": "integer"},
                },
                "required": ["question", "options", "answer_index"],
                "additionalProperties": False,
            },
        }

        prompt = f"""
Create {num_questions} MCQs from the context.
Return ONLY valid JSON.

Context:
{context}
"""
        result = self._generate(
            prompt,
            response_json_schema=schema,
        )

        try:
            cleaned = _strip_json_fences(result)
            parsed = json.loads(cleaned)
            return json.dumps(parsed, indent=2, ensure_ascii=False)
        except Exception:
            return result

    def explain_simply(self, context: str) -> str:
        prompt = f"""
Explain in simple bullet points:

{context}
"""
        return self._generate(prompt)

    def handle_agent_task(self, task: str, context: str) -> str:
        prompt = f"""
Task: {task}

Context:
{context}

Return step-by-step structured answer.
"""
        return self._generate(prompt)