import json
import os
import re
import time
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from google import genai

# Load .env locally; on Render this is harmless and environment variables still win.
load_dotenv()


def _first_env(*names: str, default: Optional[str] = None) -> Optional[str]:
    for name in names:
        value = os.getenv(name)
        if value is not None and value.strip() != "":
            return value.strip()
    return default


def _dedupe_keep_order(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _strip_json_fences(text: str) -> str:
    cleaned = text.strip()
    if "```" in cleaned:
        cleaned = cleaned.replace("```json", "").replace("```", "").strip()
    return cleaned


class AIAgent:
    def __init__(self):
        self.api_key = _first_env("GEMINI_API_KEY", "GOOGLE_API_KEY")
        self.max_retries = int(_first_env("GEMINI_MAX_RETRIES", default="2") or "2")
        self.retry_delay = float(_first_env("GEMINI_RETRY_DELAY", default="1") or "1")
        self.temperature = float(_first_env("GEMINI_TEMPERATURE", default="0.4") or "0.4")
        self.top_p = float(_first_env("GEMINI_TOP_P", default="0.9") or "0.9")
        self.top_k = int(_first_env("GEMINI_TOP_K", default="40") or "40")

        # Keep the model configurable so you can switch in Render without code changes.
        preferred_model = _first_env("GEMINI_MODEL")
        self.model_candidates = _dedupe_keep_order(
            [
                preferred_model,
                "gemini-2.5-flash",
                "gemini-2.5-pro",
                "gemini-3-flash-preview",
            ]
        )

        if self.api_key:
            self.client = genai.Client(api_key=self.api_key)
        else:
            # This can still work if GEMINI_API_KEY is present in the environment.
            self.client = genai.Client()

    def _base_config(self) -> Dict[str, Any]:
        return {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
        }

    def _generate(
        self,
        prompt: str,
        *,
        system_instruction: Optional[str] = None,
        response_json_schema: Optional[Dict[str, Any]] = None,
    ) -> str:
        if not self.api_key and not _first_env("GEMINI_API_KEY", "GOOGLE_API_KEY"):
            return "Error: GEMINI_API_KEY is not set."

        last_error: Optional[Exception] = None

        for model_name in self.model_candidates:
            for attempt in range(self.max_retries):
                try:
                    config: Dict[str, Any] = self._base_config()

                    if system_instruction:
                        config["system_instruction"] = system_instruction

                    if response_json_schema is not None:
                        config["response_mime_type"] = "application/json"
                        config["response_json_schema"] = response_json_schema

                    response = self.client.models.generate_content(
                        model=model_name,
                        contents=prompt,
                        config=config,
                    )

                    text = (response.text or "").strip()
                    if text:
                        return text
                    return "Empty response"

                except Exception as e:
                    last_error = e
                    msg = str(e).lower()

                    # If the model is unavailable in this account / API version, try the next one.
                    if (
                        "404" in msg
                        or "not found" in msg
                        or "unsupported for generatecontent" in msg
                        or "is not found for api version" in msg
                    ):
                        break

                    if attempt < self.max_retries - 1:
                        time.sleep(self.retry_delay)

        return f"Error: Gemini failed - {str(last_error)}" if last_error else "Error: Gemini failed"

    def ask_question(self, query: str, context: str) -> str:
        prompt = f"""
You are a strict document-based AI assistant.

RULES:
- Use ONLY the provided context
- If the answer is not in the context, say: Not found in document
- Be concise

CONTEXT:
{context}

QUESTION:
{query}

ANSWER:
"""
        return self._generate(prompt)

    def generate_quiz(self, context: str, num_questions: int = 5) -> str:
        quiz_schema = {
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
                    "answer_index": {"type": "integer", "minimum": 0, "maximum": 3},
                },
                "required": ["question", "options", "answer_index"],
                "additionalProperties": False,
            },
        }

        prompt = f"""
Create {num_questions} multiple-choice questions from the context.

Return only JSON that matches the provided schema.

Context:
{context}
"""
        result = self._generate(
            prompt,
            response_json_schema=quiz_schema,
        )

        try:
            cleaned = _strip_json_fences(result)
            parsed = json.loads(cleaned)
            return json.dumps(parsed, indent=2, ensure_ascii=False)
        except Exception:
            return result

    def explain_simply(self, context: str) -> str:
        prompt = f"""
Explain the following in simple bullet points.
Use short sentences and simple language.

Context:
{context}
"""
        return self._generate(prompt)

    def handle_agent_task(self, task: str, context: str) -> str:
        prompt = f"""
Task: {task}

Context:
{context}

Return a structured step-by-step response.
"""
        return self._generate(prompt)