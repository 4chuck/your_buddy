import json
import logging
import os
import time
import traceback
from typing import Any, Optional

from dotenv import load_dotenv
from google import genai
from google.genai import types

# Load local .env; Render environment variables still take priority.
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _first_env(*names: str, default: Optional[str] = None) -> Optional[str]:
    for name in names:
        value = os.getenv(name)
        if value is not None and value.strip() != "":
            return value.strip()
    return default


def _env_int(*names: str, default: int) -> int:
    value = _first_env(*names)
    try:
        return int(value) if value is not None else default
    except ValueError:
        return default


def _env_float(*names: str, default: float) -> float:
    value = _first_env(*names)
    try:
        return float(value) if value is not None else default
    except ValueError:
        return default


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


def _truncate(text: str, max_chars: int) -> str:
    return text if len(text) <= max_chars else text[:max_chars]


class AIAgent:
    def __init__(self):
        self.api_key = _first_env("GEMINI_API_KEY", "GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY (or GOOGLE_API_KEY) is not set")

        self.client = genai.Client(api_key=self.api_key)

        self.max_retries = _env_int("GEMINI_MAX_RETRIES", default=2)
        self.retry_delay = _env_float("GEMINI_RETRY_DELAY", default=1.0)

        self.temperature = _env_float("GEMINI_TEMPERATURE", default=0.4)
        self.top_p = _env_float("GEMINI_TOP_P", default=0.9)
        self.top_k = _env_int("GEMINI_TOP_K", default=40)

        self.max_context_chars = _env_int("GEMINI_MAX_CONTEXT_CHARS", default=12000)

        preferred_model = _first_env("GEMINI_MODEL")
        self.model_candidates = _dedupe_keep_order(
            [
                preferred_model,
                "gemini-2.5-flash",
                "gemini-2.5-pro",
            ]
        )

    def _build_config(
        self,
        *,
        system_instruction: Optional[str] = None,
        response_json_schema: Optional[dict[str, Any]] = None,
    ) -> types.GenerateContentConfig:
        kwargs: dict[str, Any] = {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
        }

        if system_instruction:
            kwargs["system_instruction"] = system_instruction

        if response_json_schema is not None:
            kwargs["response_mime_type"] = "application/json"
            kwargs["response_json_schema"] = response_json_schema

        return types.GenerateContentConfig(**kwargs)

    def _generate(
        self,
        prompt: str,
        *,
        system_instruction: Optional[str] = None,
        response_json_schema: Optional[dict[str, Any]] = None,
        expect_json: bool = False,
    ) -> str:
        last_error: Optional[Exception] = None

        for model_name in self.model_candidates:
            for attempt in range(self.max_retries):
                try:
                    config = self._build_config(
                        system_instruction=system_instruction,
                        response_json_schema=response_json_schema,
                    )

                    response = self.client.models.generate_content(
                        model=model_name,
                        contents=prompt,
                        config=config,
                    )

                    text = getattr(response, "text", None)
                    if text and text.strip():
                        return text.strip()

                    if expect_json:
                        return json.dumps(
                            {
                                "status": "error",
                                "message": "Empty response from AI",
                            }
                        )

                    return "Something went wrong. Please try again."

                except Exception as e:
                    last_error = e
                    logger.warning(
                        "Gemini attempt failed (model=%s, attempt=%s/%s): %s",
                        model_name,
                        attempt + 1,
                        self.max_retries,
                        str(e),
                    )

                    if attempt < self.max_retries - 1:
                        time.sleep(self.retry_delay)

        if last_error:
            logger.error("Gemini generation failed: %s", str(last_error))
            traceback.print_exc()

            if expect_json:
                return json.dumps(
                    {
                        "status": "error",
                        "message": str(last_error),
                    }
                )

            return "Something went wrong. Please try again."

        if expect_json:
            return json.dumps(
                {
                    "status": "error",
                    "message": "Unknown error",
                }
            )

        return "Something went wrong. Please try again."

    def ask_question(self, query: str, context: str) -> str:
        context = _truncate(context, self.max_context_chars)

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
        context = _truncate(context, self.max_context_chars)

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
                    "answer_index": {"type": "integer", "minimum": 0, "maximum": 3},
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
            expect_json=True,
        )

        try:
            cleaned = _strip_json_fences(result)
            parsed = json.loads(cleaned)

            if isinstance(parsed, dict) and parsed.get("status") == "error":
                return json.dumps(parsed, ensure_ascii=False)

            if not isinstance(parsed, list):
                return json.dumps(
                    {
                        "status": "error",
                        "message": "Quiz response was not a JSON array",
                    },
                    ensure_ascii=False,
                )

            return json.dumps(parsed, indent=2, ensure_ascii=False)

        except Exception as e:
            logger.error("Quiz JSON parse failed: %s", str(e))
            return json.dumps(
                {
                    "status": "error",
                    "message": "Invalid quiz JSON from AI",
                },
                ensure_ascii=False,
            )

    def explain_simply(self, context: str) -> str:
        context = _truncate(context, self.max_context_chars)

        prompt = f"""
Explain the following in simple bullet points.

Use short sentences and simple language.

Context:
{context}
"""
        return self._generate(prompt)

    def handle_agent_task(self, task: str, context: str) -> str:
        context = _truncate(context, self.max_context_chars)

        prompt = f"""
Task: {task}

Context:
{context}

Return a structured step-by-step response.
"""
        return self._generate(prompt)