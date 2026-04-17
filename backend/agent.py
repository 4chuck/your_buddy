import os
import json
import time
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if api_key:
    genai.configure(api_key=api_key)
else:
    print("WARNING: GEMINI_API_KEY not set")


class AIAgent:
    def __init__(self):
        self.model = None
        self.max_retries = 2
        self.retry_delay = 1

        try:
            self.model = genai.GenerativeModel("gemini-1.5-flash")
        except Exception as e:
            print(f"Model init error: {e}")
            self.model = None

    # ---------------- SAFE GENERATION ----------------
    def _generate(self, prompt: str) -> str:
        if not self.model:
            return "Error: Gemini model not initialized."

        for attempt in range(self.max_retries):
            try:
                response = self.model.generate_content(
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.4,
                        top_p=0.9,
                        top_k=40
                    )
                )

                if response and response.text:
                    return response.text.strip()

                return "Empty response"

            except Exception as e:
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                else:
                    return f"Error: Gemini failed - {str(e)}"

        return "Error: Failed after retries"

    # ---------------- QA ----------------
    def ask_question(self, query: str, context: str) -> str:
        prompt = f"""
You are a strict document-based AI assistant.

RULES:
- Use ONLY context
- If not found say: Not found in document
- Be concise

CONTEXT:
{context}

QUESTION:
{query}

ANSWER:
"""
        return self._generate(prompt)

    # ---------------- QUIZ ----------------
    def generate_quiz(self, context: str, num_questions: int = 5) -> str:
        prompt = f"""
Create {num_questions} MCQs.

Return ONLY valid JSON:
[
  {{
    "question": "string",
    "options": ["A", "B", "C", "D"],
    "answer_index": 0
  }}
]

Context:
{context}
"""

        result = self._generate(prompt)

        # ---------------- SAFE JSON PARSE ----------------
        try:
            cleaned = result.strip()

            # remove markdown fences if Gemini adds them
            if "```" in cleaned:
                cleaned = cleaned.replace("```json", "").replace("```", "").strip()

            parsed = json.loads(cleaned)
            return json.dumps(parsed, indent=2)

        except Exception:
            return result

    # ---------------- SIMPLIFY ----------------
    def explain_simply(self, context: str) -> str:
        prompt = f"""
Explain the following in simple bullet points:

- simple language
- short sentences

Context:
{context}
"""
        return self._generate(prompt)

    # ---------------- AGENT ----------------
    def handle_agent_task(self, task: str, context: str) -> str:
        prompt = f"""
Task: {task}

Context:
{context}

Return structured step-by-step response.
"""
        return self._generate(prompt)