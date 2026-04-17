import os
import google.generativeai as genai
from dotenv import load_dotenv
import time
import json

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

    def _generate(self, prompt: str, max_retries: int = None) -> str:
        if not self.model:
            return "Error: Gemini model not initialized."

        if max_retries is None:
            max_retries = self.max_retries

        for attempt in range(max_retries):
            try:
                response = self.model.generate_content(
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.4,
                        top_p=0.9,
                        top_k=40
                    )
                )

                return response.text if response.text else "Empty response"

            except Exception:
                if attempt < max_retries - 1:
                    time.sleep(self.retry_delay)
                else:
                    return "Error: Gemini generation failed"

        return "Error: Failed after retries"

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

    def generate_quiz(self, context: str, num_questions: int = 5) -> str:
        prompt = f"""
Create {num_questions} MCQs.

Return ONLY valid JSON:

[
  {{
    "question": "...",
    "options": ["A", "B", "C", "D"],
    "answer": 0
  }}
]

Context:
{context}
"""
        result = self._generate(prompt)

        try:
            return json.dumps(json.loads(result), indent=2)
        except:
            return result

    def explain_simply(self, context: str) -> str:
        prompt = f"""
Explain simply:

- bullet points
- simple words

Context:
{context}
"""
        return self._generate(prompt)

    def handle_agent_task(self, task: str, context: str) -> str:
        prompt = f"""
Task: {task}

Context:
{context}

Return structured explanation.
"""
        return self._generate(prompt)