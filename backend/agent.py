import os
import google.generativeai as genai
from dotenv import load_dotenv
import time
import json

# Load env
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

        # safer model selection (stable for hackathon)
        try:
            self.model = genai.GenerativeModel("gemini-1.5-flash")
        except Exception as e:
            print(f"Model init error: {e}")
            self.model = None

    # ----------------------------
    # CORE GENERATION ENGINE
    # ----------------------------
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
                        temperature=0.4,   # lower = more factual (IMPORTANT)
                        top_p=0.9,
                        top_k=40
                    )
                )

                return response.text if response.text else "Empty response"

            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(self.retry_delay)
                else:
                    return f"Error: {str(e)}"

        return "Error: Failed after retries"

    # ----------------------------
    # QA MODE (NotebookLM STYLE)
    # ----------------------------
    def ask_question(self, query: str, context: str) -> str:
        prompt = f"""
You are NotebookLM-style AI assistant.

RULES:
- Use ONLY the context
- If answer not in context say: "Not found in document"
- Be concise and factual

CONTEXT:
{context}

QUESTION:
{query}

ANSWER:
"""
        return self._generate(prompt)

    # ----------------------------
    # QUIZ MODE (IMPROVED - STRUCTURED JSON)
    # ----------------------------
    def generate_quiz(self, context: str, num_questions: int = 5) -> str:
        prompt = f"""
You are a quiz generator.

Create exactly {num_questions} MCQs from the context.

RULES:
- Only from context
- No repetition
- Medium difficulty
- STRICT JSON OUTPUT ONLY

FORMAT:
[
  {{
    "question": "...",
    "options": ["A", "B", "C", "D"],
    "answer": 0
  }}
]

Context:
{context}

OUTPUT ONLY JSON:
"""
        result = self._generate(prompt)

        # try safe JSON parsing (VERY IMPORTANT FOR FRONTEND)
        try:
            return json.dumps(json.loads(result), indent=2)
        except:
            return result  # fallback raw

    # ----------------------------
    # SIMPLIFY MODE
    # ----------------------------
    def explain_simply(self, context: str) -> str:
        prompt = f"""
Explain this content in very simple terms:

RULES:
- Use bullet points
- Use simple language
- Add small examples

CONTENT:
{context}

EXPLANATION:
"""
        return self._generate(prompt)

    # ----------------------------
    # AGENT MODE (SMART ANALYSIS)
    # ----------------------------
    def handle_agent_task(self, task: str, context: str) -> str:
        prompt = f"""
You are an expert AI study assistant.

TASK:
{task}

CONTEXT:
{context}

RULES:
- Structured response
- Step-by-step reasoning
- Clear headings
- No hallucination outside context

ANSWER:
"""
        return self._generate(prompt)