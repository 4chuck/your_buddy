import os
import google.generativeai as genai
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()

# Configure the Gemini API key
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    print("WARNING: GEMINI_API_KEY environment variable is not set. Some features will not work.")

if api_key:
    try:
        genai.configure(api_key=api_key)
    except Exception as e:
        print(f"ERROR: Failed to configure Gemini API: {e}")

class AIAgent:
    def __init__(self):
        # We can use gemini-1.5-flash for faster responses and lower cost/free tier
        self.model = None
        self.max_retries = 2
        self.retry_delay = 1
        
        try:
            self.model = genai.GenerativeModel('gemini-2.0-flash')
        except Exception as e:
            print(f"ERROR: Failed to initialize Gemini model: {e}")
            # Fallback to older model
            try:
                self.model = genai.GenerativeModel('gemini-1.5-flash')
            except Exception as e2:
                print(f"ERROR: Failed to initialize fallback model: {e2}")
                self.model = None

    def _generate(self, prompt: str, max_retries: int = None) -> str:
        if not self.model:
            return "Error: Gemini API is not available. Please check your GEMINI_API_KEY and API quota."
        
        if max_retries is None:
            max_retries = self.max_retries
        
        for attempt in range(max_retries):
            try:
                response = self.model.generate_content(
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.7,
                        top_p=0.95,
                        top_k=40,
                    )
                )
                if response.text:
                    return response.text
                else:
                    return "Error: Empty response from API. Please try again."
            except genai.types.BlockedPromptException:
                return "Error: The request was blocked by Gemini's safety filters. Please rephrase your query."
            except genai.types.StopCandidateException as e:
                return f"Error: Generation stopped unexpectedly. Reason: {str(e)}"
            except Exception as e:
                error_msg = str(e)
                if "RESOURCE_EXHAUSTED" in error_msg or "429" in error_msg:
                    return "Error: API quota exceeded. Please try again later."
                elif "UNAUTHENTICATED" in error_msg or "401" in error_msg:
                    return "Error: API key is invalid or expired. Please check your GEMINI_API_KEY."
                elif attempt < max_retries - 1:
                    print(f"Attempt {attempt + 1} failed, retrying in {self.retry_delay}s: {e}")
                    time.sleep(self.retry_delay)
                else:
                    return f"Error: Failed to generate response after {max_retries} attempts. Details: {error_msg}"
        
        return "Error: Unable to process request. Please try again."

    def ask_question(self, query: str, context: str) -> str:
        if not query or not query.strip():
            return "Error: Please provide a question."
        if not context or not context.strip():
            return "Error: No document context available. Please upload a file first."
        
        prompt = f"""You are a helpful AI study assistant. Use the following context to answer the user's question accurately and concisely.
If the answer is not found in the context, explicitly state "I cannot find the answer in the provided document."

Context:
{context}

Question: {query}

Answer:"""
        return self._generate(prompt)

    def generate_quiz(self, context: str, num_questions: int = 5) -> str:
        if not context or not context.strip():
            return "Error: No document context available. Please upload a file first."
        if num_questions < 1 or num_questions > 20:
            num_questions = 5
        
        prompt = f"""Generate exactly {num_questions} multiple choice questions based on the provided text.
Return ONLY a valid JSON array with no markdown, no code blocks, no extra text. Each object must have:
- "question": the question text
- "options": array of exactly 4 answer choices
- "answer_index": integer (0, 1, 2, or 3) for the correct answer

Context:
{context}

Return only valid JSON array, nothing else:"""
        return self._generate(prompt, max_retries=1)

    def explain_simply(self, context: str) -> str:
        if not context or not context.strip():
            return "Error: No document context available. Please upload a file first."
        
        prompt = f"""Explain the following text in very simple terms as if teaching a 10-year-old.
Use everyday analogies, simple language, and short sentences. Avoid jargon and technical terms.
Focus on the "why" and "how" in a way anyone can understand.

Context:
{context}

Simple Explanation:"""
        return self._generate(prompt)

    def handle_agent_task(self, task: str, context: str) -> str:
        """
        General agent task executor. Breaks down complex tasks into structured steps.
        Designed for multi-step analytical reasoning and synthesis.
        """
        if not task or not task.strip():
            return "Error: Please provide a task description."
        if not context or not context.strip():
            return "Error: No document context available. Please upload a file first."
        
        prompt = f"""You are an expert AI assistant solving a complex multi-step analytical task.

Task: {task}

Follow this structured approach:
1. **Analyze**: Identify key components and requirements
2. **Research**: Extract relevant information from context
3. **Plan**: Outline step-by-step approach
4. **Execute**: Perform each step with detailed reasoning
5. **Synthesize**: Provide comprehensive, well-structured final answer

Use clear formatting with sections and bullet points. Include reasoning for conclusions.

Context:
{context}

Structured Response:"""
        return self._generate(prompt)
