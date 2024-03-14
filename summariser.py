from llm import GPT
from dotenv import load_dotenv
import os

load_dotenv()
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
summary_llm = GPT(key = OPENAI_API_KEY, model = 'gpt-3.5-turbo-0125', system_prompt='Sumaise the information in 30-40 words, for the next model in chain')

def get_summary(prompt:str, summary:str):
    global summary_llm
    prompt_ =f"""
        Previous Summary: {summary}
        Prompt: {prompt}
        """
    return summary_llm.generate_response(prompt_, 0.1)