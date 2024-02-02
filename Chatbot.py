import os
import langchain
from langchain_openai import ChatOpenAI
from langchain import PromptTemplate
from langchain_community.callbacks import get_openai_callback
from dotenv import load_dotenv
from memory import ImageMemory
from llm import GPT

load_dotenv()
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

class LLM_Chain:
    def __init__(self, openai_api_key, model:str='gpt-3.5-turbo', temperature=0.7):
        self.__chatbot = ChatOpenAI(
            model = model,
            temperature = temperature,
            openai_api_key = openai_api_key,
        )
        self.memory = ImageMemory(openai_api_key)

if __name__ == "__main__":
    llm_chain = LLM_Chain(OPENAI_API_KEY)
    while True:
        query = input(">> ")
        response, tokens = llm_chain.generate_response(query)
        print(f"Response: {response}")
        print(f"Tokens: {tokens}")


