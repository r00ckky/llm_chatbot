import os
import langchain
from langchain_openai import ChatOpenAI
from langchain import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains.conversation.memory import ConversationKGMemory
from langchain_community.callbacks import get_openai_callback
from langchain.chains import LLMChain, ConversationChain
from dotenv import load_dotenv


load_dotenv()
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

class LLM_Chain:
    def __init__(self, openai_api_key, memory:str=None, model:str='gpt-3.5-turbo', temperature=0.7):
        self.chatbot = ChatOpenAI(
            model = model,
            temperature = temperature,
            openai_api_key = openai_api_key,
        )

        self.chain = ConversationChain(
            llm = self.chatbot,
            memory = ConversationKGMemory(
                llm=self.chatbot,
            ),
        )

    def generate_response(self, query):
        with get_openai_callback() as cb:
            result = self.chain.run(query)
            return result, cb.total_tokens

if __name__ == "__main__":
    llm_chain = LLM_Chain(OPENAI_API_KEY)
    while True:
        query = input(">> ")
        response, tokens = llm_chain.generate_response(query)
        print(f"Response: {response}")
        print(f"Tokens: {tokens}")


