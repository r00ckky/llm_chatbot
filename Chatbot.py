import os
from langchain_openai import ChatOpenAI
from langchain import PromptTemplate
from langchain.memory import ConversationBufferMemory
# from langchain.prompts import (
#     ChatPromptTemplate,
#     MessagesPlaceholder,
#     SystemPromptTemplate,
#     HumanMessagePromptTemplate
# )
from langchain.chains.conversation.memory import (
    ConversationBufferMemory,
    ConversationSummaryMemory,
    ConversationBufferWindowMemory,
    ConversationKGMemory
)

from langchain_community.callbacks import get_openai_callback

from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')


