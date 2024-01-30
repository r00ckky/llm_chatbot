from typing import Any, Dict, List
from cs50 import SQL
from langchain.chains import ConversationChain
from langchain.schema import BaseMemory
from langchain_openai import OpenAI
from pydantic import BaseModel

class ImageMemory(BaseModel, BaseMemory):
    def __init__(self)->None:
        pass
