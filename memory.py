from typing import Any, Dict, List
from cs50 import SQL
from langchain.chains import ConversationChain
from langchain.schema import BaseMemory
from langchain_openai import ChatOpenAI
from pydantic import BaseModel
from langchain.prompts.prompt import PromptTemplate
import cv2
import pandas as pd
import numpy as np
class ImageMemoryWithDatabase(BaseMemory, BaseMemory):
    def __init__(self, db_path:str, OPENAI_API_KEY)->None:
        super().__init__()
        self.db_path = db_path
        self.db = SQL(f"sqlite:///{self.db_path}")
        self.db.execute("""
            CREATE TABLE IF NOT EXISTS llm_info (
                id TEXT PRIMARY KEY,
                image BLOB NOT NULL,
                summary TEXT NOT NULL
            )
        """)
        template = """
            Summarise the text below in 1-2 sentences, for memory to be given to the coming generations.
            
            Previous Summary: {summary}

            Prompt: {prompt}
        """
        self.prompt_template = PromptTemplate(input_variables = ['summary', 'prompt'], template = template)
        self.llm = ChatOpenAI(
            model = "gpt-3.5-turbo",
            temperature = 0.7,
            token = OPENAI_API_KEY,
            template = self.prompt_template,
        )

    def save_new(self, image:np.ndarray, summary:str)->None:
        self.db.execute("""
            INSERT INTO llm_info (image, summary) VALUES (?, ?)
        """, (image, summary))
    
    def get_summary(self, id:str):
        result_set =  self.db.execute(f"""
            SELECT summary FROM llm_info WHERE id = :id
        """, id=id)
        
        return result_set.fetchall()
    
    def save_summary(self, id:str, summary:str)->None:
        self.db.execute("""
            UPDATE llm_info SET summary = :summary WHERE id = :id
        """, summary=summary, id=id)

    @property
    def memory_variables(self)-> List[str]:
        return ["summary"]
    
    def load_memory_variables(self, id:str)-> Dict[str, Any]:
        result_set =  self.get_summary(id)
        
        return result_set.fetchall()
    
    def save_context(self, id:str, memory_variables:Dict[str, Any])->None:
        summary = memory_variables["summary"]
        self.save_summary(id, summary)
