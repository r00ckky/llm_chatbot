from typing import Any, Dict, List
from cs50 import SQL
from langchain.chains import ConversationChain
from langchain.schema import BaseMemory
from langchain_openai import OpenAI
from pydantic import BaseModel
import cv2
import pandas as pd
import numpy as np
class ImageMemoryWithDatabase(BaseModel, BaseMemory):
    def __init__(self, db_path:str)->None:
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
    def save(self, image:np.ndarray, summary:str)->None:
        self.db.execute("""
            INSERT INTO llm_info (image, summary) VALUES (?, ?)
        """, image, summary)
    
    def get_summary(self, id:str):
        result_set =  self.db.execute(f"""
            SELECT summary FROM llm_info WHERE id = :id
        """, id=id)
        
        return result_set.fetchall()

    @property
    def memory_variables(self)-> List[str]:
        return ["summary"]
    
    def load_memory_variables(self, id:str)-> Dict[str, Any]:
        result_set =  self.get_summary(id)
        
        return result_set.fetchall()
    

