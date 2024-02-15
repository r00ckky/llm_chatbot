from llm import GPT
from uuid import uuid4
import cv2
import pandas as pd
import numpy as np
import os
from face import Face

class ImageMemory:
    def __init__(self, csv_path:str, OPENAI_API_KEY:str, dir_faces:str)->None:
        super().__init__()
        self.face = Face()
        self.csv_path = csv_path
        self.df = pd.read_csv(self.csv_path) if os.path.exists(csv_path) or csv_path==None else pd.DataFrame(columns=['id', 'image_path', 'summary'])
        self.__system_prompt = "Summarise the text below in 30-40 words, for memory to be given to the coming generations."
        self.__llm = GPT(OPENAI_API_KEY, 'gpt-3.5-turbo', self.__system_prompt)
        self.__user_prompt = """
            Previous Summary: {summary}

            Prompt: {prompt}
        """

    def __generate_uni_id(self)->str:
        return str(uuid4)
    
    def __save_new(self, image:np.ndarray, summary:str)->None:
        unique_id = self.__generate_uni_id()
        new_row = {'id': unique_id, 'image': image, 'summary': summary}
        self.df = self.df.append(new_row, ignore_index=True)
    
    def get_summary(self, face_img:np.ndarray, id:str=None): 
        """
        Here we will be calling the oneshot algo to identify the person 
        and return the id and summary for the particular person.
        """
        try:
            self.summary = self.df.loc[self.df['id']==id]['summary'].values

        except KeyError:
            self.summary = 'Someone new is here to chat with you'

        except:
            self.summary = 'Unable to extract the old summary for this person'

        return self.summary, id

    def __save_summary(self, id:str, summary:str)->None:
        self.df.loc[self.df['id'] == id, 'summary'] = summary
    
    def save_data(self, new_path:str=None):
        try:
            self.df.to_csv(new_path if new_path is not None else self.csv_path, index=False)
        except:
            raise OSError('Dataframe not stored provide path')
        
    def generate_new_summary(self, summary:str, prompt:str, id:str):
        user_prompt = self.__user_prompt.format(summary, prompt)
        new_summary = self.__llm.generate_response(user_prompt)
        self.__save_summary(id, new_summary)
        return new_summary