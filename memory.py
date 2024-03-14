from llm import GPT
from uuid import uuid4
import cv2
import pandas as pd
import numpy as np
import os
import pymongo as pym
from face import Face
from summariser import *
from dotenv import load_dotenv
load_dotenv()
MANGODB_CONNECTION_STRING = os.environ.get('MANGODB_CONNECTION_STRING')
class ImageMemory:
    def __init__(self, cliet_path:str):
        super().__init__()
        self.__client = pym.MongoClient(cliet_path)
        self.__database = self.__client["ImageMemory"]
        self.__face = Face()
        self.__face.face_name = [doc["_id"] for doc in self.__database.find({}, {'_id':1})]
        self.__face.known_face_encoding = [doc['face_encoding'] for doc in self.__database.find({},{'face_encoding':1})]
        
    def __retrieve_summary(self, img:np.array, area_max:bool):
        names, face_encodings, indexs = self.__face.get_face_info(img, area_max=area_max)
        summary = []
        for i in range(len(indexs)):
            if indexs[i]==0:
                self.__database.insert_one({
                    "_id":f"{names[i]}",
                    "face_encoding":f"{face_encodings[i]}",
                    "summary": "They are new"
                })
                summary.append("They are new")
            else:
                summary.append(self.__database.find({"_id":names[i],}, {"summary":1, "_id":0, "face_encoding":0})["summary"])
        return summary
    
    def retrieve_summary(self, img:np.array, area_max:bool, prompt:str):
        summary = self.__retrieve_summary(img, area_max=area_max)
        new_summary = get_summary(prompt=prompt, summary="\n".join(summary))
        return new_summary