import pymongo
import os
import face_recognition as fr
import cv2
import numpy as np
import torch
import torchvision
from torchvision.transforms import v2
from pymongo import MongoClient
from torch import nn
from typing import List, Dict

MONGO_CONNECTION_STRING = os.environ.get('MONGODB_ATLAS_CONNECTION_STRING')

T = v2.Compose([
        v2.ToTensor(),
        v2.Resize(256),
        v2.CenterCrop(224),
        v2.Normalize(mean = [0.48235, 0.45882, 0.40784], 
                     std=[0.00392156862745098, 0.00392156862745098, 0.00392156862745098])
    ])

class ResLink(nn.Module):
    def __init__(self, in_ch) -> None:
        super(ResLink, self).__init__()
        self.con1 = nn.Conv2d(in_ch, in_ch*2, kernel_size=(3,3), stride=(1,1), padding=(1,1), bias=True)
        self.btn = nn.BatchNorm2d(2*in_ch)
    
    def forward(self, x):
        x = self.con1(x)
        x = self.btn(x)
        return x
        
class CNNBlock(nn.Module):
    def __init__(self, in_ch) -> None:
        super(CNNBlock, self).__init__()
        self.con1_1 = nn.Conv2d(in_ch, in_ch*2, kernel_size=(3,3), stride=(1,1), padding=(1,1), bias=False)
        self.btn1_1 = nn.BatchNorm2d(in_ch*2)
        self.rel1_1 = nn.ReLU()
        self.con2_1 = nn.Conv2d(in_ch*2, in_ch*2, kernel_size=(3,3), stride=(1,1), padding=(1,1), bias=False)
        self.btn2_1 = nn.BatchNorm2d(in_ch*2)
        self.rel2_1 = nn.ReLU()
        self.res_link = ResLink(in_ch)
        self.downsample = nn.Sequential(
            nn.Conv2d(in_ch*4, out_channels=in_ch*4, kernel_size=(3,3), stride=(2,2)),
            nn.BatchNorm2d(in_ch*4),
        )

    def forward(self, x):
        x1 = self.con1_1(x)
        x1 = self.btn1_1(x1)
        x1 = self.rel1_1(x1)
        x1 = self.con2_1(x1)
        x1 = self.btn2_1(x1)
        x1 = self.rel2_1(x1)
        x2 = self.res_link(x)
        x = torch.cat([x1, x2], dim=1)
        x = self.downsample(x)
        return x

class EncoderCNN(nn.Module):
    def __init__(self, in_ch, out_in_ch) -> None:
        super(EncoderCNN, self).__init__()
        self.con_in = nn.Conv2d(in_ch, out_channels=out_in_ch, kernel_size=(3,3), stride=(1,1))
        self.btn_in = nn.BatchNorm2d(32)
        self.rel_in = nn.ReLU()
        self.cnn1 = CNNBlock(out_in_ch)
        self.cnn2 = CNNBlock(out_in_ch*4)
        # self.cnn3 = CNNBlock(out_in_ch*16)
        self.avg = nn.AdaptiveAvgPool2d((1,1))
        self.out = nn.Sequential(
            nn.Dropout(),
            nn.Linear(in_features=out_in_ch*64, out_features=out_in_ch*16),
        )
    
    def forward(self, img):
        x = self.con_in(img)
        x = self.btn_in(x)
        x = self.rel_in(x)
        x = self.cnn1(x)
        x = self.cnn2(x)
        # x = self.cnn3(x)
        x = self.avg(x)
        x = x.squeeze()
        print(x.shape)
        x = self.out(x)
        return x
    
class Model(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.pos = EncoderCNN(3, 32)
        self.neg = EncoderCNN(3,32)
    def forward(self, anc, pos, neg):
        out_pos = self.pos(pos)
        out_anc = self.pos(anc)
        out_neg = self.pos(neg)
        return out_pos, out_anc, out_neg

class FaceEmbedding(Model):
    def __init__(self) -> None:
        super().__init__()
        self.FaceEmbeddingModel = torch.load('Model/model2.pth')
        self.T = v2.Compose([
            v2.ToTensor(),
            v2.Resize(256),
            v2.CenterCrop(224),
            v2.Normalize(mean = [0.48235, 0.45882, 0.40784], 
                         std=[0.00392156862745098, 0.00392156862745098, 0.00392156862745098])
        ])
        self.dbName = "FaceSimilarity"
        self.collectionName = "Embeddings"
        self.client = MongoClient(MONGO_CONNECTION_STRING)
        self.collection = self.client[self.dbName][self.collectionName]
        
    def __makeEucEmbeddings(self, img)->np.ndarray:
        img_t = self.T(img)
        img_t = torch.unsqueeze(dim=0)
        embedding = self.FaceEmbeddingModel.pos(img_t)
        del img_t
        torch.cuda.empty_cache()
        return embedding.squeeze().cpu().numpy()

    def makeEmbeddings(self, img, k):
        face_locations = fr.face_locations(img)
        sorted(face_locations, key = lambda rect: abs(rect[2]-rect[0])*abs(rect[1]-rect[3]))
        face_locations = face_locations[:k][::-1]
        EucEmb = []
        FREmb = []
        for face in face_locations:
            face_img = img[face[0]:face[2], face[1]:face[3]]
            EucEmb.append(self.__makeEucEmbeddings(face_img).tolist())
            FREmb.append(fr.face_encodings(face_img))
            
        return EucEmb, FREmb
    
    def __make_pipeline(self, EucEmb):
        pipeline = [{
            "$vectorSearch":{
                "index":"FaceEuc",
                "path":"EuclidianEmbeddings",
                "queryVector":EucEmb,
                "numCandidates":200,
                "limit":10
            }
        }]
        return pipeline
        
    def __saveEmbedding(self, embeddings)->None:
        data = []
        for EucEmb, FREmb in embeddings:
            data.append({
                "EuclidianEmbedding_1":EucEmb,
                "FREmbedding_1":FREmb
            })
        self.collection.insert_many(data)
    
    def __saveOneEmbedding(self, FREmb, EucEmb):
        self.collection.insert({
                "EuclidianEmbedding_1":EucEmb,
                "FREmbedding_1":FREmb
            }
        )
    
    def __vectorSearch(self, img, k):
        EucEmb, FREmb = self.makeEmbeddings(img, k)
        ResEmb = self.collection.aggregate(self.__make_pipeline(EucEmb))
        RecFace = []
        NotRecFace = []
        for emb in range(len(FREmb)):
            match = fr.compare_faces([i['FREmbedding_1'] for i in ResEmb], FREmb[emb])[0]
            if True in match:
                idx = match.index(True)
                RecFace.append(ResEmb[idx])
            else:
                idx = FREmb.index(emb)
                NotRecFace.append([
                    EucEmb[emb],
                    FREmb[emb]
                ])
        return RecFace, NotRecFace
    
    def vectorSearch(self, img, k, SaveNotRecFace=False):
        RecFace, NotRecFace = self.__vectorSearch(img, k)        
        if SaveNotRecFace:
            for embedding in NotRecFace:
                self.__saveEmbedding(embedding)
                
        
            