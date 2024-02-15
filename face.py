import face_recognition as fr
import cv2
import numpy as np
import os
from uuid import uuid4
class Face:
    def __init__(self, DIR_FACES:str=None) -> None:
        self.dir = DIR_FACES if DIR_FACES is not None else 'face'
        if DIR_FACES is None:
            try:
                os.mkdir('face')
            except:
                pass
        self.face_name = [name.split('.')[0] for name in os.listdir(self.dir)]
        self.known_face = [fr.load_image_file(os.path.join(DIR_FACES, dir)) for dir in os.listdir(DIR_FACES) if os.path.exists(os.path.join(DIR_FACES, dir))] if DIR_FACES is not None else []
        self.known_face_encoding = [fr.face_encodings(face)[0] for face in self.known_face]
    
    def get_face_info(self, image:np.array, area_max:bool):
        name = None
        face_locations = fr.face_locations(image)
        if area_max:
            max_area_tuple = max(face_locations, key=lambda rect: abs(rect[2] - rect[0]) * abs(rect[3] - rect[1]))
            top, right, bottom, left = max_area_tuple
            face_image = image[top:bottom, left:right]
            face_encodings = fr.face_encodings(image, [max_area_tuple])
            match = fr.compare_faces(self.known_face_encoding, face_encodings)
        else:
            face_encodings = fr.face_encodings(image, face_locations)
            for face_encoding in face_encodings:
                match = fr.compare_faces(self.known_face_encoding, face_encoding)
            if True in match:
                index = match.index(True)
                name = self.face_name[index]
                return name, face_image
        
        if name is None:
            name = uuid4()
            return name, face_image
        
        def save_face_info(self, name:str, face_image:np.array):
            self.face_name.append(name)
            self.known_face.append()

if __name__ == '__main__':
    bruh = Face(DIR_FACES='face')
    img = cv2.imread('people.jpeg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    print(img.shape)
    name, face_image = bruh.get_face_info(img, True)
    print(name, face_image.shape)
    cv2.imwrite(f'{name}.jpeg',face_image)
    