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
            match = [fr.compare_faces(self.known_face_encoding, face_encoding)[0] for face_encoding in face_encodings]
            if True in match:
                index = match.index(True)
                name = self.face_name[index]
            else:
                name = str(uuid4())
            self.save_face_info(name, face_image)
            return name, face_image
        
        else:
            names = []
            face_images = []
            face_encodings = fr.face_encodings(image, face_locations)
            for face_encoding, face_loc in zip(face_encodings, face_locations):
                matches = fr.compare_faces(self.known_face_encoding, face_encoding)
                top, right, bottom, left = face_loc
                face_image = image[top:bottom, left:right]
                
                face_images.append(image[top:bottom, left:right])
                if True in matches:
                    first_match_index = matches.index(True)
                    name = self.face_name[first_match_index]
                    names.append(self.face_name[first_match_index])
                else: 
                    name = str(uuid4())
                    names.append(name)
                self.save_face_info(name, face_image)
            return names, face_images
        
    def save_face_info(self, names, face_images):
        if type(name)==list and type(face_image)==list:
            for name, face_image in zip(names, face_images):
                self.face_name.append(name)
                self.known_face.append(face_image)
                self.known_face_encoding(fr.face_encodings(face_image)[0])
        else:    
            self.face_name.append(names)
            self.known_face.append(face_images)
            self.known_face_encoding(fr.face_encodings(face_images)[0])

if __name__ == '__main__':
    bruh = Face(DIR_FACES='face')
    img = cv2.imread('people.jpeg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    names, face_images = bruh.get_face_info(img, False)
    print(names)
    for name, face_image in zip(names, face_images):
        cv2.imwrite(f'{name}.jpeg', face_image)
    