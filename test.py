from PIL import Image
import face_recognition

# Load the jpg file into a numpy array
image = face_recognition.load_image_file("people.jpeg")

# Find all the faces in the image using the default HOG-based model.
# This method is fairly accurate, but not as accurate as the CNN model and not GPU accelerated.
# See also: find_faces_in_picture_cnn.py
face_locations = face_recognition.face_locations(image)
max_area_tuple = max(face_locations, key=lambda rect: abs(rect[2] - rect[0]) * abs(rect[3] - rect[1]))

print(face_locations)
print(max_area_tuple)