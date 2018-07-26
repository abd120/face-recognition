import numpy as np
import cv2
import matplotlib.pyplot as plt

haar_face_cascade = cv2.CascadeClassifier('data/haarcascade_frontalface_alt.xml')
lbp_face_cascade = cv2.CascadeClassifier('data/lbpcascade_frontalface.xml')

def convertToRGB(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def detect_faces(f_cascade, colored_img, scaleFactor = 1.1):
    img_copy = np.copy(colored_img)
    gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
    faces = f_cascade.detectMultiScale(gray, scaleFactor=scaleFactor, minNeighbors=5);
    print('Faces found: ', len(faces))
    for (x, y, w, h) in faces:
        cv2.rectangle(img_copy, (x, y), (x+w, y+h), (0, 255, 0), 2)
    return img_copy

image = cv2.imread('data/image.jpg')
faces_detected_img = detect_faces(lbp_face_cascade, image)
plt.imshow(convertToRGB(faces_detected_img))
