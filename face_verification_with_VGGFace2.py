#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 08:12:25 2021

@author: samantha
"""

import matplotlib.pyplot as plt
from PIL import Image
from numpy import asarray
from scipy.spatial.distance import cosine
from mtcnn.mtcnn import MTCNN
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input
from keras.preprocessing import image
import cv2
import numpy as np

# load a face detector
detector = MTCNN()
 # create VGGFace2 for extract face embedding
model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')
       
def get_embedding(img_path) :
    # load an image from file
    img = plt.imread(img_path)
      
    # detect face in image
    detection = detector.detect_faces(img)
    x, y, w, h = detection[0]['box']
    face = img[y:y+h, x:x+w]
    
    # reshape face for input of face recognition (VGGface)
    face = image.array_to_img(face)
    face = face.resize((224,224))
    
    # convert back to array
    face_array = image.img_to_array(face)
    
    # apply VGGFace preprocess_input to the array
    face_array = preprocess_input(face_array, version=2)
    
    # create samples (incase of one image at once)
    samples = np.expand_dims(face_array, 0)
    
    # get embedding
    embedding = model.predict(samples)	

    return embedding[0]

known_img_path = "data/images/millajovovich1.jpg"
candidate_img_path = "data/images/millajovovich2.jpg"

known_emb = get_embedding(known_img_path)
candidate_emb = get_embedding(candidate_img_path)
# calculate distance between embeddings
score = cosine(known_emb, candidate_emb)
print(score)

fig, ax = plt.subplots(1,2)
fig.suptitle(f"similarity : {round(score*100,2)} %")
plt.subplot(121)
plt.imshow(plt.imread(known_img_path))
plt.axis("off")
plt.subplot(122)
plt.imshow(plt.imread(candidate_img_path))
plt.axis("off")
plt.show()

