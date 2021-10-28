"""
Celebrities Face recognition with keras-vggface
"""
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input, decode_predictions
from keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np
from mtcnn.mtcnn import MTCNN

# load a test image
img = image.load_img('data/images/halleberry.jpeg')
img = image.img_to_array(img)

# extract face from the loaded image
face_detector = MTCNN()
faces = face_detector.detect_faces(img)
x, y, w, h = faces[0]['box']
face = img[y:y+h, x:x+w]

# reshape face for input of face recognition (VGGface)
face = image.array_to_img(face)
face = face.resize((224,224))

# convert image to array
face = image.img_to_array(face)

# create batch
sample = np.expand_dims(face, 0)

# preprocess input image vgg16(version=1), resnet50(version=2), senet50(version=2)
sample = preprocess_input(sample, version=2)

# create a pre-trained model
# vgg16(2015), resnet50(2017), senet50(2017)
face_recognizer = VGGFace(model='senet50')

# make prediction on well-known celebrity image
prediction = face_recognizer.predict(sample)
print('Predicted:', decode_predictions(prediction))

# show image and prediction
plt.imshow(img/255.)
plt.axis('off')
if decode_predictions(prediction)[0][0][1] > 0.8 :
    fcolor = 'green'
else :
    fcolor = 'red'
plt.title(f"{decode_predictions(prediction)[0][0][0][3:-1]}({round(decode_predictions(prediction)[0][0][1]*100,2)}%)",
          color=fcolor)