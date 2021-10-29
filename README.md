# face_recognition_with_VGGFace

## Installation
1. I use mtcnn for face detection, it can be installed via pip; `pip install mtcnn`
2. VGGFace2 for face recognition; `pip install git+https://github.com/yaledhlab/vggface.git`

## VGGFace usage
*celebrities_face_recognition_with_pre-trained_VGGFace2.py* and *.ipynb*,  
are the example of using pre-trained VGGFace to identify the celebrity's name.  

## Face dataset
I prepared the face dataset using face_scraping_from_google.py script.  
You can change the name of people as you want,  
or you can create dataset of people faces by yourself.  

## Training face recognition model
In *training_face_recognition_with_VGGFace.ipynb*, I create face recognition model  
with VGGFace2 as an integrated feature extractor.  
