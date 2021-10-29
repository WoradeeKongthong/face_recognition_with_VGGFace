"""
face scraping of specified person name using Selenium and MTCNN
"""
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import os
import time
import cv2
from mtcnn.mtcnn import MTCNN

def get_person_images(search_name, save_name, num_images):
    # define driver
    driver = webdriver.Chrome("/home/samantha/chromedriver")
    driver.maximize_window()

    # go to the page
    driver.get("https://google.com")
    
    # find search box
    box = driver.find_element_by_xpath("/html/body/div[1]/div[3]/form/div[1]/div[1]/div[1]/div/div[2]/input")

    # search and go to next page
    box.send_keys(search_name)
    box.send_keys(Keys.ENTER)
    
    # find element (img) and save the screenshot
    driver.find_element_by_xpath('//*[@id="hdtb-msb"]/div[1]/div/div[2]/a').click()
    time.sleep(3)
    
    # scroll page for n times
    pixel = 1000
    for i in range(5):
        driver.execute_script("window.scrollTo(0,{})".format(pixel)) 
        time.sleep(3)
        pixel = pixel + 10000
    
    driver.execute_script("window.scrollTo(0,0)")
    
    folder_name = f"data/faces/{save_name}"
    if not os.path.isdir(folder_name):
        os.makedirs(folder_name)
    for i in range(1, num_images+1):
        try :
            driver.find_element_by_xpath(f'//*[@id="islrg"]/div[1]/div[{i}]/a[1]/div[1]/img').screenshot(folder_name+f'/{save_name}_{i}.png')
            time.sleep(3)
        except :
            pass
    driver.close()

def get_person_face(imageFileName, detector):
    # load image
    img = cv2.imread(imageFileName)
    # detection
    face_detect = detector.detect_faces(img)
    # extract face from image
    if len(face_detect) == 1:
        x, y, w, h = face_detect[0]['box']
        face = img[y:y+h, x:x+w]
        # save face instead of image
        cv2.imwrite(imageFileName, face)
    else:
        os.remove(imageFileName)

def create_face_dataset(faces_path):
    # get image file name list
    fileNames = []
    for root, subdir, files in os.walk(faces_path):
        for file in files:
            #if not file.startswith('haarcascade'):
            fileName = os.path.join(root, file)
            fileNames.append(fileName)
    
    # create the detector, using default weights
    detector = MTCNN()
    
    # extract faces
    for fileName in fileNames:
        get_person_face(fileName, detector)

if __name__ == '__main__':
    
    # determine search names and save names
    search_names = ['jisooblackpink','jennieblackpink','roseblackpink','lisablackpink']
    save_names = ['Jisoo', 'Jennie','Rose','Lisa']

    # get images from google
    for i in range(len(search_names)):
        get_person_images(search_names[i], save_names[i], img_num)
    
    # extract face from image and 
    create_face_dataset('data/faces')
