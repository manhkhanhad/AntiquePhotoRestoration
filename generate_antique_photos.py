import cv2
import matplotlib.pyplot as plt
import numpy as np
import random
from google.colab.patches import cv2_imshow
def get_random_crop(image,size):

    crop_height, crop_width = size
    max_x = image.shape[1] - crop_width
    max_y = image.shape[0] - crop_height

    x = np.random.randint(0, max_x)
    y = np.random.randint(0, max_y)

    crop = image[y: y + crop_height, x: x + crop_width]

    return crop

def get_rotate_image(image,angle):
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    # rotate our image by 45 degrees around the center of the image
    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h))
    # cv2.imshow("Rotated by 45 Degrees", rotated)
    # # rotate our image by -90 degrees around the image
    # M = cv2.getRotationMatrix2D((cX, cY), -90, 1.0)
    # rotated = cv2.warpAffine(image, M, (w, h))
    return rotated

def fusion(path1,path2,path3):
    #Load img
    content = cv2.imread(path1)
    texture = cv2.imread(path2)
    fold = cv2.imread(path3)

    content_size = content.shape[:2]
    
    #crop_texture
    crop = random.choice([0,1])
    if (crop == True) and (content.shape[0] < texture.shape[0]) and  (content.shape[1] < texture.shape[1]):
        print(texture.shape, "crop exture to --> ", end = " ")
        texture = get_random_crop(texture,content_size)
        print(texture.shape)
    else:
        print(texture.shape, "resize exture to --> ", end = " ")
        texture = cv2.resize(texture,content_size[::-1])
        print(texture.shape)

    

    content_gray = cv2.cvtColor(content, cv2.COLOR_BGR2GRAY)
    texture_gray = cv2.cvtColor(texture, cv2.COLOR_BGR2GRAY)

    #fusion
    alpha = round(np.random.uniform(0.4, 0.65),2)
    beta = (1.0 - alpha)
    result = cv2.addWeighted(content_gray, alpha, texture_gray, beta, 0.0)

    fold = get_rotate_image(fold,45)
    #fold = get_random_crop(fold,content_size)
    crop = random.choice([0,1])
    if (crop == True) and (content.shape[0] < fold.shape[0]) and  (content.shape[1] < fold.shape[1]):
        print(fold.shape, "crop fold to --> ", end = " ")
        fold = get_random_crop(fold,content_size)
        print(fold.shape)
    else:
        print(fold.shape, "resize fold to --> ", end = " ")
        fold = cv2.resize(fold,content_size[::-1])
        print(fold.shape)
    fold = cv2.cvtColor(fold, cv2.COLOR_BGR2GRAY)
    ret,thresh1 = cv2.threshold(fold,127,255,cv2.THRESH_BINARY)


    result = cv2.bitwise_and(result,result,mask = cv2.bitwise_not(thresh1))
    img_thresh = (np.ones_like(result) * 220)
    img_thresh = cv2.bitwise_and(img_thresh,img_thresh,mask = (thresh1))
    #cv2_imshow(img_thresh)
    result = result + img_thresh

    
    # cv2_imshow(content)
    # cv2_imshow(texture)
    # cv2_imshow(thresh1)
    
    return result


import secrets
import os
import random

content_path = "/content/drive/MyDrive/ImageRestoration/Make data/Landmark/Landmark_content/test"
texture_path = "/content/drive/MyDrive/ImageRestoration/Make data/Landmark/texture"
fold_path = "/content/drive/MyDrive/ImageRestoration/Make data/Landmark/fold"

i = 0
for content in os.listdir(content_path):
    # if i == 5000:
    #     break
    print(i)
    texture = secrets.choice(os.listdir(texture_path))
    while texture == ".ipynb_checkpoints":
        texture = secrets.choice(os.listdir(texture_path))

    fold = secrets.choice(os.listdir(fold_path))
    while fold == ".ipynb_checkpoints":
        fold = secrets.choice(os.listdir(fold_path))

    result = fusion(os.path.join(content_path,content),os.path.join(texture_path,texture),os.path.join(fold_path,fold))

    real = os.path.join(content_path,content)
    real = cv2.imread(real)
    gray = cv2.cvtColor(real, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(os.path.join("/content/drive/MyDrive/ImageRestoration/Make data/Landmark/test/data_gen",str(i)+".jpg"),result)
    cv2.imwrite(os.path.join("/content/drive/MyDrive/ImageRestoration/Make data/Landmark/test/data_real",str(i)+".jpg"),real)
    cv2.imwrite(os.path.join("/content/drive/MyDrive/ImageRestoration/Make data/Landmark/test/data_gray",str(i)+".jpg"),gray)
    i+=1
    