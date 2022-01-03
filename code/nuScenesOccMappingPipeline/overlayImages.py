import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt 

def gray2rgb(img):
    img_ = np.zeros((img.shape[0],img.shape[1],3),dtype=np.uint8)
    img_[:,:,0] = img
    img_[:,:,1] = img
    img_[:,:,2] = img
    return img_

datasetDir = "../_DATASETS_/occMapDataset/val/_scenes/scene0002/"
fileEnding = "1533201477397423.png"

imgDir1 = datasetDir + "ilmMapPatch/"
imgFiles1 = [fileName for fileName in os.listdir(imgDir1) if fileName.endswith(fileEnding)]
imgFiles1.sort()

imgDir2 = datasetDir + "l/"
imgFiles2 = [fileName for fileName in os.listdir(imgDir2) if fileName.endswith(fileEnding)]
imgFiles2.sort()

for i in range(len(imgFiles1)):
    # load images
    img1 = np.asarray(Image.open(imgDir1+imgFiles1[i]))
    # img1 = gray2rgb(img1)
    img2 = np.asarray(Image.open(imgDir2+imgFiles2[i]))

    # overlay image
    img1 = img1.copy()
    img1[np.logical_not(np.logical_and(img1[:,:,0]>= 120,img1[:,:,1]>= 120))] = np.array([0,0,0])
    img1[img2[:,:] > 0,:] = np.array([255,255,255])

    
    plt.figure()
    plt.imshow(img1)
    frame = plt.gca()
    frame.axes.get_xaxis().set_visible(False)
    frame.axes.get_yaxis().set_visible(False)
    # save the image
    # img1 = Image.fromarray(img1)
    # img1.save("./img_{0:}.png".format(i))