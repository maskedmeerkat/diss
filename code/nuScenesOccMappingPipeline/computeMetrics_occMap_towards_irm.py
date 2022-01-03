print("# Import Libraries")
import os
import sys
import time
from skimage.metrics import structural_similarity as sk_ssim
from utils.prettytable import PrettyTable
import csv
import numpy as np
from PIL import Image
from tqdm import tqdm


#=====================================
def discretizeDs(img):
    # define pixel labels
    pxLabels = np.argmax(img, axis=2)
    dyLabels = np.logical_and(img[...,0]>120, img[...,1]>120)
    
    # define discretized ground truth
    img_disc = np.zeros_like(img, dtype=np.uint8)
    img_disc[pxLabels==0,0] = 255
    img_disc[pxLabels==1,1] = 255
    img_disc[pxLabels==2,2] = 255
    img_disc[dyLabels,0] = 127
    img_disc[dyLabels,1] = 127
    img_disc[dyLabels,2] = 0    
    return img_disc


#=====================================
def computeConfusionMatrix(estimates,labels,meanConfusionMat,numSamples):
    # fill temp confusion matrix
    for iTrue in range(estimates.shape[2]):
        for iEst in range(estimates.shape[2]):
            mask = labels == iTrue
            
            if (np.sum(mask) > 0):
                confusionMat_tmp = np.sum(estimates[...,iEst] * mask.astype(float)) / np.sum(mask) * 100 
                
                # update mean confusion matrix with temp confusion matrix by weighted mean
                meanConfusionMat[iTrue,iEst] = updateMean(meanConfusionMat[iTrue,iEst],confusionMat_tmp,numSamples)        
    return meanConfusionMat


#=====================================
def updateMean(mean,newSample,numSamples):
    if (numSamples > 0):
        mean = (numSamples*mean + newSample) / (numSamples+1)
    else:
        mean = newSample
    return mean


#=====================================
def writeConfusionMatrixToCsvFile(confusionMatrix,fileWriter):
    confusionMatrix = np.round(confusionMatrix,2)
    fileWriter.writerow(["[%]", "dy_est", " fr_est ", " oc_est ", " un_est "])
    fileWriter.writerow([" dy_true ", confusionMatrix[0,0], confusionMatrix[0,1], confusionMatrix[0,2], confusionMatrix[0,3]])
    fileWriter.writerow([" fr_true ", confusionMatrix[1,0], confusionMatrix[1,1], confusionMatrix[1,2], confusionMatrix[1,3]])
    fileWriter.writerow([" oc_true ", confusionMatrix[2,0], confusionMatrix[2,1], confusionMatrix[2,2], confusionMatrix[2,3]])
    fileWriter.writerow([" un_true ", confusionMatrix[3,0], confusionMatrix[3,1], confusionMatrix[3,2], confusionMatrix[3,3]])


#=====================================
def printConfusionMat(meanConfusionMat):
    table = PrettyTable()
    meanConfusionMat = np.round(meanConfusionMat,2)
    table.field_names = ["[%]", "fr_est", "oc_est", "un_est"]
    table.add_row(["fr_true", meanConfusionMat[0,0], meanConfusionMat[0,1], meanConfusionMat[0,2]])
    table.add_row(["oc_true", meanConfusionMat[1,0], meanConfusionMat[1,1], meanConfusionMat[1,2]])
    table.add_row(["un_true", meanConfusionMat[2,0], meanConfusionMat[2,1], meanConfusionMat[2,2]])
    print(table)


#=============================================================================#
print("\n# Define Parameters")
# data directory
DATA_DIR = "../_DATASETS_/occMapDataset/val/_scenes/"
LOG_DIR = "./"
RESULT_FILE_NAME = "evNet_occMap_scores.csv"

yEstDirName = "irmMap"
yEstFileName = yEstDirName

# yEstDirName = "dirNet_ilmMapPatchDisc_r_20"
yEstDirName = "shiftNet_ilmMapPatchDisc_r_20"
yEstFileName = yEstDirName + "_map"
# yEstFileName = yEstDirName + "_mapScaled"
# yEstFileName = yEstDirName + "_mapFused"

print("\n# Compute Metrics for each Scene")
# get all directory names of scene data
sceneNames = [sceneName for sceneName in os.listdir(DATA_DIR) if sceneName.startswith('scene0')] 
sceneNames.sort()

# initialize the matrices to store accumulated masses and number of data points
confusionMat = np.zeros((3,3), dtype=np.double)
interPx = np.zeros(3)
unionPx = np.zeros(3)
mIoU = np.zeros(3)
ssim = 0
numSamples = 0

# loop thru all scenes and compute metrics
for sceneName in tqdm(sceneNames):
          
    # load targets
    l_occ = np.array(Image.open(DATA_DIR + sceneName + "/irmMap/irmMap.png"))
    l_occ = discretizeDs(l_occ)    
    mappedArea = (np.array(Image.open(DATA_DIR + sceneName + "/irmMap/irmMap.png"))[...,2]>0.3).astype(float)
    
    # load estimates
    y_est = np.array(Image.open(DATA_DIR + sceneName + "/" + yEstDirName + "/" + yEstFileName + ".png"))/255
    
    # classification results of labels and predictions
    l_occ_disc = np.argmax(l_occ, axis=2)
    
    # # update per class IoU
    l_occ_ = l_occ.copy()
    l_occ_[mappedArea==0,:] = 0
    mIoU_ = (  np.sum(np.sum(np.logical_and(y_est,l_occ_),axis=0),axis=0) 
                 / np.sum(np.sum(np.logical_or( y_est,l_occ_),axis=0),axis=0) ) * 100
    mIoU = updateMean(mIoU, mIoU_, numSamples)
    
    # update structural similarity scores
    ssim_ = sk_ssim(y_est, l_occ ,data_range=y_est.max() - y_est.min(), multichannel=True)
    ssim = updateMean(ssim, ssim_, numSamples)
    
    # remove all labels outside the mapped area
    labels = l_occ_disc.copy()
    labels[mappedArea==0] = -1
    
    # update confusion matrix
    confusionMat = computeConfusionMatrix(discretizeDs(y_est*255)/255, labels, confusionMat, numSamples)
    
    # update number of samples
    numSamples += 1
        
        
print(yEstDirName)

print("\n# IoU")
mIoU = mIoU.round(1)
print(mIoU, mIoU.mean().round(1))

print("\n# Confusion Matrices")
printConfusionMat(confusionMat)

print("\n# SSIM")
ssim = ssim.round(2)
print(ssim)

# # write hidden scores to csv
# f = open(LOG_DIR + RESULT_FILE_NAME, 'w')
# writer = csv.writer(f)
# writer.writerow(["hiddenScores"])
# writer.writerow(["[%]", " fr_est ", " oc_est ", " un_est "])
# writer.writerow([" fr_true ", confusionMat[0,0], confusionMat[0,1], confusionMat[0,2]])
# writer.writerow([" oc_true ", confusionMat[1,0], confusionMat[1,1], confusionMat[1,2]])
# writer.writerow([" un_true ", confusionMat[2,0], confusionMat[2,1], confusionMat[2,2]])
# f.close()
        
        
        
        
        
        
        
        
        
        
        
        
    