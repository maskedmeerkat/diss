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
import cv2


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
DATA_DIR = "C:/Users/Daniel/Documents/_uni/PhD/code//_DATASETS_/occMapDataset/val/_scenes/"
LOG_DIR = "./"
RESULT_FILE_NAME = "evNet_occMap_scores.csv"

# yEstDirName = "irmMap"
# yEstDirName = "irmMap_noAddFr"
yEstDirName = "irmMap"
yEstFileName = yEstDirName

yEstDirName = "shiftNet_ilmMapPatchDisc_r_20"
# yEstDirName = "shiftNet_ilmMapPatchDisc_d"
# yEstDirName = "shiftNet_ilmMapPatchDisc_dr20"
yEstFileName = yEstDirName + "_mapFused"
yEstFileName = yEstDirName + "_mapFused_prior"
# yEstFileName = yEstDirName + "_mapScaled"

# map based on fusion of deep & geo ism
# areas in this map with mu >= uMin are only allocated by deep ism
# areas with mu < uMin should be verified by geo ism
# yFusedDirName = "shiftNet_ilmMapPatchDisc_dr20"
# yFusedFileName = yFusedDirName + "_mapFused"
yFusedDirName = "irmMap"
yFusedFileName = yFusedDirName

print("\n# Compute Metrics for each Scene")
# get all directory names of scene data
sceneNames = [sceneName for sceneName in os.listdir(DATA_DIR) if sceneName.startswith('scene0')] 
sceneNames.sort()

# initialize the matrices to store accumulated masses and number of data points
confusionMat_uncertain = np.zeros((3,3), dtype=np.double)
confusionMat_certain = np.zeros((3,3), dtype=np.double)
confusionMat_certain_irm_correct = np.zeros((3,3), dtype=np.double)
confusionMat_certain_irm_false = np.zeros((3,3), dtype=np.double)
interPx = np.zeros(3)
unionPx = np.zeros(3)
mIoU = np.zeros(3)
ssim = 0
numSamples = 0
numViolations = 0
numGridCells = np.uint64(0)

# toggles either the evaluation of the occ maps inside the mapped area (True)
# OR the evaluation only within the boundary areas arounds occupied space (False)
USE_MAPPED_AREA_NOT_BOUNDARY_AREA = False
boundary_thickness =10  # in case boundary evaluation is used [pixels]

# loop thru all scenes and compute metrics
for sceneName in tqdm(sceneNames):
# for sceneName in ['scene0486']:
          
    # load targets
    l_occ = np.array(Image.open(DATA_DIR + sceneName + "/ilmMap/ilmMap.png"))
    l_occ = discretizeDs(l_occ)

    # define the area where the map shall be evaluated
    if USE_MAPPED_AREA_NOT_BOUNDARY_AREA:
        mappedArea = np.array(Image.open(DATA_DIR + sceneName + "/ilmMap/mappedArea.png")) / 255
    else:
        # use an enlarged boundary area as mapped area
        mappedArea = l_occ[:, :, 1]
        mappedArea[l_occ[:, :, 1] >= 0.8] = 255
        mappedArea[l_occ[:, :, 1] < 0.8] = 0
        mappedArea = mappedArea.astype(np.uint8)
        cnts = cv2.findContours(mappedArea, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        for c in cnts:
            cv2.drawContours(mappedArea, [c], -1, 255, thickness=boundary_thickness)
        mappedArea = mappedArea / 255
        # import matplotlib.pyplot as plt
        # plt.figure()
        # plt.imshow(mappedArea)
        # plt.show()
    
    # load estimates
    y_est = np.array(Image.open(DATA_DIR + sceneName + "/" + yEstDirName + "/" + yEstFileName + ".png"))/255
    y_est_disc = discretizeDs(y_est * 255) / 255

    # load fused map
    y_fused = np.array(Image.open(DATA_DIR + sceneName + "/" + yFusedDirName + "/" + yFusedFileName + ".png"))/255
    y_fused_disc = np.argmax(discretizeDs(y_fused * 255) / 255, axis=2)
    
    # classification results of labels and predictions
    l_occ_disc = np.argmax(l_occ, axis=2)
    
    # update per class IoU
    l_occ_ = l_occ.copy()
    l_occ_[mappedArea == 0,:] = 0  # only inside mapped area
    l_occ_[y_fused[:, :, -1] == 1, :] = 0  # only inside geo irm touched area
    mIoU_ = (np.sum(np.sum(np.logical_and(y_est_disc, l_occ_), axis=0), axis=0)
             / np.sum(np.sum(np.logical_or(y_est_disc, l_occ_), axis=0), axis=0)) * 100
    mIoU = updateMean(mIoU, mIoU_, numSamples)
    
    # update structural similarity scores
    ssim_ = sk_ssim(y_est, l_occ, data_range=y_est.max() - y_est.min(), multichannel=True)
    ssim = updateMean(ssim, ssim_, numSamples)
    
    # remove all labels outside the mapped area
    labels = l_occ_disc.copy()
    labels[mappedArea == 0] = -1
    labels_certain = labels.copy()
    labels_uncertain = labels.copy()

    # REMOVE THIS LATER
    # ignore areas where mu >= uMin
    if yEstFileName.split("_")[-1] == "prior":
        labels_certain[y_fused[:, :, -1] < 0.3] = -1
        labels_uncertain[y_fused[:, :, -1] >= 0.3] = -1
    else:
        labels_certain[y_fused[:, :, -1] >= 1.0] = -1
        labels_uncertain[y_fused[:, :, -1] < 1.0] = -1

    numViolations += np.sum(np.logical_and(y_est[:, :, 2] < 0.298, y_fused[:, :, -1] == 1.))

    # TODO: REMOVE THIS
    y_est = y_est_disc


    # update confusion matrix
    confusionMat_uncertain = computeConfusionMatrix(y_est, labels_uncertain, confusionMat_uncertain, numSamples)
    confusionMat_certain = computeConfusionMatrix(y_est, labels_certain, confusionMat_certain, numSamples)

    # update the falsification rate
    labels_certain_false = labels_certain.copy()
    labels_certain_correct = labels_certain.copy()
    labels_certain_false[np.equal(y_fused_disc, l_occ_disc)] = -1
    labels_certain_correct[np.not_equal(y_fused_disc, l_occ_disc)] = -1
    confusionMat_certain_irm_false = computeConfusionMatrix(y_est, labels_certain_false, confusionMat_certain_irm_false, numSamples)
    confusionMat_certain_irm_correct = computeConfusionMatrix(y_est, labels_certain_correct, confusionMat_certain_irm_correct, numSamples)
    
    # update number of samples
    numSamples += 1
    numGridCells = np.uint64(numGridCells + y_est.shape[0]*y_est.shape[1])
        
        
print(yEstDirName)
print(f"VIOLATIONS = {numViolations}, {numViolations/numGridCells}%")

print("\n# IoU")
mIoU = mIoU.round(1)
print(mIoU, mIoU.mean().round(1))

print("\n# Confusion Matrices UNCERTAIN")
printConfusionMat(confusionMat_uncertain)

print("\n# Confusion Matrices CERTAIN")
printConfusionMat(confusionMat_certain)

# these two conf matrices show the falsification
print("\n# Confusion Matrices CERTAIN | IRM is false")
printConfusionMat(confusionMat_certain_irm_false)

print("\n# Confusion Matrices CERTAIN | IRM is correct")
printConfusionMat(confusionMat_certain_irm_correct)

print("\n# SSIM")
ssim = ssim.round(2)
print(ssim)

# # write hidden scores to csv
# confusionMat_certain = np.round(confusionMat_certain, 1)
# confusionMat_certain[0, -1] = np.round(100 - np.sum(confusionMat_certain[0, :-1]), 2)
# confusionMat_certain[1, -1] = np.round(100 - np.sum(confusionMat_certain[1, :-1]), 2)
# confusionMat_certain[2, -1] = np.round(100 - np.sum(confusionMat_certain[2, :-1]), 2)
# confusionMat_certain = confusionMat_certain.astype(str)
# confusionMat_certain_irm_false = np.round(confusionMat_certain_irm_false, 1)
# confusionMat_certain_irm_false[0, -1] = np.round(100 - np.sum(confusionMat_certain_irm_false[0, :-1]), 2)
# confusionMat_certain_irm_false[1, -1] = np.round(100 - np.sum(confusionMat_certain_irm_false[1, :-1]), 2)
# confusionMat_certain_irm_false[2, -1] = np.round(100 - np.sum(confusionMat_certain_irm_false[2, :-1]), 2)
# confusionMat_certain_irm_false = confusionMat_certain_irm_false.astype(str)
# confusionMat_certain_irm_correct = np.round(confusionMat_certain_irm_correct, 1)
# confusionMat_certain_irm_correct[0, -1] = np.round(100 - np.sum(confusionMat_certain_irm_correct[0, :-1]), 2)
# confusionMat_certain_irm_correct[1, -1] = np.round(100 - np.sum(confusionMat_certain_irm_correct[1, :-1]), 2)
# confusionMat_certain_irm_correct[2, -1] = np.round(100 - np.sum(confusionMat_certain_irm_correct[2, :-1]), 2)
# confusionMat_certain_irm_correct = confusionMat_certain_irm_correct.astype(str)
# with open(LOG_DIR + "occMapFusion__" + yEstFileName + ".txt", 'w') as txt_file:
#     txt_file.write('\\begin{tabular}{c|c|ccc|ccc|ccc}\n')
#     # needs package \usepackage{slashbox}
#     txt_file.write("&\\backslashbox{}{\\scriptsize{$k$}} & $f$ & $o$ & $u$ & $f$ & $o$ & $u$ & $f$ & $o$ & $u$\\\\\n")
#     txt_file.write("\\hline\n")
#     txt_file.write("\\parbox[t]{2mm}{\\multirow{3}{*}{\\rotatebox[origin=c]{90}{\\scriptsize{R$_{20}$}}}} &$p(k|f)$ "
#                    "& \\textcolor{mygreen}{"+confusionMat_certain[0, 0]+"} & \\textcolor{myred}{"+confusionMat_certain[0, 1]+"} & "+confusionMat_certain[0, 2] +
#                    "& \\textcolor{mygreen}{" + confusionMat_certain_irm_correct[0, 0] + "} & \\textcolor{myred}{" + confusionMat_certain_irm_correct[0, 1] + "} & " + confusionMat_certain_irm_correct[0, 2] +
#                    "& \\textcolor{mygreen}{"+confusionMat_certain_irm_false[0, 0]+"} & \\textcolor{myred}{"+confusionMat_certain_irm_false[0, 1]+"} & "+confusionMat_certain_irm_false[0, 2]+" \\\\\n")
#     txt_file.write("&$p(k|o)$ "
#                    "& \\textcolor{myred}{"+confusionMat_certain[1, 0]+"} & \\textcolor{mygreen}{"+confusionMat_certain[1, 1]+"} & "+confusionMat_certain[1, 2] +
#                    "& \\textcolor{myred}{" + confusionMat_certain_irm_correct[1, 0] + "} & \\textcolor{mygreen}{" + confusionMat_certain_irm_correct[1, 1] + "} & " + confusionMat_certain_irm_correct[1, 2] +
#                    "& \\textcolor{myred}{"+confusionMat_certain_irm_false[1, 0]+"} & \\textcolor{mygreen}{"+confusionMat_certain_irm_false[1, 1]+"} & "+confusionMat_certain_irm_false[1, 2]+" \\\\\n")
#     txt_file.write("&$p(k|u)$ "
#                    "& "+confusionMat_certain[2, 0]+" & "+confusionMat_certain[2, 1]+" & "+confusionMat_certain[2, 2] +
#                    "& " + confusionMat_certain_irm_correct[2, 0] + " & " + confusionMat_certain_irm_correct[2, 1] + " & " + confusionMat_certain_irm_correct[2, 2] +
#                    "& "+confusionMat_certain_irm_false[2, 0]+" & "+confusionMat_certain_irm_false[2, 1]+" & "+confusionMat_certain_irm_false[2, 2]+" \\\\\n")
#     txt_file.write("\\hline\n")
#     txt_file.write("\\multicolumn{2}{c|}{\\textbf{ShiftNet}} "
#                    "& \\multicolumn{3}{c|}{\\scriptsize{geo IR$_{20}$M$(m_u) < 1$}} "
#                    "& \\multicolumn{3}{c|}{\\scriptsize{geo IR$_{20}$M$(m_u) < 1$ \\& correct}} "
#                    "& \\multicolumn{3}{c}{\\scriptsize{geo IR$_{20}$M$(m_u) < 1$ \\& false}} \n")
#     txt_file.write("\\end{tabular}")
        
        
        
        
        
        
        
        
        
        
        
        
    