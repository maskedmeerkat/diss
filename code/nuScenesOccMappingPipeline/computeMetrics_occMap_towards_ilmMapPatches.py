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


# =====================================
def discretizeDs(img):
    # define pixel labels
    pxLabels = np.argmax(img, axis=2)
    dyLabels = np.logical_and(img[..., 0] > 120, img[..., 1] > 120)

    # define discretized ground truth
    img_disc = np.zeros_like(img, dtype=np.uint8)
    img_disc[pxLabels == 0, 0] = 255
    img_disc[pxLabels == 1, 1] = 255
    img_disc[pxLabels == 2, 2] = 255
    img_disc[dyLabels, 0] = 127
    img_disc[dyLabels, 1] = 127
    img_disc[dyLabels, 2] = 0
    return img_disc


# =====================================
def computeConfusionMatrix(estimates, labels, meanConfusionMat, numSamples):
    # fill temp confusion matrix
    for iTrue in range(estimates.shape[2]):
        for iEst in range(estimates.shape[2]):
            mask = labels == iTrue

            if (np.sum(mask) > 0):
                confusionMat_tmp = np.sum(estimates[..., iEst] * mask.astype(float)) / np.sum(mask) * 100

                # update mean confusion matrix with temp confusion matrix by weighted mean
                meanConfusionMat[iTrue, iEst] = updateMean(meanConfusionMat[iTrue, iEst], confusionMat_tmp, numSamples)
    return meanConfusionMat


# =====================================
def updateMean(mean, newSample, numSamples):
    if (numSamples > 0):
        mean = (numSamples * mean + newSample) / (numSamples + 1)
    else:
        mean = newSample
    return mean


# =====================================
def writeConfusionMatrixToCsvFile(confusionMatrix, fileWriter):
    confusionMatrix = np.round(confusionMatrix, 2)
    fileWriter.writerow(["[%]", "dy_est", " fr_est ", " oc_est ", " un_est "])
    fileWriter.writerow(
        [" dy_true ", confusionMatrix[0, 0], confusionMatrix[0, 1], confusionMatrix[0, 2], confusionMatrix[0, 3]])
    fileWriter.writerow(
        [" fr_true ", confusionMatrix[1, 0], confusionMatrix[1, 1], confusionMatrix[1, 2], confusionMatrix[1, 3]])
    fileWriter.writerow(
        [" oc_true ", confusionMatrix[2, 0], confusionMatrix[2, 1], confusionMatrix[2, 2], confusionMatrix[2, 3]])
    fileWriter.writerow(
        [" un_true ", confusionMatrix[3, 0], confusionMatrix[3, 1], confusionMatrix[3, 2], confusionMatrix[3, 3]])


# =====================================
def printConfusionMat(meanConfusionMat):
    table = PrettyTable()
    meanConfusionMat = np.round(meanConfusionMat, 1)
    table.field_names = ["[%]", "fr_est", "oc_est", "un_est"]
    table.add_row(["fr_true", meanConfusionMat[0, 0], meanConfusionMat[0, 1], np.round(100-np.sum(meanConfusionMat[0, :-1]),2)])
    table.add_row(["oc_true", meanConfusionMat[1, 0], meanConfusionMat[1, 1], np.round(100-np.sum(meanConfusionMat[1, :-1]),2)])
    table.add_row(["un_true", meanConfusionMat[2, 0], meanConfusionMat[2, 1], np.round(100-np.sum(meanConfusionMat[2, :-1]),2)])
    print(table)


# =============================================================================#
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
yEstDirName = "shiftNet_ilmMapPatchDisc_d"
yEstDirName = "shiftNet_ilmMapPatchDisc_dr20"
yEstDirName = "shiftNet_ilmMapPatchDisc_l"
yEstDirName = "shiftNet_ilmMapPatchDisc_lr20"
# yEstFileName = yEstDirName + "_map"
# yEstFileName = yEstDirName + "_mapRmBias"
yEstFileName = yEstDirName + "_mapScaled"

# yEstFileName = yEstDirName + "_mapFused_overwriteGeo"
# yEstFileName = yEstDirName + "_mapFused"
# yEstFileName = yEstDirName + "_mapFused_prior"

# toggles either the evaluation of the occ maps inside the mapped area (True)
# OR the evaluation only within the boundary areas arounds occupied space (False)
boundary_thickness = 10  # in case boundary evaluation is used [pixels]

print("\n# Compute Metrics for each Scene")
# get all directory names of scene data
sceneNames = [sceneName for sceneName in os.listdir(DATA_DIR) if sceneName.startswith('scene0')]
sceneNames.sort()

# initialize the matrices to store accumulated masses and number of data points
confusionMat = np.zeros((3, 3), dtype=np.double)
confusionMat_border = np.zeros((3, 3), dtype=np.double)

interPx = np.zeros(3)
unionPx = np.zeros(3)
mIoU = np.zeros(3)
interPx_border = np.zeros(3)
unionPx_border = np.zeros(3)
mIoU_border = np.zeros(3)

ssim = 0

numSamples = 0

# loop thru all scenes and compute metrics
for sceneName in tqdm(sceneNames):
    # for iScene in range(1):

    # load targets
    l_occ = np.array(Image.open(DATA_DIR + sceneName + "/ilmMap/ilmMap.png"))
    l_occ = discretizeDs(l_occ)

    # define the area where the map shall be evaluated
    mappedArea = np.array(Image.open(DATA_DIR + sceneName + "/ilmMap/mappedArea.png")) / 255

    # use an enlarged boundary area as mapped area
    mappedArea_border = l_occ[:, :, 1]
    mappedArea_border[l_occ[:, :, 1] >= 0.8] = 255
    mappedArea_border[l_occ[:, :, 1] < 0.8] = 0
    mappedArea_border = mappedArea_border.astype(np.uint8)
    cnts = cv2.findContours(mappedArea_border, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        cv2.drawContours(mappedArea_border, [c], -1, 255, thickness=boundary_thickness)
    mappedArea_border = mappedArea_border/255
    # import matplotlib.pyplot as plt
    # plt.figure()
    # plt.imshow(mappedArea)
    # plt.show()

    # load estimates
    y_est = np.array(Image.open(DATA_DIR + sceneName + "/" + yEstDirName + "/" + yEstFileName + ".png")) / 255
    y_est_disc = discretizeDs(y_est*255)/255

    # classification results of labels and predictions
    l_occ_disc = np.argmax(l_occ, axis=2)

    # update per class IoU
    l_occ_ = l_occ.copy()
    l_occ_[mappedArea == 0, :] = 0
    mIoU_ = (np.sum(np.sum(np.logical_and(y_est_disc, l_occ_), axis=0), axis=0)
             / np.sum(np.sum(np.logical_or(y_est_disc, l_occ_), axis=0), axis=0)) * 100
    mIoU = updateMean(mIoU, mIoU_, numSamples)

    l_occ_ = l_occ.copy()
    l_occ_[mappedArea_border == 0, :] = 0
    mIoU_ = (np.sum(np.sum(np.logical_and(y_est_disc, l_occ_), axis=0), axis=0)
             / np.sum(np.sum(np.logical_or(y_est_disc, l_occ_), axis=0), axis=0)) * 100
    mIoU_border = updateMean(mIoU_border, mIoU_, numSamples)

    # update structural similarity scores
    ssim_ = sk_ssim(y_est_disc, l_occ, data_range=y_est.max() - y_est.min(), multichannel=True)
    ssim = updateMean(ssim, ssim_, numSamples)

    # remove all labels outside the mapped area
    labels = l_occ_disc.copy()
    labels_border = l_occ_disc.copy()
    labels[mappedArea == 0] = -1
    labels_border[mappedArea_border == 0] = -1

    # update confusion matrix
    confusionMat = computeConfusionMatrix(y_est_disc, labels, confusionMat, numSamples)
    confusionMat_border = computeConfusionMatrix(y_est_disc, labels_border, confusionMat_border, numSamples)

    # update number of samples
    numSamples += 1

print(yEstDirName)

print("\n# IoU (mapped area | border area)")
mIoU = mIoU.round(1)
mIoU_border = mIoU_border.round(1)
print(mIoU, mIoU.mean().round(1))
print(mIoU_border, mIoU_border.mean().round(1))

print("\n# Confusion Matrices (mapped area)")
printConfusionMat(confusionMat)

print("\n# Confusion Matrices (border area)")
printConfusionMat(confusionMat_border)

print("\n# SSIM")
ssim = ssim.round(2)
print(ssim)

# write hidden scores to csv
confusionMat = np.round(confusionMat, 1)
confusionMat[0, -1] = np.round(100 - np.sum(confusionMat[0, :-1]), 2)
confusionMat[1, -1] = np.round(100 - np.sum(confusionMat[1, :-1]), 2)
confusionMat[2, -1] = np.round(100 - np.sum(confusionMat[2, :-1]), 2)
confusionMat = confusionMat.astype(str)
confusionMat_border = np.round(confusionMat_border, 1)
confusionMat_border[0, -1] = np.round(100 - np.sum(confusionMat_border[0, :-1]), 2)
confusionMat_border[1, -1] = np.round(100 - np.sum(confusionMat_border[1, :-1]), 2)
confusionMat_border[2, -1] = np.round(100 - np.sum(confusionMat_border[2, :-1]), 2)
confusionMat_border = confusionMat_border.astype(str)

mIoU = mIoU.astype(str)
mIoU_border = mIoU_border.astype(str)
with open(LOG_DIR + "redundancyAnalysis__" + yEstFileName + ".txt", 'w') as txt_file:
    txt_file.write('\\begin{tabular}{c|c|ccc|ccc}\n')
    # needs package \usepackage{slashbox}
    txt_file.write("&\\backslashbox{}{\\scriptsize{$k$}} & $f$ & $o$ & $u$ & $f$ & $o$ & $u$\\\\\n")
    txt_file.write("\\hline\n")
    txt_file.write("\\parbox[t]{2mm}{\\multirow{3}{*}{\\rotatebox[origin=c]{90}{\\scriptsize{R$_{20}$}}}} &$p(k|f)$ "
                   "& \\textcolor{mygreen}{"+confusionMat[0, 0]+"} & \\textcolor{myred}{"+confusionMat[0, 1]+"} & "+confusionMat[0, 2] +
                   "& \\textcolor{mygreen}{"+confusionMat_border[0, 0]+"} & \\textcolor{myred}{"+confusionMat_border[0, 1]+"} & "+confusionMat_border[0, 2]+" \\\\\n")
    txt_file.write("&$p(k|o)$ "
                   "& \\textcolor{myred}{"+confusionMat[1, 0]+"} & \\textcolor{mygreen}{"+confusionMat[1, 1]+"} & "+confusionMat[1, 2] +
                   "& \\textcolor{myred}{"+confusionMat_border[1, 0]+"} & \\textcolor{mygreen}{"+confusionMat_border[1, 1]+"} & "+confusionMat_border[1, 2]+" \\\\\n")
    txt_file.write("&$p(k|u)$ "
                   "& "+confusionMat[2, 0]+" & "+confusionMat[2, 1]+" & "+confusionMat[2, 2] +
                   "& "+confusionMat_border[2, 0]+" & "+confusionMat_border[2, 1]+" & "+confusionMat_border[2, 2]+" \\\\\n")
    txt_file.write("& mIoU &"
                   + mIoU[0] + "&" + mIoU[1] + "&" + mIoU[2] + "&"
                   + mIoU_border[0] + "&" + mIoU_border[1] + "&" + mIoU_border[2]+" \\\\\n")
    txt_file.write("\\end{tabular}")
