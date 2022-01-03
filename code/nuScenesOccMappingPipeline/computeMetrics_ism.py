print("# Import Libraries")
import os
import sys
import time
from skimage.metrics import structural_similarity as sk_ssim
from utils.prettytable import PrettyTable
import csv
import numpy as np
from tqdm import tqdm
from PIL import Image
import tensorflow.compat.v1 as tf
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
tf.logging.set_verbosity(tf.logging.ERROR)
tf.disable_v2_behavior()
from tensorflow.python.platform import gfile
import cv2
import importlib
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)


# clear old graphs that might be still stored in e.g. ipython console
tf.reset_default_graph()
tf.keras.backend.clear_session()


#=====================================
def buildShiftedImg(img):
    shiftedImg = np.zeros((4,)+img.shape)
    shiftedImg[0,:-1,:] = img[1:,:]
    shiftedImg[1,1:,:]  = img[:-1,:]
    shiftedImg[2,:,:-1] = img[:,1:]
    shiftedImg[3,:,1:]  = img[:,:-1]
    return shiftedImg

def boundaryDet(img, img_shift): 
    boundImg = np.zeros_like(img)
    for iShift in range(img_shift.shape[0]):
        boundImg = np.logical_or(boundImg,np.logical_and(img,img_shift[iShift,...]))
    return boundImg


#=====================================
def restoreGraphFromPB(sess, graphFile):
    '''
    Restore the "interface" variables of the graph to perform inference.
    '''
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(graphFile.read())
    sess.graph.as_default()
    tf.import_graph_def(graph_def,
                        input_map=None,
                        return_elements=None,
                        name="",
                        op_dict=None,
                        producer_op_list=None)
    return getGraphVariables()


#=====================================
def getGraphVariables(): 
    '''
    Get the input and output nodes of the model.
    '''
    graph = tf.get_default_graph()
    x = graph.get_tensor_by_name("input_1:0")    
    y_fake = graph.get_tensor_by_name("output_0:0")
    
    return x, y_fake


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
def expandDynChannel(m):
    # [fr,oc,un] -> [dy,fr,oc,un]
    tmp = np.zeros(m.shape[:2]+(4,))
    tmp[...,3] = m[...,2]
    tmp[...,0] = 2*np.min(m[...,:-1],axis=2)
    tmp[...,1] = m[...,0] - 0.5*tmp[...,0]
    tmp[...,2] = m[...,1] - 0.5*tmp[...,0]
    return tmp


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
    table.field_names = ["[%]", "dy_est", "fr_est", "oc_est", "un_est"]
    table.add_row(["dy_true", meanConfusionMat[0,0], meanConfusionMat[0,1], meanConfusionMat[0,2], meanConfusionMat[0,3]])
    table.add_row(["fr_true", meanConfusionMat[1,0], meanConfusionMat[1,1], meanConfusionMat[1,2], meanConfusionMat[1,3]])
    table.add_row(["oc_true", meanConfusionMat[2,0], meanConfusionMat[2,1], meanConfusionMat[2,2], meanConfusionMat[2,3]])
    table.add_row(["un_true", meanConfusionMat[3,0], meanConfusionMat[3,1], meanConfusionMat[3,2], meanConfusionMat[3,3]])
    print(table)

#=============================================================================#
print("\n# Define Parameters")
# data directory
DATA_DIR = "../_DATASETS_/occMapDataset/val/_scenes/"

# for network verification
possibleInputNames = ["r_1","r_5","r_10","r_20","r_20_t1","l","d","c","lr20","dr20","s","irm_1","irm_20"]
MODEL_DIR = "./models/exp_deep_ism_comparison/"
# MODEL_NAMES = [MODEL_NAME for MODEL_NAME in os.listdir(MODEL_DIR) if MODEL_NAME.endswith(".pb")]
MODEL_NAMES = ["shiftNet_ilmMapPatchDisc_r_20_t1__20211010_004353_ckpt_250.pb",]
# MODEL_NAMES = ["_irm_1__", "_irm_20__"]
MODEL_NAMES.sort()
# MODEL_NAMES = MODEL_NAMES[7:]

for MODEL_NAME in MODEL_NAMES: 
    # retrieve input name from ckpt name
    inputName = ""
    for possibleInputName in possibleInputNames:
        if "_"+possibleInputName+"__" in MODEL_NAME:
            inputName = possibleInputName
    if inputName == "":
        inputName = MODEL_NAME
    
    print("\n# Compute Metrics for ", MODEL_NAME)
    # get all directory names of scene data
    sceneNames = [sceneName for sceneName in os.listdir(DATA_DIR) if sceneName.startswith('scene')]
    sceneNames.sort()     
    
    # store mean inference time on gpu and cpu
    mInfTime_gpu = -1
    mInfTime_cpu = -1
    
    # clear old graphs that might be still stored in e.g. ipython console
    tf.reset_default_graph()
    tf.keras.backend.clear_session()    
    
    with tf.Session() as sess:
        if MODEL_NAME[-3:] == ".pb":
            graphFile = gfile.FastGFile(MODEL_DIR + MODEL_NAME, 'rb')
            x, y_fake = restoreGraphFromPB(sess, graphFile)
        
        # loop through all scenes and compute metrics
        meanConfusionMat = np.zeros((4, 4), dtype=np.double)
        meanConfusionMat_hid = np.zeros((4, 4), dtype=np.double)
        meanConfusionMat_obs = np.zeros((4, 4), dtype=np.double)
        p_un_notUn = 0.
        p_dy_notDy = 0.
        mIoU = np.zeros(4)
        mIoU_hid = np.zeros(4)
        mIoU_obs = np.zeros(4)
        ssim = 0
        numSamples = 0
        for iScene in tqdm(range(int(len(sceneNames)))):
        # for iScene in [0]:
            # get current scene            
            sceneName = sceneNames[iScene]
            
            # loop thru all imgs in the scene and compute the metric
            IlmFiles = os.listdir(DATA_DIR + sceneName + "/ilm/")
            ilmMapPatchFiles = os.listdir(DATA_DIR + sceneName + "/ilmMapPatch/")
            xFiles = os.listdir(DATA_DIR + sceneName + "/" + inputName + "/")
            
            IlmFiles = [IlmFile for IlmFile in IlmFiles if IlmFile.startswith('ilm__')]
            ilmMapPatchFiles = [ilmMapPatchFile for ilmMapPatchFile in ilmMapPatchFiles
                                if ilmMapPatchFile.startswith('ilmMapPatch__')]
            xFiles = [xFile for xFile in xFiles if xFile.startswith(inputName+'__')]
            
            IlmFiles.sort()
            ilmMapPatchFiles.sort()
            xFiles.sort()
            
            # check that we have found the same amount of inputs als targets
            assert(len(IlmFiles) == len(ilmMapPatchFiles))
            assert(len(IlmFiles) == len(xFiles))
                
            for iSample in range(len(IlmFiles)):        
                # load targets
                l_ism = np.array(Image.open(DATA_DIR + sceneName + "/ilm/" + IlmFiles[iSample]))
                l_occ = np.array(Image.open(DATA_DIR + sceneName + "/ilmMapPatch/" + ilmMapPatchFiles[iSample]))
                x_ = np.array(Image.open(DATA_DIR + sceneName + "/"+inputName+"/" + xFiles[iSample]))        
                # l_occ = discretizeDs(l_occ, int(pF*255), int(pO*255))
                
                # discretize ground truth
                l_occ = discretizeDs(l_occ)
                
                # trafo inputs from [0,255] -> [0,1]
                if len(x_.shape) == 2:
                    x_ = x_[np.newaxis, :, :, np.newaxis]/255
                else:
                    x_ = x_[np.newaxis, :, :, :]/255
                
                # perform inference
                if MODEL_NAME[-3:] == ".pb":
                    y_est = sess.run(y_fake, feed_dict={x: x_})[0, ...]
                else:
                    y_est = x_[0, ...]
                
                # update structural similarity scores
                ssim_ = sk_ssim(y_est, l_occ, data_range=y_est.max() - y_est.min(), multichannel=True)
                ssim = updateMean(ssim, ssim_, numSamples)
                
                # trafo [fr,oc,un] -> [dy,fr,oc,un]
                l_occ = expandDynChannel(l_occ)
                l_ism = expandDynChannel(l_ism)
                y_est = expandDynChannel(y_est)

                # classify each ground truth pixel
                l_occ_disc = np.argmax(l_occ, axis=2)
                l_ism_disc = np.argmax(l_ism, axis=2)
                
                # overall confusion matrix  
                labels = l_occ_disc.copy()
                meanConfusionMat = computeConfusionMatrix(y_est, labels, meanConfusionMat, numSamples)
                  
                # observable confusion matrix
                labels = l_occ_disc.copy()
                labels[l_ism_disc == 3] = -1
                meanConfusionMat_obs = computeConfusionMatrix(y_est, labels, meanConfusionMat_obs, numSamples)
                
                # hidden confusion matrix
                labels = l_occ_disc.copy()
                labels[l_ism_disc != 3] = -1
                meanConfusionMat_hid = computeConfusionMatrix(y_est, labels, meanConfusionMat_hid, numSamples)

                # build occupied-free and dynamic-free border regions
                # dyRegion = (l_occ_disc == 0).astype(float)
                frRegion = (l_occ_disc == 1).astype(float)
                ocRegion = (l_occ_disc == 2).astype(float)
                frRegion_shifted = buildShiftedImg(frRegion)
                # dyFrBoundRegion = boundaryDet(dyRegion, frRegion_shifted)
                ocFrBoundRegion = boundaryDet(ocRegion, frRegion_shifted)
                
                # put both problamatic regions together in one mask
                # dyFrAndOcFrBoundRegion = np.logical_or(dyFrBoundRegion,ocFrBoundRegion)
                
                # make the region a bit bigger
                kernel = np.ones((5, 5), np.float32)
                ocFrBoundRegion = cv2.filter2D((ocFrBoundRegion*255).astype(np.uint8), -1, kernel)
                ocFrBoundRegion = (ocFrBoundRegion > 0)

                # probability of unknown estimate for non-unknown ground-truth
                p_un_notUn_tmp = np.sum(y_est[ocFrBoundRegion, 3]) / np.sum(ocFrBoundRegion) * 100
                p_un_notUn = updateMean(p_un_notUn, p_un_notUn_tmp, numSamples)
                
                # probability of dynamic estimate for non-dynamic ground-truth
                p_dy_notDy_tmp = np.sum(y_est[ocFrBoundRegion, 0]) / np.sum(ocFrBoundRegion) * 100
                p_dy_notDy = updateMean(p_dy_notDy, p_dy_notDy_tmp, numSamples)
                
                # mean overall IoU
                intersec = np.sum(np.sum(np.logical_and(y_est, l_occ), axis=0), axis=0)
                union = np.sum(np.sum(np.logical_or(y_est, l_occ), axis=0), axis=0)
                intersec[union == 0] = 0  # if no union is given, set iou to zero
                union[union == 0] = 1  # if no union is given, set iou to zero
                mIoU_ = intersec / union * 100
                mIoU = updateMean(mIoU, mIoU_, numSamples)
                
                # mean observable IoU 
                l_occ_obs = l_occ.copy()
                l_occ_obs[l_ism_disc == 3, :] = 0
                intersec_obs = np.sum(np.sum(np.logical_and(y_est, l_occ_obs), axis=0), axis=0)
                union_obs = np.sum(np.sum(np.logical_or(y_est, l_occ_obs), axis=0), axis=0)
                intersec_obs[union_obs == 0] = 0  # if no union is given, set iou to zero
                union_obs[union_obs == 0] = 1  # if no union is given, set iou to zero
                mIoU_obs_ = intersec_obs / union_obs * 100
                mIoU_obs = updateMean(mIoU_obs, mIoU_obs_, numSamples)
                
                # mean hidden IoU 
                l_occ_hid = l_occ.copy()
                l_occ_hid[l_ism_disc != 3, :] = 0
                intersec_hid = np.sum(np.sum(np.logical_and(y_est, l_occ_hid), axis=0), axis=0)
                union_hid = np.sum(np.sum(np.logical_or(y_est, l_occ_hid), axis=0), axis=0)
                intersec_hid[union_hid == 0] = 0  # if no union is given, set iou to zero
                union_hid[union_hid == 0] = 1  # if no union is given, set iou to zero
                mIoU_hid_ = intersec_hid / union_hid * 100
                mIoU_hid = updateMean(mIoU_hid, mIoU_hid_, numSamples)

                # update total number of processed samples (once!)
                numSamples += 1  
                
                
        print("Overall")
        printConfusionMat(meanConfusionMat)
        print("Observable")
        meanConfusionMat_obs[3,:] = -1
        printConfusionMat(meanConfusionMat_obs)
        print("Hidden")
        printConfusionMat(meanConfusionMat_hid)
        print('\033[92m',"p(un|not un) =",round(p_un_notUn,2),'\033[0m')
        print('\033[92m',"p(dy|not dy) =",round(p_dy_notDy,2),'\033[0m')
        print('\033[94m',"mIoU     =",np.round(mIoU[1:].mean()    ,2)," = mean",np.round(mIoU    ,2),'\033[0m')
        print('\033[94m',"mIoU_obs =",np.round(mIoU_obs[1:].mean(),2)," = mean",np.round(mIoU_obs,2),'\033[0m')
        print('\033[94m',"mIoU_hid =",np.round(mIoU_hid[1:].mean(),2)," = mean",np.round(mIoU_hid,2),'\033[0m')
        print("mean SSIM")
        print(ssim)
        
        print("\n Write Scores to File")
        if (MODEL_NAME[-3:] == ".pb"):
            f = open(MODEL_DIR + MODEL_NAME[:-3] + "_scores.csv", 'w')
        else:
            f = open(MODEL_DIR + MODEL_NAME + "_scores.csv", 'w')
        writer = csv.writer(f)
        writer.writerow(["mIoU    ",np.round(mIoU    ,2)])
        writer.writerow(["mIoU_obs",np.round(mIoU_obs,2)])
        writer.writerow(["mIoU_hid",np.round(mIoU_hid,2)])
        writer.writerow(["p(un|not un)",round(p_un_notUn,2)])
        writer.writerow(["p(dy|not dy)",round(p_dy_notDy,2)])
        writer.writerow(["overallScores"])
        writeConfusionMatrixToCsvFile(meanConfusionMat, writer)
        writer.writerow(["observableScores"])
        writeConfusionMatrixToCsvFile(meanConfusionMat_obs, writer)
        writer.writerow(["hiddenScores"])
        writeConfusionMatrixToCsvFile(meanConfusionMat_hid, writer)
        f.close()
        
        
            
    
        
        
        
        
        
        
        
        
        
        
    