print("# Import Libraries")
import os
import sys
import time
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

# clear old graphs that might be still stored in e.g. ipython console
tf.reset_default_graph()
tf.keras.backend.clear_session()


#===========================#
def cart2polarImg(cartImg, mDim, pDim, aDim):
    # init polar image
    if (len(cartImg.shape) == 3):
        polImg = np.zeros((int(pDim/2),aDim,3))
        polImg[...,2] = 1.
    else:
        polImg = np.zeros((int(pDim/2),aDim))
    # define trafo vcf -> icf
    R_V2I = np.array([[1,0],[0,-1]])
    t_V2I = np.ones((2,1)) * mDim/2
    # get all indices of ranges and angles in image coordinates
    iRanges, iAngles = np.meshgrid(np.arange(polImg.shape[0]), np.arange(polImg.shape[1]))
    iRanges = iRanges.flatten()
    iAngles = iAngles.flatten()
    # get pixels range and angles
    ranges = iRanges/pDim*mDim#*2
    angles = iAngles/aDim*2*np.pi
    # trafo polar to cartesian coordinates
    pts = np.array([[ranges * np.cos(angles)],
                    [ranges * np.sin(angles)]])[:,0,:]
    # trafo from vcf -> icf
    pts = np.dot(R_V2I,pts) + t_V2I
    # trafo icf -> pcf
    iCols = np.floor(pts[0,:]/mDim*pDim).astype(np.int)
    iRows = np.floor(pts[1,:]/mDim*pDim).astype(np.int)
    # check boundaries
    iRows_mask = np.logical_and((iRows>=0), (iRows<cartImg.shape[0]))
    iCols_mask = np.logical_and((iCols>=0), (iCols<cartImg.shape[1]))
    mask = np.logical_and(iRows_mask,iCols_mask)
    iRows = iRows[mask]
    iCols = iCols[mask]
    iRanges = iRanges[mask]
    iAngles = iAngles[mask]
    
    if (len(cartImg.shape) == 3):
        polImg[iRanges,iAngles,:] = cartImg[iRows,iCols,:]
    else:
        polImg[iRanges,iAngles] = cartImg[iRows,iCols]
    polImg = np.flip(polImg,axis=0)
    
    return polImg


#===========================#
def polar2cartImg(polarImg, mDim, pDim, aDim, offset=np.zeros((2,1))):
    polarImg = np.flip(polarImg,axis=0)
    # init cartesian image
    if (len(polarImg.shape) == 3):
        cartImg = np.zeros((pDim,pDim,3))
        cartImg[:,:,2] = 1.
    else:
        cartImg = np.zeros((pDim,pDim,1))
    # define trafo vcf -> icf
    R_V2I = np.array([[1,0],[0,-1]])
    t_V2I = np.ones((2,1)) * mDim/2
    # get all indices of ranges and angles in image coordinates
    iRanges, iAngles = np.meshgrid(np.arange(polarImg.shape[0]), np.arange(polarImg.shape[1]))
    iRanges = iRanges.flatten()
    iAngles = iAngles.flatten()
    # get pixels range and angles
    ranges = iRanges/pDim*mDim#*2
    angles = iAngles/aDim*2*np.pi
    # trafo polar to cartesian coordinates
    pts = np.array([[ranges * np.cos(angles)],
                    [ranges * np.sin(angles)]])[:,0,:]
    # add offset
    pts -= offset
    # trafo from vcf -> icf
    pts = np.dot(R_V2I,pts) + t_V2I
    # trafo icf -> pcf
    iCols = np.floor(pts[0,:]/mDim*pDim).astype(np.int)
    iRows = np.floor(pts[1,:]/mDim*pDim).astype(np.int)
    # check boundaries
    iRows_mask = np.logical_and((iRows>=0), (iRows<cartImg.shape[0]))
    iCols_mask = np.logical_and((iCols>=0), (iCols<cartImg.shape[1]))
    mask = np.logical_and(iRows_mask,iCols_mask)
    iRows = iRows[mask]
    iCols = iCols[mask]
    iRanges = iRanges[mask]
    iAngles = iAngles[mask]
    # in case unasigned
    if (cartImg[iRows,iCols,:] == np.array([0,0,1])).all():
        cartImg[iRows,iCols,:] = polarImg[iRanges,iAngles,:]
    # higher priority to keep occupied pixels
    elif ((cartImg[iRows,iCols,1] == 0) and (polarImg[iRanges,iAngles,1] > 0)):
        cartImg[iRows,iCols,:] = polarImg[iRanges,iAngles,:]
    # higher priority to keep dynamic pixels
    elif ((cartImg[iRows,iCols,0] == 0) and (cartImg[iRows,iCols,1] > 0) and 
          (polarImg[iRanges,iAngles,0] > 0) and (polarImg[iRanges,iAngles,1] > 0)):
        cartImg[iRows,iCols,:] = polarImg[iRanges,iAngles,:]
    return cartImg


#===========================#
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


#===========================#
def getGraphVariables(): 
    '''
    Get the input and output nodes of the model.
    '''
    graph = tf.get_default_graph()
    x = graph.get_tensor_by_name("input_1:0")    
    y_fake = graph.get_tensor_by_name("output_0:0")
    
    return x, y_fake


#===========================#
def discretizeDs(img, pF, pO):
    discImg = np.zeros_like(img)
    
    # discretize the image according to threshold
    discImg[img[...,0] > pF, 0] = 255
    discImg[img[...,1] > pO, 1] = 255
    
    # in case pixels are both free & occupied after discretization
    mask = np.zeros_like(img[...,0])
    mask = np.logical_or( mask, discImg[...,0] + discImg[...,1] == 500)
    frMask = np.logical_and(mask, img[...,0] >= img[...,1])
    ocMask = np.logical_and(mask, img[...,0] <  img[...,1])
    
    discImg[frMask,0] = 255
    discImg[ocMask,1] = 255
    
    discImg[...,2] = 255 - discImg[...,0] - discImg[...,1]
    
    return discImg


#===========================#
def expandDynChannel(m):
    # [fr,oc,un] -> [dy,fr,oc,un]
    tmp = np.zeros(m.shape[:2]+(4,))
    tmp[...,3] = m[...,2]
    tmp[...,0] = 2*np.min(m[...,:-1],axis=2)
    tmp[...,1] = m[...,0] - 0.5*tmp[...,0]
    tmp[...,2] = m[...,1] - 0.5*tmp[...,0]
    return tmp


#=============================================================================#
print("\n# Define Parameters")
# data directory
DATA_DIR = "../_DATASETS_/occMapDataset/val/_scenes/"

# for network verification
polarFlag = False
inputName = "r_1"
MODEL_DIR = "./models/exp_deep_ism_comparison/softNet_ilmMapPatch_r_20__20201218_125246/"
MODEL_NAMES = [MODEL_NAME for MODEL_NAME in os.listdir(MODEL_DIR) if MODEL_NAME.endswith(".pb")]
# MODEL_NAMES = ["occNet_100_subrate025_maxFeatures128_maxComFeat16_ilmMapPatch_r_1__20201211_152215_ckpt_106.pb"]
MODEL_NAMES.sort()

for MODEL_NAME in MODEL_NAMES: 
    # for image verification
    # inputName = "ilm"
    # MODEL_NAME = inputName
    # MODEL_DIR = "./"
    
    
    print("\n# Compute Metrics for ",MODEL_NAME)
    # get all directory names of scene data
    sceneNames = [sceneName for sceneName in os.listdir(DATA_DIR)  if sceneName.startswith('scene')]
    sceneNames.sort() 
    
    # initialize the matrices to store accumulated masses and number of data points
    confusionMat_obs = np.zeros((3,4), dtype=np.double)
    confusionMat_hid = np.zeros((4,4), dtype=np.double)
    interPx_observed = np.zeros(3)
    unionPx_observed = np.zeros(3)
    interPx_hidden = np.zeros(4)
    unionPx_hidden = np.zeros(4)
    interPx_all = np.zeros(4)
    unionPx_all = np.zeros(4)
    
    # store mean inference time on gpu and cpu
    mInfTime_gpu = -1
    mInfTime_cpu = -1
    
    with tf.Session() as sess:
        if (MODEL_NAME[-3:] == ".pb"):
            graphFile = gfile.FastGFile(MODEL_DIR + MODEL_NAME,'rb')
            x, y_fake = restoreGraphFromPB(sess, graphFile)
        
        # loop thru all scenes and compute metrics
        for iScene in tqdm(range(len(sceneNames))):
        # for iScene in [0]:
            # get current scene
            sceneName = sceneNames[iScene]
            
            # loop thru all imgs in the scene and compute the metric
            IlmFiles = os.listdir(DATA_DIR + sceneName + "/ilm/")
            ilmMapPatchFiles = os.listdir(DATA_DIR + sceneName + "/ilmMapPatch/")
            xFiles = os.listdir(DATA_DIR + sceneName + "/" + inputName + "/")
            
            IlmFiles = [IlmFile for IlmFile in IlmFiles if IlmFile.startswith('ilm__')]
            ilmMapPatchFiles = [ilmMapPatchFile for ilmMapPatchFile in ilmMapPatchFiles if ilmMapPatchFile.startswith('ilmMapPatch__')]
            xFiles = [xFile for xFile in xFiles if xFile.startswith(inputName+'__')]
            
            IlmFiles.sort()
            ilmMapPatchFiles.sort()
            xFiles.sort()
            
            # check that we have found the same amount of inputs als targets
            assert(len(IlmFiles) == len(ilmMapPatchFiles))
            assert(len(IlmFiles) == len(xFiles))
                
            infTimes = np.zeros(len(xFiles))
            for iSample in range(len(IlmFiles)):        
                # load targets
                l_ism = np.array(Image.open(DATA_DIR + sceneName + "/ilm/" + IlmFiles[iSample]))
                l_occ = np.array(Image.open(DATA_DIR + sceneName + "/ilmMapPatch/" + ilmMapPatchFiles[iSample]))
                x_ = np.array(Image.open(DATA_DIR + sceneName + "/"+inputName+"/" + xFiles[iSample]))        
                # l_occ = discretizeDs(l_occ, int(pF*255), int(pO*255))
                
                if (MODEL_NAME[-3:] == ".pb"):                                        
                    # trafo inputs from [0,255] -> [0,1]
                    if (len(x_.shape)==2):
                        x_ = x_[np.newaxis,:,:,np.newaxis]/255
                    else:
                        x_ = x_[np.newaxis,:,:,:]/255
                    
                    # perform inference
                    t0 = time.time()   
                    if (polarFlag):
                        x_ = cart2polarImg(x_[0,:,:,0], 40., 128, 720)[np.newaxis,:,:,np.newaxis]
                    y_est = sess.run(y_fake, feed_dict = {x: x_})[0,...]
                    if (polarFlag):
                        y_est = polar2cartImg(y_est, 40., 128, 720)
                    if (iScene == 3):
                        infTimes[iSample] = time.time()-t0
                else:
                    y_est = x_
                
                # trafo [fr,oc,un] -> [dy,fr,oc,un]
                # l_occ = expandDynChannel(l_occ)
                # l_ism = expandDynChannel(l_ism)
                # y_est = expandDynChannel(y_est)
                l_occ = np.append(np.zeros(l_occ.shape[:-1]+(1,)), l_occ, axis=2)
                l_ism = np.append(np.zeros(l_ism.shape[:-1]+(1,)), l_ism, axis=2)
                y_est = np.append(np.zeros(y_est.shape[:-1]+(1,)), y_est, axis=2)
                
                # classification results of labels and predictions
                lOccDisc = np.argmax(l_occ, axis=2)
                lIsmDisc = np.argmax(l_ism, axis=2)
                yEstDisc = np.argmax(y_est, axis=2)
            
                # get masks for dynamic, free and occupied ground truth pixels
                dyMask_gt_observed = (lIsmDisc == 0)
                frMask_gt_observed = (lIsmDisc == 1)
                ocMask_gt_observed = (lIsmDisc == 2)
                
                dyMask_gt_hidden = (lOccDisc == 0) * (lIsmDisc == 3)
                frMask_gt_hidden = (lOccDisc == 1) * (lIsmDisc == 3)
                ocMask_gt_hidden = (lOccDisc == 2) * (lIsmDisc == 3)
                unMask_gt_hidden = (lOccDisc == 3) * (lIsmDisc == 3)
                
                # get masks for free and occupied estimate pixels
                dyMask_est_observed = (yEstDisc == 0)
                frMask_est_observed = (yEstDisc == 1)
                ocMask_est_observed = (yEstDisc == 2)
                
                dyMask_est_hidden = (yEstDisc == 0) * (lIsmDisc == 3)
                frMask_est_hidden = (yEstDisc == 1) * (lIsmDisc == 3)
                ocMask_est_hidden = (yEstDisc == 2) * (lIsmDisc == 3)
                unMask_est_hidden = (yEstDisc == 3) * (lIsmDisc == 3)
                
                # update per class IoU
                interPx_observed[0] += np.sum(np.logical_and(dyMask_est_observed, dyMask_gt_observed))
                interPx_observed[1] += np.sum(np.logical_and(frMask_est_observed, frMask_gt_observed))
                interPx_observed[2] += np.sum(np.logical_and(ocMask_est_observed, ocMask_gt_observed))
                unionPx_observed[0] += np.sum(np.logical_or(dyMask_est_observed, dyMask_gt_observed))
                unionPx_observed[1] += np.sum(np.logical_or(frMask_est_observed, frMask_gt_observed))
                unionPx_observed[2] += np.sum(np.logical_or(ocMask_est_observed, ocMask_gt_observed))
                
                interPx_hidden[0] += np.sum(np.logical_and(dyMask_est_hidden, dyMask_gt_hidden))
                interPx_hidden[1] += np.sum(np.logical_and(frMask_est_hidden, frMask_gt_hidden))
                interPx_hidden[2] += np.sum(np.logical_and(ocMask_est_hidden, ocMask_gt_hidden))
                interPx_hidden[3] += np.sum(np.logical_and(unMask_est_hidden, unMask_gt_hidden))
                unionPx_hidden[0] += np.sum(np.logical_or(dyMask_est_hidden, dyMask_gt_hidden))
                unionPx_hidden[1] += np.sum(np.logical_or(frMask_est_hidden, frMask_gt_hidden))
                unionPx_hidden[2] += np.sum(np.logical_or(ocMask_est_hidden, ocMask_gt_hidden))
                unionPx_hidden[3] += np.sum(np.logical_or(unMask_est_hidden, unMask_gt_hidden))
                
                interPx_all[0] += np.sum(np.logical_and((yEstDisc == 0), (lOccDisc == 0)))
                interPx_all[1] += np.sum(np.logical_and((yEstDisc == 1), (lOccDisc == 1)))
                interPx_all[2] += np.sum(np.logical_and((yEstDisc == 2), (lOccDisc == 2)))
                interPx_all[3] += np.sum(np.logical_and((yEstDisc == 3), (lOccDisc == 3)))
                unionPx_all[0] += np.sum(np.logical_or((yEstDisc == 0), (lOccDisc == 0)))
                unionPx_all[1] += np.sum(np.logical_or((yEstDisc == 1), (lOccDisc == 1)))
                unionPx_all[2] += np.sum(np.logical_or((yEstDisc == 2), (lOccDisc == 2)))
                unionPx_all[3] += np.sum(np.logical_or((yEstDisc == 3), (lOccDisc == 3)))
                
                # update observable metrics
                confusionMat_obs[0,0] += np.sum(y_est[...,0] * dyMask_gt_observed)
                confusionMat_obs[0,1] += np.sum(y_est[...,1] * dyMask_gt_observed)
                confusionMat_obs[0,2] += np.sum(y_est[...,2] * dyMask_gt_observed)
                confusionMat_obs[0,3] += np.sum(y_est[...,3] * dyMask_gt_observed)
                
                confusionMat_obs[1,0] += np.sum(y_est[...,0] * frMask_gt_observed)
                confusionMat_obs[1,1] += np.sum(y_est[...,1] * frMask_gt_observed)
                confusionMat_obs[1,2] += np.sum(y_est[...,2] * frMask_gt_observed)
                confusionMat_obs[1,3] += np.sum(y_est[...,3] * frMask_gt_observed)
                
                confusionMat_obs[2,0] += np.sum(y_est[...,0] * ocMask_gt_observed)
                confusionMat_obs[2,1] += np.sum(y_est[...,1] * ocMask_gt_observed)
                confusionMat_obs[2,2] += np.sum(y_est[...,2] * ocMask_gt_observed)
                confusionMat_obs[2,3] += np.sum(y_est[...,3] * ocMask_gt_observed)
                
                # update hidden metrics
                confusionMat_hid[0,0] += np.sum(y_est[...,0] * dyMask_gt_hidden)
                confusionMat_hid[0,1] += np.sum(y_est[...,1] * dyMask_gt_hidden)
                confusionMat_hid[0,2] += np.sum(y_est[...,2] * dyMask_gt_hidden)
                confusionMat_hid[0,3] += np.sum(y_est[...,3] * dyMask_gt_hidden)
                
                confusionMat_hid[1,0] += np.sum(y_est[...,0] * frMask_gt_hidden)
                confusionMat_hid[1,1] += np.sum(y_est[...,1] * frMask_gt_hidden)
                confusionMat_hid[1,2] += np.sum(y_est[...,2] * frMask_gt_hidden)
                confusionMat_hid[1,3] += np.sum(y_est[...,3] * frMask_gt_hidden)
                
                confusionMat_hid[2,0] += np.sum(y_est[...,0] * ocMask_gt_hidden)
                confusionMat_hid[2,1] += np.sum(y_est[...,1] * ocMask_gt_hidden)
                confusionMat_hid[3,2] += np.sum(y_est[...,2] * ocMask_gt_hidden)
                confusionMat_hid[3,3] += np.sum(y_est[...,3] * ocMask_gt_hidden)
                
                confusionMat_hid[3,0] += np.sum(y_est[...,0] * unMask_gt_hidden)
                confusionMat_hid[3,1] += np.sum(y_est[...,1] * unMask_gt_hidden)
                confusionMat_hid[3,2] += np.sum(y_est[...,2] * unMask_gt_hidden)
                confusionMat_hid[3,3] += np.sum(y_est[...,3] * unMask_gt_hidden)
                
            if (iScene == 3) and (MODEL_NAME[-3:] == ".pb"):
                mInfTime_gpu = np.mean(infTimes)
    
    if (MODEL_NAME[-3:] == ".pb"):
        # clear old graphs that might be still stored in e.g. ipython console
        tf.reset_default_graph()
        tf.keras.backend.clear_session()
        # get the cpu inference time
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
        os.environ["CUDA_VISIBLE_DEVICES"] = ""          # hide gpus
        with tf.Session() as sess:
            graphFile = gfile.FastGFile(MODEL_DIR + MODEL_NAME,'rb')
            x, y_fake = restoreGraphFromPB(sess, graphFile)
            sceneName = sceneNames[0]
                
            # get input file names
            xFiles = os.listdir(DATA_DIR + sceneName + "/" + inputName + "/")
            xFiles = [xFile for xFile in xFiles if xFile.startswith(inputName+'__')]
            xFiles.sort()
            
            # perform inference on each file
            infTimes = np.zeros(len(xFiles))
            for iSample in range(len(xFiles)): 
                # load input file
                x_ = np.array(Image.open(DATA_DIR + sceneName + "/"+inputName+"/" + xFiles[iSample]))   
                
                # trafo inputs from [0,255] -> [0,1]
                if (len(x_.shape)==2):
                    x_ = x_[np.newaxis,:,:,np.newaxis]/255
                else:
                    x_ = x_[np.newaxis,:,:,:]/255
                
                # perform inference
                t0 = time.time()
                if (polarFlag):
                    x_ = cart2polarImg(x_[0,:,:,0], 40., 128, 720)[np.newaxis,:,:,np.newaxis]
                y_est = sess.run(y_fake, feed_dict = {x: x_})[0,...]
                if (polarFlag):
                    y_est = polar2cartImg(y_est, 40., 128, 720)
                infTimes[iSample] = time.time()-t0
                
            mInfTime_cpu = np.mean(infTimes)
            
    
    print("\n# Model Stats")
    print("inf. time GPU: {0:.4f}sec".format(mInfTime_gpu))
    print("inf. time CPU: {0:.4f}sec".format(mInfTime_cpu))
    if (MODEL_NAME[-3:] == ".pb"):
        modelSize = os.path.getsize(MODEL_DIR + MODEL_NAME)/1e6
    else:
        modelSize = 0
    print("size: {0:.2f}MB".format(modelSize))
    
    print("\n# Compute per Class IoU")
    print("observable")
    print("dy IoU: {0:.2f}%".format(interPx_observed[0]/unionPx_observed[0]*100))
    print("fr IoU: {0:.2f}%".format(interPx_observed[1]/unionPx_observed[1]*100))
    print("oc IoU: {0:.2f}%".format(interPx_observed[2]/unionPx_observed[2]*100))
    # print("overall IoU: {0:.2f}%".format((interPx_observed[0]/unionPx_observed[0] +
    #                                       interPx_observed[1]/unionPx_observed[1] +
    #                                       interPx_observed[2]/unionPx_observed[2])/3*100))
    print("overall IoU: {0:.2f}%".format((interPx_observed[1]/unionPx_observed[1] +
                                          interPx_observed[2]/unionPx_observed[2])/2*100))
    print("hidden")
    print("dy IoU: {0:.2f}%".format(interPx_hidden[0]/unionPx_hidden[0]*100))
    print("fr IoU: {0:.2f}%".format(interPx_hidden[1]/unionPx_hidden[1]*100))
    print("oc IoU: {0:.2f}%".format(interPx_hidden[2]/unionPx_hidden[2]*100))
    print("un IoU: {0:.2f}%".format(interPx_hidden[3]/unionPx_hidden[3]*100))
    
    print("all")
    print("dy IoU: {0:.2f}%".format(interPx_all[0]/unionPx_all[0]*100))
    print("fr IoU: {0:.2f}%".format(interPx_all[1]/unionPx_all[1]*100))
    print("oc IoU: {0:.2f}%".format(interPx_all[2]/unionPx_all[2]*100))
    print("un IoU: {0:.2f}%".format(interPx_all[3]/unionPx_all[3]*100))
    print('\033[94m',"mean IoU: {0:.2f}%".format((interPx_all[1]/unionPx_all[1]+
                                       interPx_all[2]/unionPx_all[2]+
                                       interPx_all[3]/unionPx_all[3])/3*100),'\033[0m')
    
    
    print("\n# Compute Confusion Matrices")
    confusionMat_obs[0,:] /= np.sum(confusionMat_obs[0,:])
    confusionMat_obs[1,:] /= np.sum(confusionMat_obs[1,:])
    confusionMat_obs[2,:] /= np.sum(confusionMat_obs[2,:])
    confusionMat_obs = confusionMat_obs * 100
    confusionMat_obs = np.around(confusionMat_obs, decimals=2)
    
    confusionMat_hid[0,:] /= np.sum(confusionMat_hid[0,:])
    confusionMat_hid[1,:] /= np.sum(confusionMat_hid[1,:])
    confusionMat_hid[2,:] /= np.sum(confusionMat_hid[2,:])
    confusionMat_hid[3,:] /= np.sum(confusionMat_hid[3,:])
    confusionMat_hid = confusionMat_hid * 100
    confusionMat_hid = np.around(confusionMat_hid, decimals=2)
    
    
    print("\n Observable Scores")
    table = PrettyTable()
    table.field_names = ["[%]", "dy_est", "fr_est", "oc_est", "un_est"]
    table.add_row(["dy_true", confusionMat_obs[0,0], confusionMat_obs[0,1], confusionMat_obs[0,2], confusionMat_obs[0,3]])
    table.add_row(["fr_true", confusionMat_obs[1,0], confusionMat_obs[1,1], confusionMat_obs[1,2], confusionMat_obs[1,3]])
    table.add_row(["oc_true", confusionMat_obs[2,0], confusionMat_obs[2,1], confusionMat_obs[2,2], confusionMat_obs[2,3]])
    print(table)
    
    print("\n Hidden Scores")
    table = PrettyTable()
    table.field_names = ["[%]", "dy_est", "fr_est", "oc_est", "un_est"]
    table.add_row(["dy_true", confusionMat_hid[0,0], confusionMat_hid[0,1], confusionMat_hid[0,2], confusionMat_hid[0,3]])
    table.add_row(["fr_true", confusionMat_hid[1,0], confusionMat_hid[1,1], confusionMat_hid[1,2], confusionMat_hid[1,3]])
    table.add_row(["oc_true", confusionMat_hid[2,0], confusionMat_hid[2,1], confusionMat_hid[2,2], confusionMat_hid[2,3]])
    table.add_row(["un_true", confusionMat_hid[3,0], confusionMat_hid[3,1], confusionMat_hid[3,2], confusionMat_hid[3,3]])
    print(table)
    
    print("\n Write Scores to File")
    if (MODEL_NAME[-3:] == ".pb"):
        f = open(MODEL_DIR + MODEL_NAME[:-3] + "_scores.csv", 'w')
    else:
        f = open(MODEL_DIR + MODEL_NAME + "_scores.csv", 'w')
    writer = csv.writer(f)
    writer.writerow(["gpu time [sec]",round(mInfTime_gpu,4)])
    writer.writerow(["cpu time [sec]",round(mInfTime_cpu,4)])
    writer.writerow(["model size [MB]",round(modelSize,2)])
    writer.writerow([" "])
    writer.writerow(["observableScores"])
    writer.writerow(["mIoU dy",round(interPx_observed[0]/unionPx_observed[0]*100,2)])
    writer.writerow(["mIoU fr",round(interPx_observed[1]/unionPx_observed[1]*100,2)])
    writer.writerow(["mIoU oc",round(interPx_observed[2]/unionPx_observed[2]*100,2)])
    # writer.writerow(["mIoU fr&oc",round((interPx_observed[0]/unionPx_observed[0] + 
    #                                      interPx_observed[1]/unionPx_observed[1] +
    #                                      interPx_observed[2]/unionPx_observed[2])/3*100,2)])
    writer.writerow(["mIoU fr&oc",round((interPx_observed[1]/unionPx_observed[1] +
                                         interPx_observed[2]/unionPx_observed[2])/2*100,2)])
    writer.writerow(["[%]", "dy_est", "fr_est ", " oc_est ", " un_est "])
    writer.writerow([" dy_true ", confusionMat_obs[0,0], confusionMat_obs[0,1], confusionMat_obs[0,2], confusionMat_obs[0,3]])
    writer.writerow([" fr_true ", confusionMat_obs[1,0], confusionMat_obs[1,1], confusionMat_obs[1,2], confusionMat_obs[1,3]])
    writer.writerow([" oc_true ", confusionMat_obs[2,0], confusionMat_obs[2,1], confusionMat_obs[2,2], confusionMat_obs[2,3]])
    writer.writerow([" "])
    writer.writerow(["hiddenScores"])
    writer.writerow(["mIoU dy",round(interPx_hidden[0]/unionPx_hidden[0]*100,2)])
    writer.writerow(["mIoU fr",round(interPx_hidden[1]/unionPx_hidden[1]*100,2)])
    writer.writerow(["mIoU oc",round(interPx_hidden[2]/unionPx_hidden[2]*100,2)])
    writer.writerow(["mIoU un",round(interPx_hidden[3]/unionPx_hidden[3]*100,2)])
    writer.writerow(["[%]", "dy_est", " fr_est ", " oc_est ", " un_est "])
    writer.writerow([" dy_true ", confusionMat_hid[0,0], confusionMat_hid[0,1], confusionMat_hid[0,2], confusionMat_hid[0,3]])
    writer.writerow([" fr_true ", confusionMat_hid[1,0], confusionMat_hid[1,1], confusionMat_hid[1,2], confusionMat_hid[1,3]])
    writer.writerow([" oc_true ", confusionMat_hid[2,0], confusionMat_hid[2,1], confusionMat_hid[2,2], confusionMat_hid[2,3]])
    writer.writerow([" un_true ", confusionMat_hid[3,0], confusionMat_hid[3,1], confusionMat_hid[3,2], confusionMat_hid[3,3]])
    writer.writerow(["overall"])
    writer.writerow(["mIoU dy",round(interPx_all[0]/unionPx_all[0]*100,2)])
    writer.writerow(["mIoU fr",round(interPx_all[1]/unionPx_all[1]*100,2)])
    writer.writerow(["mIoU oc",round(interPx_all[2]/unionPx_all[2]*100,2)])
    writer.writerow(["mIoU un",round(interPx_all[3]/unionPx_all[3]*100,2)])
    writer.writerow(["mIoU",round((interPx_all[1]/unionPx_all[1]+
                                   interPx_all[2]/unionPx_all[2]+
                                   interPx_all[3]/unionPx_all[3])/3*100,2)])
    f.close()
        
        
        
        
        
        
        
        
        
        
        
    