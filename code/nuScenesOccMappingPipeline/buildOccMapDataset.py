from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud, RadarPointCloud
import numpy as np
import csv
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import os.path as osp
from tqdm import tqdm
from pyquaternion import Quaternion
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import sys
import utils.interUtils as interUtils
import utils.sceneAttribUtils as sceneAttribUtils
import utils.cameraUtils as cameraUtils
import utils.mapUtils as mapUtils
from collections import deque
import time
import itertools
import multiprocessing as mp
import tensorflow.compat.v1 as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.logging.set_verbosity(tf.logging.ERROR)
tf.disable_v2_behavior()
from tensorflow.python.platform import gfile

plt.close("all")

VEHICLE_BOARDERS = np.array([0.75, -0.75, 1.9, -1.2]) * 1.07
# HEIGHT_THRESHOLDS = np.array([0.5, 3.0])
HEIGHT_THRESHOLDS = np.array([0.6, 3.0])
NUM_INTERPOL_PTS = 20
ADD_TO_BOUNDING_BOX_DIAMETER = 0.1


# =====================================
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


# =====================================
def getGraphVariables():
    '''
    Get the input and output nodes of the model.
    '''
    graph = tf.get_default_graph()
    x = graph.get_tensor_by_name("input_1:0")
    y_fake = graph.get_tensor_by_name("output_0:0")

    return x, y_fake


# ===========================#
def pose2HomogTrafo_2D(p):
    T = np.eye(3)
    T[:2, :2] = rotMat(p[2])
    T[:2, 2] = p[:2]
    return T


# ===========================#
def invertHomogTrafo(T):
    T_ = np.eye(3)
    R = T[:2, :2]
    t = T[:2, 2]
    T_[:2, :2] = R.T
    T_[:2, 2] = -np.dot(R.T, t)
    return T_


# ===========================#
def homogTrafo2Pose_2D(T):
    p = np.array([T[0, -1], T[1, -1], np.arctan2(T[1, 0], T[0, 0])])
    return p


# ===========================#
def rotMat(angle):
    R = np.array([[np.cos(angle), -np.sin(angle)],
                  [np.sin(angle), np.cos(angle)]])
    return R


# ===========================#
def vizPose(p, scale=1.):
    R = rotMat(p[2])
    xEndPt = np.dot(R, scale * np.array([[1], [0]])) + p[:2, np.newaxis]
    yEndPt = np.dot(R, scale * np.array([[0], [1]])) + p[:2, np.newaxis]

    plt.plot([p[0], xEndPt[0]], [p[1], xEndPt[1]], 'r')
    plt.plot([p[0], yEndPt[0]], [p[1], yEndPt[1]], 'g')


# ===========================#
def dyn_static_obj_masks(pc_wcf, boundingBoxesOfDynObjs_wcf):
    """ segments objects from point clouds based on the provided annotation
        boxes. 
        pcl: point cloud
        sensorName:'CAM_FRONT','CAM_BACK', ...
                'CAM_BACK_LEFT','CAM_FRONT_LEFT','CAM_FRONT_RIGHT','CAM_BACK_RIGHT',...
                'LIDAR_TOP', ...
                'RADAR_FRONT', ...
                'RADAR_FRONT_RIGHT','RADAR_FRONT_LEFT','RADAR_BACK_RIGHT','RADAR_BACK_LEFT'
    """
    # initialize the masks        
    dynamic_mask = np.zeros_like(pc_wcf[0, :]).astype("bool")

    # loop over all annotation boxes and mark moving objects in them in a mask
    for iBox in range(boundingBoxesOfDynObjs_wcf.shape[0]):
        annBox = boundingBoxesOfDynObjs_wcf[iBox]
        # get the points corresponding to the bounding box
        # get transformation into box coordiantes
        t = annBox.center[:, np.newaxis]
        R = annBox.orientation.rotation_matrix

        # transform the sensor to the box coordinate system
        pc_bcf = np.dot(R.T, pc_wcf - t)

        # get the bounding box diameters
        width_box, length_box, height_box = annBox.wlh

        # filter out all points within the box         
        mask = np.ones_like(pc_bcf[0, :])
        mask = np.logical_and(mask, pc_bcf[0, :] < length_box / 2 + ADD_TO_BOUNDING_BOX_DIAMETER)
        mask = np.logical_and(mask, pc_bcf[0, :] > -length_box / 2 - ADD_TO_BOUNDING_BOX_DIAMETER)
        mask = np.logical_and(mask, pc_bcf[1, :] < width_box / 2 + ADD_TO_BOUNDING_BOX_DIAMETER)
        mask = np.logical_and(mask, pc_bcf[1, :] > - width_box / 2 - ADD_TO_BOUNDING_BOX_DIAMETER)
        #        mask = np.logical_and(mask,pc_bcf[2,:]<height_box/2)
        #        mask = np.logical_and(mask,pc_bcf[2,:]>-height_box/2)

        # decide wether to put obj points to dynamic or static mask
        dynamic_mask = np.logical_or(dynamic_mask, mask)

    return dynamic_mask


# ===========================#
def pc2VehicleCenteredBevImg(pc, pDim=512, mDim=40.):
    # trafo vcf -> icf     
    pc.points[1, :] = -pc.points[1, :]
    pc.points = pc.points[[1, 0, 2], :]
    pc.points = pc.points + np.array([[mDim / 2, mDim / 2, 0]]).T

    # trafo icf -> pcf
    pc.points = (pc.points * pDim / mDim).astype(int)

    # filter out all pts outside image dimension
    mask = (pc.points[0, :] >= 0) * (pc.points[1, :] >= 0) * (pc.points[0, :] < pDim) * (pc.points[1, :] < pDim)
    pc.points = pc.points[:, mask]

    # create brid's eye view lidar image
    bevImg = np.zeros((pDim, pDim))

    # mark detections in the image
    bevImg[pc.points[0, :], pc.points[1, :]] = 1

    return bevImg


# ===========================#
def vehicleCenteredBevImg2Pc(bevImg, pDim=512, mDim=40.):
    # trafo xy coordinates of marked cells back to meters -> discretized pointcloud
    pcDisc = np.transpose(np.transpose((bevImg > 0).nonzero())).astype(float)
    pcDisc = np.append(pcDisc, np.zeros((1, pcDisc.shape[1])), axis=0)
    pcDisc *= mDim / pDim

    # trafo icf -> vcf
    pcDisc = pcDisc - np.array([[mDim / 2, mDim / 2, 0]]).T
    pcDisc = pcDisc[[1, 0, 2], :]
    pcDisc[1, :] = -pcDisc[1, :]

    return pcDisc


# ===========================#
def getPcWithMotionStatusInRcf(nusc, sample, vPose, boundingBoxesOfDynObjs_rcf, dataPath_and_sensorName,
                               sweepFlag=False):
    """
    sensorName: 'CAM_FRONT','CAM_BACK', ...
                'CAM_BACK_LEFT','CAM_FRONT_LEFT','CAM_FRONT_RIGHT','CAM_BACK_RIGHT', ...
                'LIDAR_TOP', ...
                'RADAR_FRONT', ...
                'RADAR_FRONT_RIGHT','RADAR_FRONT_LEFT','RADAR_BACK_RIGHT','RADAR_BACK_LEFT'
    """
    (dataPath, sensorName) = dataPath_and_sensorName

    # load point cloud and motion status
    if (sensorName == 'LIDAR_TOP'):
        pc = LidarPointCloud.from_file(dataPath)
    else:
        pc = RadarPointCloud.from_file(dataPath)

    # remove false alarm detections
    #    if (sensorName[:5] == 'RADAR'):
    #        pc.points = pc.points[:,pc.points[15,:] < 2]

    # remove the detections corresponding to the ego vehicle
    if (sensorName == 'LIDAR_TOP'):
        mask = np.zeros_like(pc.points[0, :])
        mask = np.logical_or(mask, pc.points[0, :] > VEHICLE_BOARDERS[0])
        mask = np.logical_or(mask, pc.points[0, :] < VEHICLE_BOARDERS[1])
        mask = np.logical_or(mask, pc.points[1, :] > VEHICLE_BOARDERS[2])
        mask = np.logical_or(mask, pc.points[1, :] < VEHICLE_BOARDERS[3])
        pc.points = pc.points[:3, mask]

        # trafo scf -> vcf
    sample_data = nusc.get('sample_data', sample['data'][sensorName])
    cal_sensor = nusc.get('calibrated_sensor', sample_data['calibrated_sensor_token'])
    pc.rotate(Quaternion(cal_sensor['rotation']).rotation_matrix)
    pc.translate(np.array(cal_sensor['translation']))

    # remove ground plane points through thresholding
    if (sensorName == 'LIDAR_TOP'):
        mask = np.zeros_like(pc.points[0, :]).astype(int)
        mask = np.logical_or(mask, np.abs(pc.points[2, :]) > HEIGHT_THRESHOLDS[0])
        mask = np.logical_and(mask, pc.points[2, :] < HEIGHT_THRESHOLDS[1])
        pc.points = pc.points[:, mask]

    if (sensorName == 'LIDAR_TOP'):
        pDim = 512
        mDim = 40.
        bevImg = pc2VehicleCenteredBevImg(pc, pDim=pDim, mDim=mDim)
        pcDisc = vehicleCenteredBevImg2Pc(bevImg, pDim=pDim, mDim=mDim)

    else:
        pcDisc = pc.points

    # trafo vcf -> rcf
    R = np.eye(3)
    R[:2, :2] = rotMat(vPose[2])
    t = np.zeros((3, 1))
    t[:2, 0] = vPose[:2]
    pcDisc[:3, :] = np.dot(R, pcDisc[:3, :]) + t

    # identify moving objects
    if (sensorName == 'LIDAR_TOP'):
        # mark all lidar points inside moving obj bounding boxes as dynamic
        isDynamic = dyn_static_obj_masks(pcDisc, boundingBoxesOfDynObjs_rcf)
        isStationary = (isDynamic == 0).astype(int)
    else:
        isStationary = ((pc.points[3, :] != 0) * (pc.points[3, :] != 2) * (pc.points[3, :] != 6)).astype(int)
    #        isStationary = np.logical_or(isStationary, pc.points[3,:])
    #        isStationary = ((pc.points[3,:] == 7)).astype(int)
    #        isStationary = (np.linalg.norm(pc.points[8,:] + pc.points[9,:]) < 20.).astype(int)

    return pcDisc, isStationary


# ===========================#
def getSampleDataPath(sample, sensorName):
    return osp.join(nusc.dataroot, nusc.get('sample_data', sample['data'][sensorName])['filename']), sensorName


# ===========================#
def getSweepDataPath(sweepName, sensorName):
    return nusc.dataroot + 'sweeps' + '/' + sensorName + '/' + sweepName, sensorName


# ===========================#
def getPoseAndTimeFromSample(nusc, scene, sample, sensorName):
    # get sensor's meta data
    sample_data = nusc.get('sample_data', sample['data'][sensorName])

    # get timestamp from meta data
    timestamp = sample_data['timestamp']

    # trafo scf -> vcf
    cal_sensor = nusc.get('calibrated_sensor', sample_data['calibrated_sensor_token'])
    R = Quaternion(cal_sensor['rotation']).rotation_matrix
    t = np.array(cal_sensor['translation'])[..., np.newaxis]
    rotAngle = np.arctan2(R[1, 0], R[0, 0])
    sensorPose = np.array([t[0, 0], t[1, 0], rotAngle])

    # trafo vcf -> wcf
    ego_pose = nusc.get('ego_pose', sample_data['ego_pose_token'])
    R = Quaternion(ego_pose['rotation']).rotation_matrix
    t = np.array(ego_pose['translation'])[..., np.newaxis]
    T_v2w = np.eye(4)
    T_v2w[:3, :3] = R
    T_v2w[:3, [3]] = t

    # trafo from wcf -> rcf
    sample0 = nusc.get('sample', scene['first_sample_token'])
    sample0_data = nusc.get('sample_data', sample0['data']['LIDAR_TOP'])
    ego_pose = nusc.get('ego_pose', sample0_data['ego_pose_token'])
    R = Quaternion(ego_pose['rotation']).rotation_matrix
    t = np.array(ego_pose['translation'])[..., np.newaxis]
    T_w2r = np.eye(4)
    T_w2r[:3, :3] = R.T
    T_w2r[:3, [3]] = -np.dot(R.T, t)

    # trafo from vcf -> rcf
    T_v2r = np.dot(T_w2r, T_v2w)
    rotAngle = np.arctan2(T_v2r[1, 0], T_v2r[0, 0])
    vehiclePose = np.array([T_v2r[0, 3], T_v2r[1, 3], rotAngle])

    return sensorPose, vehiclePose, timestamp


# ===========================#
def findNextIndex(currIdx, refSweepName, sweepNames):
    # extract the reference timestamp from the reference sweep name
    refTimestamp = int(refSweepName[-20:-4])

    isSearching = True
    while (isSearching):
        # in case the last entry is reached, break the loop
        if (currIdx == (sweepNames.shape[0] - 1)):
            isSearching = False
            break

        # extract the current and next timestamps from the sweep names
        currTimestamp = int(sweepNames[currIdx][-20:-4])
        nextTimestamp = int(sweepNames[currIdx + 1][-20:-4])

        # compute distance between current and next timestamp towards reference timestamp
        diff = np.abs(refTimestamp - currTimestamp)
        diff_ = np.abs(refTimestamp - nextTimestamp)

        # test whether the next timestamp is closer to the reference timestamp or not        
        if (diff >= diff_):
            currIdx += 1
        else:
            isSearching = False

    return currIdx


# ===========================#
def findAllSweepsForTheScene(scene, sensorNames):
    """
    sensorType: 'CAM_FRONT','CAM_BACK', ...
                'CAM_BACK_LEFT','CAM_FRONT_LEFT','CAM_FRONT_RIGHT','CAM_BACK_RIGHT', ...
                'LIDAR_TOP', ...
                'RADAR_FRONT', ...
                'RADAR_FRONT_RIGHT','RADAR_FRONT_LEFT','RADAR_BACK_RIGHT','RADAR_BACK_LEFT'
    """
    allSweepNames = []
    for sensorName in sensorNames:
        # get the first & last sample
        sample0 = nusc.get('sample', scene['first_sample_token'])
        sample_ = nusc.get('sample', scene['last_sample_token'])

        # get first and last sample timestamp
        if (sensorName == 'LIDAR_TOP'):
            sampleT0 = int(nusc.get('sample_data', sample0['data'][sensorName])['filename'][-24:-8])
            sampleT_ = int(nusc.get('sample_data', sample_['data'][sensorName])['filename'][-24:-8])
        else:
            sampleT0 = int(nusc.get('sample_data', sample0['data'][sensorName])['filename'][-20:-4])
            sampleT_ = int(nusc.get('sample_data', sample_['data'][sensorName])['filename'][-20:-4])

        # get all sweep names
        sweepNames = [sweepName for sweepName in os.listdir(nusc.dataroot + 'sweeps/' + sensorName + '/')
                      if (sweepName.endswith('.jpg') or sweepName.endswith('.pcd') or sweepName.endswith('.pcd.bin'))]
        sweepNames.sort()
        sweepNames = np.asarray(sweepNames)

        # discard all sweeps outside the scenes timespan
        isInsideTimeSpan = np.zeros_like(sweepNames, dtype=bool)
        for iSweep in range(sweepNames.shape[0]):
            if (sensorName == 'LIDAR_TOP'):
                isInsideTimeSpan[iSweep] = int(sweepNames[iSweep][-24:-8]) > sampleT0
                isInsideTimeSpan[iSweep] *= int(sweepNames[iSweep][-24:-8]) < sampleT_
            else:
                isInsideTimeSpan[iSweep] = int(sweepNames[iSweep][-20:-4]) > sampleT0
                isInsideTimeSpan[iSweep] *= int(sweepNames[iSweep][-20:-4]) < sampleT_
        allSweepNames.append(sweepNames[isInsideTimeSpan])

    return allSweepNames


# ===========================#
def findAllSweepsBetweenTimesteps(t0, t1, sweepNames, sensorName):
    # discard all sweeps outside the scenes timespan
    isInsideTimeSpan = np.zeros_like(sweepNames, dtype=bool)
    for iSweep in range(sweepNames.shape[0]):
        if (sensorName == 'LIDAR_TOP'):
            isInsideTimeSpan[iSweep] = int(sweepNames[iSweep][-24:-8]) > t0
            isInsideTimeSpan[iSweep] *= int(sweepNames[iSweep][-24:-8]) < t1
        else:
            isInsideTimeSpan[iSweep] = int(sweepNames[iSweep][-20:-4]) > t0
            isInsideTimeSpan[iSweep] *= int(sweepNames[iSweep][-20:-4]) < t1

    sweepNames = sweepNames[isInsideTimeSpan]
    sweepNames.sort()

    return sweepNames


# ===========================#
def writePcSensorToCsv(csvWriter, pc, sensorPose, vehiclePose, timestamp_us, isStationary, sensorName):
    # define name of sensor in csv file
    name = 'None'
    if (sensorName == 'LIDAR_TOP'):
        name = 'l__'
    elif (sensorName == 'RADAR_FRONT'):
        name = 'rf_'
    elif (sensorName == 'RADAR_FRONT_RIGHT'):
        name = 'rfr'
    elif (sensorName == 'RADAR_FRONT_LEFT'):
        name = 'rfl'
    elif (sensorName == 'RADAR_BACK_RIGHT'):
        name = 'rbr'
    elif (sensorName == 'RADAR_BACK_LEFT'):
        name = 'rbl'

    # store the timestamp in us
    csvWriter.writerow([name + '_timestamp_us', timestamp_us])

    # trafo vcf -> wcf    
    row = vehiclePose.tolist()
    row.insert(0, name + '_vehicle_pose')
    csvWriter.writerow(row)

    # trafo scf -> vcf
    row = sensorPose.tolist()
    row.insert(0, name + '_sensor_pose')
    csvWriter.writerow(row)

    # x-pts in wcf
    row = pc[0, :].tolist()
    row.insert(0, name + '_pts_wcf_x')
    csvWriter.writerow(row)

    # y-pts in wcf
    row = pc[1, :].tolist()
    row.insert(0, name + '_pts_wcf_y')
    csvWriter.writerow(row)

    # is stationary flag 
    row = isStationary.tolist()
    row.insert(0, name + '_is_stationary')
    csvWriter.writerow(row)


# ===========================#
def writeBoundingBoxesToCsv(csvWriter, vehiclePose, dynBoundingBoxes_rcf, statBoundingBoxes_rcf, timestamp_us):
    numBoundingBoxes = dynBoundingBoxes_rcf.shape[0] + statBoundingBoxes_rcf.shape[0]

    if (numBoundingBoxes > 0):
        name = 'b__'

        # store the timestamp in us
        csvWriter.writerow([name + '_timestamp_us', timestamp_us])

        # define motion status
        is_stationary = np.ones((numBoundingBoxes))
        is_stationary[:dynBoundingBoxes_rcf.shape[0]] = 0

        # put static and dynamic boxes togehter into one array
        allBoundingBoxes_rcf = np.append(dynBoundingBoxes_rcf, statBoundingBoxes_rcf)

        # define centers, rotations and box diameters as arrays
        center_x = np.zeros((numBoundingBoxes))
        center_y = np.zeros((numBoundingBoxes))
        rotation_rad = np.zeros((numBoundingBoxes))
        diameter_x = np.zeros((numBoundingBoxes))
        diameter_y = np.zeros((numBoundingBoxes))
        for iBox in range(allBoundingBoxes_rcf.shape[0]):
            center_x[iBox] = allBoundingBoxes_rcf[iBox].center[0]
            center_y[iBox] = allBoundingBoxes_rcf[iBox].center[1]
            R = allBoundingBoxes_rcf[iBox].orientation.rotation_matrix
            rotation_rad[iBox] = np.arctan2(R[1, 0], R[0, 0])
            diameter_x[iBox] = allBoundingBoxes_rcf[iBox].wlh[1]
            diameter_y[iBox] = allBoundingBoxes_rcf[iBox].wlh[0]

        # trafo vcf -> wcf    
        row = vehiclePose.tolist()
        row.insert(0, name + '_vehicle_pose')
        csvWriter.writerow(row)

        # bounding box center x
        row = center_x.tolist()
        row.insert(0, name + '_center_x')
        csvWriter.writerow(row)

        # bounding box center y
        row = center_y.tolist()
        row.insert(0, name + '_center_y')
        csvWriter.writerow(row)

        # bounding box center x
        row = rotation_rad.tolist()
        row.insert(0, name + '_rotation_rad')
        csvWriter.writerow(row)

        # bounding box center x
        row = diameter_x.tolist()
        row.insert(0, name + '_diameter_x')
        csvWriter.writerow(row)

        # bounding box center x
        row = diameter_y.tolist()
        row.insert(0, name + '_diameter_y')
        csvWriter.writerow(row)

        # bounding box center x
        row = is_stationary.tolist()
        row.insert(0, name + '_is_stationary')
        csvWriter.writerow(row)


# ===========================#
def writeCamToCsv(csvWriter, vehiclePose, timestamp_us):
    name = 'c__'
    # store the timestamp in us
    csvWriter.writerow([name + '_timestamp_us', timestamp_us])
    # trafo vcf -> wcf    
    row = vehiclePose.tolist()
    row.insert(0, name + '_vehicle_pose')
    csvWriter.writerow(row)


# ===========================#
def findRefCamName(c_sweepNames):
    numSweepsForScene = [len(c_sweepName) for c_sweepName in c_sweepNames]
    refIdx = np.argmin(numSweepsForScene)
    return camNames[refIdx], c_sweepNames[refIdx]


# ===========================#
def vizPc(pc, isStat, vPose, maxDist=20., statColor='b', dynColor='c', zorder=1):
    # remove all points further away from current vehicle pose than maxDist
    l_range = np.sqrt((pc[0, :] - vPose[0]) ** 2 + (pc[1, :] - vPose[1]) ** 2)

    # split points into static and dynamic ones
    pltIdx_stat = np.logical_and(isStat.astype(np.bool), l_range < maxDist)
    pltIdx_dyna = np.logical_and(np.logical_not(isStat.astype(np.bool)), l_range < maxDist)

    plt.plot(pc[0, pltIdx_stat], pc[1, pltIdx_stat], statColor + '.', ms=1.5, zorder=zorder)
    plt.plot(pc[0, pltIdx_dyna], pc[1, pltIdx_dyna], dynColor + '.', ms=4.5, zorder=zorder)


# ===========================#
def loadPcWithMotionStatusInRcf(nusc, scene, sample, sensorName, minVisThres):
    # get sensor and vehicle pose and timestamp info
    sPose, vPose, t = getPoseAndTimeFromSample(nusc, scene, sample, sensorName)

    # get bounding boxes in wcf of dynamic objects for current and previous sample
    if (sensorName == 'LIDAR_TOP'):
        dynBoundingBoxes_rcf, statBoundingBoxes_rcf = interUtils.getDynObjBoundingBoxes_rcf(nusc, scene, sample,
                                                                                            sensorName, minVisThres)
    else:
        dynBoundingBoxes_rcf = []

    # get the point clouds with motion status in reference coordinate system (first lidar pose in scene) 
    pc, isStat = getPcWithMotionStatusInRcf(nusc, sample, vPose, dynBoundingBoxes_rcf,
                                            getSampleDataPath(sample, sensorName))
    pc[2, :] = isStat
    pcWithPose = {"pc": pc, "vPose": vPose, "sPose": sPose}
    return pcWithPose


# ===========================#
def processSweepOfPcSensor(nusc, sample, scene, sweepNames, vPoses01, t01, sensorNames, buffers, detMap, ismMap, t_r2i,
                           pF, pO, pD, pDim, mDim, aDim, numColsPerCone, storeIsmMapImg):
    # get the sensor poses
    sPoses = []
    for sensorName in sensorNames:
        sPose, _, _ = getPoseAndTimeFromSample(nusc, scene, sample, sensorName)
        sPoses.append(sPose)

    # find all sweep names between current and next sample
    sweepNames_ = []
    for i, sensorName in enumerate(sensorNames):
        sweepNames_.append(findAllSweepsBetweenTimesteps(t01[0], t01[-1], sweepNames[i], sensorName))

    if (sensorNames[0] == 'LIDAR_TOP'):
        # interpolate the 3D bounding box poses
        boundingBoxOfDynObjs_01_wcf = interUtils.getInterpolBoundingBoxes_wcf(nusc, scene, sample,
                                                                              numInterPts=NUM_INTERPOL_PTS)

    # find sensor with least sweeps
    iSensMin = 0
    for iSens in range(len(sensorNames)):
        if (len(sweepNames_[iSens]) <= len(sweepNames_[iSensMin])):
            iSensMin = iSens

    # process each sweep
    iSweeps = [0] * len(sensorNames)
    for iSweep in range(len(sweepNames_[iSensMin])):
        # get current ref timestamp
        if (sensorNames[0] == 'LIDAR_TOP'):
            t_ref = int(sweepNames_[iSensMin][iSweep][-24:-8])
        else:
            t_ref = int(sweepNames_[iSensMin][iSweep][-20:-4])

        # fill the buffers for each sensor up to the current ref timestamp
        for iSens, sensorName in enumerate(sensorNames):
            while (True):
                # find interpolated pose with closest timestamp
                if (sensorName == 'LIDAR_TOP'):
                    t = int(sweepNames_[iSens][iSweeps[iSens]][-24:-8])
                else:
                    t = int(sweepNames_[iSens][iSweeps[iSens]][-20:-4])
                if (t > t_ref):
                    break
                idx = np.argmin(abs(t01 - t))
                vPose = vPoses01[idx, :]

                boundingBoxesOfDynObjs_wcf = []
                if (sensorName == 'LIDAR_TOP'):
                    boundingBoxesOfDynObjs_wcf = boundingBoxOfDynObjs_01_wcf.flatten()

                    # get the point clouds with motion status in reference coordinate system (first lidar pose in scene)
                pc, isStat = getPcWithMotionStatusInRcf(nusc, sample, vPose, boundingBoxesOfDynObjs_wcf,
                                                        getSweepDataPath(sweepNames_[iSens][iSweeps[iSens]],
                                                                         sensorName), sweepFlag=True)
                pc[2, :] = isStat

                # add bev detection image to detection map
                detMap = mapUtils.markInGlobalImg(pc, detMap, t_r2i, mDim, pDim)

                # append new pointcloud with pose to the data buffer                
                pcWithPose = {"pc": pc, "vPose": vPose, "sPose": sPoses[iSens]}
                buffers[iSens].append(pcWithPose)

                # get the next sweep
                if (iSweeps[iSens] < (len(sweepNames_[iSens]) - 1)):
                    iSweeps[iSens] += 1
                else:
                    break

        # update map with current state of buffers
        eachSensHasAtleastOneRecording = True
        for iSens in range(len(buffers)):
            eachSensHasAtleastOneRecording = eachSensHasAtleastOneRecording and len(buffers[iSens])
        if (eachSensHasAtleastOneRecording and storeIsmMapImg):
            vPose = buffers[0][-1]["vPose"]
            ismImg = mapUtils.rayCastingBev(buffers, pDim, mDim, aDim, pF, pO, pD,
                                            numColsPerCone, vPose, 0, "",
                                            lidarFlag=(sensorName == 'LIDAR_TOP'), noDynFlag=True)
            # trafo pose to global image coordinates
            R_r2i = np.array([[0., -1.],
                              [1., 0.]])
            imgCenterPt = np.matmul(R_r2i, vPose[:2, np.newaxis] + t_r2i)[:, 0]
            # trafo icf -> pcf
            imgCenterPt = (imgCenterPt * pDim / mDim).astype(int)

            # map area which will be updated
            xLim = [imgCenterPt[0] - pDim // 2, imgCenterPt[0] + pDim // 2]
            yLim = [imgCenterPt[1] - pDim // 2, imgCenterPt[1] + pDim // 2]
            xLim_ = [0, pDim]
            yLim_ = [0, pDim]

            # check map boundaries
            if (xLim[0] < 0):
                dx = 0 - xLim[0]
                xLim[0] += dx
                xLim_[0] += dx
            if (yLim[0] < 0):
                dy = 0 - yLim[0]
                yLim[0] += dy
                yLim_[0] += dy
            if (xLim[1] > ismMap.shape[0]):
                dx = ismMap.shape[0] - xLim[1]
                xLim[1] += dx
                xLim_[1] += dx
            if (yLim[1] > ismMap.shape[1]):
                dy = ismMap.shape[1] - yLim[1]
                yLim[1] += dy
                yLim_[1] += dy

            # fuse new ism into each global map
            ismMap[xLim[0]:xLim[1], yLim[0]:yLim[1], :] = \
                mapUtils.fuseImgs(ismImg[xLim_[0]:xLim_[1], yLim_[0]:yLim_[1], :],
                                  ismMap[xLim[0]:xLim[1], yLim[0]:yLim[1], :])

    return buffers, detMap, ismMap


# ===========================#
def processSweepOfDeepIsm(nusc, sample, scene, sweepNames, vPoses01, t01, sensorNames, buffers, ismMaps, t_r2i,
                          pF, pO, pD, pDim, mDim, aDim, numColsPerCone, y_fake, uMin):
    [deep_ismMap, deep_ismMap_rescaled, deepGeo_ismMap] = ismMaps

    # get the sensor poses
    sPoses = []
    for sensorName in sensorNames:
        sPose, _, _ = getPoseAndTimeFromSample(nusc, scene, sample, sensorName)
        sPoses.append(sPose)

    # find all sweep names between current and next sample
    sweepNames_ = []
    for i, sensorName in enumerate(sensorNames):
        sweepNames_.append(findAllSweepsBetweenTimesteps(t01[0], t01[-1], sweepNames[i], sensorName))

    if (sensorNames[0] == 'LIDAR_TOP'):
        # interpolate the 3D bounding box poses
        boundingBoxOfDynObjs_01_wcf = interUtils.getInterpolBoundingBoxes_wcf(nusc, scene, sample,
                                                                              numInterPts=NUM_INTERPOL_PTS)

    # find sensor with least sweeps
    iSensMin = 0
    for iSens in range(len(sensorNames)):
        if (len(sweepNames_[iSens]) <= len(sweepNames_[iSensMin])):
            iSensMin = iSens

    # process each sweep
    iSweeps = [0] * len(sensorNames)
    for iSweep in range(len(sweepNames_[iSensMin])):
        # get current ref timestamp
        if (sensorNames[0] == 'LIDAR_TOP'):
            t_ref = int(sweepNames_[iSensMin][iSweep][-24:-8])
        else:
            t_ref = int(sweepNames_[iSensMin][iSweep][-20:-4])

        # fill the buffers for each sensor up to the current ref timestamp
        for iSens, sensorName in enumerate(sensorNames):
            while (True):
                # find interpolated pose with closest timestamp
                if (sensorName == 'LIDAR_TOP'):
                    t = int(sweepNames_[iSens][iSweeps[iSens]][-24:-8])
                else:
                    t = int(sweepNames_[iSens][iSweeps[iSens]][-20:-4])
                if (t > t_ref):
                    break
                idx = np.argmin(abs(t01 - t))
                vPose = vPoses01[idx, :]

                boundingBoxesOfDynObjs_wcf = []
                if (sensorName == 'LIDAR_TOP'):
                    boundingBoxesOfDynObjs_wcf = boundingBoxOfDynObjs_01_wcf.flatten()

                    # get the point clouds with motion status in reference coordinate system (first lidar pose in scene)
                pc, isStat = getPcWithMotionStatusInRcf(nusc, sample, vPose, boundingBoxesOfDynObjs_wcf,
                                                        getSweepDataPath(sweepNames_[iSens][iSweeps[iSens]],
                                                                         sensorName), sweepFlag=True)
                pc[2, :] = isStat

                # append new pointcloud with pose to the data buffer                
                pcWithPose = {"pc": pc, "vPose": vPose, "sPose": sPoses[iSens]}
                buffers[iSens].append(pcWithPose)

                # get the next sweep
                if (iSweeps[iSens] < (len(sweepNames_[iSens]) - 1)):
                    iSweeps[iSens] += 1
                else:
                    break

        # update map with current state of buffers
        eachSensHasAtleastOneRecording = True
        for iSens in range(len(buffers)):
            eachSensHasAtleastOneRecording = eachSensHasAtleastOneRecording and len(buffers[iSens])
        if (eachSensHasAtleastOneRecording):
            vPose = buffers[0][-1]["vPose"]

            # perform inference            
            x_ = pc2Bev(buffers, pDim, mDim, vPose, 0, None)[np.newaxis, :, :, np.newaxis] / 255
            deepIsmImg = sess.run(y_fake, feed_dict={x: x_})[0, ...]
            geoIsmImg = mapUtils.rayCastingBev(buffers, pDim, mDim, aDim, pF, pO, pD,
                                               numColsPerCone, vPose, 0, "",
                                               lidarFlag=(sensorName == 'LIDAR_TOP'), noDynFlag=True)
            # trafo pose to global image coordinates
            R_r2i = np.array([[0., -1.],
                              [1., 0.]])
            imgCenterPt = np.matmul(R_r2i, vPose[:2, np.newaxis] + t_r2i)[:, 0]
            # trafo icf -> pcf
            imgCenterPt = (imgCenterPt * pDim / mDim).astype(int)

            # map area which will be updated
            xLim = [imgCenterPt[0] - pDim // 2, imgCenterPt[0] + pDim // 2]
            yLim = [imgCenterPt[1] - pDim // 2, imgCenterPt[1] + pDim // 2]
            xLim_ = [0, pDim]
            yLim_ = [0, pDim]

            # check map boundaries
            if (xLim[0] < 0):
                dx = 0 - xLim[0]
                xLim[0] += dx
                xLim_[0] += dx
            if (yLim[0] < 0):
                dy = 0 - yLim[0]
                yLim[0] += dy
                yLim_[0] += dy
            if (xLim[1] > ismMaps[0].shape[0]):
                dx = ismMaps[0].shape[0] - xLim[1]
                xLim[1] += dx
                xLim_[1] += dx
            if (yLim[1] > ismMaps[0].shape[1]):
                dy = ismMaps[0].shape[1] - yLim[1]
                yLim[1] += dy
                yLim_[1] += dy

            # deep ism map
            deepIsmImg_ = deepIsmImg.copy()
            deep_ismMap[xLim[0]:xLim[1], yLim[0]:yLim[1], :] = \
                mapUtils.fuseImgs(deepIsmImg_[xLim_[0]:xLim_[1], yLim_[0]:yLim_[1], :],
                                  deep_ismMap[xLim[0]:xLim[1], yLim[0]:yLim[1], :],
                                  comb_rule=0, entropy_scaling=False, u_min=0.)

            # deep ism map with entropy rescaling
            # comb_rule 0:Yager, 1:YaDer, 2:Yager mu>uMin & YaDer mu<=uMinelse, else:Dempster
            deepIsmImg_ = deepIsmImg.copy()
            deep_ismMap_rescaled[xLim[0]:xLim[1], yLim[0]:yLim[1], :] = \
                mapUtils.fuseImgs(deepIsmImg_[xLim_[0]:xLim_[1], yLim_[0]:yLim_[1], :],
                                  deep_ismMap_rescaled[xLim[0]:xLim[1], yLim[0]:yLim[1], :],
                                  comb_rule=0, entropy_scaling=True, u_min=0.)

            # deep ism map with entropy rescaling and lower threshold on u
            deepIsmImg_ = deepIsmImg.copy()
            deepGeo_ismMap[xLim[0]:xLim[1], yLim[0]:yLim[1], :] = \
                mapUtils.fuseImgs(deepIsmImg_[xLim_[0]:xLim_[1], yLim_[0]:yLim_[1], :],
                                  deepGeo_ismMap[xLim[0]:xLim[1], yLim[0]:yLim[1], :],
                                  comb_rule=0, entropy_scaling=True, u_min=uMin)

            # geo ism
            deepGeo_ismMap[xLim[0]:xLim[1], yLim[0]:yLim[1], :] = \
                mapUtils.fuseImgs(geoIsmImg[xLim_[0]:xLim_[1], yLim_[0]:yLim_[1], :],
                                  deepGeo_ismMap[xLim[0]:xLim[1], yLim[0]:yLim[1], :],
                                  comb_rule=2, entropy_scaling=False, u_min=0.)

    return buffers, [deep_ismMap, deep_ismMap_rescaled, deepGeo_ismMap]


# ===========================#
def addImgTo360Img_inRcf(img, img_360, homographyMat, vPose, croppPortion, imgMask, pDim):
    # pcf -> vcf
    T_p2v = np.eye(3)
    T_p2v[1, 1] = -1.
    T_p2v[0, 2] = -pDim // 2
    T_p2v[1, 2] = pDim // 2

    # trafo vcf -> rcf in homogeneous coordinates
    T_v2r = np.eye(3)
    T_v2r[:2, :2] = rotMat(vPose[2])
    T_v2r[:2, 2] = vPose[:2]

    # only translation back to vcf
    T_r2v_ = np.eye(3)
    T_r2v_[:2, 2] = -vPose[:2]

    T_all = np.dot(np.linalg.inv(T_p2v), np.dot(T_r2v_, np.dot(T_v2r, T_p2v)))

    # crop camera image to remove infinite distance points
    img = img[int(img.shape[0] * croppPortion):, ...]
    img = img.copy()
    if (len(img.shape) == 3):
        img[imgMask, :] = np.zeros(3)
    else:
        img[imgMask] = 0

    # warp from camera to bird's eye view perspective (in rcf)
    img = cv2.warpPerspective(img, np.dot(T_all, homographyMat), (pDim, pDim), flags=cv2.INTER_NEAREST)
    # img = cameraUtils.warpPerspective(img, np.dot(T_all, homographyMat), pDim)
    if (len(img.shape) <= 2):
        img = img[..., np.newaxis]

    # replace all non empty pixels in the 360 bird's eye view image
    img_360[np.sum(img, axis=2) > 0, :] = img[np.sum(img, axis=2) > 0, :]

    return img_360


# ===========================#
def applyCityScapesColorMap(img):
    colormap = np.zeros((256, 3), dtype=np.uint8)
    colormap[0] = [128, 64, 128]
    colormap[1] = [244, 35, 232]
    colormap[2] = [70, 70, 70]
    colormap[3] = [102, 102, 156]
    colormap[4] = [190, 153, 153]
    colormap[5] = [153, 153, 153]
    colormap[6] = [250, 170, 30]
    colormap[7] = [220, 220, 0]
    colormap[8] = [107, 142, 35]
    colormap[9] = [152, 251, 152]
    colormap[10] = [70, 130, 180]
    colormap[11] = [220, 20, 60]
    colormap[12] = [255, 0, 0]
    colormap[13] = [0, 0, 142]
    colormap[14] = [0, 0, 70]
    colormap[15] = [0, 60, 100]
    colormap[16] = [0, 80, 100]
    colormap[17] = [0, 0, 230]
    colormap[18] = [119, 11, 32]

    img_rgb = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)

    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            img_rgb[x, y, :] = colormap[img[x, y, 0]]
    return img_rgb


# ===========================#
def surroundCam2Bev(nusc, scene, sample, imgStorDir,
                    vPose_ref, t_ref, camNames, homographyMats, croppPortion, imgMasks, pDim, semSegFlag=False):
    # project all cameras into one bird's eye view image
    img_360 = np.zeros((pDim, pDim, 3), dtype=np.uint8)
    for iCam in range(len(camNames)):
        # load camera image
        cam_data = nusc.get('sample_data', sample['data'][camNames[iCam]])
        camFileName = cam_data['filename']
        if (semSegFlag):
            camFileName = camFileName[:8] + 'SEMSEG_' + camFileName[
                                                        8:8 + len(camNames[iCam]) + 32] + 'SEMSEG_' + camFileName[
                                                                                                      8 + len(camNames[
                                                                                                                  iCam]) + 32:]
        img = np.asarray(Image.open(osp.join(nusc.dataroot, camFileName)))

        # add camera image into 360 bird's eye view image
        img_360 = addImgTo360Img_inRcf(img, img_360, homographyMats[iCam], vPose_ref, croppPortion, imgMasks[iCam],
                                       pDim)

    if (semSegFlag):
        # apply cityscapes colormap 
        img_360 = applyCityScapesColorMap(img_360)

    # store 360 bird's eye view image
    img_360 = Image.fromarray(img_360)
    if (semSegFlag):
        img_360.save(imgStorDir + imgStorDir.split("/")[-2] + "__{:}.png".format(t_ref))
    else:
        img_360.save(imgStorDir + imgStorDir.split("/")[-2] + "__{:}.png".format(t_ref))

    # ===========================#


def processSweepOf360Camera(nusc, sweepNames_ref, sweepNames, vPoses01, t01, csvWriter, imgStorDir, store360ImgFlag,
                            storeCsvFlag,
                            refCamName, camNames, homographyMats, croppPortion, pDim, semSegFlag=False):
    # find all sweep names between current and next sample
    sweepNames_ref_ = findAllSweepsBetweenTimesteps(t01[0], t01[-1], np.asarray(sweepNames_ref), refCamName)

    sweepNames_ = []
    for iCam in range(len(sweepNames)):
        sweepNames_.append(findAllSweepsBetweenTimesteps(t01[0], t01[-1], np.asarray(sweepNames[iCam]), refCamName))

    # initialize sweep indices all cameras 
    sweepIdx = np.zeros(len(sweepNames), dtype=int)

    # process each sweep
    for iSweep in range(len(sweepNames_ref_)):
        # find interpolated pose with closest timestamp
        t_ref = int(sweepNames_ref_[iSweep][-20:-4])
        idx = np.argmin(abs(t01 - t_ref))
        vPose_ref = vPoses01[idx, :]

        # store current sweep
        if (storeCsvFlag):
            writeCamToCsv(csvWriter, vPoses01[idx, :], t_ref)

        # project all cameras into one bird's eye view image
        img_360 = np.zeros((pDim, pDim, 3), dtype=np.uint8)
        for iCam in range(len(camNames)):
            # find closest camera image
            sweepIdx[iCam] = findNextIndex(sweepIdx[iCam], sweepNames_ref_[iSweep], sweepNames_[iCam])

            # load camera image
            img = np.asarray(
                Image.open(nusc.dataroot + 'sweeps/' + camNames[iCam] + '/' + sweepNames_[iCam][sweepIdx[iCam]]))

            # add camera image into 360 bird's eye view image
            img_360 = addImgTo360Img_inRcf(img, img_360, homographyMats[iCam], vPose_ref, croppPortion, pDim)

        # store 360 bird's eye view image
        if (store360ImgFlag):
            img_360 = Image.fromarray(img_360)
            if (semSegFlag):
                img_360.save(imgStorDir + "s__{:}.png".format(t_ref))
            else:
                img_360.save(imgStorDir + "c__{:}.png".format(t_ref))

            # ===========================#


def processSampleOfMonodepth(nusc, scene, sample, imgStorDir,
                             refCamName, camNames, pDim, mDim):
    # get sensor and vehicle pose and timestamp info
    _, vPose_ref, t_ref = getPoseAndTimeFromSample(nusc, scene, sample, refCamName)

    # construct bev image based on monodepth of all cameras
    bevImg = np.zeros((pDim, pDim), dtype=np.uint8)
    for camName in camNames:
        # get the depth file
        cam_data = nusc.get('sample_data', sample['data'][camName])
        depthFileName = cam_data['filename'][:-4] + ".npy"
        depthFileName = depthFileName[:len("samples/")] + "DEPTH_" + depthFileName[len("samples/"):]

        # load depth
        depthImg = np.load(nusc.dataroot + depthFileName)[:, :, 0]
        # img = np.asarray(Image.open(osp.join(nusc.dataroot, cam_data['filename'])))

        # get point cloud without ground plane
        pc = cameraUtils.monoDepth2PcInVcf(nusc, depthImg, sample['data'][camName])

        # trafo vcf -> rcf in homogeneous coordinates
        pc[:2, :] = np.dot(rotMat(vPose_ref[2]), pc[:2, :])

        # trafo point cloud to bev image
        bevImg_ = cameraUtils.pc2BevImg(pc, pDim, mDim)

        # put new bev image entries into overall bev image
        bevImg[bevImg_ > 0] = bevImg_[bevImg_ > 0]

    bevImg = Image.fromarray(bevImg)
    bevImg.save(imgStorDir + imgStorDir.split("/")[-2] + "__{:}.png".format(t_ref))


# ===========================#
def processSampleOfDeepIsm(t_ref, inputName, modelName, sceneDir, y_fake, x, ismMap, ismMapScaled, t_r2i, vPose_ref,
                           storeDeepIsm, storeProgress):
    x_ = np.array(Image.open(sceneDir + inputName + "/" + inputName + "__{:}.png".format(t_ref)))

    # trafo inputs from [0,255] -> [0,1]
    if (len(x_.shape) == 2):
        x_ = x_[np.newaxis, :, :, np.newaxis] / 255
    else:
        x_ = x_[np.newaxis, :, :, :] / 255

    # perform inference
    ismImg = sess.run(y_fake, feed_dict={x: x_})[0, ...]

    # trafo pose to global image coordinates
    R_r2i = np.array([[0., -1.],
                      [1., 0.]])
    imgCenterPt = np.matmul(R_r2i, vPose_ref[:2, np.newaxis] + t_r2i)[:, 0]
    # trafo icf -> pcf
    imgCenterPt = (imgCenterPt * pDim / mDim).astype(int)

    # map area which will be updated
    xLim = [imgCenterPt[0] - pDim // 2, imgCenterPt[0] + pDim // 2]
    yLim = [imgCenterPt[1] - pDim // 2, imgCenterPt[1] + pDim // 2]
    xLim_ = [0, pDim]
    yLim_ = [0, pDim]

    # check map boundaries
    if (xLim[0] < 0):
        dx = 0 - xLim[0]
        xLim[0] += dx
        xLim_[0] += dx
    if (yLim[0] < 0):
        dy = 0 - yLim[0]
        yLim[0] += dy
        yLim_[0] += dy
    if (xLim[1] > ismMap.shape[0]):
        dx = ismMap.shape[0] - xLim[1]
        xLim[1] += dx
        xLim_[1] += dx
    if (yLim[1] > ismMap.shape[1]):
        dy = ismMap.shape[1] - yLim[1]
        yLim[1] += dy
        yLim_[1] += dy

    # fuse new ism into global map
    ismMap[xLim[0]:xLim[1], yLim[0]:yLim[1], :] = \
        mapUtils.fuseImgs(ismImg[xLim_[0]:xLim_[1], yLim_[0]:yLim_[1], :], ismMap[xLim[0]:xLim[1], yLim[0]:yLim[1], :])

    # perform entropy scaling on ism
    ismImg_scaled = ismImg.copy()
    ismMapScaled[xLim[0]:xLim[1], yLim[0]:yLim[1], :] = \
        mapUtils.fuseImgs(ismImg_scaled[xLim_[0]:xLim_[1], yLim_[0]:yLim_[1], :],
                          ismMapScaled[xLim[0]:xLim[1], yLim[0]:yLim[1], :], entropy_scaling=True, u_min=0.)

    # split the channels
    # ismImg_ = np.append(ismImg[...,0],ismImg[...,1],axis=1)
    # ismImg = np.append(ismImg_, ismImg[...,2],axis=1)

    if (storeDeepIsm):
        ismImg = Image.fromarray((ismImg * 255).astype(np.uint8))
        ismImg.save(sceneDir + modelName + "/" + modelName + "__{:}.png".format(t_ref))
    if (storeProgress):
        ismImg_scaled = Image.fromarray((ismImg_scaled * 255).astype(np.uint8))
        ismImg_scaled.save(sceneDir + modelName + "/" + modelName + "_scaled__{:}.png".format(t_ref))

    return ismMap, ismMapScaled


# ===========================#
def pc2Bev(buffers, pDim, mDim, vPose_ref, t_ref, imgStorDir, lidarFlag=False, wholeTempInfo=False):
    bevImg = np.zeros((pDim, pDim), dtype=np.uint8)
    for buffer in buffers:
        for iBuff, pcWithPose in enumerate(buffer):
            # trafo rcf -> vcf
            pc = pcWithPose["pc"][:2, :] - vPose_ref[:2, np.newaxis]
            isStat = 0
            if lidarFlag:
                isStat = np.ones_like(pcWithPose["pc"][2, :])
            else:
                isStat = pcWithPose["pc"][2, :]
            # compute bev image
            if wholeTempInfo:
                # encode whole temporal information without discounting
                bevImg_ = mapUtils.pc2VehicleCenteredBevImg(pc[:2, :], isStat,
                                                            pDim=pDim, mDim=mDim)
                # encode all static pixels as odd and dynamic pixels as even numbers
                bevImg_[bevImg_ == 255] = 2 * (iBuff + 1) - 1
                bevImg_[bevImg_ == 128] = 2 * (iBuff + 1)
            else:
                # discount dynamic detections over time
                bevImg_ = mapUtils.pc2VehicleCenteredBevImg(pc[:2, :], isStat,
                                                            pDim=pDim, mDim=mDim, dynDiscount=(iBuff + 1) / len(buffer))
            # mark latest bevImg in overall bev img
            newEntryMask = (bevImg_ > 0)
            bevImg[newEntryMask] = bevImg_[newEntryMask]
    # store bev image
    if (imgStorDir == None):
        return bevImg
    else:

        # cmap = plt.get_cmap('jet')
        # norm = plt.Normalize(0, 40)
        #
        # bevImg_dyn = np.zeros_like(bevImg)
        # bevImg_stat = np.zeros_like(bevImg)
        #
        # stat_mask = np.logical_and(bevImg % 2, bevImg > 0)
        # dyn_mask = np.logical_and(np.logical_not(bevImg % 2), bevImg > 0)
        #
        # bevImg_stat[stat_mask] = bevImg[stat_mask]
        # bevImg_dyn[dyn_mask] = bevImg[dyn_mask]
        #
        # plt.figure(1)
        # plt.subplot(2, 2, 1)
        # plt.imshow(cmap(norm(bevImg_stat)))
        # plt.subplot(2, 2, 2)
        # plt.imshow(cmap(norm(bevImg_dyn)))
        # plt.subplot(2, 2, 3)
        # plt.imshow(bevImg > 0)
        # plt.subplot(2, 2, 4)
        # plt.imshow(bevImg_dyn > 0)
        # plt.pause(0.1)

        bevImg = Image.fromarray(bevImg)
        if wholeTempInfo:
            bevImg.save(imgStorDir + imgStorDir.split("/")[-2] + "_all__{:}.png".format(t_ref))
        else:
            bevImg.save(imgStorDir + imgStorDir.split("/")[-2] + "__{:}.png".format(t_ref))
    # ===========================#


def createImgStorDir(dirName):
    if not os.path.exists(dirName):
        try:
            os.makedirs(dirName)
        except:
            pass
    return dirName


# ===========================#
def saveImg(img, dirPath):
    img_ = Image.fromarray((img * 255).astype(np.uint8))
    img_.save(dirPath)


# ================================================= MAIN ==============================================================#
# flags to toggle visualization
storeProgress = False
# radar
storeRadarImg = False
storeRadarMap = False
storeIrmImg = False
storeIrmMap = False
# lidar
storeLidarImg = False
storeLidarMap = False
storeIlmImg = False
storeIlmMap = True
storeIlmMapPatches = False
# camera
store360CamImg = False
store360SemSegImg = False
store360MonoDepthImg = False
# deep ism
modelDir = "./models/exp_deep_ism_comparison/"
# modelName = "dirNet_ilmMapPatchDisc_r_1__20201223_215714_ckpt_180.pb"
# modelName = "dirNet_ilmMapPatchDisc_r_20__20201223_215858_ckpt_683.pb"
# modelName = "dirNet_ilmMapPatchDisc_l__20201226_082638_ckpt_723.pb"
# modelName = "dirNet_ilmMapPatchDisc_lr20__20201227_222824_ckpt_318.pb"
# modelName = "dirNet_ilmMapPatchDisc_d__20201226_082043_ckpt_402.pb"
# modelName = "dirNet_ilmMapPatchDisc_dr20__20201227_222639_ckpt_244.pb"
# modelName = "shiftNet_ilmMapPatchDisc_r_1__20201223_215050_ckpt_198.pb"
modelName = "shiftNet_ilmMapPatchDisc_r_20__20201223_215231_ckpt_321.pb"
# modelName = "softNet_ilmMapPatchDisc_r_1__20201223_215415_ckpt_688.pb"
deepIsmInputName = modelName[modelName.find("_") + 17:modelName.find("__")]
storeDeepIsm = False
storeDeepIsmMap = True
uMin = 0.3
if (storeDeepIsm or storeDeepIsmMap):
    tf.reset_default_graph()
    tf.keras.backend.clear_session()
    sess = tf.Session()
    graphFile = gfile.FastGFile(modelDir + modelName, 'rb')
    x, y_fake = restoreGraphFromPB(sess, graphFile)

# minimum distance to move within a scene to use it for mapping
minDistTravelledThres = 20.  # [m]

# disregard all bounding boxes with a visibility lower than certain threshold
# visibility is defined as the fraction of pixels of a particular annotation that are visible over the 6 camera feeds, grouped into 4 bins.
minVisThres = 3

# load data set
DATA_DIR = 'C:/Users/Daniel/Documents/_uni/PhD/code/_DATASETS_/NuScenes/'
STORAGE_DIR = 'C:/Users/Daniel/Documents/_uni/PhD/code/_DATASETS_/occMapDataset_/'
if not os.path.exists(STORAGE_DIR):
    os.makedirs(STORAGE_DIR)
# nuscVersion = "v1.0-mini"
nuscVersion = "v1.0-trainval"
if not ('nusc' in locals()):
    nusc = NuScenes(version=nuscVersion, dataroot=DATA_DIR, verbose=True)

# define all camera and radar names
camNames = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
            'CAM_BACK', 'CAM_BACK_RIGHT', 'CAM_BACK_LEFT']
radarNames = ['RADAR_FRONT', 'RADAR_FRONT_RIGHT', 'RADAR_FRONT_LEFT',
              'RADAR_BACK_RIGHT', 'RADAR_BACK_LEFT']
lidarNames = ['LIDAR_TOP']

# portion of image from ground up used for bird's eye view (SPECIFIC FOR HOMOGRAPHY MATRIX)
croppPortion = 0.55

# image dimension in bird's eye view
pDim = 128  # [px]
mDim = 40.  # [m]
aRes_deg = 0.4
aDim = int(360 / aRes_deg)  # [deg]

# lidar params
maxBuffSize_l = 1
pF_lMap = [0.025]
pO_lMap = 0.5
pD_lMap = 0.3
pF_ilm = [0.9]
pO_ilm = 0.9
pD_ilm = 0.5
angleResCone_l = np.array([3])  # [deg]
numColsPerCone_l = (angleResCone_l / aRes_deg / 2).astype(int) * 2

# radar params
# buffSizes_r = [1,5,20] # from smallest to biggest number, always!!!
# maxBuffSize_r = int(np.max(buffSizes_r))
# pF_rMap = [0.1,0.1]
# pO_rMap = 0.3
# pD_rMap = 0.3
# pF_irm = [0.9,0.9]
# pO_irm = 0.9
# pD_irm = 0.5
# angleResCone_r = np.array([5,30.]) # [deg]
# numColsPerCone_r = (angleResCone_r/aRes_deg/2).astype(int)*2
buffSizes_r = [20]  # from smallest to biggest number, always!!!
maxBuffSize_r = int(np.max(buffSizes_r))
pF_rMap = [0.1]
pO_rMap = 0.3
pD_rMap = 0.3
pF_irm = [0.9, 0.9]
pO_irm = 0.9
pD_irm = 0.5
angleResCone_r = np.array([5])  # [deg]
numColsPerCone_r = (angleResCone_r / aRes_deg / 2).astype(int) * 2


# ================#
# Process Scenes  
# ================#
# for iScene in tqdm(sceneIdxs):
def main(iScene):
    # lidar and radar buffers
    buff_l = [deque(maxlen=maxBuffSize_l)]
    buff_r = [deque(maxlen=maxBuffSize_r) for _ in range(len(radarNames))]

    # get current scene
    scene = nusc.scene[iScene]
    print(iScene)

    # only process scene, iff vehicle has travelled a certain distance
    travelledDist = sceneAttribUtils.computeTravelledDistanceForScene(nusc, scene)
    if travelledDist < minDistTravelledThres:
        return 1

    # only use day scenes
    if (("night" in scene["description"].lower()) or
            ("difficult lighting" in scene["description"].lower())):
        return 1

    # check whether the scene is in train, val or test
    setName = ""
    if scene["name"] in trainNames:
        setName = "train/_scenes/"
    elif scene["name"] in valNames:
        setName = "val/_scenes/"
    else:
        setName = "test/_scenes/"

    # check, if scene has already been processed
    # if os.path.exists(STORAGE_DIR + setName + "scene{:04}".format(iScene) + '/'):
    #     return 1
    # create folders to store images
    if (store360CamImg):
        c_storDir = createImgStorDir(STORAGE_DIR + setName + "scene{:04}".format(iScene) + '/c/')
    if (storeDeepIsm or storeDeepIsmMap):
        deepIsm_storDir = createImgStorDir(
            STORAGE_DIR + setName + "scene{:04}".format(iScene) + "/" + modelName[:modelName.find("__")] + "/")
    if (store360SemSegImg):
        s_storDir = createImgStorDir(STORAGE_DIR + setName + "scene{:04}".format(iScene) + '/s/')
    if (store360MonoDepthImg):
        d_storDir = createImgStorDir(STORAGE_DIR + setName + "scene{:04}".format(iScene) + '/d/')
    r_storDirs = []
    irm_storDirs = []
    for buffSize_r in buffSizes_r:
        if (storeRadarImg or storeRadarMap):
            r_storDirs.append(
                createImgStorDir(STORAGE_DIR + setName + "scene{:04}".format(iScene) + '/r_{0:}/'.format(buffSize_r)))
        if (storeIrmImg):
            irm_storDirs.append(
                createImgStorDir(STORAGE_DIR + setName + "scene{:04}".format(iScene) + '/irm_{0:}/'.format(buffSize_r)))
    if (storeIrmMap):
        irmMap_storDir = createImgStorDir(STORAGE_DIR + setName + "scene{:04}".format(iScene) + '/irmMap/')
    if (storeLidarImg or storeLidarMap):
        l_storDir = createImgStorDir(STORAGE_DIR + setName + "scene{:04}".format(iScene) + '/l/')
    if (storeIlmImg):
        ilm_storDir = createImgStorDir(STORAGE_DIR + setName + "scene{:04}".format(iScene) + '/ilm/')
    if (storeIlmMap):
        ilmMap_storDir = createImgStorDir(STORAGE_DIR + setName + "scene{:04}".format(iScene) + '/ilmMap/')
        ilmMaps_storDir = createImgStorDir(STORAGE_DIR + setName + "ilmMaps/")
    if (storeIlmMapPatches):
        ilmMapPatch_storDir = createImgStorDir(STORAGE_DIR + setName + "scene{:04}".format(iScene) + '/ilmMapPatch/')

    # get all sweep names inside the scene's timespan
    c_sweepNames = findAllSweepsForTheScene(scene, camNames)
    if ((storeRadarImg) or (storeRadarMap) or (storeIrmImg) or (storeIrmMap) or (storeDeepIsmMap)):
        r_sweepNames = findAllSweepsForTheScene(scene, radarNames)
    if ((storeLidarImg) or (storeLidarMap) or (storeIlmImg) or (storeIlmMapPatches) or (storeIlmMap)):
        l_sweepNames = findAllSweepsForTheScene(scene, lidarNames)

    # choose reference camera for current scene as the one with the least sweeps    
    refCamName, sweepNames_ref = findRefCamName(c_sweepNames)

    # compute homography matrices for each camera for the current scene 
    homographyMats = [0] * len(camNames)
    imgMasks = [0] * len(camNames)
    if (store360CamImg or store360SemSegImg):
        for iCam, camName in enumerate(camNames):
            homographyMat, imgMask = cameraUtils.computeHomography(nusc, scene, camName, pDim=pDim, mDim=mDim,
                                                                   croppPortion=croppPortion, heightEps=0.02)
            if (homographyMat.shape[0] != 0):
                homographyMats[iCam] = homographyMat
                imgMasks[iCam] = imgMask

    # create map images
    detMap, ismMap, t_r2i = mapUtils.initGlobalImg(nusc, scene, mDim, pDim)
    l_detMap = detMap.copy()
    r_detMap = detMap.copy()
    l_ismMap = ismMap.copy()
    r_ismMap = ismMap.copy()
    deep_ismMap = ismMap.copy()
    deep_ismMap_rescaled = ismMap.copy()
    deepGeo_ismMap = ismMap.copy()
    mappedArea = detMap.copy()

    # all ref vehicle positions
    vPoses_ref = []
    ts_ref = []
    bboxes = []

    # ================#
    # Process Samples  
    # ================#
    sample = nusc.get('sample', scene['first_sample_token'])
    # for iSample in tqdm(range(scene['nbr_samples'] - 1)):
    for iSample in tqdm(range(5)):
        #     get sensor and vehicle pose and timestamp info
        _, vPose_ref, t_ref = getPoseAndTimeFromSample(nusc, scene, sample, refCamName)
        if (iSample > 0):
            # store the vehicle position in image map coordinates
            vPoses_ref.append(vPose_ref)
            ts_ref.append(t_ref)

            # process lidar
            if storeLidarImg or storeIlmImg:
                buff_l[0].append(loadPcWithMotionStatusInRcf(nusc, scene, sample, 'LIDAR_TOP', minVisThres))
            if storeLidarImg:
                pc2Bev(buff_l, pDim, mDim, vPose_ref, t_ref, l_storDir, lidarFlag=True)
            if storeIlmImg:
                mapUtils.rayCastingBev(buff_l, pDim, mDim, aDim, pF_ilm, pO_ilm, pD_ilm,
                                       numColsPerCone_l, vPose_ref, t_ref, ilm_storDir, lidarFlag=True)
            if storeIlmMapPatches:
                dynBoundingBoxes_rcf, _ = \
                    interUtils.getDynObjBoundingBoxes_rcf(nusc, scene, sample, "LIDAR_TOP", minVisThres)
                bboxes.append(dynBoundingBoxes_rcf)

            # process 360 radars
            if (storeRadarImg or storeIrmImg):
                for iRadar, radarName in enumerate(radarNames):
                    buff_r[iRadar].append(loadPcWithMotionStatusInRcf(nusc, scene, sample, radarName, minVisThres))
                for iBuffSize in range(len(buffSizes_r)):
                    # only take the last buffSize_r samples from the buffer
                    buffSize_r = buffSizes_r[iBuffSize]
                    buff_r_ = []
                    for iSens in range(len(buff_r)):
                        sliceLowIdx = int(np.clip(len(buff_r[iSens]) - buffSize_r, 0, None))
                        sliceUpIdx = int(len(buff_r[iSens]))
                        buff_r_.append(list(itertools.islice(buff_r[iSens], sliceLowIdx, sliceUpIdx)))

                    if (storeRadarImg):
                        # pc2Bev(buff_r_, pDim, mDim, vPose_ref, t_ref, r_storDirs[iBuffSize])
                        pc2Bev(buff_r_, pDim, mDim, vPose_ref, t_ref, r_storDirs[iBuffSize], wholeTempInfo=False)
                    if (storeIrmImg):
                        mapUtils.rayCastingBev(buff_r_, pDim, mDim, aDim, pF_irm, pO_irm, pD_irm,
                                               numColsPerCone_r, vPose_ref, t_ref, irm_storDirs[iBuffSize],
                                               lidarFlag=False, allPrevFlag=True)

            # process 360 cameras
            if (store360CamImg):
                surroundCam2Bev(nusc, scene, sample, c_storDir,
                                vPose_ref, t_ref, camNames, homographyMats, croppPortion, imgMasks, pDim)

            # process semseg of 360 cameras  
            if (store360SemSegImg):
                surroundCam2Bev(nusc, scene, sample, s_storDir,
                                vPose_ref, t_ref, camNames, homographyMats, croppPortion, imgMasks, pDim, True)

            # process 360 monodepth
            if (store360MonoDepthImg):
                processSampleOfMonodepth(nusc, scene, sample, d_storDir, refCamName, camNames, pDim, mDim)

            # process deep ism
            if (storeDeepIsm or storeDeepIsmMap):
                sceneDir = STORAGE_DIR + setName + "scene{:04}".format(iScene) + "/"
                deep_ismMap, deep_ismMap_rescaled = processSampleOfDeepIsm(t_ref, deepIsmInputName, modelName[:modelName.find("__")], sceneDir,
                                                                           y_fake, x, deep_ismMap, deep_ismMap_rescaled, t_r2i, vPose_ref,
                                                                           storeDeepIsm, storeProgress)

        # get next sample
        sample = nusc.get('sample', sample['next'])

        # ================#
        # Process Sweeps  
        # ================#
        # interpolate the poses between the current and next refcamera
        _, vPose_ref_next, t_ref_next = getPoseAndTimeFromSample(nusc, scene, sample, refCamName)
        if ((storeLidarImg) or (storeLidarMap) or (storeIlmImg) or (storeIlmMapPatches) or (storeIlmMap) or
                (storeRadarImg) or (storeRadarMap) or (storeIrmImg) or (storeIrmMap) or (storeDeepIsmMap)):
            vPoses01, t01 = interUtils.interpolate2DPoses(vPose_ref, vPose_ref_next, t_ref, t_ref_next,
                                                          numInterPts=NUM_INTERPOL_PTS)

        # process sweep data using the interpolated poses
        if ((storeLidarImg) or (storeLidarMap) or (storeIlmImg) or (storeIlmMapPatches) or (storeIlmMap)):
            buff_l, l_detMap, l_ismMap = processSweepOfPcSensor(nusc, sample, scene, l_sweepNames, vPoses01, t01,
                                                                lidarNames, buff_l, l_detMap, l_ismMap, t_r2i,
                                                                pF_lMap, pO_lMap, pD_lMap, pDim, mDim, aDim,
                                                                numColsPerCone_l, storeIlmMap or storeIlmMapPatches)
        if ((storeRadarImg) or (storeRadarMap) or (storeIrmImg) or (storeIrmMap)):
            buff_r, r_detMap, r_ismMap = processSweepOfPcSensor(nusc, sample, scene, r_sweepNames, vPoses01, t01,
                                                                radarNames, buff_r, r_detMap, r_ismMap, t_r2i,
                                                                pF_rMap, pO_rMap, pD_rMap, pDim, mDim, aDim,
                                                                numColsPerCone_r, storeIrmMap)
        if (storeDeepIsmMap):
            buff_r, ismMaps = processSweepOfDeepIsm(nusc, sample, scene, r_sweepNames, vPoses01, t01,
                                                    radarNames, buff_r,
                                                    [deep_ismMap, deep_ismMap_rescaled, deepGeo_ismMap], t_r2i,
                                                    pF_rMap, pO_rMap, pD_rMap, pDim, mDim, aDim, numColsPerCone_r,
                                                    y_fake, uMin)
            [deep_ismMap, deep_ismMap_rescaled, deepGeo_ismMap] = ismMaps

        # store mapping progress
        if (storeLidarMap) and (storeProgress):
            saveImg(l_detMap, l_storDir + l_storDir.split("/")[-2] + "_map__{0:}.png".format(t_ref))
        if (storeIlmMap) and (storeProgress):
            saveImg(l_ismMap, ilmMap_storDir + ilmMap_storDir.split("/")[-2] + "__{0:}.png".format(t_ref))
        if (storeRadarMap) and (storeProgress):
            saveImg(r_detMap, r_storDirs[0] + r_storDirs[0].split("/")[-2] + "_map__{0:}.png".format(t_ref))
        if (storeIrmMap) and (storeProgress):
            saveImg(r_ismMap, irmMap_storDir + irmMap_storDir.split("/")[-2] + "__{0:}.png".format(t_ref))
        if (storeDeepIsmMap) and (storeProgress):
            saveImg(deep_ismMap, deepIsm_storDir + deepIsm_storDir.split("/")[-2] + "_map__{0:}.png".format(t_ref))
            saveImg(deep_ismMap_rescaled,
                    deepIsm_storDir + deepIsm_storDir.split("/")[-2] + "_mapScaled__{0:}.png".format(t_ref))
            saveImg(deepGeo_ismMap,
                    deepIsm_storDir + deepIsm_storDir.split("/")[-2] + "_mapFused__{0:}.png".format(t_ref))

    # store map images
    if (storeLidarMap):
        saveImg(l_detMap, l_storDir + l_storDir.split("/")[-2] + "_map.png")
    if (storeIlmMap):
        saveImg(l_ismMap, ilmMap_storDir + ilmMap_storDir.split("/")[-2] + ".png")
        saveImg(l_ismMap, ilmMaps_storDir + "map_{:04}.png".format(iScene))
    if (storeRadarMap):
        saveImg(r_detMap, r_storDirs[0] + r_storDirs[0].split("/")[-2] + "_map.png")
    if (storeIrmMap):
        saveImg(r_ismMap, irmMap_storDir + irmMap_storDir.split("/")[-2] + ".png")
    if (storeDeepIsmMap):
        saveImg(deep_ismMap, deepIsm_storDir + deepIsm_storDir.split("/")[-2] + "_map.png")
        saveImg(deep_ismMap_rescaled, deepIsm_storDir + deepIsm_storDir.split("/")[-2] + "_mapScaled.png")
        saveImg(deepGeo_ismMap, deepIsm_storDir + deepIsm_storDir.split("/")[-2] + "_mapFused.png")

    # observed area map
    if (storeIlmMap):
        # create observed area map
        mappedArea = mapUtils.createObservedAreaMap(mappedArea, t_r2i, vPoses_ref, mDim, pDim, aDim)
        # store observed area map
        saveImg(mappedArea, ilmMap_storDir + "mappedArea.png")

    # store ilm map patches
    if (storeIlmMapPatches):
        mapUtils.saveMapPatches(l_ismMap, t_r2i, vPoses_ref, ts_ref, ilmMapPatch_storDir, mDim, pDim, bboxes)


# get the names of train, val and test scenes
trainNames, valNames, trainIdxs, valIdxs = sceneAttribUtils.getTrainValTestSplitNames()

# all scenes
# sceneIdxs = np.arange(len(nusc.scene))

# train scenes
# sceneIdxs = trainIdxs

# val scenes
sceneIdxs = valIdxs

# specific scenes
# sceneIdxs = [0]

t0 = time.time()
for iScene, sceneIdx in enumerate(sceneIdxs):
    print("Progress: {0:.2f} %".format(iScene / len(sceneIdxs) * 100))
    main(sceneIdx)
# pool = mp.Pool(mp.cpu_count())  
# pool.map(main, sceneIdxs)   
# pool.close()
# pool.join()  
print("Done after {:.2f} min!".format((time.time() - t0) / 60))
