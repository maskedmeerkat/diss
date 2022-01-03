from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud, RadarPointCloud
import numpy as np
import csv
import os
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
plt.close("all")

# VEHICLE_BOARDERS = np.array([0.75, -0.75, 1.9, -1.2])*1.07
# HEIGHT_THRESHOLDS = np.array([0.5, 3.0])
NUM_INTERPOL_PTS = 20
ADD_TO_BOUNDING_BOX_DIAMETER = 0.1
    
#===========================#
def pose2HomogTrafo_2D(p):
    T = np.eye(3)
    T[:2,:2] = rotMat(p[2])
    T[:2, 2] = p[:2]
    return T


#===========================#
def invertHomogTrafo(T):
    T_ = np.eye(3)
    R = T[:2,:2]
    t = T[:2, 2]
    T_[:2,:2] = R.T
    T_[:2, 2] = -np.dot(R.T, t)
    return T_

    
#===========================#
def homogTrafo2Pose_2D(T):  
    p = np.array([T[0,-1], T[1,-1], np.arctan2(T[1,0],T[0,0])])
    return p
    
    
#===========================#
def rotMat(angle):
    R = np.array([[np.cos(angle), -np.sin(angle)],
		          [np.sin(angle),  np.cos(angle)]])
    return R

  
#===========================#
def vizPose(p, scale=1.):
    R = rotMat(p[2])    
    xEndPt = np.dot(R, scale*np.array([[1],[0]])) + p[:2,np.newaxis]
    yEndPt = np.dot(R, scale*np.array([[0],[1]])) + p[:2,np.newaxis]
    
    plt.plot([p[0], xEndPt[0]], [p[1], xEndPt[1]],'r')
    plt.plot([p[0], yEndPt[0]], [p[1], yEndPt[1]],'g')

    
#===========================#
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
    dynamic_mask = np.zeros_like(pc_wcf[0,:]).astype("bool")      
    
    # loop over all annotation boxes and mark moving objects in them in a mask
    for iBox in range(boundingBoxesOfDynObjs_wcf.shape[0]):
        annBox = boundingBoxesOfDynObjs_wcf[iBox]           
        # get the points corresponding to the bounding box
        # get transformation into box coordiantes
        t = annBox.center[:,np.newaxis]
        R = annBox.orientation.rotation_matrix             
     
        # transform the sensor to the box coordinate system
        pc_bcf = np.dot(R.T, pc_wcf - t)
        
        # get the bounding box diameters
        width_box, length_box, height_box = annBox.wlh
        
        # filter out all points within the box         
        mask = np.ones_like(pc_bcf[0,:])
        mask = np.logical_and(mask,pc_bcf[0,:]< length_box/2 + ADD_TO_BOUNDING_BOX_DIAMETER)
        mask = np.logical_and(mask,pc_bcf[0,:]>-length_box/2 - ADD_TO_BOUNDING_BOX_DIAMETER)
        mask = np.logical_and(mask,pc_bcf[1,:]<  width_box/2 + ADD_TO_BOUNDING_BOX_DIAMETER)
        mask = np.logical_and(mask,pc_bcf[1,:]>- width_box/2 - ADD_TO_BOUNDING_BOX_DIAMETER)
#        mask = np.logical_and(mask,pc_bcf[2,:]<height_box/2)
#        mask = np.logical_and(mask,pc_bcf[2,:]>-height_box/2)
                
        # decide wether to put obj points to dynamic or static mask
        dynamic_mask = np.logical_or(dynamic_mask,mask)
            
    return dynamic_mask


#===========================#
def pc2VehicleCenteredBevImg(pc, pDim=512, mDim=40.):
    # trafo vcf -> icf     
    pc.points[1,:] = -pc.points[1,:]
    pc.points = pc.points[[1,0,2],:]
    pc.points = pc.points + np.array([[mDim/2,mDim/2,0]]).T        
    
    # trafo icf -> pcf
    pc.points = (pc.points * pDim/mDim).astype(np.int)
    
    # filter out all pts outside image dimension
    mask = (pc.points[0,:] >= 0) * (pc.points[1,:] >= 0) * (pc.points[0,:] < pDim) * (pc.points[1,:] < pDim)
    pc.points = pc.points[:,mask]
    
    # create brid's eye view lidar image
    bevImg = np.zeros((pDim,pDim))
    
    # mark detections in the image
    bevImg[pc.points[0,:],pc.points[1,:]] = 1
    
    return bevImg


#===========================#
def vehicleCenteredBevImg2Pc(bevImg, pDim=512, mDim=40.):     
    # trafo xy coordinates of marked cells back to meters -> discretized pointcloud
    pcDisc = np.transpose(np.transpose((bevImg>0).nonzero())).astype(np.float)
    pcDisc = np.append(pcDisc, np.zeros((1,pcDisc.shape[1])),axis=0)
    pcDisc *= mDim/pDim
    
    # trafo icf -> vcf
    pcDisc = pcDisc - np.array([[mDim/2,mDim/2,0]]).T  
    pcDisc = pcDisc[[1,0,2],:]
    pcDisc[1,:] = -pcDisc[1,:]
    
    return pcDisc
    
    
#===========================#
def getPcWithMotionStatusInRcf(nusc, sample, vPose, boundingBoxesOfDynObjs_rcf, dataPath_and_sensorName, sweepFlag=False, loadStatus=0, heightThresholds = [0.5,3.0]):
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
    # if (sensorName == 'LIDAR_TOP'):
    #     mask = np.zeros_like(pc.points[0,:])
    #     mask = np.logical_or(mask, pc.points[0,:]>VEHICLE_BOARDERS[0])
    #     mask = np.logical_or(mask, pc.points[0,:]<VEHICLE_BOARDERS[1])
    #     mask = np.logical_or(mask, pc.points[1,:]>VEHICLE_BOARDERS[2])
    #     mask = np.logical_or(mask, pc.points[1,:]<VEHICLE_BOARDERS[3])
    #     pc.points = pc.points[:3,mask] 
    
    # trafo scf -> vcf
    sample_data = nusc.get('sample_data', sample['data'][sensorName])
    cal_sensor = nusc.get('calibrated_sensor', sample_data['calibrated_sensor_token'])
    pc.rotate(Quaternion(cal_sensor['rotation']).rotation_matrix)
    pc.translate(np.array(cal_sensor['translation']))    
    
    # remove ground plane points through thresholding
    if (sensorName == 'LIDAR_TOP'):
        # load lidar semantic labels
        lidarseg_labels_filename = osp.join(nusc.dataroot,
                                    nusc.get('lidarseg', sample['data']['LIDAR_TOP'])['filename'])
        points_label = np.fromfile(lidarseg_labels_filename, dtype=np.uint8)
        
        # remove points belonging to the following categories
        mask = (points_label != 31) # ego vehicle
        if (loadStatus==0):
            mask = np.logical_and( mask, np.abs(pc.points[2,:]) > heightThresholds[0])
            mask = np.logical_and( mask, pc.points[2,:] < heightThresholds[1])
        if (loadStatus==1):
            mask = np.logical_and(mask, points_label != 24) # flat driveable surface
        if (loadStatus==2):
            mask = np.logical_and(mask, points_label != 24) # flat driveable surface
            mask = np.logical_and(mask, points_label != 25) # flat other
        if (loadStatus==3):
            mask = np.logical_and(mask, points_label != 24) # flat driveable surface
            mask = np.logical_and(mask, points_label != 25) # flat other
            mask = np.logical_and(mask, points_label != 26) # flat sidewalk
        if (loadStatus==4):
            mask = np.logical_and(mask, points_label != 24) # flat driveable surface
            mask = np.logical_and(mask, points_label != 25) # flat other
            mask = np.logical_and(mask, points_label != 26) # flat sidewalk
            mask = np.logical_and(mask, points_label != 27) # flat terrain
        if (loadStatus==5):
            mask = np.logical_and(mask, points_label != 25) # flat other
        if (loadStatus==6):
            mask = np.logical_and(mask, points_label != 25) # flat other
            mask = np.logical_and(mask, points_label != 26) # flat sidewalk
        if (loadStatus==7):
            mask = np.logical_and(mask, points_label != 25) # flat other
            mask = np.logical_and(mask, points_label != 26) # flat sidewalk
            mask = np.logical_and(mask, points_label != 27) # flat terrain
        if (loadStatus==8):
            mask = np.logical_and(mask, points_label != 26) # flat sidewalk
        if (loadStatus==9):
            mask = np.logical_and(mask, points_label != 26) # flat sidewalk
            mask = np.logical_and(mask, points_label != 27) # flat terrain
        if (loadStatus==10):
            mask = np.logical_and(mask, points_label != 27) # flat sidewalk
        
        pc.points = pc.points[:,mask] 
        
        # subsample the point cloud according to the bev image resolution
        pDim = 512
        mDim = 40.
        bevImg = pc2VehicleCenteredBevImg(pc, pDim=pDim, mDim=mDim)
        pcDisc = vehicleCenteredBevImg2Pc(bevImg, pDim=pDim, mDim=mDim)
        
    else:
        pcDisc = pc.points

    # trafo vcf -> rcf
    R = np.eye(3)
    R[:2,:2] = rotMat(vPose[2])
    t = np.zeros((3,1))
    t[:2,0] = vPose[:2]
    pcDisc[:3,:] = np.dot(R,pcDisc[:3,:]) + t
    
    # identify moving objects
    if (sensorName == 'LIDAR_TOP'):
        # mark all lidar points inside moving obj bounding boxes as dynamic
        isDynamic = dyn_static_obj_masks(pcDisc, boundingBoxesOfDynObjs_rcf)
        isStationary = (isDynamic==0).astype(np.int)
    else:
        isStationary = ((pc.points[3,:] != 0) * (pc.points[3,:] != 2) * (pc.points[3,:] != 6)).astype(np.int) 
#        isStationary = np.logical_or(isStationary, pc.points[3,:])
#        isStationary = ((pc.points[3,:] == 7)).astype(np.int) 
#        isStationary = (np.linalg.norm(pc.points[8,:] + pc.points[9,:]) < 20.).astype(np.int)
    
    return pcDisc, isStationary   


#===========================#
def getSampleDataPath(sample, sensorName):
    return osp.join(nusc.dataroot, nusc.get('sample_data', sample['data'][sensorName])['filename']), sensorName


#===========================#
def getSweepDataPath(sweepName, sensorName):    
    return nusc.dataroot+'sweeps'+'/'+sensorName+'/'+sweepName, sensorName

#===========================#
def getPoseAndTimeFromSample(nusc, scene, sample, sensorName):
    # get sensor's meta data
    sample_data = nusc.get('sample_data', sample['data'][sensorName])
    
    # get timestamp from meta data
    timestamp = sample_data['timestamp']
    
    # trafo scf -> vcf
    cal_sensor = nusc.get('calibrated_sensor', sample_data['calibrated_sensor_token'])
    R = Quaternion(cal_sensor['rotation']).rotation_matrix
    t = np.array(cal_sensor['translation'])[...,np.newaxis]
    rotAngle = np.arctan2(R[1,0],R[0,0])
    sensorPose = np.array([t[0,0], t[1,0], rotAngle])
    
    # trafo vcf -> wcf
    ego_pose = nusc.get('ego_pose', sample_data['ego_pose_token'])
    R = Quaternion(ego_pose['rotation']).rotation_matrix
    t = np.array(ego_pose['translation'])[...,np.newaxis]    
    T_v2w = np.eye(4)
    T_v2w[:3,:3] = R
    T_v2w[:3, [3]] = t
    
    # trafo from wcf -> rcf
    sample0  = nusc.get('sample', scene['first_sample_token'])
    sample0_data = nusc.get('sample_data', sample0['data']['LIDAR_TOP'])
    ego_pose = nusc.get('ego_pose', sample0_data['ego_pose_token'])
    R = Quaternion(ego_pose['rotation']).rotation_matrix
    t = np.array(ego_pose['translation'])[...,np.newaxis]
    T_w2r = np.eye(4)
    T_w2r[:3,:3] = R.T
    T_w2r[:3, [3]] = -np.dot(R.T, t)
    
    # trafo from vcf -> rcf
    T_v2r = np.dot(T_w2r, T_v2w)
    rotAngle = np.arctan2(T_v2r[1,0],T_v2r[0,0])
    vehiclePose = np.array([T_v2r[0,3], T_v2r[1,3], rotAngle])
    
    return sensorPose, vehiclePose, timestamp



#===========================#
def findNextIndex(currIdx, refSweepName, sweepNames):
    # extract the reference timestamp from the reference sweep name
    refTimestamp = int(refSweepName[-20:-4]) 
    
    isSearching = True
    while (isSearching):
        # in case the last entry is reached, break the loop
        if ( currIdx == (sweepNames.shape[0]-1) ):
            isSearching = False
            break
            
        # extract the current and next timestamps from the sweep names
        currTimestamp = int(sweepNames[currIdx][-20:-4])
        nextTimestamp = int(sweepNames[currIdx+1][-20:-4])
            
        # compute distance between current and next timestamp towards reference timestamp
        diff  = np.abs(refTimestamp - currTimestamp)
        diff_ = np.abs(refTimestamp - nextTimestamp)
        
        # test whether the next timestamp is closer to the reference timestamp or not        
        if (diff >= diff_):
            currIdx += 1
        else:
            isSearching = False
            
    return currIdx
        

#===========================#
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
        sample_ = nusc.get('sample', scene['last_sample_token' ])
        
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
                isInsideTimeSpan[iSweep]  = int(sweepNames[iSweep][-24:-8]) > sampleT0
                isInsideTimeSpan[iSweep] *= int(sweepNames[iSweep][-24:-8]) < sampleT_
            else:
                isInsideTimeSpan[iSweep]  = int(sweepNames[iSweep][-20:-4]) > sampleT0
                isInsideTimeSpan[iSweep] *= int(sweepNames[iSweep][-20:-4]) < sampleT_
        
        allSweepNames.append(sweepNames[isInsideTimeSpan])
    
    return allSweepNames


#===========================#
def findAllSweepsBetweenTimesteps(t0, t1, sweepNames, sensorName):        
    # discard all sweeps outside the scenes timespan
    isInsideTimeSpan = np.zeros_like(sweepNames, dtype=bool)
    for iSweep in range(sweepNames.shape[0]):
        if (sensorName == 'LIDAR_TOP'):
            isInsideTimeSpan[iSweep]  = int(sweepNames[iSweep][-24:-8]) > t0
            isInsideTimeSpan[iSweep] *= int(sweepNames[iSweep][-24:-8]) < t1
        else:
            isInsideTimeSpan[iSweep]  = int(sweepNames[iSweep][-20:-4]) > t0
            isInsideTimeSpan[iSweep] *= int(sweepNames[iSweep][-20:-4]) < t1
    
    sweepNames = sweepNames[isInsideTimeSpan]
    sweepNames.sort()
    
    return sweepNames
    

#===========================#
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
    row.insert(0,name + '_vehicle_pose')
    csvWriter.writerow( row )
    
    # trafo scf -> vcf
    row = sensorPose.tolist()
    row.insert(0,name + '_sensor_pose')
    csvWriter.writerow( row )
    
    # x-pts in wcf
    row = pc[0,:].tolist()
    row.insert(0,name + '_pts_wcf_x')
    csvWriter.writerow( row )
    
    # y-pts in wcf
    row = pc[1,:].tolist()
    row.insert(0,name + '_pts_wcf_y')
    csvWriter.writerow( row )
    
    # is stationary flag 
    row = isStationary.tolist()
    row.insert(0,name + '_is_stationary')
    csvWriter.writerow( row ) 
    
    
#===========================#    
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
        allBoundingBoxes_rcf = np.append(dynBoundingBoxes_rcf,statBoundingBoxes_rcf)
        
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
            rotation_rad[iBox] = np.arctan2(R[1,0],R[0,0])
            diameter_x[iBox] = allBoundingBoxes_rcf[iBox].wlh[1]
            diameter_y[iBox] = allBoundingBoxes_rcf[iBox].wlh[0]
            
        # trafo vcf -> wcf    
        row = vehiclePose.tolist()
        row.insert(0,name + '_vehicle_pose')
        csvWriter.writerow( row )
                
        # bounding box center x
        row = center_x.tolist()
        row.insert(0,name + '_center_x')
        csvWriter.writerow( row )
        
        # bounding box center y
        row = center_y.tolist()
        row.insert(0,name + '_center_y')
        csvWriter.writerow( row )
        
        # bounding box center x
        row = rotation_rad.tolist()
        row.insert(0,name + '_rotation_rad')
        csvWriter.writerow( row )
        
        # bounding box center x
        row = diameter_x.tolist()
        row.insert(0,name + '_diameter_x')
        csvWriter.writerow( row )
        
        # bounding box center x
        row = diameter_y.tolist()
        row.insert(0,name + '_diameter_y')
        csvWriter.writerow( row )
        
        # bounding box center x
        row = is_stationary.tolist()
        row.insert(0,name + '_is_stationary')
        csvWriter.writerow( row )
    
    
#===========================#
def writeCamToCsv(csvWriter, vehiclePose, timestamp_us):
    name = 'c__'    
    # store the timestamp in us
    csvWriter.writerow([name + '_timestamp_us', timestamp_us])    
    # trafo vcf -> wcf    
    row = vehiclePose.tolist()
    row.insert(0,name + '_vehicle_pose')
    csvWriter.writerow( row )
            
            
#===========================#
def findRefCamName(c_sweepNames):   
    numSweepsForScene = [len(c_sweepName) for c_sweepName in c_sweepNames]
    refIdx = np.argmin(numSweepsForScene)        
    return camNames[refIdx], c_sweepNames[refIdx]


#===========================#
def vizPc(pc, isStat, vPose, maxDist=20., statColor='b', dynColor='c', zorder=1):
    # remove all points further away from current vehicle pose than maxDist
    l_range = np.sqrt((pc[0,:]-vPose[0])**2 + (pc[1,:]-vPose[1])**2)
    
    # split points into static and dynamic ones
    pltIdx_stat = np.logical_and(isStat.astype(np.bool), l_range < maxDist)
    pltIdx_dyna = np.logical_and(np.logical_not(isStat.astype(np.bool)), l_range < maxDist)
    
    plt.plot(pc[0,pltIdx_stat], pc[1,pltIdx_stat],statColor+'.',ms=1.5, zorder=zorder)
    plt.plot(pc[0,pltIdx_dyna], pc[1,pltIdx_dyna],dynColor +'.',ms=4.5, zorder=zorder)


#===========================#
def loadPcWithMotionStatusInRcf(nusc, scene, sample, sensorName, minVisThres, loadStatus=0, heightThresholds=[0.5,3.0]):
    # get sensor and vehicle pose and timestamp info
    sPose, vPose, t = getPoseAndTimeFromSample(nusc, scene, sample, sensorName)
    
    # get bounding boxes in wcf of dynamic objects for current and previous sample
    if (sensorName == 'LIDAR_TOP'):        
        dynBoundingBoxes_rcf, statBoundingBoxes_rcf = interUtils.getDynObjBoundingBoxes_rcf(nusc, scene, sample, sensorName, minVisThres)
    else:
        dynBoundingBoxes_rcf = []
        
    # get the point clouds with motion status in reference coordinate system (first lidar pose in scene)
    pc, isStat = getPcWithMotionStatusInRcf(nusc, sample, vPose, dynBoundingBoxes_rcf,
                                            getSampleDataPath(sample, sensorName),loadStatus=loadStatus,heightThresholds=heightThresholds)
    pc[2,:] = isStat
    pcWithPose = {"pc":pc, "vPose":vPose, "sPose":sPose}
    return pcWithPose


#===========================#
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
        boundingBoxOfDynObjs_01_wcf = interUtils.getInterpolBoundingBoxes_wcf(nusc, scene, sample, numInterPts=NUM_INTERPOL_PTS)
    
    # find sensor with least sweeps
    iSensMin = 0
    for iSens in range(len(sensorNames)):
        if (len(sweepNames_[iSens]) <= len(sweepNames_[iSensMin])):
            iSensMin = iSens
            
    
    # process each sweep
    iSweeps = [0]*len(sensorNames)
    for iSweep in range(len(sweepNames_[iSensMin])):
        # get current ref timestamp
        if (sensorNames[0] == 'LIDAR_TOP'):
            t_ref = int(sweepNames_[iSensMin][iSweep][-24:-8])
        else:
            t_ref = int(sweepNames_[iSensMin][iSweep][-20:-4])
    
        # fill the buffers for each sensor up to the current ref timestamp
        for iSens, sensorName in enumerate(sensorNames): 
            while(True):
                # find interpolated pose with closest timestamp
                if (sensorName == 'LIDAR_TOP'):
                    break
                    t = int(sweepNames_[iSens][iSweeps[iSens]][-24:-8])
                else:
                    t = int(sweepNames_[iSens][iSweeps[iSens]][-20:-4])
                if (t > t_ref):
                    break
                idx = np.argmin(abs(t01 - t))
                vPose = vPoses01[idx,:]
                
                boundingBoxesOfDynObjs_wcf = []
                if (sensorName == 'LIDAR_TOP'):
                    boundingBoxesOfDynObjs_wcf = boundingBoxOfDynObjs_01_wcf.flatten()                        
                
                # get the point clouds with motion status in reference coordinate system (first lidar pose in scene) 
                pc, isStat = getPcWithMotionStatusInRcf(nusc, sample, vPose, boundingBoxesOfDynObjs_wcf, 
                                                        getSweepDataPath(sweepNames_[iSens][iSweeps[iSens]], sensorName), sweepFlag=True)        
                pc[2,:] = isStat
                
                # add bev detection image to detection map
                detMap = mapUtils.markInGlobalImg(pc, detMap, t_r2i, mDim, pDim)                
                
                # append new pointcloud with pose to the data buffer                
                pcWithPose = {"pc":pc, "vPose":vPose, "sPose":sPoses[iSens]}
                buffers[iSens].append(pcWithPose)
                
                # get the next sweep
                if (iSweeps[iSens] < (len(sweepNames_[iSens])-1)):
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
                              [1. , 0.]])
            imgCenterPt = np.matmul(R_r2i, vPose[:2,np.newaxis] + t_r2i)[:,0]
            # trafo icf -> pcf
            imgCenterPt = (imgCenterPt * pDim/mDim).astype(int)
            
            # map area which will be updated
            xLim = [imgCenterPt[0]-pDim//2,imgCenterPt[0]+pDim//2]
            yLim = [imgCenterPt[1]-pDim//2,imgCenterPt[1]+pDim//2]
            xLim_ = [0,pDim]
            yLim_ = [0,pDim]
            
            # check map boundaries
            if (xLim[0] < 0):
                dx = 0 - xLim[0]
                xLim[0]  += dx
                xLim_[0] += dx
            if (yLim[0] < 0):
                dy = 0 - yLim[0]
                yLim[0]  += dy
                yLim_[0] += dy
            if (xLim[1] > ismMap.shape[0]):
                dx = ismMap.shape[0] - xLim[1]
                xLim[1]  += dx
                xLim_[1] += dx
            if (yLim[1] > ismMap.shape[1]):
                dy = ismMap.shape[1] - yLim[1]
                yLim[1]  += dy
                yLim_[1] += dy
            
            # fuse new ism into global map
            ismMap[xLim[0]:xLim[1],yLim[0]:yLim[1],:] = \
                mapUtils.fuseImgs(ismImg[xLim_[0]:xLim_[1], yLim_[0]:yLim_[1], :],
                                  ismMap[xLim[0]:xLim[1], yLim[0]:yLim[1], :])
        
            if (sensorName == 'LIDAR_TOP'):
                return buffers, detMap, ismMap
        
    return buffers, detMap, ismMap
            

#===========================# 
def addImgTo360Img_inRcf(img, img_360, homographyMat, vPose, croppPortion, imgMask, pDim): 
    # pcf -> vcf
    T_p2v = np.eye(3)
    T_p2v[1,1] = -1. 
    T_p2v[0, 2] = -pDim//2
    T_p2v[1, 2] = pDim//2
    
    # trafo vcf -> rcf in homogeneous coordinates
    T_v2r = np.eye(3)
    T_v2r[:2,:2] = rotMat(vPose[2])
    T_v2r[:2, 2] = vPose[:2]
    
    # only translation back to vcf
    T_r2v_ = np.eye(3)
    T_r2v_[:2, 2] = -vPose[:2]
    
    T_all = np.dot(np.linalg.inv(T_p2v), np.dot(T_r2v_, np.dot(T_v2r, T_p2v)))
    
    # crop camera image to remove infinite distance points
    img = img[int(img.shape[0] * croppPortion):,...]
    img = img.copy()
    if (len(img.shape)==3):
        img[imgMask,:] = np.zeros(3)
    else:
        img[imgMask] = 0
    
    # warp from camera to bird's eye view perspective (in rcf)
    img = cv2.warpPerspective(img, np.dot(T_all, homographyMat), (pDim, pDim), flags=cv2.INTER_NEAREST) 
    # img = cameraUtils.warpPerspective(img, np.dot(T_all, homographyMat), pDim)
    if (len(img.shape) <= 2):
        img = img[...,np.newaxis]
    
    # replace all non empty pixels in the 360 bird's eye view image
    img_360[np.sum(img, axis=2)>0,:] = img[np.sum(img, axis=2)>0,:]  

    return img_360         


#===========================# 
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
    
    img_rgb = np.zeros((img.shape[0],img.shape[1],3),dtype=np.uint8)
    
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            img_rgb[x,y,:] = colormap[img[x,y,0]]
    return img_rgb
    

#===========================#          
def surroundCam2Bev(nusc, scene, sample, imgStorDir,
                             vPose_ref, t_ref, camNames, homographyMats, croppPortion, imgMasks, pDim, semSegFlag=False):        
    # project all cameras into one bird's eye view image
    img_360 = np.zeros((pDim,pDim,3), dtype=np.uint8)
    for iCam in range(len(camNames)):
        # load camera image
        cam_data = nusc.get('sample_data', sample['data'][camNames[iCam]] )  
        camFileName = cam_data['filename']
        if (semSegFlag):
            camFileName = camFileName[:8] + 'SEMSEG_' + camFileName[8:8+len(camNames[iCam])+32] + 'SEMSEG_' + camFileName[8+len(camNames[iCam])+32:]
        img = np.asarray(Image.open(osp.join(nusc.dataroot, camFileName)))
    
        # add camera image into 360 bird's eye view image
        img_360 = addImgTo360Img_inRcf(img, img_360, homographyMats[iCam], vPose_ref, croppPortion, imgMasks[iCam], pDim)
        
    if (semSegFlag):
        # apply cityscapes colormap 
        img_360 = applyCityScapesColorMap(img_360)
    
    # store 360 bird's eye view image
    img_360 = Image.fromarray(img_360)
    if (semSegFlag):
        img_360.save(imgStorDir+imgStorDir.split("/")[-2]+"__{:}.png".format(t_ref))       
    else:
        img_360.save(imgStorDir+imgStorDir.split("/")[-2]+"__{:}.png".format(t_ref)) 


#===========================#          
def processSweepOf360Camera(nusc, sweepNames_ref, sweepNames, vPoses01, t01, csvWriter, imgStorDir, store360ImgFlag, storeCsvFlag,
                             refCamName, camNames, homographyMats, croppPortion, pDim, semSegFlag=False):
    # find all sweep names between current and next sample
    sweepNames_ref_ = findAllSweepsBetweenTimesteps(t01[0], t01[-1], np.asarray(sweepNames_ref), refCamName)
    
    sweepNames_ = []
    for iCam in range(len(sweepNames)):
        sweepNames_.append( findAllSweepsBetweenTimesteps(t01[0], t01[-1], np.asarray(sweepNames[iCam]), refCamName) )
        
    # initialize sweep indices all cameras 
    sweepIdx = np.zeros(len(sweepNames), dtype=np.int)
    
    # process each sweep
    for iSweep in range(len(sweepNames_ref_)):
        # find interpolated pose with closest timestamp
        t_ref = int(sweepNames_ref_[iSweep][-20:-4])
        idx = np.argmin(abs(t01 - t_ref))
        vPose_ref = vPoses01[idx,:]
    
        # store current sweep
        if (storeCsvFlag):
            writeCamToCsv(csvWriter, vPoses01[idx,:], t_ref)
    
        # project all cameras into one bird's eye view image
        img_360 = np.zeros((pDim,pDim,3), dtype=np.uint8)
        for iCam in range(len(camNames)):
            # find closest camera image
            sweepIdx[iCam] = findNextIndex(sweepIdx[iCam], sweepNames_ref_[iSweep], sweepNames_[iCam])
            
            # load camera image
            img = np.asarray(Image.open(nusc.dataroot + 'sweeps/' + camNames[iCam] + '/' + sweepNames_[iCam][sweepIdx[iCam]]))
        
            # add camera image into 360 bird's eye view image
            img_360 = addImgTo360Img_inRcf(img, img_360, homographyMats[iCam], vPose_ref, croppPortion, pDim)
    
        # store 360 bird's eye view image
        if (store360ImgFlag):
            img_360 = Image.fromarray(img_360)
            if (semSegFlag):
                img_360.save(imgStorDir+"s__{:}.png".format(t_ref))       
            else:
                img_360.save(imgStorDir+"c__{:}.png".format(t_ref))       
   
    
#===========================#
def processSampleOfMonodepth(nusc, scene, sample, imgStorDir,
                             refCamName, camNames, pDim, mDim):
    # get sensor and vehicle pose and timestamp info
    _, vPose_ref, t_ref = getPoseAndTimeFromSample(nusc, scene, sample, refCamName)
    
    # construct bev image based on monodepth of all cameras
    bevImg = np.zeros((pDim,pDim),dtype=np.uint8)
    for camName in camNames:
        # get the depth file
        cam_data = nusc.get('sample_data', sample['data'][camName] )  
        depthFileName = cam_data['filename'][:-4] + ".npy"
        depthFileName = depthFileName[:len("samples/")] + "DEPTH_" + depthFileName[len("samples/"):]
        
        # load depth
        depthImg = np.load(nusc.dataroot + depthFileName)[:,:,0]
        img = np.asarray(Image.open(osp.join(nusc.dataroot, cam_data['filename'])))
        
        # get point cloud without ground plane
        pc = cameraUtils.monoDepth2PcInVcf(nusc, depthImg, sample['data'][camName])
    
        # trafo vcf -> rcf in homogeneous coordinates
        pc[:2,:] = np.dot(rotMat(vPose_ref[2]), pc[:2,:])
    
        # trafo point cloud to bev image
        bevImg_ = cameraUtils.pc2BevImg(pc, pDim, mDim)
        
        # put new bev image entries into overall bev image
        bevImg[bevImg_>0] = bevImg_[bevImg_>0]
        
    
    bevImg = Image.fromarray(bevImg)
    bevImg.save(imgStorDir+imgStorDir.split("/")[-2]+"__{:}.png".format(t_ref))
    
    
#===========================#     
def pc2Bev(buffers, pDim, mDim, vPose_ref, t_ref, imgStorDir):
    bevImg = np.zeros((pDim,pDim),dtype = np.uint8)
    for buffer in buffers:
        for pcWithPose in buffer:
            # trafo rcf -> vcf
            pc = pcWithPose["pc"][:2,:] - vPose_ref[:2,np.newaxis]
            # compute bev image
            bevImg_ = mapUtils.pc2VehicleCenteredBevImg(pc[:2,:],pcWithPose["pc"][2,:],pDim,mDim)
            # mark latest bevImg in overall bev img
            newEntryMask = (bevImg_>0)
            bevImg[newEntryMask] = bevImg_[newEntryMask]
    # store bev image
    bevImg = Image.fromarray(bevImg)
    bevImg.save(imgStorDir+imgStorDir.split("/")[-2]+"__{:}.png".format(t_ref))    


#===========================#  
def createImgStorDir(dirName):
    if not os.path.exists(dirName):
        try:
            os.makedirs(dirName)
        except:
            pass
    return dirName


#===========================#  
def saveImg(img, dirPath):
    img_ = Image.fromarray((img*255).astype(np.uint8))
    img_.save(dirPath)
    

#===========================#  
def computeIoU(estImg, gtImg, mappedArea):
    interUnionPx = np.zeros((3,2))
    
    # classification results of labels and predictions
    classes_gt = np.argmax(gtImg, axis=2)
    classes_est = np.argmax(estImg, axis=2)
    
    # get masks for free, occupied and unknown ground truth pixels
    frMask_gt = (classes_gt == 0) * (mappedArea == 1)
    ocMask_gt = (classes_gt == 1) * (mappedArea == 1)
    unMask_gt = (classes_gt == 2) * (mappedArea == 1)
    
    # get masks for free, occupied and unknown predicted pixels
    frMask_est = (classes_est == 0) * (mappedArea == 1)
    ocMask_est = (classes_est == 1) * (mappedArea == 1)
    unMask_est = (classes_est == 2) * (mappedArea == 1)
    
    # update per class IoU
    interUnionPx[0,0] += np.sum(np.logical_and(frMask_est, frMask_gt))
    interUnionPx[1,0] += np.sum(np.logical_and(ocMask_est, ocMask_gt))
    interUnionPx[2,0] += np.sum(np.logical_and(unMask_est, unMask_gt))
    interUnionPx[0,1] += np.sum(np.logical_or(frMask_est, frMask_gt))
    interUnionPx[1,1] += np.sum(np.logical_or(ocMask_est, ocMask_gt))
    interUnionPx[2,1] += np.sum(np.logical_or(unMask_est, unMask_gt))
    
    return interUnionPx

    
    
#========================== MAIN =============================================#
# minimum distance to move within a scene to use it for mapping
minDistTravelledThres = 20. # [m]

# disregard all bounding boxes with a visibility lower than certain threshold
# visibility is defined as the fraction of pixels of a particular annotation that are visible over the 6 camera feeds, grouped into 4 bins.
minVisThres = 3

# load data set
DATA_DIR = '../_DATASETS_/NuScenes/'
STORAGE_DIR = '../_DATASETS_/occMapDataset/'
os.makedirs(STORAGE_DIR,exist_ok=True)
# nuscVersion = "v1.0-mini"
nuscVersion = "v1.0-trainval"
if not('nusc' in locals()):
    nusc = NuScenes(version=nuscVersion, dataroot=DATA_DIR, verbose=True)  

# define all camera and radar names
camNames = ['CAM_FRONT','CAM_FRONT_RIGHT','CAM_FRONT_LEFT',
            'CAM_BACK','CAM_BACK_RIGHT','CAM_BACK_LEFT']
radarNames = ['RADAR_FRONT','RADAR_FRONT_RIGHT','RADAR_FRONT_LEFT',
              'RADAR_BACK_RIGHT','RADAR_BACK_LEFT']
lidarNames = ['LIDAR_TOP']

# image dimension in bird's eye view
pDim = 128 # [px]
mDim = 40. # [m]
aRes_deg = 0.4
aDim = int(360/aRes_deg) # [deg]

# lidar params
maxBuffSize_l = 1
pF_lMap = [0.5]
pO_lMap = 0.9
pD_lMap = 0.5
pF_ilm = [0.9]
pO_ilm = 0.9
pD_ilm = 0.5
angleResCone_l = np.array([3]) # [deg]
numColsPerCone_l = (angleResCone_l/aRes_deg/2).astype(int)*2

# radar params
buffSizes_r = [20] # from smallest to biggest number, always!!!
maxBuffSize_r = int(np.max(buffSizes_r))
pF_rMap = [0.1,0.1]
pO_rMap = 0.8
pD_rMap = 0.3
pF_irm = [0.9,0.9]
pO_irm = 0.9
pD_irm = 0.5
angleResCone_r = np.array([5,30.]) # [deg]
numColsPerCone_r = (angleResCone_r/aRes_deg/2).astype(int)*2


heightThresholds = np.array([0.025,0.05,0.075,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0])[np.newaxis,:]
heightThresholds = np.append(heightThresholds,np.ones_like(heightThresholds)*3.0,axis=0)


#================#
# Process Scenes  
#================#
# for iScene in tqdm(sceneIdxs):
def main(iScene):         
    # lidar and radar buffers
    buff_l_heightThres = [deque(maxlen=maxBuffSize_l) for _ in range(heightThresholds.shape[1])]
    buff_l_noStreet = [deque(maxlen=maxBuffSize_l)]
    buff_l_noStreetOther = [deque(maxlen=maxBuffSize_l)]
    buff_l_noStreetOtherSidewalk = [deque(maxlen=maxBuffSize_l)]
    buff_l_noStreetOtherSidewalkTerrain = [deque(maxlen=maxBuffSize_l)]
    buff_l_noOther = [deque(maxlen=maxBuffSize_l)]
    buff_l_noOtherSidewalk = [deque(maxlen=maxBuffSize_l)]
    buff_l_noOtherSidewalkTerrain = [deque(maxlen=maxBuffSize_l)]
    buff_l_noSidewalk = [deque(maxlen=maxBuffSize_l)]
    buff_l_noSidewalkTerrain = [deque(maxlen=maxBuffSize_l)]
    buff_l_noTerrain = [deque(maxlen=maxBuffSize_l)]
    buff_r = [deque(maxlen=maxBuffSize_r) for _ in range(len(radarNames))]
    
    # get current scene
    scene = nusc.scene[iScene]
    print(iScene)    
    
    # only process scene, iff vehicle has travelled a certain distance
    travelledDist = sceneAttribUtils.computeTravelledDistanceForScene(nusc, scene)    
    if (travelledDist < minDistTravelledThres):
        return 1
    
    # only use day scenes
    if (("night" in scene["description"].lower()) or 
        ("difficult lighting" in scene["description"].lower())):
        return 1
    
    # check whether the scene is in train, val or test
    setName = "gtVerification/"
    
    # create folders to store images
    sceneStorDir = createImgStorDir(STORAGE_DIR + setName + "scene{:04}".format(iScene) + '/')
    
    # get all sweep names inside the scene's timespan
    c_sweepNames = findAllSweepsForTheScene(scene, camNames)
    r_sweepNames = findAllSweepsForTheScene(scene, radarNames)
    l_sweepNames = findAllSweepsForTheScene(scene, lidarNames)
            
    # create map images
    detMap, ismMap, t_r2i = mapUtils.initGlobalImg(nusc, scene, mDim, pDim)
    # lMap_heightThres = np.zeros((heightThresholds.shape[1],)+ismMap.shape)
    # lMap_heightThres[...,2] = 1.
    # lMap_noStreet  = ismMap.copy()
    # lMap_noStreetOther  = ismMap.copy()
    # lMap_noStreetOtherSidewalk  = ismMap.copy()
    # lMap_noStreetOtherSidewalkTerrain  = ismMap.copy()
    lMap_noOther  = ismMap.copy()
    lMap_noOtherSidewalk  = ismMap.copy()
    lMap_noOtherSidewalkTerrain  = ismMap.copy()
    lMap_noSidewalk  = ismMap.copy()
    lMap_noSidewalkTerrain  = ismMap.copy()
    lMap_noTerrain  = ismMap.copy()
    # rMap = ismMap.copy()
    # mappedArea = detMap.copy()
    
    # all ref vehicle positions
    vPoses_ref = []
    ts_ref = []
    bboxes = []
    
    # choose reference camera for current scene as the one with the least sweeps
    refCamName, sweepNames_ref = findRefCamName(c_sweepNames)
    
    #================#
    # Process Samples  
    #================#
    sample = nusc.get('sample', scene['first_sample_token'])
    for iSample in tqdm(range(scene['nbr_samples']-1)):
    # for iSample in range(2):
        # get sensor and vehicle pose and timestamp info
        _, vPose_ref, t_ref = getPoseAndTimeFromSample(nusc, scene, sample, refCamName)
        if (iSample > 0):   
            # store the vehicle position in image map coordinates
            vPoses_ref.append(vPose_ref)
            ts_ref.append(t_ref)
                                
            # process lidar
            # loadStatus in [0,4]
            # for i in range(heightThresholds.shape[1]):
            #     buff_l_heightThres[i].append(loadPcWithMotionStatusInRcf(nusc, scene, sample, 'LIDAR_TOP', minVisThres, loadStatus=0,heightThresholds=heightThresholds[:,i]))
            # buff_l_noStreet[0].append(loadPcWithMotionStatusInRcf(nusc, scene, sample, 'LIDAR_TOP', minVisThres,loadStatus=1))
            # buff_l_noStreetOther[0].append(loadPcWithMotionStatusInRcf(nusc, scene, sample, 'LIDAR_TOP', minVisThres,loadStatus=2))
            # buff_l_noStreetOtherSidewalk[0].append(loadPcWithMotionStatusInRcf(nusc, scene, sample, 'LIDAR_TOP', minVisThres,loadStatus=3))
            # buff_l_noStreetOtherSidewalkTerrain[0].append(loadPcWithMotionStatusInRcf(nusc, scene, sample, 'LIDAR_TOP', minVisThres,loadStatus=4))
            buff_l_noOther[0].append(loadPcWithMotionStatusInRcf(nusc, scene, sample, 'LIDAR_TOP', minVisThres,loadStatus=5))
            buff_l_noOtherSidewalk[0].append(loadPcWithMotionStatusInRcf(nusc, scene, sample, 'LIDAR_TOP', minVisThres,loadStatus=6))
            buff_l_noOtherSidewalkTerrain[0].append(loadPcWithMotionStatusInRcf(nusc, scene, sample, 'LIDAR_TOP', minVisThres,loadStatus=7))
            buff_l_noSidewalk[0].append(loadPcWithMotionStatusInRcf(nusc, scene, sample, 'LIDAR_TOP', minVisThres,loadStatus=8))
            buff_l_noSidewalkTerrain[0].append(loadPcWithMotionStatusInRcf(nusc, scene, sample, 'LIDAR_TOP', minVisThres,loadStatus=9))
            buff_l_noTerrain[0].append(loadPcWithMotionStatusInRcf(nusc, scene, sample, 'LIDAR_TOP', minVisThres,loadStatus=10))
            
            # process 360 radars
            # for iRadar, radarName in enumerate(radarNames):
            #     buff_r[iRadar].append(loadPcWithMotionStatusInRcf(nusc, scene, sample, radarName, minVisThres))
                
        
        # get next sample
        sample = nusc.get('sample', sample['next'])
        
        
        
        #================#
        # Process Sweeps  
        #================#
        # interpolate the poses between the current and next refcamera
        _, vPose_ref_next, t_ref_next = getPoseAndTimeFromSample(nusc, scene, sample, refCamName)
        vPoses01, t01 = interUtils.interpolate2DPoses(vPose_ref, vPose_ref_next, t_ref, t_ref_next, numInterPts=NUM_INTERPOL_PTS)
        
        # process sweep data using the interpolated poses
        # for i in range(heightThresholds.shape[1]):
        #     _, _, lMap_heightThres[i,...] = processSweepOfPcSensor(nusc, sample, scene, l_sweepNames, vPoses01, t01,
        #                                                            lidarNames, [buff_l_heightThres[i]], detMap, lMap_heightThres[i,...], t_r2i,
        #                                                            pF_lMap, pO_lMap, pD_lMap, pDim, mDim, aDim, numColsPerCone_l,True)
        # _, _, lMap_noStreet = processSweepOfPcSensor(nusc, sample, scene, l_sweepNames, vPoses01, t01,
        #                                                     lidarNames, buff_l_noStreet, detMap, lMap_noStreet, t_r2i,
        #                                                     pF_lMap, pO_lMap, pD_lMap, pDim, mDim, aDim, numColsPerCone_l,True)
        # _, _, lMap_noStreetOther = processSweepOfPcSensor(nusc, sample, scene, l_sweepNames, vPoses01, t01,
        #                                                     lidarNames, buff_l_noStreetOther, detMap, lMap_noStreetOther, t_r2i,
        #                                                     pF_lMap, pO_lMap, pD_lMap, pDim, mDim, aDim, numColsPerCone_l,True)
        # _, _, lMap_noStreetOtherSidewalk = processSweepOfPcSensor(nusc, sample, scene, l_sweepNames, vPoses01, t01,
        #                                                     lidarNames, buff_l_noStreetOtherSidewalk, detMap, lMap_noStreetOtherSidewalk, t_r2i,
        #                                                     pF_lMap, pO_lMap, pD_lMap, pDim, mDim, aDim, numColsPerCone_l,True)
        # _, _, lMap_noStreetOtherSidewalkTerrain = processSweepOfPcSensor(nusc, sample, scene, l_sweepNames, vPoses01, t01,
        #                                                     lidarNames, buff_l_noStreetOtherSidewalkTerrain, detMap, lMap_noStreetOtherSidewalkTerrain, t_r2i,
        #                                                     pF_lMap, pO_lMap, pD_lMap, pDim, mDim, aDim, numColsPerCone_l,True)
        _, _, lMap_noOther = processSweepOfPcSensor(nusc, sample, scene, l_sweepNames, vPoses01, t01,
                                                            lidarNames, buff_l_noOther, detMap, lMap_noOther, t_r2i,
                                                            pF_lMap, pO_lMap, pD_lMap, pDim, mDim, aDim, numColsPerCone_l,True)
        _, _, lMap_noOtherSidewalk = processSweepOfPcSensor(nusc, sample, scene, l_sweepNames, vPoses01, t01,
                                                            lidarNames, buff_l_noOtherSidewalk, detMap, lMap_noOtherSidewalk, t_r2i,
                                                            pF_lMap, pO_lMap, pD_lMap, pDim, mDim, aDim, numColsPerCone_l,True)
        _, _, lMap_noOtherSidewalkTerrain = processSweepOfPcSensor(nusc, sample, scene, l_sweepNames, vPoses01, t01,
                                                            lidarNames, buff_l_noOtherSidewalkTerrain, detMap, lMap_noOtherSidewalkTerrain, t_r2i,
                                                            pF_lMap, pO_lMap, pD_lMap, pDim, mDim, aDim, numColsPerCone_l,True)
        _, _, lMap_noSidewalk = processSweepOfPcSensor(nusc, sample, scene, l_sweepNames, vPoses01, t01,
                                                            lidarNames, buff_l_noSidewalk, detMap, lMap_noSidewalk, t_r2i,
                                                            pF_lMap, pO_lMap, pD_lMap, pDim, mDim, aDim, numColsPerCone_l,True)
        _, _, lMap_noSidewalkTerrain = processSweepOfPcSensor(nusc, sample, scene, l_sweepNames, vPoses01, t01,
                                                            lidarNames, buff_l_noSidewalkTerrain, detMap, lMap_noSidewalkTerrain, t_r2i,
                                                            pF_lMap, pO_lMap, pD_lMap, pDim, mDim, aDim, numColsPerCone_l,True)
        _, _, lMap_noTerrain = processSweepOfPcSensor(nusc, sample, scene, l_sweepNames, vPoses01, t01,
                                                            lidarNames, buff_l_noTerrain, detMap, lMap_noTerrain, t_r2i,
                                                            pF_lMap, pO_lMap, pD_lMap, pDim, mDim, aDim, numColsPerCone_l,True)
        
        # buff_r, _, rMap = processSweepOfPcSensor(nusc, sample, scene, r_sweepNames, vPoses01, t01,
        #                                                     radarNames, buff_r, detMap, rMap, t_r2i,
        #                                                     pF_rMap, pO_rMap, pD_rMap, pDim, mDim, aDim, numColsPerCone_r, True)
        
        # # store mapping progress
        # if (storeLidarMap) and (storeProgress):
        #     saveImg(l_detMap, l_storDir+l_storDir.split("/")[-2]+"_map__{0:}.png".format(t_ref))
        # if (storeIlmMap) and (storeProgress):
        #     saveImg(l_ismMap, ilmMap_storDir+ilmMap_storDir.split("/")[-2]+"__{0:}.png".format(t_ref))
        # if (storeRadarMap) and (storeProgress):
        #     saveImg(r_detMap, r_storDirs[0]+r_storDirs[0].split("/")[-2]+"_map__{0:}.png".format(t_ref))
        # if (storeIrmMap) and (storeProgress):
        #     saveImg(r_ismMap, irmMap_storDir+irmMap_storDir.split("/")[-2]+"__{0:}.png".format(t_ref))
        
    # store map images
    # saveImg(rMap, sceneStorDir+"r_map.png") 
    # for i in range(heightThresholds.shape[1]):
    #     saveImg(lMap_heightThres[i,...], sceneStorDir+"l_heightThres_{:.3f}_map.png".format(heightThresholds[0,i])) 
    # saveImg(lMap_noStreet, sceneStorDir+"l_noStreetOther_map.png") 
    # saveImg(lMap_noStreetOther, sceneStorDir+"l_noStreetOtherSidewalk_map.png") 
    # saveImg(lMap_noStreetOtherSidewalk, sceneStorDir+"l_noStreetOtherSidewalk_map.png") 
    # saveImg(lMap_noStreetOtherSidewalkTerrain, sceneStorDir+"l_noStreetOtherSidewalkTerrain_map.png")
    saveImg(lMap_noOther, sceneStorDir+"l_noOther.png")
    saveImg(lMap_noOtherSidewalk, sceneStorDir+"l_noOtherSidewalk.png")
    saveImg(lMap_noOtherSidewalkTerrain, sceneStorDir+"l_noOtherSidewalkTerrain.png")
    saveImg(lMap_noSidewalk, sceneStorDir+"l_noSidewalk.png")
    saveImg(lMap_noSidewalkTerrain, sceneStorDir+"l_noSidewalkTerrain.png")
    saveImg(lMap_noTerrain, sceneStorDir+"l_noTerrain.png")
            
    # create observed area map
    # mappedArea = mapUtils.createObservedAreaMap(mappedArea, t_r2i, vPoses_ref, mDim, pDim, aDim)
    # # store observed area map
    # saveImg(mappedArea, sceneStorDir+"mappedArea.png")
    

# get the names of train, val and test scenes
trainNames, valNames, trainIdxs, valIdxs = sceneAttribUtils.getTrainValTestSplitNames()

# all scenes
# sceneIdxs = np.arange(len(nusc.scene))

# train scenes
sceneIdxs = trainIdxs

# val scenes
# sceneIdxs = valIdxs

# specific scenes
# sceneIdxs = [0]
               
t0 = time.time()
for sceneIdx in sceneIdxs:
    main(sceneIdx)
# main(0)
# pool = mp.Pool(mp.cpu_count())  
# pool.map(main, sceneIdxs)   
# pool.close()
# pool.join()  

# compute mIoU
interUnionPx = np.zeros((10+heightThresholds.shape[1],3,2))
for iScene in sceneIdxs:
    # get current scene
    scene = nusc.scene[iScene]
    print(iScene)
 
    # only process scene, iff vehicle has travelled a certain distance
    travelledDist = sceneAttribUtils.computeTravelledDistanceForScene(nusc, scene)    
    if (travelledDist < minDistTravelledThres):
        continue
    
    # only use day scenes
    if (("night" in scene["description"].lower()) or 
        ("difficult lighting" in scene["description"].lower())):
        continue
    
    # check whether the scene is in train, val or test
    setName = "gtVerification/"
    
    # create folders to store images
    sceneStorDir = createImgStorDir(STORAGE_DIR + setName + "scene{:04}".format(iScene) + '/')
    
    # load maps
    rMap = np.asarray(Image.open(sceneStorDir+"r_map.png"))/255
    lMap_heightThres = np.zeros((heightThresholds.shape[1],)+rMap.shape)
    for i in range(heightThresholds.shape[1]):
        lMap_heightThres[i,...] = np.asarray(Image.open(sceneStorDir+"l_heightThres_{:.3f}_map.png".format(heightThresholds[0,i])))/255
    lMap_noStreet = np.asarray(Image.open(sceneStorDir+"l_noStreetOther_map.png"))/255
    lMap_noStreetOther = np.asarray(Image.open(sceneStorDir+"l_noStreetOtherSidewalk_map.png"))/255
    lMap_noStreetOtherSidewalk = np.asarray(Image.open(sceneStorDir+"l_noStreetOtherSidewalk_map.png"))/255
    lMap_noStreetOtherSidewalkTerrain = np.asarray(Image.open(sceneStorDir+"l_noStreetOtherSidewalkTerrain_map.png"))/255
    lMap_noOther= np.asarray(Image.open(sceneStorDir+"l_noOther_map.png"))/255
    lMap_noOtherSidewalk = np.asarray(Image.open(sceneStorDir+"l_noOtherSidewalk_map.png"))/255
    lMap_noOtherSidewalkTerrain = np.asarray(Image.open(sceneStorDir+"l_noOtherSidewalkTerrain_map.png"))/255
    lMap_noSidewalk = np.asarray(Image.open(sceneStorDir+"l_noSidewalk_map.png"))/255
    lMap_noSidewalkTerrain = np.asarray(Image.open(sceneStorDir+"l_noSidewalkTerrain_map.png"))/255
    lMap_noTerrain = np.asarray(Image.open(sceneStorDir+"l_noTerrain_map.png"))/255
    mappedArea = np.asarray(Image.open(sceneStorDir+"mappedArea.png"))/255
    
    # compute the scores
    for i in range(heightThresholds.shape[1]):
        interUnionPx[i,:,:] += computeIoU(lMap_heightThres[i,...], rMap, mappedArea)
    interUnionPx[heightThresholds.shape[1],:,:] += computeIoU(lMap_noStreet, rMap, mappedArea)
    interUnionPx[heightThresholds.shape[1]+1,:,:] += computeIoU(lMap_noStreetOther, rMap, mappedArea)
    interUnionPx[heightThresholds.shape[1]+2,:,:] += computeIoU(lMap_noStreetOtherSidewalk, rMap, mappedArea)
    interUnionPx[heightThresholds.shape[1]+3,:,:] += computeIoU(lMap_noStreetOtherSidewalkTerrain, rMap, mappedArea)
    interUnionPx[heightThresholds.shape[1]+4,:,:] += computeIoU(lMap_noOther, rMap, mappedArea)
    interUnionPx[heightThresholds.shape[1]+5,:,:] += computeIoU(lMap_noOtherSidewalk, rMap, mappedArea)
    interUnionPx[heightThresholds.shape[1]+6,:,:] += computeIoU(lMap_noOtherSidewalkTerrain, rMap, mappedArea)
    interUnionPx[heightThresholds.shape[1]+7,:,:] += computeIoU(lMap_noSidewalk, rMap, mappedArea)
    interUnionPx[heightThresholds.shape[1]+8,:,:] += computeIoU(lMap_noSidewalkTerrain, rMap, mappedArea)
    interUnionPx[heightThresholds.shape[1]+9,:,:] += computeIoU(lMap_noTerrain, rMap, mappedArea)

print("mIoU                                    fr  |  oc  |  un  | (fr+oc)/2")
for i in range(heightThresholds.shape[1]):
    print("mIoU height threshold {4:.3f}m           {0:.2f} | {1:.2f} | {2:.2f} | {3:.2f}".format(interUnionPx[i,0,0]/interUnionPx[i,0,1],
                                                                                      interUnionPx[i,1,0]/interUnionPx[i,1,1],
                                                                                      interUnionPx[i,2,0]/interUnionPx[i,2,1],
                                                                                      (interUnionPx[i,0,0]/interUnionPx[i,0,1]+
                                                                                      interUnionPx[i,1,0]/interUnionPx[i,1,1])/2,
                                                                                      heightThresholds[0,i]))
print("mIoU noStreet                          {0:.2f} | {1:.2f} | {2:.2f} | {3:.2f}".format(interUnionPx[heightThresholds.shape[1],0,0]/interUnionPx[heightThresholds.shape[1],0,1],
                                                                                  interUnionPx[heightThresholds.shape[1],1,0]/interUnionPx[heightThresholds.shape[1],1,1],
                                                                                  interUnionPx[heightThresholds.shape[1],2,0]/interUnionPx[heightThresholds.shape[1],2,1],
                                                                                  (interUnionPx[heightThresholds.shape[1],0,0]/interUnionPx[heightThresholds.shape[1],0,1]+
                                                                                  interUnionPx[heightThresholds.shape[1],1,0]/interUnionPx[heightThresholds.shape[1],1,1])/2 ))
print("mIoU noStreetOther                     {0:.2f} | {1:.2f} | {2:.2f} | {3:.2f}".format(interUnionPx[heightThresholds.shape[1]+1,0,0]/interUnionPx[heightThresholds.shape[1]+1,0,1],
                                                                                  interUnionPx[heightThresholds.shape[1]+1,1,0]/interUnionPx[heightThresholds.shape[1]+1,1,1],
                                                                                  interUnionPx[heightThresholds.shape[1]+1,2,0]/interUnionPx[heightThresholds.shape[1]+1,2,1],
                                                                                  (interUnionPx[heightThresholds.shape[1]+1,0,0]/interUnionPx[heightThresholds.shape[1]+1,0,1]+
                                                                                  interUnionPx[heightThresholds.shape[1]+1,1,0]/interUnionPx[heightThresholds.shape[1]+1,1,1])/2))
print("mIoU noStreetOtherSidewalk             {0:.2f} | {1:.2f} | {2:.2f} | {3:.2f}".format(interUnionPx[heightThresholds.shape[1]+2,0,0]/interUnionPx[heightThresholds.shape[1]+2,0,1],
                                                                                  interUnionPx[heightThresholds.shape[1]+2,1,0]/interUnionPx[heightThresholds.shape[1]+2,1,1],
                                                                                  interUnionPx[heightThresholds.shape[1]+2,2,0]/interUnionPx[heightThresholds.shape[1]+2,2,1],
                                                                                  (interUnionPx[heightThresholds.shape[1]+2,0,0]/interUnionPx[heightThresholds.shape[1]+2,0,1]+
                                                                                  interUnionPx[heightThresholds.shape[1]+2,1,0]/interUnionPx[heightThresholds.shape[1]+2,1,1])/2))
print("mIoU noStreetOtherSidewalkTerrain      {0:.2f} | {1:.2f} | {2:.2f} | {3:.2f}".format(interUnionPx[heightThresholds.shape[1]+3,0,0]/interUnionPx[heightThresholds.shape[1]+3,0,1],
                                                                                  interUnionPx[heightThresholds.shape[1]+3,1,0]/interUnionPx[heightThresholds.shape[1]+3,1,1],
                                                                                  interUnionPx[heightThresholds.shape[1]+3,2,0]/interUnionPx[heightThresholds.shape[1]+3,2,1],
                                                                                  (interUnionPx[heightThresholds.shape[1]+3,0,0]/interUnionPx[heightThresholds.shape[1]+3,0,1]+
                                                                                  interUnionPx[heightThresholds.shape[1]+3,1,0]/interUnionPx[heightThresholds.shape[1]+3,1,1])/2))
print("mIoU noOther                           {0:.2f} | {1:.2f} | {2:.2f} | {3:.2f}".format(interUnionPx[heightThresholds.shape[1]+4,0,0]/interUnionPx[heightThresholds.shape[1]+4,0,1],
                                                                                  interUnionPx[heightThresholds.shape[1]+4,1,0]/interUnionPx[heightThresholds.shape[1]+4,1,1],
                                                                                  interUnionPx[heightThresholds.shape[1]+4,2,0]/interUnionPx[heightThresholds.shape[1]+4,2,1],
                                                                                  (interUnionPx[heightThresholds.shape[1]+4,0,0]/interUnionPx[heightThresholds.shape[1]+4,0,1]+
                                                                                  interUnionPx[heightThresholds.shape[1]+4,1,0]/interUnionPx[heightThresholds.shape[1]+4,1,1])/2))
print("mIoU noOtherSidewalk                  {0:.2f} | {1:.2f} | {2:.2f} | {3:.2f}".format(interUnionPx[heightThresholds.shape[1]+5,0,0]/interUnionPx[heightThresholds.shape[1]+5,0,1],
                                                                                  interUnionPx[heightThresholds.shape[1]+5,1,0]/interUnionPx[heightThresholds.shape[1]+5,1,1],
                                                                                  interUnionPx[heightThresholds.shape[1]+5,2,0]/interUnionPx[heightThresholds.shape[1]+5,2,1],
                                                                                  (interUnionPx[heightThresholds.shape[1]+5,0,0]/interUnionPx[heightThresholds.shape[1]+5,0,1]+
                                                                                  interUnionPx[heightThresholds.shape[1]+5,1,0]/interUnionPx[heightThresholds.shape[1]+5,1,1])/2))
print("mIoU noOtherSidewalkTerrain           {0:.2f} | {1:.2f} | {2:.2f} | {3:.2f}".format(interUnionPx[heightThresholds.shape[1]+6,0,0]/interUnionPx[heightThresholds.shape[1]+6,0,1],
                                                                                  interUnionPx[heightThresholds.shape[1]+6,1,0]/interUnionPx[heightThresholds.shape[1]+6,1,1],
                                                                                  interUnionPx[heightThresholds.shape[1]+6,2,0]/interUnionPx[heightThresholds.shape[1]+6,2,1],
                                                                                  (interUnionPx[heightThresholds.shape[1]+6,0,0]/interUnionPx[heightThresholds.shape[1]+6,0,1]+
                                                                                  interUnionPx[heightThresholds.shape[1]+6,1,0]/interUnionPx[heightThresholds.shape[1]+6,1,1])/2))
print("mIoU noSidewalk                       {0:.2f} | {1:.2f} | {2:.2f} | {3:.2f}".format(interUnionPx[heightThresholds.shape[1]+7,0,0]/interUnionPx[heightThresholds.shape[1]+7,0,1],
                                                                                  interUnionPx[heightThresholds.shape[1]+7,1,0]/interUnionPx[heightThresholds.shape[1]+7,1,1],
                                                                                  interUnionPx[heightThresholds.shape[1]+7,2,0]/interUnionPx[heightThresholds.shape[1]+7,2,1],
                                                                                  (interUnionPx[heightThresholds.shape[1]+7,0,0]/interUnionPx[heightThresholds.shape[1]+7,0,1]+
                                                                                  interUnionPx[heightThresholds.shape[1]+7,1,0]/interUnionPx[heightThresholds.shape[1]+7,1,1])/2))
print("mIoU noSidewalkTerrain                {0:.2f} | {1:.2f} | {2:.2f} | {3:.2f}".format(interUnionPx[heightThresholds.shape[1]+8,0,0]/interUnionPx[heightThresholds.shape[1]+8,0,1],
                                                                                  interUnionPx[heightThresholds.shape[1]+8,1,0]/interUnionPx[heightThresholds.shape[1]+8,1,1],
                                                                                  interUnionPx[heightThresholds.shape[1]+8,2,0]/interUnionPx[heightThresholds.shape[1]+8,2,1],
                                                                                  (interUnionPx[heightThresholds.shape[1]+8,0,0]/interUnionPx[heightThresholds.shape[1]+8,0,1]+
                                                                                  interUnionPx[heightThresholds.shape[1]+8,1,0]/interUnionPx[heightThresholds.shape[1]+8,1,1])/2))
print("mIoU noTerrain                       {0:.2f} | {1:.2f} | {2:.2f} | {3:.2f}".format(interUnionPx[heightThresholds.shape[1]+9,0,0]/interUnionPx[heightThresholds.shape[1]+9,0,1],
                                                                                  interUnionPx[heightThresholds.shape[1]+9,1,0]/interUnionPx[heightThresholds.shape[1]+9,1,1],
                                                                                  interUnionPx[heightThresholds.shape[1]+9,2,0]/interUnionPx[heightThresholds.shape[1]+9,2,1],
                                                                                  (interUnionPx[heightThresholds.shape[1]+9,0,0]/interUnionPx[heightThresholds.shape[1]+9,0,1]+
                                                                                  interUnionPx[heightThresholds.shape[1]+9,1,0]/interUnionPx[heightThresholds.shape[1]+9,1,1])/2))
print("Done after {:.2f} min!".format((time.time()-t0)/60 ))   
np.save("./mIouGtVerification.npy",interUnionPx)
            


