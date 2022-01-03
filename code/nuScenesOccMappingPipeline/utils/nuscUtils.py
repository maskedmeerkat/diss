import os
import numpy as np
import utils.interUtils as interUtils
from pyquaternion import Quaternion
import os.path as osp
from nuscenes.utils.data_classes import LidarPointCloud, RadarPointCloud
import utils.mapUtils as mapUtils

VEHICLE_BOARDERS = np.array([0.75, -0.75, 1.9, -1.2])
HEIGHT_THRESHOLDS = np.array([0.5, 3.0])
NUM_INTERPOL_PTS = 20
ADD_TO_BOUNDING_BOX_DIAMETER = 0.1


#===========================#
def rotMat(angle):
    R = np.array([[np.cos(angle), -np.sin(angle)],
		          [np.sin(angle),  np.cos(angle)]])
    return R


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
def getPcWithMotionStatusInRcf(nusc, scene, sample, sensorName, dynBoundingBoxes_rcf, pDim, mDim, minVisThres, sweepFlag=False):
    """
    sensorName: 'CAM_FRONT','CAM_BACK', ...
                'CAM_BACK_LEFT','CAM_FRONT_LEFT','CAM_FRONT_RIGHT','CAM_BACK_RIGHT', ...
                'LIDAR_TOP', ...
                'RADAR_FRONT', ...
                'RADAR_FRONT_RIGHT','RADAR_FRONT_LEFT','RADAR_BACK_RIGHT','RADAR_BACK_LEFT'
    """    
    
    # get sensor and vehicle pose and timestamp info
    sPose, vPose, t = getPoseAndTimeFromSample(nusc, scene, sample, sensorName)    
    
    # get data path
    if (sweepFlag):
        dataPath = nusc.dataroot+'sweeps'+'/'+sensorName+'/'+sweepName
    else:
        dataPath = osp.join(nusc.dataroot, nusc.get('sample_data', sample['data'][sensorName])['filename'])
    
    # load point cloud and motion status
    if (sensorName == 'LIDAR_TOP'):
        pc = LidarPointCloud.from_file(dataPath)
    else:
        pc = RadarPointCloud.from_file(dataPath)
        
    # remove the detections corresponding to the ego vehicle
    if (sensorName == 'LIDAR_TOP'):
        mask = np.zeros_like(pc.points[0,:])
        mask = np.logical_or(mask, pc.points[0,:]>VEHICLE_BOARDERS[0])
        mask = np.logical_or(mask, pc.points[0,:]<VEHICLE_BOARDERS[1])
        mask = np.logical_or(mask, pc.points[1,:]>VEHICLE_BOARDERS[2])
        mask = np.logical_or(mask, pc.points[1,:]<VEHICLE_BOARDERS[3])
        pc.points = pc.points[:3,mask] 
    
    # trafo scf -> vcf
    sample_data = nusc.get('sample_data', sample['data'][sensorName])
    cal_sensor = nusc.get('calibrated_sensor', sample_data['calibrated_sensor_token'])
    pc.rotate(Quaternion(cal_sensor['rotation']).rotation_matrix)
    pc.translate(np.array(cal_sensor['translation']))    
    
    # remove ground plane points through thresholding
    if (sensorName == 'LIDAR_TOP'):
        mask = np.zeros_like(pc.points[0,:]).astype(np.int)
        mask = np.logical_or( mask, np.abs(pc.points[2,:]) > HEIGHT_THRESHOLDS[0])
        mask = np.logical_and(mask, pc.points[2,:] < HEIGHT_THRESHOLDS[1])
        pc.points = pc.points[:,mask] 

    # discretize the lidar point cloud
    if (sensorName == 'LIDAR_TOP'):
        bevImg = mapUtils.pc2VehicleCenteredBevImg(pc.points, pDim=pDim*2, mDim=mDim)
        pcDisc = mapUtils.vehicleCenteredBevImg2Pc(bevImg, pDim=pDim*2, mDim=mDim)        
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
        isDynamic = dyn_static_obj_masks(pcDisc, dynBoundingBoxes_rcf)
        isStationary = (isDynamic==0).astype(np.int)
    else:
        isStationary = ((pc.points[3,:] != 0) * (pc.points[3,:] != 2) * (pc.points[3,:] != 6)).astype(np.int) 
    
    # replace the z coordinate with the stationary information
    pcDisc[2,:] = isStationary
    
    return pcDisc


#===========================#
def findAllSweepsForTheScene(nusc, scene, sensorName):
    """
    sensorType: 'CAM_FRONT','CAM_BACK', ...
                'CAM_BACK_LEFT','CAM_FRONT_LEFT','CAM_FRONT_RIGHT','CAM_BACK_RIGHT', ...
                'LIDAR_TOP', ...
                'RADAR_FRONT', ...
                'RADAR_FRONT_RIGHT','RADAR_FRONT_LEFT','RADAR_BACK_RIGHT','RADAR_BACK_LEFT'
    """
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
    sweepNames = os.listdir(nusc.dataroot + '/sweeps/' + sensorName + '/')
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
    
    sweepNames = sweepNames[isInsideTimeSpan]
    
    return sweepNames


#===========================#
def findRefCamName(camNames, c_sweepNames):            
    numSweepsForScene = np.array([len(c_sweepNames[0]),
                                  len(c_sweepNames[1]),
                                  len(c_sweepNames[2]),
                                  len(c_sweepNames[3]),
                                  len(c_sweepNames[4]),
                                  len(c_sweepNames[5])])
    refIdx = np.argmin(numSweepsForScene)        
    return camNames[refIdx], c_sweepNames[refIdx]