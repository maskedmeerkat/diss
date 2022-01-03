import os.path as osp
from nuscenes.utils.data_classes import LidarPointCloud
from pyquaternion import Quaternion
import numpy as np
from PIL import Image
from nuscenes.utils.geometry_utils import view_points
import cv2
import sys
import matplotlib.pyplot as plt

#===========================#
def getLidarPcInVcf(nusc, scene, sample):
    # use sensor's meta data to find data path for current sample
    sample_data = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
    data_path = osp.join(nusc.dataroot, sample_data['filename']) 
    
    # load point cloud
    pc = LidarPointCloud.from_file(data_path)
        
    pointsensor = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
    cs_record = nusc.get('calibrated_sensor', pointsensor['calibrated_sensor_token'])
    pc.rotate(Quaternion(cs_record['rotation']).rotation_matrix)
    pc.translate(np.array(cs_record['translation']))
    
    return pc.points[:3,:]


#===========================#
def scaleIntrinsics(K, x_scale, y_scale):
    """Scale intrinsics given x_scale and y_scale factors"""
    K[..., 0, 0] *= x_scale
    K[..., 1, 1] *= y_scale
    K[..., 0, 2] = (K[..., 0, 2] + 0.5) * x_scale - 0.5
    K[..., 1, 2] = (K[..., 1, 2] + 0.5) * y_scale - 0.5
    return K


#===========================#
def pc2BevImg(pc, pDim, mDim):     
    # extract the height
    height = pc[2,:]
    
    # trafo to image coordinate system    
    pc[1,:] = -pc[1,:]
    pc = pc[[1,0,2],:]
    pc = pc + np.array([[mDim/2,mDim/2,0]]).T     
    
    # trafo into pixel coordinates
    pc = (pc * pDim/mDim).astype(np.int)
    
    # filter out all pts outside image dimension
    mask = (pc[0,:] >= 0) * (pc[1,:] >= 0) * (pc[0,:] < pDim) * (pc[1,:] < pDim)
    pc = pc[:,mask]
    height = height[mask]
    
    # create brid's eye view lidar image
    bevImg = np.zeros((pDim,pDim))
    
    # trafo height to color
    clipBounds = (-.5,1.)
    color = (((np.clip(height, clipBounds[0], clipBounds[1])-clipBounds[0]) / (clipBounds[1]-clipBounds[0])) * 255).astype(np.int)
    
    # mark detections in the image
    bevImg[pc[0,:], pc[1,:]] = color
    
    return bevImg


#===========================#
def monoDepth2PcInVcf(nusc, depthImg, camera_token, origScale=(900,1600)):
    """
    Given a point sensor (lidar/radar) token and camera sample_data token, load point-cloud and map it to the image
    plane.
    :param pointsensor_token: Lidar/radar sample_data token.
    :param camera_token: Camera sample_data token.
    :param min_dist: Distance from the camera below which points are discarded.
    :param render_intensity: Whether to render lidar intensity instead of point depth.
    :return (pointcloud <np.float: 2, n)>, coloring <np.float: n>, image <Image>).
    """    
    # get camera info  
    cam = nusc.get('sample_data', camera_token)
    sensorRecord = nusc.get('calibrated_sensor', cam['calibrated_sensor_token'])
    
    # create xyz coordinates for each image pixel
    xy = np.meshgrid(np.arange(int(depthImg.shape[0]//2),depthImg.shape[0]), np.arange(0,depthImg.shape[1]))
    x = xy[0].flatten()[np.newaxis,:]
    y = xy[1].flatten()[np.newaxis,:]
    z = depthImg[x,y]
    pc = np.append(y, np.append(x,z, axis=0), axis=0)
    
    # trafo icf to scf
    pc[:-1,:] *= z
    
    K = np.eye(4)
    K[:3,:3] = np.array(sensorRecord['camera_intrinsic'])
    K = scaleIntrinsics(K, depthImg.shape[1]/origScale[1], depthImg.shape[0]/origScale[0])
    
    pc = np.append(pc, np.ones((1,pc.shape[1])), axis=0)
    pc = np.matmul(np.linalg.inv(K), pc)
    
    # trafo scf -> vcf
    T_s2v = np.eye(4)
    T_s2v[:3, 3] = np.array(sensorRecord['translation'])
    T_s2v[:3,:3] = Quaternion(sensorRecord['rotation']).rotation_matrix
    pc = np.matmul(T_s2v, pc) 

    return pc[:-1,:]


#===========================#
def map_pointcloud_to_image(nusc,
                            pointsensor_token: str,
                            camera_token: str,
                            min_dist: float = 1.0,
                            render_intensity: bool = False):
    """
    Given a point sensor (lidar/radar) token and camera sample_data token, load point-cloud and map it to the image
    plane.
    :param pointsensor_token: Lidar/radar sample_data token.
    :param camera_token: Camera sample_data token.
    :param min_dist: Distance from the camera below which points are discarded.
    :param render_intensity: Whether to render lidar intensity instead of point depth.
    :return (pointcloud <np.float: 2, n)>, coloring <np.float: n>, image <Image>).
    """
    cam = nusc.get('sample_data', camera_token)
    pointsensor = nusc.get('sample_data', pointsensor_token)
    pcl_path = osp.join(nusc.dataroot, pointsensor['filename'])
    pc = LidarPointCloud.from_file(pcl_path)
    im = Image.open(osp.join(nusc.dataroot, cam['filename']))

    # Points live in the point sensor frame. So they need to be transformed via global to the image plane.
    # First step: transform the point-cloud to the ego vehicle frame for the timestamp of the sweep.
    cs_record = nusc.get('calibrated_sensor', pointsensor['calibrated_sensor_token'])
    pc.rotate(Quaternion(cs_record['rotation']).rotation_matrix)
    pc.translate(np.array(cs_record['translation']))

    # Second step: transform to the global frame.
    poserecord = nusc.get('ego_pose', pointsensor['ego_pose_token'])
    pc.rotate(Quaternion(poserecord['rotation']).rotation_matrix)
    pc.translate(np.array(poserecord['translation']))

    # Third step: transform into the ego vehicle frame for the timestamp of the image.
    poserecord = nusc.get('ego_pose', cam['ego_pose_token'])
    pc.translate(-np.array(poserecord['translation']))
    pc.rotate(Quaternion(poserecord['rotation']).rotation_matrix.T)

    # Fourth step: transform into the camera.
    cs_record = nusc.get('calibrated_sensor', cam['calibrated_sensor_token'])
    pc.translate(-np.array(cs_record['translation']))
    pc.rotate(Quaternion(cs_record['rotation']).rotation_matrix.T)

    # Fifth step: actually take a "picture" of the point cloud.
    # Grab the depths (camera frame z axis points away from the camera).
    depths = pc.points[2, :]

    if render_intensity:
        assert pointsensor['sensor_modality'] == 'lidar', 'Error: Can only render intensity for lidar!'
        # Retrieve the color from the intensities.
        # Performs arbitary scaling to achieve more visually pleasing results.
        intensities = pc.points[3, :]
        intensities = (intensities - np.min(intensities)) / (np.max(intensities) - np.min(intensities))
        intensities = intensities ** 0.1
        intensities = np.maximum(0, intensities - 0.5)
        coloring = intensities
    else:
        # Retrieve the color from the depth.
        coloring = depths

    # Take the actual picture (matrix multiplication with camera-matrix + renormalization).
    points = view_points(pc.points[:3, :], np.array(cs_record['camera_intrinsic']), normalize=True)
    
    # In the following, we are going to remove lidar pts ouside of camera view. We are however interested in which of 
    # the original points remain. Therefore, we define the pts indices
    l_indices = np.arange(points.shape[1])

    # Remove points that are either outside or behind the camera. Leave a margin of 1 pixel for aesthetic reasons.
    # Also make sure points are at least 1m in front of the camera to avoid seeing the lidar points on the camera
    # casing for non-keyframes which are slightly out of sync.
    mask = np.ones(depths.shape[0], dtype=bool)
    mask = np.logical_and(mask, depths > min_dist)
    mask = np.logical_and(mask, points[0, :] > 1)
    mask = np.logical_and(mask, points[0, :] < im.size[0] - 1)
    mask = np.logical_and(mask, points[1, :] > 1)
    mask = np.logical_and(mask, points[1, :] < im.size[1] - 1)
    points = points[:, mask]
    coloring = coloring[mask]
    l_indices = l_indices[mask]

    return points, coloring, np.asarray(im), l_indices


#===========================#
def lidar2BirdsEyeView(nusc, scene, sample, pDim, mDim):
    # load lidar points in vehicle coordinates
    l_pc = getLidarPcInVcf(nusc, scene, sample)
    
    # define lidar points in bird's eye view
    l_b = l_pc.copy()
    
    # trafo to image coordinate system    
    l_b[1,:] = -l_b[1,:]
    l_b = l_b[[1,0,2],:]
    l_b = l_b + np.array([[mDim/2,mDim/2,0]]).T        
    
    # trafo into pixel coordinates
    l_b = (l_b * pDim/mDim).astype(np.int)
    
    # define indices of each lidar point
    indices_b = np.arange(l_b.shape[1])
    
    # filter out all pts outside image dimension
    mask = (l_b[0,:] >= 0) * (l_b[1,:] >= 0) * (l_b[0,:] < pDim) * (l_b[1,:] < pDim)
    l_b = l_b[:,mask]
    indices_b = indices_b[mask]
    
    # create brid's eye view lidar image
    l_img = np.zeros((pDim,pDim))
    
    # mark detections in the image
    l_img[l_b[0,:],l_b[1,:]] = 1
    
    return l_b, indices_b, l_img


#===========================#
def findSceneWithEnoughGndPts(nusc, scene, sample, pDim, mDim, camName, heightEps, numWantedGndPts=100, minNumGndPts=30): 
    sampleWithMostNumGndPts = -1
    mostNumGndPts = 0
    for iScene in range(scene['nbr_samples']-1):    
        # project lidar to bev image"
        l_b, indices_b, l_img = lidar2BirdsEyeView(nusc, scene, sample, pDim, mDim)
        
        # project lidar to camera
        pointsensor_token = sample['data']['LIDAR_TOP']
        camera_token = sample['data'][camName]    
        l_c, coloring, c_img, indices_c = map_pointcloud_to_image(nusc, pointsensor_token, camera_token)
        
        # check, if enough lidar points are on the flat ground
        l_pc = getLidarPcInVcf(nusc, scene, sample)
        height = l_pc[2,indices_c]
        mask = np.abs(height) < heightEps
        numGndPts = np.sum(mask)
        if (mostNumGndPts < numGndPts) and (minNumGndPts < numGndPts):            
            sampleWithMostNumGndPts = sample
            mostNumGndPts = numGndPts            
        if (numGndPts < numWantedGndPts):
            sample = nusc.get('sample', sample['next']) 
        else:
            break
    return sampleWithMostNumGndPts

#===========================#
def computeHomography(nusc, scene, camName, pDim=128, mDim=40., croppPortion = 0.55, heightEps = 0.02):
    numWantedGndPts=100
    minNumGndPts=30
    # search for a sample that has enough ground points to compute the homography matrix
    sample  = nusc.get('sample', scene['first_sample_token']) 
    sampleWithMostNumGndPts = findSceneWithEnoughGndPts(nusc, scene, sample, pDim, mDim, 
                                                        camName, heightEps,
                                                        numWantedGndPts=numWantedGndPts, minNumGndPts=minNumGndPts)
    # in case no sample had enough ground points
    if (sampleWithMostNumGndPts == -1):
        return np.array([]), np.array([])
    
    # else, compute the homography matrix by...
    sample = sampleWithMostNumGndPts
    
    # project lidar to bev image"
    l_b, indices_b, l_img = lidar2BirdsEyeView(nusc, scene, sample, pDim, mDim)
    
    # project lidar to camera
    pointsensor_token = sample['data']['LIDAR_TOP']
    camera_token = sample['data'][camName]    
    l_c, coloring, c_img, indices_c = map_pointcloud_to_image(nusc, pointsensor_token, camera_token)
    
    # check, if enough lidar points are on the flat ground
    l_pc = getLidarPcInVcf(nusc, scene, sample)
    height = l_pc[2,indices_c]
    mask = np.abs(height) < heightEps
    numGndPts = np.sum(mask)
    # find index of lidar points in camera, which are on the ground
    idx_c = np.arange(indices_c.shape[0])
    indices_c = indices_c[mask]
    idx_c = idx_c[mask]
    
    # find corresponding points in bird's eye view
    idx_b = []
    mask = np.ones_like(idx_c).astype(bool)
    for i in range(idx_c.shape[0]):
        tmp = np.where(indices_b == indices_c[i])
        if (tmp[0].shape[0] > 0):
            idx_b = np.append(idx_b, tmp[0][0])
        else:
            mask[i] = False 
    
    idx_c = idx_c[mask]
    idx_b = idx_b.astype(np.int)
    
    # get source and destination points
    pts_src = l_c[:-1,idx_c]
    pts_dst = l_b[:-1,idx_b]
    pts_dst = pts_dst[[1,0],:]
    
    # crop camera image to remove infinite distance points
    cropped_img = c_img.copy()
    cropped_img = cropped_img[int(c_img.shape[0] * croppPortion):,:,:]
    
    # account source points for cropping
    pts_src[1,:] -= int(c_img.shape[0] * croppPortion)
    
    homography, mask = cv2.findHomography(pts_src.T, pts_dst.T, cv2.RANSAC)
    
    x,y = np.meshgrid(np.arange(cropped_img.shape[1]),np.arange(cropped_img.shape[0]))
    x = x.flatten()
    y = y.flatten()
    pts = np.ones((3,x.shape[0]))
    pts[0,:] = x
    pts[1,:] = y
    pts = pts.astype(np.int)
    
    pts_ = np.dot(homography,pts)
    pts_[:-1,:] /= pts_[2,:]
    
    # all points that fall inside image boundaries 
    mask = np.logical_and((pts_[0,:] >= 0), (pts_[0,:] < pDim))
    mask = np.logical_and(mask, np.logical_and((pts_[1,:] >= 0), (pts_[1,:] < pDim)))
    
    imgMask = np.ones((cropped_img.shape[0],cropped_img.shape[1])).astype(np.uint8)
    imgMask[pts[1,mask],pts[0,mask]] = 0
    
    # firstMaxInRow = np.argmax(imgMask,axis=0)
    for iCol in range(imgMask.shape[1]):
        rowsWithMax = np.where(imgMask[:,iCol])[0]
        if (rowsWithMax.shape[0] > 0):
            firstRowWithMax = np.min(rowsWithMax)
            imgMask[:firstRowWithMax,iCol] = 2  
    imgMask[imgMask==1] = 0
    imgMask[imgMask==2] = 1
    return homography, imgMask.astype(bool)
        
        
#===========================#          
def processSampleOf360Camera(nusc, scene, sample, imgStorDir,
                             refCamName, camNames, homographyMats, croppPortion, pDim, semSegFlag=False):
    # get sensor and vehicle pose and timestamp info
    _, vPose_ref, t_ref = getPoseAndTimeFromSample(nusc, scene, sample, refCamName)
    
    # store data to csv file
    if (storeCsvFlag):    
        writeCamToCsv(csvWriter, vPose_ref, t_ref)
    
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
        img_360 = addImgTo360Img_inRcf(img, img_360, homographyMats[iCam], vPose_ref, croppPortion, pDim)
        
    if (semSegFlag):
        # apply cityscapes colormap 
        img_360 = applyCityScapesColorMap(img_360)
    
    # store 360 bird's eye view image
    if (store360ImgFlag):
        img_360 = Image.fromarray(img_360)
        if (semSegFlag):
            img_360.save(imgStorDir+"s__{:}.png".format(t_ref))       
        else:
            img_360.save(imgStorDir+"c__{:}.png".format(t_ref))
    
    return vPose_ref, t_ref     