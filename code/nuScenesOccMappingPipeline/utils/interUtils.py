from pyquaternion import Quaternion
import numpy as np

#===========================#
def homogTrafo2Pose_2D(T):  
    p = np.array([T[0,2], T[1,2], np.arctan2(T[1,0],T[0,0])])
    return p


#===========================#
def rotMat(angle):
    R = np.array([[np.cos(angle), -np.sin(angle)],
		          [np.sin(angle),  np.cos(angle)]])
    return R


#===========================#
def pose2HomogTrafo_2D(p):
    T = np.eye(3)
    T[:2,:2] = rotMat(p[2])
    T[:2, 2] = p[:2]
    return T


#===========================#
def getY(x, poly):
    y = np.zeros_like(x)
    for i in range(x.shape[0]):
        y[i] = np.dot([x[i]**3 , x[i]**2 , x[i] , 1], poly) 
        
    return y


#===========================#
def getAngle(x, poly):
    y = np.zeros_like(x)
    for i in range(x.shape[0]):
        y[i] = np.arctan(np.dot([3*x[i]**2 , 2*x[i] , 1 , 0], poly))
        
    return y 


#===========================#
def poly3rdOrder(p01, angle):
    x1 = p01[0]
    y1 = p01[1]
    dydx1 = np.tan(angle)
    
    Y = np.array([[0., y1, 0., dydx1]]).T
    X = np.array([[0.     , 0.   , 0., 1],                  
                  [x1**3  , x1**2, x1, 1],
                  [0.     , 0.   , 1., 0],
                  [3*x1**2, 2*x1 , 1., 0]])
    X_ = np.linalg.inv(X)
    poly = np.dot(X_, Y)
    
    return poly

#===========================#  
def linDistBetweenPoses(p):
    length = np.zeros(p.shape[0])
    
    # recursively measure the linearized distance between the poses
    for i in range(p.shape[0]-1):
        length[i+1] = length[i] + np.linalg.norm( p[i+1,:] - p[i,:])
        
    return length


#===========================#
def interpolate2DPts(p01, angle, t0, t1, numInterPts=100, minDist=0.1):    
    # only interpolate, if points are more than minDist appart from each other    
    if (np.linalg.norm(p01) > minDist) and (angle > 0.01):
        # interpolate between the two poses by a 3rd degree polynom
        poly = poly3rdOrder(p01, angle)        
        interPts = np.zeros((numInterPts,2))
        interPts[:,0] = np.linspace(0., p01[0], numInterPts)
        interPts[:,1] = getY(interPts[:,0], poly)  

        # integrate along the curve
        length = linDistBetweenPoses(interPts)
        
        # normalize the length
        length /= length[-1]
        
        # interpolate timestamps
        t01 = np.ones_like(length) * t0
        t01 += length * (t1 - t0)              
    
    else:
        interPts = np.zeros((numInterPts, 2))
        interPts[:,0] = np.linspace(0,p01[0],numInterPts)
        interPts[:,1] = np.linspace(0,p01[1],numInterPts)
        
        t01 = np.linspace(t0,t1,numInterPts)
    
    return interPts, t01


#===========================#
def interpolate2DPoses(p0, p1, t0, t1, numInterPts=100, minDist=0.1):    
    # compute relative pose in homogeneous coordinates
    T1 = pose2HomogTrafo_2D(p1)    
    T0 = pose2HomogTrafo_2D(p0)    
    T10 = np.dot( np.linalg.inv(T0), T1)
    
    # represent all poses in p0 coordinate frame
    p1 = homogTrafo2Pose_2D(T10)
    p0 = np.array([0,0,0])
    
    if (np.sqrt(p1[0]**2 + p1[1]**2) > minDist):                
        # interpolate between the two poses by a 3rd degree polynom
        pol01 = poly3rdOrder(p1[:2], p1[2])
        p01 = np.zeros((numInterPts,3))
        p01[:,0] = np.linspace(p0[0], p1[0], p01.shape[0])
        p01[:,1] = getY(p01[:,0], pol01) 
        
        # the orientation is given by the arctan of the polynom's derivative
        p01[:,2] = getAngle(p01[:,0], pol01)                
    
    else:
        p01 = np.zeros((numInterPts, 3))
        p01[:int(numInterPts//2),:] = p0
        p01[int(numInterPts//2):,:] = p1
        
    # trafo all poses back to wcf
    for iPose in range(numInterPts):
        T_ = np.dot(T0, pose2HomogTrafo_2D(p01[iPose,:]))        
        p01[iPose,:] = homogTrafo2Pose_2D(T_)
        
    # integrate along the curve
    length = linDistBetweenPoses(p01)
    
    # normalize the length
    length /= length[-1]
    
    # interpolate timestamps
    t01 = np.ones_like(length) * t0
    t01 += length * (t1 - t0)
    
    return p01, t01


#===========================#
def interpolate3DPoses(T0, T1, t0, t1, numInterPts=100, minDist=0.3):
    # get relative pose in T0 coordinate frame
    T01 = np.dot(np.linalg.inv(T0), T1)
    
    # initialize interpolation points object
    interPts = np.zeros((numInterPts,3))
    T_inter = [None] * numInterPts
    
    # if poses in both directions are too close to eachother
    if (np.linalg.norm(T01[:3,3]) < minDist):
        for iPt in range(numInterPts):
            if (iPt <= numInterPts//2):
                T_inter[iPt] = T0
            else:
                T_inter[iPt] = T1
        
        t01 = np.linspace(t0,t1,numInterPts)
    
    else:   
        # xy plane 2D position interpolation
        yAngle = np.arctan2(T01[1,0],T01[0,0])
        if (np.linalg.norm(T01[[0,1],3]) > minDist) and (yAngle > 0.01):            
            interPts[:,[0,1]], _ = interpolate2DPts(T01[[0,1],3], yAngle, t0, t1, numInterPts=numInterPts, minDist=minDist)
        else:
            interPts[:,0] = np.linspace(0,T01[0,3],numInterPts)
            interPts[:,1] = np.linspace(0,T01[1,3],numInterPts)
        
        # xz plane 2D position interpolation
        zAngle = np.arctan2(T01[2,0],T01[0,0])
        if (np.linalg.norm(T01[[0,2],3]) > minDist) and (zAngle > 0.01):                        
            interPts[:,[0,2]], _ = interpolate2DPts(T01[[0,2],3], zAngle, t0, t1, numInterPts=numInterPts, minDist=minDist)
        else:
            interPts[:,0] = np.linspace(0,T01[0,3],numInterPts)
            interPts[:,2] = np.linspace(0,T01[2,3],numInterPts)
        
        # integrate along the curve
        length = linDistBetweenPoses(interPts)
        
        # normalize the length
        length /= length[-1]
        
        # interpolate timestamps
        t01 = np.ones_like(length) * t0
        t01 += length * (t1 - t0)
        
        # interpolate the orientation & use interpolated positions to define homogeneous trafos in world coordinates
        q0 = Quaternion(matrix=np.eye(3))
        q1 = Quaternion(matrix=T01[:3,:3])
        T_ = np.eye(4)
        for iPt in range(numInterPts):
            # define the current pose in homogeneous coordinates
            T_[:3,:3] = Quaternion.slerp(q0, q1, t01[iPt]/(t1-t0) ).rotation_matrix
            T_[:3, 3] = interPts[iPt,:]
            
            # trafo pose back to world coordinates
            T_inter[iPt] = np.dot(T0, T_)
    
    return T_inter, t01


#===========================#
def getDynObjBoundingBoxes_rcf(nusc, scene, sample, sensorName = 'LIDAR_TOP', minVisThres=0):
    sample_data = nusc.get('sample_data', sample['data'][sensorName])
    
    # trafo from sensor to vehicle coordinates
    T_s2v = np.eye(4)
    cal_sensor = nusc.get('calibrated_sensor', sample_data['calibrated_sensor_token'])
    T_s2v[:3,:3] = Quaternion(cal_sensor['rotation']).rotation_matrix
    T_s2v[:3, 3] = np.array(cal_sensor['translation'])
    
    # trafo from vehicle to world coordinates
    T_v2w = np.eye(4)
    ego_pose = nusc.get('ego_pose', sample_data['ego_pose_token'])
    T_v2w[:3,:3] = Quaternion(ego_pose['rotation']).rotation_matrix
    T_v2w[:3, 3] = np.array(ego_pose['translation'])
    
    # trafo from sensor to world coordinates
    T_s2w = np.dot(T_v2w, T_s2v)
    
    # trafo from wcf -> rcf
    sample0  = nusc.get('sample', scene['first_sample_token'])
    sample0_data = nusc.get('sample_data', sample0['data']['LIDAR_TOP'])
    ego_pose = nusc.get('ego_pose', sample0_data['ego_pose_token'])
    T_w2r = np.eye(4)
    T_w2r[:3,:3] = Quaternion(ego_pose['rotation']).rotation_matrix
    T_w2r[:3, 3] = np.array(ego_pose['translation'])
    T_w2r = np.linalg.inv(T_w2r)
    
    T_s2r = np.dot(T_w2r, T_s2w)
    
    # get lidar and annotation tokens
    sensorToken = sample['data'][sensorName]
    annTokens = sample['anns']     
    
    # loop over all annotation boxes and decide wether they belong to dynamic objs
    dynObjBoundingBoxes_rcf = []
    statnObjBoundingBoxes_rcf = []
    
    for it in range(len(annTokens)):
        # get the annotation box object
        _, box, _ = nusc.get_sample_data(sensorToken, selected_anntokens=[annTokens[it]])
        
        # use the token to get the actual annotation info        
        annotation =  nusc.get('sample_annotation', annTokens[it])
        
        # check if annotation box in fov of selected sensor
        if (len(box)==0):
            continue
        
        # check whether visibility is bigger 60% 
        if (int(annotation['visibility_token']) < minVisThres):
            continue
        
        # trafo bounding box to rcf
        T = np.eye(4)
        T[:3,:3] = box[0].orientation.rotation_matrix
        T[:3, 3] = box[0].center
        
        T = np.dot(T_s2r, T)
        box[0].orientation = Quaternion(matrix=T[:3,:3])
        box[0].center = T[:3, 3]
        
        # check the annotation name to see if it can be dynamic
        category_name = annotation['category_name']
        dynFlag = False
        
        # all humans ("h") and animals ("a") are marked as dynamic
        if (category_name[0]=="h") or (category_name[0]=="a"):    
            dynFlag = True
            
        # all vehicles ("v") which are not parked cars or driverless bikes are dynamic
        elif (category_name[0]=="v") and len(annotation['attribute_tokens']): 
            # in case of vehicles, additionally check the status
            status = nusc.get('attribute', annotation['attribute_tokens'][0])['name']
            if not((status[-6:]=="parked") or (status[-13:]=="without_rider")):
                dynFlag = True
            
        # append bounding boxes of dynamic objs
        if (dynFlag):                        
            dynObjBoundingBoxes_rcf.append(box[0])
        else:
            statnObjBoundingBoxes_rcf.append(box[0])
            
    return np.asarray(dynObjBoundingBoxes_rcf), np.asarray(statnObjBoundingBoxes_rcf)


#===========================#
def getInterpolBoundingBoxes_wcf(nusc, scene, sample1, numInterPts=10, sensorName='LIDAR_TOP'):
    # get previous sample
    sample0 = nusc.get('sample', sample1['prev'])    
    sample_data0 = nusc.get('sample_data', sample0['data'][sensorName])
    
    # trafo from scf -> vcf
    T0_s2v = np.eye(4)
    cal_sensor = nusc.get('calibrated_sensor', sample_data0['calibrated_sensor_token'])
    T0_s2v[:3,:3] = Quaternion(cal_sensor['rotation']).rotation_matrix
    T0_s2v[:3, 3] = np.array(cal_sensor['translation'])
    
    # trafo from vehicle to world coordinates
    T0_v2w = np.eye(4)
    ego_pose = nusc.get('ego_pose', sample_data0['ego_pose_token'])
    T0_v2w[:3,:3] = Quaternion(ego_pose['rotation']).rotation_matrix
    T0_v2w[:3, 3] = np.array(ego_pose['translation'])
    
    # trafo from sensor to world coordinates
    T0_s2w = np.dot(T0_v2w, T0_s2v)
    
    # trafo from wcf -> rcf
    sampleRef  = nusc.get('sample', scene['first_sample_token'])
    sampleRef_data = nusc.get('sample_data', sampleRef['data']['LIDAR_TOP'])
    ego_pose = nusc.get('ego_pose', sampleRef_data['ego_pose_token'])
    T_w2r = np.eye(4)
    T_w2r[:3,:3] = Quaternion(ego_pose['rotation']).rotation_matrix
    T_w2r[:3, 3] = np.array(ego_pose['translation'])
    T_w2r = np.linalg.inv(T_w2r)
    
    T0_s2r = np.dot(T_w2r, T0_s2w)
    
    #=================#
    # Obtain Bounding Boxes Of Current And Previous Samples
    #=================#
    # get the dynamic obj's bounding boxes of the current sample
    dynObjBoundingBoxes_1_tmp, _ = getDynObjBoundingBoxes_rcf(nusc, scene, sample1, sensorName=sensorName)
    
    # find bounding box correspondences in previous sample and filter out the ones without correspondences
    dynObjBoundingBoxes_0_rcf = []
    dynObjBoundingBoxes_1_rcf = []
    for iBox in range(len(dynObjBoundingBoxes_1_tmp)):
        # get annotation info cooresponding to bounding box
        ann = nusc.get('sample_annotation', dynObjBoundingBoxes_1_tmp[iBox].token)
        
        # check for previous obj occurences
        if (ann['prev'] == ''):
            continue
        
        # if box has previous counterpart, reteive it
        _, box0, _ = nusc.get_sample_data(sample0['data'][sensorName], selected_anntokens=[ann['prev']])
        
        # trafo the bounding box of previous sample to rcf
        T = np.eye(4)
        T[:3,:3] = box0[0].rotation_matrix
        T[:3, 3] = box0[0].center
        
        T = np.dot(T0_s2r, T)
        box0[0].orientation = Quaternion(matrix=T[:3,:3])
        box0[0].center = T[:3, 3]
        
        # add corresponding boxes of current and previous sample to the output buffers
        dynObjBoundingBoxes_0_rcf.append(box0[0])
        dynObjBoundingBoxes_1_rcf.append(dynObjBoundingBoxes_1_tmp[iBox])
        
    #=================#
    # Interpolate The 3D Bounding Box Poses
    #=================#
    T0 = np.eye(4)
    T1 = np.eye(4)
    dynObjBoundingBoxes_01_rcf = []
    for iBox in range(len(dynObjBoundingBoxes_0_rcf)):
        # get pose of current and previous bounding boxes
        T0[:3,:3] = dynObjBoundingBoxes_0_rcf[iBox].rotation_matrix
        T0[:3, 3] = dynObjBoundingBoxes_0_rcf[iBox].center
        T1[:3,:3] = dynObjBoundingBoxes_1_rcf[iBox].rotation_matrix
        T1[:3, 3] = dynObjBoundingBoxes_1_rcf[iBox].center
        
        # initialize temp bounding box buffer
        dynObjBoundingBoxes_tmp = []
        
        # interpolate 3D poses
        T01, t01 = interpolate3DPoses(T0, T1, 0., 1., numInterPts=numInterPts)
        
        # write the interpolated poses to the temp bounding box buffer
        for iInterBox in range(numInterPts):
            if (iInterBox < numInterPts//2):
                box = dynObjBoundingBoxes_0_rcf[iBox].copy()
            else:
                box = dynObjBoundingBoxes_1_rcf[iBox].copy()
            
            box.orientation = Quaternion(matrix=T01[iInterBox][:3,:3])
            box.center = T01[iInterBox][:3, 3]
            dynObjBoundingBoxes_tmp.append(box)
            
        # append temp bounding box buffer to the overall buffer
        dynObjBoundingBoxes_01_rcf.append(dynObjBoundingBoxes_tmp)
    
    return np.asarray(dynObjBoundingBoxes_01_rcf)