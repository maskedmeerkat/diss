import numpy as np
from PIL import Image
import time
from pyquaternion import Quaternion
import cv2


def limit_certainty(m, u_min):
    # unknown mass cannot fall beneath u_min
    m[:, :, :-1] = (1 - u_min) * m[:, :, :-1]
    m[:, :, 2] = 1 - m[:, :, 0] - m[:, :, 1]
    return m


def compute_ev_conflict(m1, m2):
    return m1[:, :, [0]] * m2[:, :, [1]] + m1[:, :, [1]] * m2[:, :, [0]]


def yager_rule(m1, m2):
    # init the fused mass
    m = np.zeros_like(m1)
    # compute the conflict
    k = compute_ev_conflict(m1, m2)[:, :, 0]
    # fuse the masses using Yager's rule
    m[:, :, 0] = m1[:, :, 0] * m2[:, :, 0] + m1[:, :, 0] * m2[:, :, 2] + m1[:, :, 2] * m2[:, :, 0]
    m[:, :, 1] = m1[:, :, 1] * m2[:, :, 1] + m1[:, :, 1] * m2[:, :, 2] + m1[:, :, 2] * m2[:, :, 1]
    m[:, :, 2] = m1[:, :, 2] * m2[:, :, 2] + k
    return m


def yader_rule(m1, m2):
    # init the fused mass
    m = np.zeros_like(m1)
    # compute the conflict
    k = compute_ev_conflict(m1, m2)[:, :, 0]
    # fuse the masses using Yager's rule
    m[:, :, 0] = m1[:, :, 0] * m2[:, :, 0] + m1[:, :, 0] * m2[:, :, 2] + m1[:, :, 2] * m2[:, :, 0] + k / 2
    m[:, :, 1] = m1[:, :, 1] * m2[:, :, 1] + m1[:, :, 1] * m2[:, :, 2] + m1[:, :, 2] * m2[:, :, 1] + k / 2
    m[:, :, 2] = m1[:, :, 2] * m2[:, :, 2]
    return m


def dempster_rule(m1, m2):
    # init the fused mass
    m = np.zeros_like(m1)
    # compute the conflict
    k = compute_ev_conflict(m1, m2)[:, :, 0]
    # fuse the masses using Yager's rule
    m[:, :, 0] = (m1[:, :, 0] * m2[:, :, 0] + m1[:, :, 0] * m2[:, :, 2] + m1[:, :, 2] * m2[:, :, 0]) / (1 - k)
    m[:, :, 1] = (m1[:, :, 1] * m2[:, :, 1] + m1[:, :, 1] * m2[:, :, 2] + m1[:, :, 2] * m2[:, :, 1]) / (1 - k)
    m[:, :, 2] = (m1[:, :, 2] * m2[:, :, 2]) / (1 - k)
    return m


def fuseImgs(m1, m2, comb_rule=0, entropy_scaling=False, u_min=0.3, eps=1e-4, storeRescaledPth=None):
    """
    Fuses two masses defined as m = [{fr},{oc},{fr,oc}] according to the
    Dempster's Rule of Combination.
    """
    if entropy_scaling:
        # FIRST: rescale mass to have at least uMin unknown mass
        m1 = limit_certainty(m1, u_min)

        # SECOND: rescale mass to account for redundant information
        # current conflict
        k = compute_ev_conflict(m1, m2)
        # difference in certainty
        du = m2[:, :, [2]] - m1[:, :, [2]]
        du[du < 0.] = 0.
        # difference in information
        h1_2 = np.clip(1. * (du + k), 0., 1.)

        # limit the difference in information
        h1_2_max = np.clip((u_min - m2[:, :, [2]]) / (m2[:, :, [2]] * m1[:, :, [2]] - m2[:, :, [2]] + k + eps), 0., 1.)
        limit_condition = (m2[:, :, [2]] * m1[:, :, [2]] - m2[:, :, [2]] + k) < 0
        h1_2[limit_condition] = np.minimum(h1_2[limit_condition], h1_2_max[limit_condition])
        h1_2[h1_2 < 0.001] = 0

        # move redundant info from fr & oc to unknown
        m1[:, :, :-1] *= h1_2
        m1[:, :, 2] = 1. - np.sum(m1[:, :, :-1], axis=2)

        if not(storeRescaledPth is None):
            Image.fromarray( (m1*255).astype(np.uint8) ).save(storeRescaledPth)

    # Yager
    m = np.zeros_like(m1)
    if comb_rule == 0:
        m = yager_rule(m1, m2)
    # YaDer (conflict equally assigned to fr & occ classes)
    elif comb_rule == 1:
        m = yader_rule(m1, m2)
    # use Yager or YaDer depending on unknown mass
    elif comb_rule == 2:
        # compute both yager & yader rule (i know, inefficient...so what. sue me!)
        m_yager = yager_rule(m1, m2)
        m_yader = yader_rule(m1, m2)
        # take yager for mu > u_min and yader elsewise
        m = m_yager
        m[m2[:, :, 2] <= u_min, :] = m_yader[m2[:, :, 2] <= u_min, :]
    # Dempster
    else:
        m = dempster_rule(m1, m2)

    # norm masses 
    m /= np.sum(m, axis=2, keepdims=True)

    return m


# ===========================#
def pc2VehicleCenteredBevImg(pc, isStat, pDim=512, mDim=40., dynDiscount=1.):
    # trafo vcf -> icf     
    pc[1, :] = -pc[1, :]
    pc = pc[[1, 0], :]
    pc = pc + np.array([[mDim / 2, mDim / 2]]).T

    # trafo icf -> pcf
    pc = (pc * pDim / mDim).astype(np.int)

    # filter out all pts outside image dimension
    mask = (pc[0, :] >= 0) * (pc[1, :] >= 0) * (pc[0, :] < pDim) * (pc[1, :] < pDim)
    pc = pc[:, mask]
    isStat = isStat[mask]

    # create brid's eye view lidar image
    bevImg = np.zeros((pDim, pDim), dtype=np.uint8)

    # mark detections in the image
    for iPt in range(pc.shape[1]):
        if isStat[iPt]:
            bevImg[pc[0, iPt], pc[1, iPt]] = 255
        elif bevImg[pc[0, iPt], pc[1, iPt]] == 0:
            bevImg[pc[0, iPt], pc[1, iPt]] = int(128 * dynDiscount)
    return bevImg


# ===========================#
def vehicleCenteredBevImg2Pc(bevImg, pDim=512, mDim=40.):
    # trafo xy coordinates of marked cells back to meters -> discretized pointcloud
    pcDisc = np.transpose(np.transpose((bevImg > 0).nonzero())).astype(np.float)
    pcDisc = np.append(pcDisc, np.zeros((1, pcDisc.shape[1])), axis=0)
    pcDisc *= mDim / pDim

    # trafo icf -> vcf
    pcDisc = pcDisc - np.array([[mDim / 2, mDim / 2, 0]]).T
    pcDisc = pcDisc[[1, 0, 2], :]
    pcDisc[1, :] = -pcDisc[1, :]

    return pcDisc


# ===========================#
def pc2PolarBevImg(pc, isStat, pDim, aDim, mDim):
    # build polar detection image
    polDetImg = np.zeros((int(pDim / 2), aDim), dtype=np.uint8)
    # mark each point in the polar detection image
    for i in range(pc.shape[1]):
        # trafo cartesian to polar coordinates
        iAngle = np.floor((np.arctan2(pc[1, i], pc[0, i]) + np.pi) / np.pi / 2 * aDim).astype(np.int)
        # iRange = np.floor(pDim/2 - np.sqrt(pc[0,i]**2 + pc[1,i]**2) / mDim * pDim/2).astype(np.int)
        iRange = np.floor(np.sqrt(pc[0, i] ** 2 + pc[1, i] ** 2) / mDim * pDim).astype(np.int)
        # check, if image boundaries are met
        if ((iRange > 0) and (iRange < polDetImg.shape[0]) and
                (iAngle > 0) and (iAngle < polDetImg.shape[1])):
            # static
            if (isStat[i]):
                polDetImg[iRange, iAngle] = 255
            else:
                polDetImg[iRange, iAngle] = 128
    polDetImg = np.flip(polDetImg, axis=0)
    return polDetImg


# #===========================#
# def cart2polarImg(cartImg, mDim, pDim, aDim):
#     # init polar image
#     if (len(cartImg.shape) == 3):
#         polImg = np.zeros((int(pDim/2),aDim,3),dtype=np.uint8)
#     else:
#         polImg = np.zeros((int(pDim/2),aDim),dtype=np.uint8)

#     # get all indices of x and y in icf
#     iX, iY = np.meshgrid(np.arange(cartImg.shape[0]), np.arange(cartImg.shape[1]))
#     iX = iX.flatten()
#     iY = iY.flatten()

#     # get pixel value
#     color = cartImg[iX,iY]

#     # trafo icf -> vcf
#     iX = iX/pDim*mDim - mDim/2
#     iY = mDim/2 - iY/pDim*mDim

#     # build point cloud
#     pc = np.append(iX[np.newaxis,:],iY[np.newaxis,:],axis=0)

#     # mark each point in the polar detection image
#     for i in range(pc.shape[1]):
#         # trafo cartesian to polar coordinates
#         # iAngle = np.floor((np.arctan2(pc[1,i],pc[0,i]) + np.pi)/np.pi/2*aDim).astype(np.int)
#         iAngle = np.floor((np.arctan2(-pc[0,i],pc[1,i]) + np.pi)/np.pi/2*aDim).astype(np.int)
#         iRange = np.floor(np.sqrt(pc[0,i]**2 + pc[1,i]**2) / mDim * pDim).astype(np.int)        
#         # check, if image boundaries are met
#         if ((iRange>0) and (iRange<polImg.shape[0]) and 
#             (iAngle>0) and (iAngle<polImg.shape[1])):
#             # static
#             if (len(cartImg.shape) == 3):
#                 polImg[iRange,iAngle,:] = color[i,:]
#             else:
#                 polImg[iRange,iAngle] = color[i]
#     polImg = np.flip(polImg,axis=0)
#     polImg = np.flip(polImg,axis=1)
#     return polImg

# ===========================#
def cart2polarImg(cartImg, mDim, pDim, aDim):
    # init polar image
    if (len(cartImg.shape) == 3):
        polImg = np.zeros((int(pDim / 2), aDim, 3))
        polImg[..., 2] = 1.
    else:
        polImg = np.zeros((int(pDim / 2), aDim))
    # define trafo vcf -> icf
    R_V2I = np.array([[1, 0], [0, -1]])
    t_V2I = np.ones((2, 1)) * mDim / 2
    # get all indices of ranges and angles in image coordinates
    iRanges, iAngles = np.meshgrid(np.arange(polImg.shape[0]), np.arange(polImg.shape[1]))
    iRanges = iRanges.flatten()
    iAngles = iAngles.flatten()
    # get pixels range and angles
    ranges = iRanges / pDim * mDim  # *2
    angles = iAngles / aDim * 2 * np.pi
    # trafo polar to cartesian coordinates
    pts = np.array([[ranges * np.cos(angles)],
                    [ranges * np.sin(angles)]])[:, 0, :]
    # trafo from vcf -> icf
    pts = np.dot(R_V2I, pts) + t_V2I
    # trafo icf -> pcf
    iCols = np.floor(pts[0, :] / mDim * pDim).astype(np.int)
    iRows = np.floor(pts[1, :] / mDim * pDim).astype(np.int)
    # check boundaries
    iRows_mask = np.logical_and((iRows >= 0), (iRows < cartImg.shape[0]))
    iCols_mask = np.logical_and((iCols >= 0), (iCols < cartImg.shape[1]))
    mask = np.logical_and(iRows_mask, iCols_mask)
    iRows = iRows[mask]
    iCols = iCols[mask]
    iRanges = iRanges[mask]
    iAngles = iAngles[mask]

    if (len(cartImg.shape) == 3):
        polImg[iRanges, iAngles, :] = cartImg[iRows, iCols, :]
    else:
        polImg[iRanges, iAngles] = cartImg[iRows, iCols]
    polImg = np.flip(polImg, axis=0)

    return polImg


# ===========================#
def polar2cartImg(polarImg, mDim, pDim, aDim, offset=np.zeros((2, 1))):
    polarImg = np.flip(polarImg, axis=0)
    # init cartesian image
    if (len(polarImg.shape) == 3):
        cartImg = np.zeros((pDim, pDim, 3))
        cartImg[:, :, 2] = 1.
    else:
        cartImg = np.zeros((pDim, pDim, 1))
    # define trafo vcf -> icf
    R_V2I = np.array([[1, 0], [0, -1]])
    t_V2I = np.ones((2, 1)) * mDim / 2
    # get all indices of ranges and angles in image coordinates
    iRanges, iAngles = np.meshgrid(np.arange(polarImg.shape[0]), np.arange(polarImg.shape[1]))
    iRanges = iRanges.flatten()
    iAngles = iAngles.flatten()
    # get pixels range and angles
    ranges = iRanges / pDim * mDim  # *2
    angles = iAngles / aDim * 2 * np.pi
    # trafo polar to cartesian coordinates
    pts = np.array([[ranges * np.cos(angles)],
                    [ranges * np.sin(angles)]])[:, 0, :]
    # add offset
    pts -= offset
    # trafo from vcf -> icf
    pts = np.dot(R_V2I, pts) + t_V2I
    # trafo icf -> pcf
    iCols = np.floor(pts[0, :] / mDim * pDim).astype(np.int)
    iRows = np.floor(pts[1, :] / mDim * pDim).astype(np.int)
    # check boundaries
    iRows_mask = np.logical_and((iRows >= 0), (iRows < cartImg.shape[0]))
    iCols_mask = np.logical_and((iCols >= 0), (iCols < cartImg.shape[1]))
    mask = np.logical_and(iRows_mask, iCols_mask)
    iRows = iRows[mask]
    iCols = iCols[mask]
    iRanges = iRanges[mask]
    iAngles = iAngles[mask]
    # in case unasigned
    if (cartImg[iRows, iCols, :] == np.array([0, 0, 1])).all():
        cartImg[iRows, iCols, :] = polarImg[iRanges, iAngles, :]
    # higher priority to keep occupied pixels
    elif ((cartImg[iRows, iCols, 1] == 0) and (polarImg[iRanges, iAngles, 1] > 0)):
        cartImg[iRows, iCols, :] = polarImg[iRanges, iAngles, :]
    # higher priority to keep dynamic pixels
    elif ((cartImg[iRows, iCols, 0] == 0) and (cartImg[iRows, iCols, 1] > 0) and
          (polarImg[iRanges, iAngles, 0] > 0) and (polarImg[iRanges, iAngles, 1] > 0)):
        cartImg[iRows, iCols, :] = polarImg[iRanges, iAngles, :]
    return cartImg


# ===========================#
def rayCastingBev(buffers, pDim, mDim, aDim, pF, pO, pD, numColsPerCone, vPose_ref, t_ref, storDir,
                  lidarFlag=False, noDynFlag=False, allPrevFlag=False):
    t0 = time.time()
    maxNumColsPerCone = np.max(numColsPerCone)
    # compute the free space cone probability distribution (triangular impuls with min = -pF)
    pFs = np.zeros((numColsPerCone.shape[0], maxNumColsPerCone + 1))
    for iNumCols in range(numColsPerCone.shape[0]):
        for i in range(pFs.shape[1]):
            if (i <= numColsPerCone[iNumCols] / 2):
                pFs[iNumCols, i] = 1 / (numColsPerCone[iNumCols] / 2 + 1) * (i + 1);
            else:
                pFs[iNumCols, i] = pFs[iNumCols, numColsPerCone[iNumCols] - i]

    cartImg = np.zeros((pDim, pDim, 3))
    cartImg[:, :, 2] = 1.

    # create ism for each sensor
    for iSens in range(len(buffers)):
        # current sensor pose
        sPosit = buffers[iSens][-1]["sPose"][:2, np.newaxis]
        vPose = buffers[iSens][-1]["vPose"][:2, np.newaxis]

        # create polar detection images
        polarAccDetImgBool = np.zeros((int(pDim / 2), aDim)).astype(np.bool)
        polarAccDetImg = np.zeros((int(pDim / 2), aDim))
        polarCurrDetImg = np.zeros((int(pDim / 2), aDim))
        polarCurrDetImgBool = np.zeros((int(pDim / 2), aDim)).astype(np.bool)
        for iSens_, buffer in enumerate(buffers):
            for iTime, pcWithPose in enumerate(buffer):
                # trafo rcf -> vcf
                pc = pcWithPose["pc"][:2, :] - vPose  # vPose_ref[:2,np.newaxis]
                # trafo vcf -> current scf
                pc = pc - sPosit
                # create polar bev detection image
                polarDetImg = pc2PolarBevImg(pc, pcWithPose["pc"][2, :], pDim, aDim, mDim)
                # trafo to boolean array
                polarDetImg_ = (polarDetImg > 0)
                # add current detection image to the accumulated image
                if (polarAccDetImgBool.shape[0] == 1):
                    polarAccDetImgBool = polarDetImg_.copy()
                    polarAccDetImg = polarDetImg.copy()
                else:
                    polarAccDetImgBool = np.logical_or(polarAccDetImgBool, polarDetImg_)
                    polarAccDetImg[polarDetImg > 0] = polarDetImg[polarDetImg > 0]
                # if pc is latest pc of current sensor
                if (iSens_ == iSens) and (iTime == (len(buffer) - 1)):
                    polarCurrDetImg = polarDetImg.copy()
                    polarCurrDetImgBool = (polarCurrDetImg > 0)
        # pad the polarDetImg
        numPad = maxNumColsPerCone // 2
        polarAccDetImgBool = np.pad(polarAccDetImgBool, ((0, 0), (numPad, numPad)), mode="wrap")
        polarCurrDetImgBool = np.pad(polarCurrDetImgBool, ((0, 0), (numPad, numPad)), mode="wrap")

        # mark area around detection as occupied
        blockColWdith = maxNumColsPerCone // 4
        if not (lidarFlag):
            detIdxs = np.argwhere(polarAccDetImgBool)
            if (len(detIdxs) > 0):
                for detIdx in detIdxs:
                    polarAccDetImgBool[detIdx[0],
                    np.clip(detIdx[1] - blockColWdith, 0, None):detIdx[1] + 1 + blockColWdith] = True

        # cast inverse detection models
        polarRayIsmImg = np.zeros((polarAccDetImgBool.shape[0], polarAccDetImgBool.shape[1]))
        # go thru each column and mark free space
        for iCol in range(polarAccDetImgBool.shape[1] - maxNumColsPerCone):
            # if there is a detection in current column or lidar ism is computed
            if np.sum(polarCurrDetImgBool[:, iCol + numPad]) or lidarFlag:
                # cast a free space cone
                for iNumCols in range(numColsPerCone.shape[0]):
                    # find the earliest row in which the cone hits a detection
                    firstRowHit = np.where(np.sum(polarAccDetImgBool[:, iCol:iCol + numColsPerCone[iNumCols]], axis=1))[
                        0]
                    for i in range(pFs.shape[1]):
                        if (len(firstRowHit) > 0):
                            polarRayIsmImg[firstRowHit[-1]:, iCol + i] -= pFs[iNumCols, i] * pF[iNumCols]
                        else:
                            polarRayIsmImg[:, iCol + i] -= pFs[iNumCols, i] * pF[iNumCols]

        # remove padding
        polarRayIsmImg = polarRayIsmImg[:, numPad:-numPad]
        # trafo to evidential
        polarRayIsmImg_ = np.zeros((polarRayIsmImg.shape[0], polarRayIsmImg.shape[1], 3))
        polarRayIsmImg_[:, :, 2] = 1
        # mark free
        frIdx = polarRayIsmImg < 0
        polarRayIsmImg_[frIdx, 0] = np.clip(-polarRayIsmImg[frIdx], 0, 1)
        # mark occupied
        # remove all occluded detections
        # build an image with each column number increasing from 0 to maxNumRow
        rowIdexImg = np.tile(np.arange(0, polarAccDetImg.shape[0], 1), (polarAccDetImg.shape[1], 1)).T
        # put all indices to zeros where no detection is and find the first occurance for each column
        rowIdx = np.argmax(rowIdexImg * (polarAccDetImg > 0).astype(int), axis=0)
        colIdx = np.arange(polarAccDetImg.shape[1])
        # remove all rows, where no detection was found at all
        mask = (rowIdx > 0)
        rowIdx = rowIdx[mask]
        colIdx = colIdx[mask]
        # remove occluded detections
        for i in range(colIdx.shape[0]):
            polarAccDetImg[:rowIdx[i], colIdx[i]] = 0

        ocIdx = polarAccDetImg == 255  # polarCurrDetImg==255
        polarRayIsmImg_[ocIdx, 1] = np.clip(pO, 0, 1)
        polarRayIsmImg_[ocIdx, 0] = 0.
        # mark dynamic
        if not (noDynFlag):
            dynIdx = polarAccDetImg == 128  # polarCurrDetImg==128
            polarRayIsmImg_[dynIdx, :2] = np.clip(pD, 0, 0.5)
        # normalize the unknown portion
        polarRayIsmImg_[:, :, 2] = 1 - polarRayIsmImg_[:, :, 0] - polarRayIsmImg_[:, :, 1]
        # for testing: only show sensor positions
        # polarRayIsmImg_[...,0] = 0.
        # polarRayIsmImg_[...,1] = 0.
        # polarRayIsmImg_[...,2] = 1.
        # polarRayIsmImg_[-1,:,0] = 1.
        # trafo polar to cartesian
        cartImg_ = polar2cartImg(polarRayIsmImg_, mDim, pDim, aDim, offset=sPosit + vPose - vPose_ref[:2, np.newaxis])
        # fuse current sensors ism into accumulated ism
        cartImg = fuseImgs(cartImg_, cartImg)
    # flip car image
    cartImg = np.flip(cartImg, axis=0)
    cartImg = np.flip(cartImg, axis=1)

    # store bev image
    if (len(storDir)):
        cartImg_ = Image.fromarray((cartImg * 255).astype(np.uint8))
        cartImg_.save(storDir + storDir.split("/")[-2] + "__{:}.png".format(t_ref))
    return cartImg


# ===========================#
def initGlobalImg(nusc, scene, mDim, pDim):
    # trafo from wcf -> rcf
    sample = nusc.get('sample', scene['first_sample_token'])
    sample_data = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
    ego_pose = nusc.get('ego_pose', sample_data['ego_pose_token'])
    T_w2r = np.eye(4)
    T_w2r[:3, :3] = Quaternion(ego_pose['rotation']).rotation_matrix
    T_w2r[:3, 3] = np.array(ego_pose['translation'])
    T_w2r = np.linalg.inv(T_w2r)

    # obtain all vehicle positions in reference coordinates
    p_all = np.zeros((2, scene['nbr_samples']))
    for iSample in range(scene['nbr_samples'] - 1):
        # get ego position in homogeneous coordinates
        sample_data = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        ego_pose = nusc.get('ego_pose', sample_data['ego_pose_token'])
        p_ = np.append(np.array(ego_pose['translation']), 1)

        # trafo ego position into ref coordinates
        p_all[:, iSample] = np.matmul(T_w2r, p_)[:2]

        sample = nusc.get('sample', sample['next'])

    # find image borders by finding max vehicle positions and adding the perception range
    xMax = np.max(p_all[0, :]) + mDim // 2 + 1
    xMin = np.min(p_all[0, :]) - mDim // 2 - 1
    yMax = np.max(p_all[1, :]) + mDim // 2 + 1
    yMin = np.min(p_all[1, :]) - mDim // 2 - 1

    xDim = ((xMax - xMin) * pDim / mDim).astype(np.int)
    yDim = ((yMax - yMin) * pDim / mDim).astype(np.int)

    t_r2i = -np.array([[xMin], [yMax]])
    detMap = np.zeros((int(yDim), int(xDim)))
    ismMap = np.zeros((int(yDim), int(xDim), 3))
    ismMap[:, :, 2] = 1.
    return detMap, ismMap, t_r2i


# ===========================#
def markInGlobalImg(pc, img, t_r2i, mDim, pDim):
    isStat = pc[2, :].astype(bool)

    # trafo rcf -> icf
    R_r2i = np.array([[0., -1.],
                      [1., 0.]])
    pc = np.matmul(R_r2i, pc[:2, :] + t_r2i)

    # trafo icf -> pcf
    pc = (pc * pDim / mDim).astype(int)

    # filter out all pts outside image dimension
    mask = (pc[0, :] >= 0) * (pc[1, :] >= 0) * (pc[0, :] < img.shape[0]) * (pc[1, :] < img.shape[1])
    pc = pc[:, mask]
    isStat = isStat[mask]

    # mark detections in the image
    img[pc[0, :], pc[1, :]] = 0.5
    img[pc[0, isStat], pc[1, isStat]] = 1.

    return img


# ===========================#
def discEvImg(img, pF, pO, pD):
    discImg = np.zeros_like(img)

    # discretize the image according to threshold
    discImg[img[..., 0] > pF, 0] = 255
    discImg[img[..., 1] > pO, 1] = 255

    # in case pixels are both free & occupied after discretization
    mask = np.zeros_like(img[..., 0])
    mask = np.logical_or(mask, discImg[..., 0] + discImg[..., 1] == 500)
    frMask = np.logical_and(mask, img[..., 0] >= img[..., 1])
    ocMask = np.logical_and(mask, img[..., 0] < img[..., 1])

    discImg[frMask, 0] = 255
    discImg[ocMask, 1] = 255

    discImg[..., 2] = 255 - discImg[..., 0] - discImg[..., 1]

    return discImg


# ===========================#
def saveMapPatches(mapImg, t_r2i, vPoses, ts, storePath, mDim, pDim, bboxes):
    # trafo pose to global image coordinates
    R_r2i = np.array([[0., -1.],
                      [1., 0.]])

    for iRef in range(len(vPoses)):
        imgCenterPt = np.matmul(R_r2i, vPoses[iRef][:2, np.newaxis] + t_r2i)[:, 0]
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
        if (xLim[1] > mapImg.shape[0]):
            dx = mapImg.shape[0] - xLim[1]
            xLim[1] += dx
            xLim_[1] += dx
        if (yLim[1] > mapImg.shape[1]):
            dy = mapImg.shape[1] - yLim[1]
            yLim[1] += dy
            yLim_[1] += dy

        bevImg = np.zeros((pDim, pDim, 3))
        bevImg[:, :, 2] = 1
        bevImg[xLim_[0]:xLim_[1], yLim_[0]:yLim_[1], :] = \
            mapImg[xLim[0]:xLim[1], yLim[0]:yLim[1], :].copy()

        # mark dynamic objects
        if (len(bboxes) > 0):
            for bbox in bboxes[iRef]:
                center = bbox.center[:2] - vPoses[iRef][:2]
                if ((center[0] > -mDim / 2) and (center[0] < mDim / 2) and
                        (center[1] > -mDim / 2) and (center[1] < mDim / 2)):
                    # define rectangle from bounding box
                    center = np.matmul(np.array([[1., 0.], [0., -1.]]), center[:2, np.newaxis])[:, 0]
                    center = (center * pDim / mDim).astype(int) + np.ones(2) * pDim // 2
                    R = bbox.orientation.rotation_matrix
                    angle = (np.arctan2(R[0, 0], R[1, 0])) / np.pi * 180
                    wlh = (bbox.wlh * pDim / mDim).astype(int)
                    rect = ((center[0], center[1]), (wlh[0], wlh[1]), angle)

                    # get box points
                    box = np.int0(cv2.boxPoints(rect))

                    # draw the bounding box into the image
                    bevImg = cv2.drawContours(bevImg, [box], 0, [0.5, 0.5, 0], cv2.FILLED)

        bevImg = Image.fromarray((bevImg * 255).astype(np.uint8))
        bevImg.save(storePath + storePath.split("/")[-2] + "__{:}.png".format(ts[iRef]))

    # ===========================#


def createObservedAreaMap(mapImg, t_r2i, vPoses, mDim, pDim, aDim):
    # create free space area
    polarRayIsmImg = np.ones((int(pDim / 2), aDim, 1))
    bevImg = polar2cartImg(polarRayIsmImg, mDim, pDim, aDim)[..., 0]

    # trafo pose to global image coordinates
    R_r2i = np.array([[0., -1.],
                      [1., 0.]])

    for iRef in range(len(vPoses)):
        imgCenterPt = np.matmul(R_r2i, vPoses[iRef][:2, np.newaxis] + t_r2i)[:, 0]
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
        if (xLim[1] > mapImg.shape[0]):
            dx = mapImg.shape[0] - xLim[1]
            xLim[1] += dx
            xLim_[1] += dx
        if (yLim[1] > mapImg.shape[1]):
            dy = mapImg.shape[1] - yLim[1]
            yLim[1] += dy
            yLim_[1] += dy

        mapImg[xLim[0]:xLim[1], yLim[0]:yLim[1]] += bevImg

    return np.clip(mapImg, 0., 1.)
