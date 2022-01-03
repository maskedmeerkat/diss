import cv2
import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud, RadarPointCloud
from nuscenes.utils.geometry_utils import view_points
import os
import os.path as osp
from PIL import Image
from pyquaternion import Quaternion
import sys
from collections import Counter
from tqdm import tqdm

from packnet_sfm import ModelWrapper
from packnet_sfm.datasets.augmentations import resize_image, to_tensor
from packnet_sfm.utils.horovod import hvd_init, rank, world_size, print0
from packnet_sfm.utils.image import load_image
from packnet_sfm.utils.config import parse_test_file
from packnet_sfm.utils.load import set_debug
from packnet_sfm.utils.depth import viz_inv_depth, inv2depth
from packnet_sfm.utils.logging import pcolor
import torch
from torchvision import transforms, utils, datasets

data_transforms = {'predict': transforms.Compose([ transforms.ToTensor() ])}

#==============================================================================
# Prepare PackNet-SfM for Inference
#==============================================================================
# trained model checkpoint
ckpt_file = './pre_trained_models/PackNet_NuSc_trainedWith_gtDepth_sweepDist_1.ckpt'

# Initialize horovod    
hvd_init()

# Parse arguments
config, state_dict = parse_test_file(ckpt_file)
image_shape = config.datasets.augmentation.image_shape

# Initialize model wrapper from checkpoint arguments
model_wrapper = ModelWrapper(config, load_datasets=False)

# Restore monodepth_model state
model_wrapper.load_state_dict(state_dict)

# Send model to GPU if available
if torch.cuda.is_available():
    model_wrapper = model_wrapper.to('cuda:{}'.format(rank()))

    
#==============================================================================
# MonoDepth Prediction
#==============================================================================
camNames = ['CAM_FRONT','CAM_FRONT_RIGHT','CAM_FRONT_LEFT',
            'CAM_BACK','CAM_BACK_RIGHT','CAM_BACK_LEFT'] 

# NuScenes path
DATA_DIR = '../_DATASETS_/NuScenes/samples/'    

# batch size
batchSize = 16

for camName in camNames:
    print(camName)
        
    # load all files for current camera
    camImgDir = DATA_DIR + camName + "/"
    # camDepthDir = DATA_DIR + "DEPTH_" + camName + "/"
    camDepthDir = "./logs/"
    
    # camFiles = [fileName for fileName in os.listdir(camImgDir) if fileName.endswith(".jpg")]
    # print(len(camFiles))
    
    # dataset = {'predict' : datasets.ImageFolder(camImgDir, data_transforms['predict'])}
    # dataloader = {'predict': torch.utils.data.DataLoader(dataset['predict'], batch_size = batchSize, shuffle=False, num_workers=4)}
    
    # for inputs, labels in dataloader['predict']:
    #     inputs = inputs.to(device)
    #     output = model_wrapper.depth(inputs)
    #     depthImg = inv2depth(invDepthImg)
    #     depthImg = depthImg.detach().cpu().numpy()
    #     print(depthImg.shape)
    #     sys.exit(0)
    
    camFiles = [fileName for fileName in os.listdir(camImgDir) if fileName.endswith(".jpg")]
    
    # loop over all batches of camera images for current camera
    numBatches = len(camFiles) // batchSize + 1
    for iBatch in tqdm(range(numBatches)):
        # get cam files of current batch
        currCamFiles = camFiles[iBatch*batchSize:(iBatch+1)*batchSize]
        
        # load batch of cam imgs
        batchImgs = np.zeros((len(currCamFiles),) + image_shape + (3,))
        for iBatchImg, currCamFile in enumerate(currCamFiles):
            img = Image.open(osp.join(camImgDir, currCamFile))
            
            # Resize and to tensor
            batchImgs[iBatchImg,...] = resize_image(img, image_shape)
        
        print(batchImgs.shape)
        # batchImgs = to_tensor(batchImgs).unsqueeze(0)
        batchImgs = to_tensor(batchImgs)
    
        # Send image to GPU if available
        if torch.cuda.is_available():
            batchImgs = batchImgs.to('cuda:{}'.format(rank()))
    
        # Inverse depth inference
        invDepthImg = model_wrapper.depth(batchImgs)[0]
        depthImg = inv2depth(invDepthImg)
        # depthImg = depthImg.detach().cpu().numpy()[0,0,...,np.newaxis]
        depthImg = depthImg.detach().cpu().numpy()
        print(depthImg.shape)
        sys.exit(0)
        # np.save(camDepthDir+camFile[:-4]+".npy", depthImg)
    print("")