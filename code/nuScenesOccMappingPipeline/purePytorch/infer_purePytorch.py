from utils.config import parse_test_file
from models.model_wrapper import ModelWrapper
import torch
from utils.augmentations import resize_image, to_tensor
from PIL import Image
from utils.depth import inv2depth, viz_inv_depth
import numpy as np
import matplotlib.pyplot as plt

#==============================================================================
# Prepare PackNet-SfM for Inference
#==============================================================================
ckpt_file = './pre_trained_models/PackNet_NuSc_trainedWith_gtDepth_sweepDist_1.ckpt'

# Parse arguments
config, state_dict = parse_test_file(ckpt_file)
image_shape = config.datasets.augmentation.image_shape

# Initialize model wrapper from checkpoint arguments
model_wrapper = ModelWrapper(config)

# Restore monodepth_model state
model_wrapper.load_state_dict(state_dict)

# Send model to GPU if available
if torch.cuda.is_available():
    model_wrapper = model_wrapper.to('cuda:0')


img = Image.open("./testImg.jpg")
        
# Resize and to tensor
img_ = resize_image(img, image_shape)
img_ = to_tensor(img_).unsqueeze(0)

# Send image to GPU if available
if torch.cuda.is_available():
    img_ = img_.to('cuda:0')

# Inverse depth inference
invDepthImg = model_wrapper.depth(img_)[0]
depthImg = inv2depth(invDepthImg)
depthImg = depthImg.detach().cpu().numpy()[0,0,...,np.newaxis]

# viz inverse depth image
invDepthImg_rgb = viz_inv_depth(invDepthImg[0])
plt. figure()
plt.imshow(invDepthImg_rgb)
