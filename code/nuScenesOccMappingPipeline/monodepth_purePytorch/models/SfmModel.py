# Copyright 2020 Toyota Research Institute.  All rights reserved.

import random
import torch.nn as nn
from utils.image import flip_model, interpolate_scales
from utils.pose import Pose

def make_list(var, n=None):
    """
    Wraps the input into a list, and optionally repeats it to be size n

    Parameters
    ----------
    var : Any
        Variable to be wrapped in a list
    n : int
        How much the wrapped variable will be repeated

    Returns
    -------
    var_list : list
        List generated from var
    """
    var = var if isinstance(var, list) else [var]
    if n is None:
        return var
    else:
        assert len(var) == 1 or len(var) == n, 'Wrong list length for make_list'
        return var * n if len(var) == 1 else var


class SfmModel(nn.Module):
    """
    Model class encapsulating a pose and depth networks.

    Parameters
    ----------
    depth_net : nn.Module
        Depth network to be used
    pose_net : nn.Module
        Pose network to be used
    rotation_mode : str
        Rotation mode for the pose network
    flip_lr_prob : float
        Probability of flipping when using the depth network
    upsample_depth_maps : bool
        True if detph map scales are upsampled to highest resolution
    kwargs : dict
        Extra parameters
    """
    def __init__(self, depth_net=None, pose_net=None,
                 rotation_mode='euler', flip_lr_prob=0.0,
                 upsample_depth_maps=False, **kwargs):
        super().__init__()
        self.depth_net = depth_net
        self.pose_net = pose_net
        self.rotation_mode = rotation_mode
        self.flip_lr_prob = flip_lr_prob
        self.upsample_depth_maps = upsample_depth_maps
        self._logs = {}
        self._losses = {}

    @property
    def logs(self):
        """Return logs."""
        return self._logs

    @property
    def losses(self):
        """Return metrics."""
        return self._losses

    def add_loss(self, key, val):
        """Add a new loss to the dictionary and detaches it."""
        self._losses[key] = val.detach()

    @property
    def requires_depth_net(self):
        return True

    @property
    def requires_pose_net(self):
        return True

    @property
    def requires_gt_depth(self):
        return False

    @property
    def requires_gt_pose(self):
        return False

    def add_depth_net(self, depth_net):
        self.depth_net = depth_net

    def add_pose_net(self, pose_net):
        self.pose_net = pose_net

    def compute_inv_depths(self, image):
        """Computes inverse depth maps from single images"""
        # Randomly flip and estimate inverse depth maps
        flip_lr = random.random() < self.flip_lr_prob if self.training else False
        inv_depths = make_list(flip_model(self.depth_net, image, flip_lr))
        # If upsampling depth maps
        if self.upsample_depth_maps:
            inv_depths = interpolate_scales(
                inv_depths, mode='nearest', align_corners=None)
        # Return inverse depth maps
        return inv_depths

    def compute_poses(self, image, contexts):
        """Compute poses from image and a sequence of context images"""
        pose_vec = self.pose_net(image, contexts)
        return [Pose.from_vec(pose_vec[:, i], self.rotation_mode)
                for i in range(pose_vec.shape[1])]

    def forward(self, batch, return_logs=False):
        """
        Processes a batch.

        Parameters
        ----------
        batch : dict
            Input batch
        return_logs : bool
            True if logs are stored

        Returns
        -------
        output : dict
            Dictionary containing predicted inverse depth maps and poses
        """
        # Generate inverse depth predictions
        inv_depths = self.compute_inv_depths(batch['rgb'])
        # Generate pose predictions if available
        pose = None
        if 'rgb_context' in batch and self.pose_net is not None:
            pose = self.compute_poses(batch['rgb'],
                                      batch['rgb_context'])
        # Return output dictionary
        return {
            'inv_depths': inv_depths,
            'poses': pose,
        }
