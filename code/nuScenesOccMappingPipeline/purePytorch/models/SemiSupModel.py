# Copyright 2020 Toyota Research Institute.  All rights reserved.

from models.SfmModel import SfmModel
from models.SelfSupModel import SelfSupModel


class SemiSupModel(SelfSupModel):
    """
    Model that inherits a depth and pose networks, plus the self-supervised loss from
    SelfSupModel and includes a supervised loss for semi-supervision.

    Parameters
    ----------
    supervised_loss_weight : float
        Weight for the supervised loss
    kwargs : dict
        Extra parameters
    """
    def __init__(self, supervised_loss_weight=0.9, **kwargs):
        super().__init__(**kwargs)
        self.supervised_loss_weight = supervised_loss_weight

    @property
    def requires_depth_net(self):
        return True

    @property
    def requires_pose_net(self):
        return self.supervised_loss_weight < 1.

    @property
    def requires_gt_depth(self):
        return self.supervised_loss_weight > 0.

    @property
    def requires_gt_pose(self):
        return False

    def supervised_loss(self, inv_depths, gt_inv_depths,
                        return_logs=False, progress=0.0):
        """
        Calculates the supervised loss.

        Parameters
        ----------
        inv_depths : torch.Tensor [B,1,H,W]
            Predicted inverse depth maps from the original image
        gt_inv_depths : torch.Tensor [B,1,H,W]
            Ground-truth inverse depth maps from the original image
        return_logs : bool
            True if logs are stored
        progress :
            Training progress percentage

        Returns
        -------
        output : dict
            Dictionary containing a "loss" scalar a "metrics" dictionary
        """
        return self._supervised_loss(
            inv_depths, gt_inv_depths,
            return_logs=return_logs, progress=progress)

    def forward(self, batch, return_logs=False, progress=0.0):
        """
        Processes a batch.

        Parameters
        ----------
        batch : dict
            Input batch
        return_logs : bool
            True if logs are stored
        progress :
            Training progress percentage

        Returns
        -------
        output : dict
            Dictionary containing a "loss" scalar and different metrics and predictions
            for logging and downstream usage.
        """
        # If not training, no need for self-supervised loss
        return SfmModel.forward(self, batch)
