# Copyright 2020 Toyota Research Institute.  All rights reserved.
import torch
import importlib
from inspect import signature
from collections import OrderedDict


class ModelWrapper(torch.nn.Module):
    """
    Top-level torch.nn.Module wrapper around a SfmModel (pose+depth networks).
    Designed to use models with high-level Trainer classes (cf. trainers/).

    Parameters
    ----------
    config : CfgNode
        Model configuration (cf. configs/default_config.py)
    """

    def __init__(self, config, resume=None, logger=None):
        super().__init__()

        # Store configuration, checkpoint and logger
        self.config = config
        self.logger = logger
        self.resume = resume

        # Model, optimizers, schedulers and datasets are None for now
        self.model = self.optimizer = self.scheduler = None
        self.train_dataset = self.validation_dataset = self.test_dataset = None
        self.current_epoch = 0

        # Prepare model
        self.prepare_model(resume)

        # Preparations done
        self.config.prepared = True

    def prepare_model(self, resume=None):
        """Prepare self.model (incl. loading previous state)"""
        print('### Preparing Model')
        self.model = setup_model(self.config.model, self.config.prepared)
        # Resume model if available
        if resume:
            print('### Resuming from {}'.format(resume['file']))
            self.model = load_network(
                self.model, resume['state_dict'], 'model')
            if 'epoch' in resume:
                self.current_epoch = resume['epoch']

    @property
    def depth_net(self):
        """Returns depth network."""
        return self.model.depth_net

    @property
    def pose_net(self):
        """Returns pose network."""
        return self.model.pose_net

    def forward(self, *args, **kwargs):
        """Runs the model and returns the output."""
        assert self.model is not None, 'Model not defined'
        return self.model(*args, **kwargs)

    def depth(self, *args, **kwargs):
        """Runs the pose network and returns the output."""
        assert self.depth_net is not None, 'Depth network not defined'
        return self.depth_net(*args, **kwargs)

    def pose(self, *args, **kwargs):
        """Runs the depth network and returns the output."""
        assert self.pose_net is not None, 'Pose network not defined'
        return self.pose_net(*args, **kwargs)


def setup_depth_net(config, prepared, **kwargs):
    """
    Create a depth network

    Parameters
    ----------
    config : CfgNode
        Network configuration
    prepared : bool
        True if the network has been prepared before
    kwargs : dict
        Extra parameters for the network

    Returns
    -------
    depth_net : nn.Module
        Create depth network
    """
    print('DepthNet: %s' % config.name)
    depth_net = load_class_args_create(config.name,
        paths=['networks.depth',],
        args={**config, **kwargs},
    )
    if not prepared and config.checkpoint_path != '':
        depth_net = load_network(depth_net, config.checkpoint_path,
                                 ['depth_net', 'disp_network'])
    return depth_net


def setup_pose_net(config, prepared, **kwargs):
    """
    Create a pose network

    Parameters
    ----------
    config : CfgNode
        Network configuration
    prepared : bool
        True if the network has been prepared before
    kwargs : dict
        Extra parameters for the network

    Returns
    -------
    pose_net : nn.Module
        Created pose network
    """
    print('PoseNet: %s' % config.name)
    pose_net = load_class_args_create(config.name,
        paths=['networks.pose',],
        args={**config, **kwargs},
    )
    if not prepared and config.checkpoint_path != '':
        pose_net = load_network(pose_net, config.checkpoint_path,
                                ['pose_net', 'pose_network'])
    return pose_net


def setup_model(config, prepared, **kwargs):
    """
    Create a model

    Parameters
    ----------
    config : CfgNode
        Model configuration (cf. configs/default_config.py)
    prepared : bool
        True if the model has been prepared before
    kwargs : dict
        Extra parameters for the model

    Returns
    -------
    model : nn.Module
        Created model
    """
    print('Model: %s' % config.name)
    model = load_class(config.name, paths=['models',])(
        **{**config.loss, **kwargs})
    # Add depth network if required
    if model.requires_depth_net:
        model.add_depth_net(setup_depth_net(config.depth_net, prepared))
    # Add pose network if required
    if model.requires_pose_net:
        model.add_pose_net(setup_pose_net(config.pose_net, prepared))
    # If a checkpoint is provided, load pretrained model
    if not prepared and config.checkpoint_path != '':
        model = load_network(model, config.checkpoint_path, 'model')
    # Return model
    return model


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


def load_class(filename, paths, concat=True):
    """
    Look for a file in different locations and return its method with the same name
    Optionally, you can use concat to search in path.filename instead

    Parameters
    ----------
    filename : str
        Name of the file we are searching for
    paths : str or list of str
        Folders in which the file will be searched
    concat : bool
        Flag to concatenate filename to each path during the search

    Returns
    -------
    method : Function
        Loaded method
    """
    # for each path in paths
    for path in make_list(paths):
        # Create full path
        full_path = '{}.{}'.format(path, filename) if concat else path
        if importlib.util.find_spec(full_path):
            # Return method with same name as the file
            return getattr(importlib.import_module(full_path), filename)
    raise ValueError('Unknown class {}'.format(filename))
    
def filter_args(func, keys):
    """
    Filters a dictionary so it only contains keys that are arguments of a function

    Parameters
    ----------
    func : Function
        Function for which we are filtering the dictionary
    keys : dict
        Dictionary with keys we are filtering

    Returns
    -------
    filtered : dict
        Dictionary containing only keys that are arguments of func
    """
    filtered = {}
    sign = list(signature(func).parameters.keys())
    for k, v in {**keys}.items():
        if k in sign:
            filtered[k] = v
    return filtered    
    
def filter_args_create(func, keys):
    """
    Filters a dictionary so it only contains keys that are arguments of a function
    and creates a function with those arguments

    Parameters
    ----------
    func : Function
        Function for which we are filtering the dictionary
    keys : dict
        Dictionary with keys we are filtering

    Returns
    -------
    func : Function
        Function with filtered keys as arguments
    """
    return func(**filter_args(func, keys))    
    
def load_class_args_create(filename, paths, args={}, concat=True):
    """Loads a class (filename) and returns an instance with filtered arguments (args)"""
    class_type = load_class(filename, paths, concat)
    return filter_args_create(class_type, args)    

def same_shape(shape1, shape2):
    """
    Checks if two shapes are the same

    Parameters
    ----------
    shape1 : tuple
        First shape
    shape2 : tuple
        Second shape

    Returns
    -------
    flag : bool
        True if both shapes are the same (same length and dimensions)
    """
    if len(shape1) != len(shape2):
        return False
    for i in range(len(shape1)):
        if shape1[i] != shape2[i]:
            return False
    return True

def load_network(network, path, prefixes=''):
    """
    Loads a pretrained network

    Parameters
    ----------
    network : nn.Module
        Network that will receive the pretrained weights
    path : str
        File containing a 'state_dict' key with pretrained network weights
    prefixes : str or list of str
        Layer name prefixes to consider when loading the network

    Returns
    -------
    network : nn.Module
        Updated network with pretrained weights
    """
    prefixes = make_list(prefixes)
    # If path is a string
    if isinstance(path, str):
        saved_state_dict = torch.load(path, map_location='cpu')['state_dict']
        if path.endswith('.pth.tar'):
            saved_state_dict = backwards_state_dict(saved_state_dict)
    # If state dict is already provided
    else:
        saved_state_dict = path
    # Get network state dict
    network_state_dict = network.state_dict()

    updated_state_dict = OrderedDict()
    n, _ = 0, len(network_state_dict.keys())
    for key, val in saved_state_dict.items():
        for prefix in prefixes:
            prefix = prefix + '.'
            if prefix in key:
                idx = key.find(prefix) + len(prefix)
                key = key[idx:]
                if key in network_state_dict.keys() and \
                        same_shape(val.shape, network_state_dict[key].shape):
                    updated_state_dict[key] = val
                    n += 1

    network.load_state_dict(updated_state_dict, strict=False)
    return network

def backwards_state_dict(state_dict):
    """
    Modify the state dict of older models for backwards compatibility

    Parameters
    ----------
    state_dict : dict
        Model state dict with pretrained weights

    Returns
    -------
    state_dict : dict
        Updated model state dict with modified layer names
    """
    # List of layer names to change
    changes = (('model.model', 'model'),
               ('pose_network', 'pose_net'),
               ('disp_network', 'depth_net'))
    # Iterate over all keys and values
    updated_state_dict = OrderedDict()
    for key, val in state_dict.items():
        # Ad hoc changes due to version changes
        key = '{}.{}'.format('model', key)
        if 'disp_network' in key:
            key = key.replace('conv3.0.weight', 'conv3.weight')
            key = key.replace('conv3.0.bias', 'conv3.bias')
        # Change layer names
        for change in changes:
            key = key.replace('{}.'.format(change[0]),
                              '{}.'.format(change[1]))
        updated_state_dict[key] = val
    # Return updated state dict
    return updated_state_dict
