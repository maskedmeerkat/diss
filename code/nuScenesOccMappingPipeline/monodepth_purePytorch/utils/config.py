import torch
from yacs.config import CfgNode
import importlib
from collections import OrderedDict

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

def is_cfg(data):
    """Checks if data is a configuration node"""
    return type(data) == CfgNode

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

# !
def prep_dataset(config):
    """
    Expand dataset configuration to match split length

    Parameters
    ----------
    config : CfgNode
        Dataset configuration

    Returns
    -------
    config : CfgNode
        Updated dataset configuration
    """
    # If there is no dataset, do nothing
    if len(config.path) == 0:
        return config
    # Get split length and expand other arguments to the same length
    n = len(config.split)
    config.dataset = make_list(config.dataset, n)
    config.path = make_list(config.path, n)
    config.depth_type = make_list(config.depth_type, n)
    if 'repeat' in config:
        config.repeat = make_list(config.repeat, n)
    # Return updated configuration
    return config


# def set_name(config):
#     """
#     Set run name based on available information

#     Parameters
#     ----------
#     config : CfgNode
#         Model configuration

#     Returns
#     -------
#     name : str
#         Updated run name
#     """
#     # If there is a name already, do nothing
#     if config.name is not '':
#         return config.name
#     else:
#         # Create a name based on available information
#         return '{}-{}-{}'.format(
#             os.path.basename(config.default),
#             os.path.splitext(os.path.basename(config.config))[0],
#             datetime.now().strftime("%Y.%m.%d-%Hh%Mm%Ss"))


# def set_checkpoint(config):
#     """
#     Set checkpoint information

#     Parameters
#     ----------
#     config : CfgNode
#         Model configuration

#     Returns
#     -------
#     config : CfgNode
#         Updated model configuration
#     """
#     # If checkpoint is enabled
#     if config.checkpoint.filepath is not '':
#         # Create proper monitor string
#         config.checkpoint.monitor = os.path.join('{}-{}'.format(
#             prepare_dataset_prefix(config.datasets.validation,
#                                    config.checkpoint.monitor_index),
#             config.checkpoint.monitor))
#         # Join checkpoint folder with run name
#         config.checkpoint.filepath = os.path.join(
#             config.checkpoint.filepath, config.name,
#             '{epoch:02d}_{%s:.3f}' % config.checkpoint.monitor)
#         # Set s3 url
#         if config.checkpoint.s3_path is not '':
#             config.checkpoint.s3_url = s3_url(config)
#     else:
#         # If not saving checkpoint, do not sync to s3
#         config.checkpoint.s3_path = ''
#     return config.checkpoint


# def prep_logger_and_checkpoint(model):
#     """
#     Use logger and checkpoint information to update configuration

#     Parameters
#     ----------
#     model : nn.Module
#         Module to update
#     """
#     # Change run name to be the wandb assigned name
#     if model.logger and not model.config.wandb.dry_run:
#         model.config.name = model.config.wandb.name = model.logger.run_name
#         model.config.wandb.url = model.logger.run_url
#         # If we are saving models we need to update the path
#         if model.config.checkpoint.filepath is not '':
#             # Change checkpoint filepath
#             filepath = model.config.checkpoint.filepath.split('/')
#             filepath[-2] = model.config.name
#             model.config.checkpoint.filepath = '/'.join(filepath)
#             # Change callback dirpath
#             dirpath = os.path.join(os.path.dirname(
#                 model.trainer.checkpoint.dirpath), model.config.name)
#             model.trainer.checkpoint.dirpath = dirpath
#             os.makedirs(dirpath, exist_ok=True)
#             model.config.checkpoint.s3_url = s3_url(model.config)
#         # Log updated configuration
#         model.logger.log_config(model.config)


# !
def get_default_config(cfg_default):
    """Get default configuration from file"""
    config = load_class('get_cfg_defaults',
                         paths=[cfg_default.replace('/', '.')],
                         concat=False)()
    config.merge_from_list(['default', cfg_default])
    return config

# !
def merge_cfg_file(config, cfg_file=None):
    """Merge configuration file"""
    if cfg_file is not None:
        config.merge_from_file(cfg_file)
        config.merge_from_list(['config', cfg_file])
    return config

# !
def merge_cfgs(original, override):
    """
    Updates CfgNode with information from another one

    Parameters
    ----------
    original : CfgNode
        Original configuration node
    override : CfgNode
        Another configuration node used for overriding

    Returns
    -------
    updated : CfgNode
        Updated configuration node
    """
    for key, value in original.items():
        if key in override.keys():
            if is_cfg(value): # If it's a configuration node, recursion
                original[key] = merge_cfgs(original[key], override[key])
            else: # Otherwise, simply update key
                original[key] = override[key]
    return original

# !
def backwards_config(config):
    """
    Add or update configuration for backwards compatibility
    (no need for it right now, pretrained models are up-to-date with configuration files).

    Parameters
    ----------
    config : CfgNode
        Model configuration

    Returns
    -------
    config : CfgNode
        Updated model configuration
    """
    # Return updated configuration
    return config

# def parse_train_file(file):
#     """
#     Parse file for training

#     Parameters
#     ----------
#     file : str
#         File, can be either a
#         **.yaml** for a yacs configuration file or a
#         **.ckpt** for a pre-trained checkpoint file

#     Returns
#     -------
#     config : CfgNode
#         Parsed model configuration
#     ckpt : str
#         Parsed checkpoint file
#     """
#     # If it's a .yaml configuration file
#     if file.endswith('yaml'):
#         cfg_default = 'configs/default_config'
#         return parse_train_config(cfg_default, file), None
#     # If it's a .ckpt checkpoint file
#     elif file.endswith('ckpt'):
#         checkpoint = torch.load(file, map_location='cpu')
#         config = checkpoint.pop('config')
#         checkpoint['file'] = file
#         return config, checkpoint
#     # We have a problem
#     else:
#         raise ValueError('You need to provide a .yaml or .ckpt to train')


# def parse_train_config(cfg_default, cfg_file):
#     """
#     Parse model configuration for training

#     Parameters
#     ----------
#     cfg_default : str
#         Default **.py** configuration file
#     cfg_file : str
#         Configuration **.yaml** file to override the default parameters

#     Returns
#     -------
#     config : CfgNode
#         Parsed model configuration
#     """
#     # Loads default configuration
#     config = get_default_config(cfg_default)
#     # Merge configuration file
#     config = merge_cfg_file(config, cfg_file)
#     # Return prepared configuration
#     return prepare_train_config(config)


# def prepare_train_config(config):
#     """
#     Prepare model configuration for training

#     Parameters
#     ----------
#     config : CfgNode
#         Model configuration

#     Returns
#     -------
#     config : CfgNode
#         Prepared model configuration
#     """
#     # If arguments have already been prepared, don't prepare
#     if config.prepared:
#         return config

#     # Asserts
#     assert config.wandb.dry_run or config.wandb.entity is not '', \
#         'You need a wandb entity'
#     assert config.wandb.dry_run or config.wandb.project is not '', \
#         'You need a wandb project'
#     assert config.checkpoint.filepath is '' or \
#            (config.checkpoint.monitor_index < len(config.datasets.validation.split)), \
#         'You need to monitor a valid dataset'

#     # Prepare datasets
#     config.datasets.train = prep_dataset(config.datasets.train)
#     config.datasets.validation = prep_dataset(config.datasets.validation)
#     config.datasets.test = prep_dataset(config.datasets.test)
#     # Set name and checkpoint
#     config.name = set_name(config)
#     config.checkpoint = set_checkpoint(config)
#     # Return configuration
#     return config


def parse_test_file(ckpt_file, cfg_file=None):
    """
    Parse model configuration for testing

    Parameters
    ----------
    ckpt_file : str
        Checkpoint file, with pretrained model
    cfg_file :
        Configuration file, to update pretrained model configuration

    Returns
    -------
    config : CfgNode
        Parsed model configuration
    state_dict : dict
        Model state dict with pretrained weights
    """
    assert ckpt_file.endswith('.ckpt') or ckpt_file.endswith('.pth.tar'), \
        'You need to provide a .ckpt or .pth.tar file for checkpoint, not {}'.format(ckpt_file)
    assert cfg_file is None or cfg_file.endswith('yaml'), \
        'You need to provide a .yaml file for configuration, not {}'.format(cfg_file)
    cfg_default = 'configs/default_config'
    return parse_test_config(ckpt_file, cfg_default, cfg_file)

def parse_test_config(ckpt_file, cfg_default, cfg_file):
    """
    Parse model configuration for testing

    Parameters
    ----------
    ckpt_file : str
        Checkpoint file, with pretrained model
    cfg_default : str
        Default configuration file, with default values
    cfg_file : str
        Configuration file with updated information

    Returns
    -------
    Returns
    -------
    config : CfgNode
        Parsed model configuration
    state_dict : dict
        Model state dict with pretrained weights
    """
    if ckpt_file.endswith('.ckpt'):
        # Load checkpoint
        ckpt = torch.load(ckpt_file, map_location='cpu')
        # Get base configuration
        config_default = get_default_config(cfg_default)
        # Extract configuration and model state
        config_model, state_dict = ckpt['config'], ckpt['state_dict']
        # Override default configuration with model configuration
        config = merge_cfgs(config_default, config_model)
        # Update configuration for backwards compatibility
        config = backwards_config(config)
        # If another config file is provided, use it
        config = merge_cfg_file(config, cfg_file)
    # Backwards compatibility with older models
    elif ckpt_file.endswith('.pth.tar'):
        # Load model state and update it for backwards compatibility
        state_dict = torch.load(ckpt_file, map_location='cpu')['state_dict']
        state_dict = backwards_state_dict(state_dict)
        # Get default configuration
        config = get_default_config(cfg_default)
        # If config file is present, update configuration
        config = merge_cfg_file(config, cfg_file)
    else:
        raise ValueError('Unknown checkpoint {}'.format(ckpt_file))
    # Set pretrained model name
    config.save.pretrained = ckpt_file
    # Return prepared configuration and model state
    return prepare_test_config(config), state_dict


def prepare_test_config(config):
    """
    Prepare model configuration for testing

    Parameters
    ----------
    config : CfgNode
        Model configuration

    Returns
    -------
    config : CfgNode
        Prepared model configuration
    """
    # Remove train and validation datasets
    config.datasets.train.path = config.datasets.validation.path = []
    config.datasets.test = prep_dataset(config.datasets.test)
    # Don't save models or log to wandb
    config.wandb.dry_run = True
    config.checkpoint.filepath = ''
    # Return updated configuration
    return config
