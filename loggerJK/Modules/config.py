# ---------------------------------------------------------------------
# Some code is from official Swin-Transformer Github
# https://github.com/microsoft/Swin-Transformer 
# ---------------------------------------------------------------------
import os
import yaml
from yacs.config import CfgNode as CN

_C = CN()

_C.MODEL = CN()
_C.MODEL.S = 7
_C.MODEL.B = 2
_C.MODEL.C = 20
_C.MODEL.BATCH_SIZE = 4
_C.MODEL.EPOCH = 150
_C.MODEL.FINETUNE_EPOCH = 5
_C.MODEL.HEIGHT = 384
_C.MODEL.WIDTH = 384

_C.DATASET = CN()
_C.DATASET.PATH='../'

_C.TRAINING = CN()
_C.TRAINING.LR = 1e-5
_C.TRAINING.OPTIMIZER = 'Adam'
_C.TRAINING.SCHEDULER = None
_C.TRAINING.SAVE_PATH = './'
_C.TRAINING.DEVICE = 'cpu'

_C.LOSS = CN()
_C.LOSS.LAMBDA_COORD = 5.
_C.LOSS.LAMBDA_NOOBJ = 0.5

_C.MISC = CN()
_C.MISC.SEED = 42

_C.WANDB = CN()
_C.WANDB.USE = False

def _update_config_from_file(config, cfg_file):
    config.defrost()
    # with open(cfg_file, 'r') as f:
    #     yaml_cfg: dict = yaml.load(f, Loader=yaml.FullLoader) 

    # for cfg in yaml_cfg.setdefault('BASE', ['']):
    #     if cfg:
    #         _update_config_from_file(
    #             config, os.path.join(os.path.dirname(cfg_file), cfg)
    #         )

    print('=> merge config from {}'.format(cfg_file))
    config.merge_from_file(cfg_file)
    config.freeze()

def update_config(config, args):
    _update_config_from_file(config, args.cfg)

def get_config(args):
    """Get a yacs CfgNode object with default values."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    config = _C.clone()
    update_config(config, args)
    with open(args.cfg, 'r') as f:
        yaml_cfg: dict = yaml.load(f, Loader=yaml.FullLoader) 

    return config, yaml_cfg


