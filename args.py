import os
from pickle import NONE
import torch

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DATA_PATH = os.path.join(".", "datasets")
RESULT_PATH = os.path.join(".", "results")
MODEL_PATH = os.path.join(".", "models")
LOG_PATH = os.path.join(".", "runs")
BATCH_SIZE = 64

LENGTHS = [1.0, 1.0, 1.0] 
MIN_BOUND = 0.0
MAX_BOUND = 0.5

WEIGHT_DECAY = 1e-3
LEARNING_RATE = 1e-4
WEIGHTS = {"ee_pos":1.0, "ee_rot":1.0, "rot_norm":1.0, "rot":0.0} # ee_pos, ee_rot, rot_norm, rot
LOSS = {"ee_pos":"L1", "ee_rot":"L1", "rot_norm":"L1", "rot":"L1"} # "L2"
OPTIMIZER = "Adam"
SCHEDULER = "Exponential"

NORMALIZE = False

REPR = "COSSIN" # "ANGLE"
DATA_LABEL = {"ee_pos":(0, 2), "ee_rot":(2, 4), "rot":(0, 6)} if REPR == "COSSIN" else {"ee_pos":(0, 2), "ee_rot":(2, 3), "rot":(0, 3)}

IK_VER = 0

MAX_EPOCH = 500
SAVE_FREQ = 50
VERBOSE_FREQ = 1000
START_EPOCH = 0

class Args():
    def __init__(self,
                device=DEVICE,
                data_path=DATA_PATH,
                result_path=RESULT_PATH,
                model_path=MODEL_PATH,
                log_path=LOG_PATH,
                lengths=LENGTHS,
                min_bound=MIN_BOUND,
                max_bound=MAX_BOUND,
                batch_size=BATCH_SIZE,
                weights=WEIGHTS,
                weight_decay=WEIGHT_DECAY,
                loss=LOSS,
                optimizer=OPTIMIZER,
                scheduler=SCHEDULER,
                repr=REPR,
                normalize=NORMALIZE,
                ik_ver = IK_VER,
                epoch=START_EPOCH,
                learning_rate=LEARNING_RATE,
                max_epoch=MAX_EPOCH,
                save_freq=SAVE_FREQ,
                verbose_freq = VERBOSE_FREQ,
                data_label=DATA_LABEL
                ):
        self.device = device
        self.data_path = data_path
        self.result_path = result_path
        self.model_path = model_path
        self.log_path = log_path
        self.lengths = lengths
        self.min_bound = min_bound
        self.max_bound = max_bound
        self.batch_size = batch_size
        self.weights = weights
        self.weight_decay = weight_decay
        self.loss = loss
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.pos_scale = None
        self.rot_scale = None
        self.repr = repr
        self.normalize = normalize
        self.ik_ver = ik_ver
        self.epoch = epoch
        self.learning_rate = learning_rate
        self.max_epoch = max_epoch
        self.save_freq = save_freq
        self.verbose_freq = verbose_freq
        self.data_label = data_label

        self.set_dir_name()
        
    def set_dir_name(self):
        dir_name = ""
        dir_name += f"REPR[{self.repr}]"
        dir_name += f"RANGE[{self.min_bound}{self.max_bound}]"
        dir_name += f"LEN[{self.lengths[0]},{self.lengths[1]},{self.lengths[2]}]"
        dir_name += f"WEIGHTS[{self.weights['ee_pos']},{self.weights['ee_rot']},{self.weights['rot_norm']},{self.weights['rot']}]"
        self.dir_name = dir_name
        self.result_path = os.path.join(self.result_path, dir_name)
        self.model_path = os.path.join(self.model_path, dir_name)
        self.log_path = os.path.join(self.log_path, dir_name)
        if not(os.path.isdir(self.result_path)):
            os.makedirs(self.result_path)
        if not(os.path.isdir(self.model_path)):
            os.makedirs(self.model_path)
        if not(os.path.isdir(self.log_path)):
            os.makedirs(self.log_path)
        

