import os
import torch

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DATA_PATH = os.path.join(".", "datasets")
RESULT_PATH = os.path.join(".", "results")
MODEL_PATH = os.path.join(".", "models")
LOG_PATH = os.path.join(".", "runs")

# LENGTHS = [0.0, 2.0, 0.67] # l0=0, l1=mean, l2=std for (gaussian)random lengths
LENGTHS = [2.0, 2.0, 2.0]
MIN_BOUND = 0.0
MAX_BOUND = 1.0

WEIGHTS = {"ee_pos":1.0, "ee_rot":1.0, "rot_norm":0.0, "rot":1.0} # ee_pos, ee_rot, rot_norm, rot
LOSS = {"ee_pos":"L1", "ee_rot":"L1", "rot_norm":"L1", "rot":"L1"} # "L1"/"MSE" Let's keep it simple. L1.
ACTIVATION = "LeakyReLU" # "ReLU"/"LeakyReLU"/"Tanh" Tanh is bad. ReLU and LeakyReLU are similar. /  "Sigmoid" for gating net?
WEIGHT_DECAY = 1e-3
LEARNING_RATE = 1e-4
OPTIMIZER = "Adam"
SCHEDULER = "Exponential"
BATCH_SIZE = 128

NORMALIZE = True

REPR = "COSSIN" # "COSSIN"/"ANGLE"
DATA_LABEL = {"ee_pos":(0, 2), "ee_rot":(2, 4), "rot":(0, 6)} if REPR == "COSSIN" else {"ee_pos":(0, 2), "ee_rot":(2, 3), "rot":(0, 3)}

IK_VER = 1

MAX_EPOCH = 1000

MODEL_SAVE_FREQ = 100
LOG_SAVE_FREQ = 1
VERBOSE_FREQ = 1000
START_EPOCH = "latest"
E2E_EPOCH = 100

# MIXTURE OF EXPERTS
GATING_VER = 1
NUM_NETS = 0

class Args():
    def __init__(self,
                device=DEVICE,
                data_path=DATA_PATH,
                result_path=RESULT_PATH,
                model_path=MODEL_PATH,
                log_path=LOG_PATH,

                lengths=LENGTHS, # different configuration for each expert
                min_bound=MIN_BOUND, # different configuration for each expert
                max_bound=MAX_BOUND, # different configuration for each expert
                
                weights=WEIGHTS,
                loss=LOSS,
                activation=ACTIVATION,
                weight_decay=WEIGHT_DECAY,
                learning_rate=LEARNING_RATE,
                optimizer=OPTIMIZER,
                scheduler=SCHEDULER,
                batch_size=BATCH_SIZE,
                
                normalize=NORMALIZE,

                repr=REPR,
                data_label=DATA_LABEL,
                
                ik_ver = IK_VER,
                
                max_epoch=MAX_EPOCH,
                model_save_freq=MODEL_SAVE_FREQ,
                log_save_freq=LOG_SAVE_FREQ,
                verbose_freq = VERBOSE_FREQ,
                epoch=START_EPOCH,
                e2e_epoch=E2E_EPOCH,

                gating_ver=GATING_VER,
                num_nets=NUM_NETS,
                ):

        self.device = device
        self.data_path = data_path
        self.result_path = result_path
        self.model_path = model_path
        self.log_path = log_path

        self.lengths = lengths
        self.min_bound = min_bound
        self.max_bound = max_bound

        self.weights = weights
        self.loss = loss
        self.activation = activation        
        self.weight_decay = weight_decay
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.batch_size = batch_size

        self.normalize = normalize
        
        self.repr = repr
        self.data_label = data_label
        
        self.ik_ver = ik_ver
        
        self.max_epoch = max_epoch
        self.model_save_freq = model_save_freq
        self.log_save_freq = log_save_freq
        self.verbose_freq = verbose_freq
        self.epoch = epoch
        self.e2e_epoch = e2e_epoch

        self.gating_ver = gating_ver
        self.num_nets = num_nets

        self.set_dirs()
        
    def set_dirs(self):
        dir_name = ""

        if self.num_nets != 0:
            dir_name += f"GatingNet[{self.gating_ver}]"
        else:
            dir_name += f"IKNet[{self.ik_ver}]"

        dir_name += f"REPR[{self.repr},{self.normalize}]"
        dir_name += f"RANGE[{self.min_bound},{self.max_bound}]"

        dir_name += f"LEN[{self.lengths[0]},{self.lengths[1]},{self.lengths[2]}]"

        dir_name += f"LOSS[{self.loss['ee_pos']},{self.loss['ee_rot']},{self.loss['rot_norm']},{self.loss['rot']}]"
        dir_name += f"WEIGHTS[{self.weights['ee_pos']},{self.weights['ee_rot']},{self.weights['rot_norm']},{self.weights['rot']}]"
        dir_name += f"ACTIVATION[{self.activation}]"

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
        

