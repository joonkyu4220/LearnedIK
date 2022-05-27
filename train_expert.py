from args import *
from expert import *
from dataset import *
from logger import *
from nets import *

if __name__ == "__main__":
    args = Args()
    expert = Expert(args)
    trainer = ExpertTrainer(expert)
    trainer.train()