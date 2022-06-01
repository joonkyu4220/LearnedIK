from args import *
from expert import *

if __name__ == "__main__":
    args = Args()
    expert = Expert(args)
    trainer = ExpertTrainer(expert)
    trainer.train()