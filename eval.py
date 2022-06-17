from args import *
from expert import *

if __name__ == "__main__":
    args = Args(epoch="FINETUNE1201")
    expert = Expert(args)
    trainer = ExpertTrainer(expert)
    trainer.eval()