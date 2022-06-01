from args import *
from expert import *
from mixture_of_experts import *

if __name__ == "__main__":
    args1 = Args(lengths=[2.0, 2.0, 2.0], min_bound=0.0, max_bound=0.5)
    expert1 = Expert(args1)


    args = Args(num_nets=1)
    moe = MixtureOfExperts(args, [expert1])
    trainer = MOETrainer(moe)
    trainer.train()