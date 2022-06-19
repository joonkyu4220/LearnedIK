from args import *
from expert import *
from mixture_of_experts import *

if __name__ == "__main__":

    experts = []
    
    # args = Args(lengths=[2.0, 2.0, 2.0])
    # experts.append(Expert(args))

    # args = Args(lengths=[1.0, 2.0, 2.0])
    # experts.append(Expert(args))

    # args = Args(lengths=[2.0, 1.0, 2.0])
    # experts.append(Expert(args))

    # args = Args(lengths=[2.0, 2.0, 1.0])
    # experts.append(Expert(args))

    # args = Args(lengths=[3.0, 2.0, 2.0])
    # experts.append(Expert(args))

    # args = Args(lengths=[2.0, 3.0, 2.0])
    # experts.append(Expert(args))

    # args = Args(lengths=[2.0, 2.0, 3.0])
    # experts.append(Expert(args))

    args = Args(lengths=[2.0, 2.0, 2.0], min_bound=-1.0, max_bound=0.0, epoch=300)
    experts.append(Expert(args))

    args = Args(lengths=[2.0, 2.0, 2.0], min_bound=-0.5, max_bound=0.5, epoch=300)
    experts.append(Expert(args))

    args = Args(lengths=[2.0, 2.0, 2.0], min_bound=0.0, max_bound=1.0, epoch=300)
    experts.append(Expert(args))

    args = Args(lengths=[2.0, 2.0, 2.0], min_bound=0.5, max_bound=1.5, epoch=300)
    experts.append(Expert(args))

    args = Args(num_nets=4)
    moe = MixtureOfExperts(args, experts)
    trainer = MOETrainer(moe)

    trainer.train()