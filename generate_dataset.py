import torch
import os
from args import Args

DATASET_SIZE = 1000000
THRESHOLD = 0.1

PREFIX = "REPR[{}]RANGE[{},{}][{}]"
INPUT_SUFFIX = "input_{}_{}_{}.pt"
GT_SUFFIX = "gt_{}_{}_{}.pt"

def equals(x, y):
    if abs(x - y) < THRESHOLD:
        return True
    else:
        return False

def new_data(data0, data1, repr):
    if ((repr == "ANGLE") and equals(data0[0], data1[0]) and equals(data0[1], data1[1]) and equals(data0[2], data1[2])) \
        or ((repr == "COSSIN") and equals(data0[0], data1[0]) and equals(data0[1], data1[1]) and equals(data0[2], data1[2]) and equals(data0[3], data1[3])):
        return False
    else:
        return True

def generate_dataset(l0, l1, l2, dataset_size, min_bound, max_bound, repr, path):
    if l0 == 0:
        mean = torch.ones(dataset_size, 3) * l1
        std = torch.ones(dataset_size, 3) * l2
        lengths = torch.clamp(torch.normal(mean=mean, std=std), l1 - 3 * l2, l1 + 3 * l2)
    else:
        lengths = torch.ones(dataset_size, 3) * torch.tensor([[l0, l1, l2]])
    
    q = torch.rand(dataset_size, 3) * torch.pi * (max_bound - min_bound) + torch.pi * min_bound
    q00 = torch.sum(q[:, 0:1], dim=1, keepdim=True)
    q01 = torch.sum(q[:, 0:2], dim=1, keepdim=True)
    q02 = torch.sum(q[:, 0:3], dim=1, keepdim=True)
    x = lengths[:, 0:1] * torch.cos(q00) + lengths[:, 1:2] * torch.cos(q01) + lengths[:, 2:3] * torch.cos(q02)
    y = lengths[:, 0:1] * torch.sin(q00) + lengths[:, 1:2] * torch.sin(q01) + lengths[:, 2:3] * torch.sin(q02)
    if repr == "ANGLE":
        input_data = torch.cat([lengths, x, y, q02], dim=1)
        gt_data = q
    else:
        input_data = torch.cat([lengths, x, y, torch.cos(q02), torch.sin(q02)], dim=1)
        gt_data = torch.cat([torch.cos(q), torch.sin(q)], dim=1)[:, [0, 3, 1, 4, 2, 5]]
    
    test_size = dataset_size // 10
    # validation_size = test_size * 2

    if not(os.path.isdir(path)):
        os.makedirs(path)

    torch.save(input_data[:test_size], os.path.join(path, PREFIX.format(repr, min_bound, max_bound, "TEST")+INPUT_SUFFIX.format(l0, l1, l2)))
    torch.save(gt_data[:test_size], os.path.join(path, PREFIX.format(repr, min_bound, max_bound, "TEST")+GT_SUFFIX.format(l0, l1, l2)))

    # torch.save(input_data[test_size:test_size+validation_size], os.path.join(path, PREFIX.format(repr, min_bound, max_bound, "VALIDATION")+INPUT_SUFFIX.format(l0, l1, l2)))
    # torch.save(gt_data[test_size:test_size+validation_size], os.path.join(path, PREFIX.format(repr, min_bound, max_bound, "VALIDATION")+GT_SUFFIX.format(l0, l1, l2)))
    
    torch.save(input_data[test_size:], os.path.join(path, PREFIX.format(repr, min_bound, max_bound, "TRAIN")+INPUT_SUFFIX.format(l0, l1, l2)))
    torch.save(gt_data[test_size:], os.path.join(path, PREFIX.format(repr, min_bound, max_bound, "TRAIN")+GT_SUFFIX.format(l0, l1, l2)))


if __name__ == "__main__":
    args = Args()
    l0, l1, l2 = args.lengths
    generate_dataset(l0, l1, l2, DATASET_SIZE, args.min_bound, args.max_bound, args.repr, args.data_path)

    
    
