import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from generate_dataset import PREFIX, INPUT_SUFFIX, GT_SUFFIX
import os

class IKDataset(Dataset):
    def __init__(self, input_path, gt_path, device):
        print("Loading data...")
        self.input = torch.load(input_path, map_location = device)
        self.gt = torch.load(gt_path, map_location = device)
        print(f"Dataset contains: {self.input.shape[0]} data")

    def __len__(self):
        return self.input.shape[0]

    def __getitem__(self, idx):
        return self.input[idx], self.gt[idx]

class DataManager():
    def __init__(self, args):
        self.args = args
        self.datasets = {}
        self.dataloaders = {}
        self.num_batches = {}
        self.init("TRAIN")
        # self.init("VALIDATION", False)
        self.init("TEST", False)

    def init(self, mode, shuffle=True):
        prefix = os.path.join(self.args.data_path, PREFIX.format(self.args.repr, self.args.min_bound, self.args.max_bound, mode))
        input_suffix = INPUT_SUFFIX.format(self.args.lengths[0], self.args.lengths[1], self.args.lengths[2])
        gt_suffix = GT_SUFFIX.format(self.args.lengths[0], self.args.lengths[1], self.args.lengths[2])
        self.datasets[mode] = IKDataset(prefix+input_suffix, prefix+gt_suffix, self.args.device)
        self.dataloaders[mode] = DataLoader(self.datasets[mode], batch_size=self.args.batch_size, shuffle=shuffle)
        self.num_batches = {key:len(val) for (key, val) in self.dataloaders.items()}
        return
    
    def get_all_data(self, mode):
        return self.datasets[mode].input, self.datasets[mode].gt


if __name__ == "__main__":
    from args import Args
    args = Args()
    datamanager = DataManager(args)

    print("INPUT")
    print("MIN")
    print(torch.min(datamanager.datasets["TRAIN"].input, dim=0))
    print("MAX")
    print(torch.max(datamanager.datasets["TRAIN"].input, dim=0))
    print("MEAN")
    print(torch.mean(datamanager.datasets["TRAIN"].input, dim=0))
    print("STD")
    print(torch.std(datamanager.datasets["TRAIN"].input, dim=0))

    print("GT")
    print("MIN")
    print(torch.min(datamanager.datasets["TRAIN"].gt, dim=0))
    print("MAX")
    print(torch.max(datamanager.datasets["TRAIN"].gt, dim=0))
    print("MEAN")
    print(torch.mean(datamanager.datasets["TRAIN"].gt, dim=0))
    print("STD")
    print(torch.std(datamanager.datasets["TRAIN"].gt, dim=0))

