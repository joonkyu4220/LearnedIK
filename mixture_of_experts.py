from dataset import DataManager
from expert import LossManager
from nets import *
import os

from torch.nn import L1Loss
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR

from logger import Logger

class MixtureOfExperts():
    def __init__(self, args, experts):
        self.args = args
        self.experts = experts
        self.num_nets = len(self.experts)

        self.init_nets()
        self.init_optimizer()

        if self.args.epoch != 0:
            if self.args.epoch == "latest":
                suf_len = 3
                epochs = [int(epoch[:-suf_len]) for epoch in os.listdir(self.args.model_path)]
                self.args.epoch = max(epochs) if len(epochs) else 0
            load_model_path = os.path.join(self.args.model_path, str(self.args.epoch) + ".pt")
            print(f"Trying to load: {load_model_path}")
            if os.path.isfile(load_model_path):
                model = torch.load(load_model_path)
                self.gatingnet.load_state_dict(model["gatingnet_state_dict"])
                self.optimizer.load_state_dict(model["optimizer_state_dict"])
                self.scheduler.load_state_dict(model["scheduler_state_dict"])
                print(f"Successfully loaded {load_model_path}")
            else:
                print("!PRETRAINED MODEL NOT FOUND!\nSTARTING FROM THE TOP...")
                self.args.epoch = 0
        self.args.epoch += 1

    def init_nets(self):
        if self.args.gating_ver == 0:
            self.gatingnet = GatingNet(self.args).to(self.args.device)
        self.fknet = FKNet(self.args).to(self.args.device)
        return

    def init_optimizer(self):
        if self.args.optimizer == "Adam":
            self.optimizer = Adam(self.gatingnet.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
        if self.args.scheduler == "Exponential":
            self.scheduler = ExponentialLR(optimizer=self.optimizer, gamma=0.9)
        return
    
    def gate(self, x):
        return self.gatingnet(x)

    def fk(self, rot, lengths):
        return self.fknet(rot, lengths)

    def normalize(self, rots):
        idx = 0
        while idx < rots.shape[1]:
            rots[:, idx:idx+6] = self.fknet.normalized(rots[:, idx:idx+6])
        return rots

    def save_dict(self):
        torch.save({'epoch': self.args.epoch,
                    'gatingnet_state_dict': self.gatingnet.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict()},
                    os.path.join(self.args.model_path, f"{self.args.epoch}.pt"))
        return

class MOETrainer():
    def __init__(self, moe):
        self.moe = moe
        self.args = self.moe.args
        self.logger = Logger(self.args)
        self.loss_manager = LossManager(self.args)
        self.data_manager = DataManager(self.args)

    def weight(self, weights, rots):
        if self.args.repr == "COSSIN":
            weighted_q = weights.repeat_interleave(3, dim=1) * torch.arctan2(rots[:, [1, 3, 5]], rots[:, [0, 2, 4]])
            return torch.cat([torch.cos(weighted_q), torch.sin(weighted_q)], dim=1)[:, [0, 3, 1, 4, 2, 5]]
        else:
            weighted_q = weights.reshape(1, -1) * rots
            return weighted_q
        
    def train_step(self):
        self.moe.gatingnet.train()
        for batch, (x, y) in enumerate(self.data_manager.dataloaders["TRAIN"]):
            weights = self.moe.gatingnet(x)
            rots = []
            for expert in self.moe.experts:
                rots.append(expert.ik(x[:, 3:]))
            rot = self.weight(weights, torch.cat(rots, dim=1))
            recon = self.moe.fk(rot, x[:, :3])
            loss_dic = self.loss_manager.compute_losses(x[:, 3:], y, rot, recon)
            self.logger.add_loss({key: val.item() if val else val for (key, val) in loss_dic.items()})
            self.moe.optimizer.zero_grad()
            
            # for expert in self.experts:
            #     expert.optimizer.zero_grad()

            loss_dic["total"].backward()
            self.moe.optimizer.step()

            # for expert in self.experts:
            #     expert.optimizer.step()

            if (batch % self.args.verbose_freq == 0):
                print(f"====================BATCH {batch}====================")
                for key, val in loss_dic.items():
                    print(f"{key}: {val.item() if val else val:>7f}")
        self.logger.average_loss(self.data_manager.num_batches["TRAIN"])
        self.logger.write_loss("TRAIN")
        self.args.epoch += 1

        # for expert in self.experts:
        #     expert.args.epoch += 1
        
        self.logger.reset()

    def test(self):
        self.moe.gatingnet.eval()
        with torch.no_grad():
            for x, y in self.data_manager.dataloaders["TEST"]:
                weights = self.moe.gatingnet(x)
                rots = []
                for expert in self.moe.experts:
                    rots.append(expert.ik(x[:, 3:]))
                rot = self.weight(weights, torch.cat(rots, dim=1))
                recon = self.moe.fk(rot, x[:, :3])
                loss_dic = self.loss_manager.compute_losses(x[:, 3:], y, rot, recon)
                self.logger.add_loss({key: val.item() if val else val for (key, val) in loss_dic.items()})
                self.logger.add_result({"x": x[:, 3:], "y": y, "recon": recon, "rot": rot})
        self.logger.average_loss(self.data_manager.num_batches["TEST"])
        self.logger.write_loss("TEST")
        if (self.args.epoch % self.args.log_save_freq == 0):
            self.logger.save_txt()
            self.logger.save_fig()
        
        self.logger.reset()

    def train(self):
        while (self.args.epoch <= self.args.max_epoch):
            print(f"====================EPOCH {self.args.epoch}====================")
            self.train_step()
            self.test()
            if (self.args.epoch % self.args.model_save_freq == 0):
                self.moe.save_dict()
        print("DONE!")

