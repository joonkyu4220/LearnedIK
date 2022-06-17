from dataset import DataManager
from nets import *
import os

from torch.nn import L1Loss, MSELoss
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR


from logger import Logger


class Expert():
    def __init__(self, args):
        self.args = args
        self.init_nets()
        self.init_optimizer()
        if self.args.epoch != 0:
            if self.args.epoch == "latest":
                suf_len = 3
                epochs = [int(epoch[:-suf_len]) for epoch in os.listdir(self.args.model_path)]
                self.args.epoch = max(epochs) if len(epochs) else 0
            load_model_path = os.path.join(self.args.model_path, str(self.args.epoch)+".pt")
            print(f"Trying to load: {load_model_path}")
            if os.path.isfile(load_model_path):
                model = torch.load(load_model_path)
                self.iknet.load_state_dict(model["iknet_state_dict"])
                self.optimizer.load_state_dict(model["optimizer_state_dict"])
                self.scheduler.load_state_dict(model["scheduler_state_dict"])
                print(f"Successfully loaded {load_model_path}")
            else:
                print("!PRETRAINED MODEL NOT FOUND!\nSTARTING FROM THE TOP...")
                self.args.epoch = 0
            if isinstance(self.args.epoch, str):
                self.args.epoch = int(self.args.epoch[8:])
        self.args.epoch += 1
    
    def init_nets(self):
        if self.args.ik_ver == 0:
            self.iknet = IKNet(self.args).to(self.args.device)
        elif self.args.ik_ver == 1:
            self.iknet = IKNet1(self.args).to(self.args.device)
        elif self.args.ik_ver == 2:
            self.iknet = IKNet2(self.args).to(self.args.device)
        self.fknet = FKNet(self.args).to(self.args.device)
        return
    
    def init_optimizer(self):
        if self.args.optimizer == "Adam":
            self.optimizer = Adam(self.iknet.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
        if self.args.scheduler == "Exponential":
            self.scheduler = ExponentialLR(optimizer=self.optimizer, gamma=0.9)
        return

    def ik(self, x):
        return self.iknet(x)
    
    def fk(self, rot, lengths):
        return self.fknet(rot, lengths)
    
    def save_dict(self, fine_tune = False):
        file_name = f"{self.args.epoch}.pt" if not(fine_tune) else f"FINETUNE{self.args.epoch}.pt"
        torch.save({'epoch': self.args.epoch,
                    'iknet_state_dict': self.iknet.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    "scheduler_state_dict": self.scheduler.state_dict()},
                    os.path.join(self.args.model_path, file_name))
        return

class LossManager():
    def __init__(self, args):
        self.device = args.device
        self.repr = args.repr
        self.loss_fn = {}
        for (key, val) in args.loss.items():
            if val == "L1":
                self.loss_fn[key] = L1Loss()
            elif val == "MSE":
                self.loss_fn[key] = MSELoss()
        self.weights = args.weights
    
    def compute_losses(self, x, y, pred, recon):
        ee_pos_loss = self.loss_fn["ee_pos"](x[:2], recon[:2]) if self.weights["ee_pos"] > 0.0 else 0
        ee_rot_loss = self.loss_fn["ee_rot"](x[2:], recon[2:]) if self.weights["ee_rot"] > 0.0 else 0
        rot_norm_loss = self.loss_fn["rot_norm"](torch.norm(pred, dim=1), torch.ones_like(pred[:, 0]).to(self.device))\
                        if (self.weights["rot_norm"] > 0.0 and self.repr == "COSSIN") \
                        else 0
        rot_loss = self.loss_fn["rot"](y, pred) if self.weights["rot"] > 0.0 else 0
        weighted_sum = self.weights["ee_pos"] * ee_pos_loss + self.weights["ee_rot"] * ee_rot_loss + self.weights["rot_norm"] * rot_norm_loss + self.weights["rot"] * rot_loss
        return {"ee_pos": ee_pos_loss, "ee_rot": ee_rot_loss, "rot_norm": rot_norm_loss, "rot": rot_loss, "total": weighted_sum}

class ExpertTrainer():
    def __init__(self, expert):
        self.expert = expert
        self.args = self.expert.args

        self.logger = Logger(self.args)
        self.loss_manager = LossManager(self.args)
        self.data_manager = DataManager(self.args)
        

    def train_step(self):
        self.expert.iknet.train()
        for batch, (x, y) in enumerate(self.data_manager.dataloaders["TRAIN"]):
            rot = self.expert.ik(x)
            recon = self.expert.fk(rot, x[:, :3])
            loss_dic = self.loss_manager.compute_losses(x[:, 3:], y, rot, recon)
            self.logger.add_loss({key: val.item() if val else val for (key, val) in loss_dic.items()})
            self.expert.optimizer.zero_grad()
            loss_dic["total"].backward()
            self.expert.optimizer.step()
            if (batch % self.args.verbose_freq == 0):
                print(f"====================BATCH {batch}====================")
                for key, val in loss_dic.items():
                    print(f"{key}: {val.item() if val else val:>7f}")
        self.logger.average_loss(self.data_manager.num_batches["TRAIN"])
        self.logger.write_loss("TRAIN")

        self.logger.reset()
    
    def test(self):
        self.expert.iknet.eval()
        with torch.no_grad():
            for x, y in self.data_manager.dataloaders["TEST"]:
                rot = self.expert.ik(x)
                recon = self.expert.fk(rot, x[:, :3])
                loss_dic = self.loss_manager.compute_losses(x[:, 3:], y, rot, recon)
                self.logger.add_loss({key: val.item() if val else val for (key, val) in loss_dic.items()})
                self.logger.add_result({"x": x[:, 3:], "y": y, "recon": recon, "rot": rot})
        self.logger.average_loss(self.data_manager.num_batches["TEST"])
        self.logger.write_loss("TEST")
        if (self.args.epoch % self.args.log_save_freq == 0):
            # self.logger.save_txt()
            self.logger.save_fig()

        self.logger.reset()


    def train(self):
        while (self.args.epoch <= self.args.max_epoch):
            print(f"====================EPOCH {self.args.epoch}====================")
            self.train_step()
            self.test()
            if (self.args.epoch % self.args.model_save_freq == 0):
                self.expert.save_dict()
            self.args.epoch += 1
        print("DONE!")

    def eval(self):
        self.test()
        print("DONE!")