import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from vis_util import pos_to_color

class Logger():
    def __init__(self, args):
        self.args = args
        self.result_path = args.result_path
        self.log_path = args.log_path
        self.loss_dic = {}
        self.result_dic = {}
        self.writer = SummaryWriter(log_dir=self.log_path)
        self.reset()
        self.new = True

    def reset(self):
        self.loss_dic.clear()
        self.result_dic.clear()
        return

    def add_loss(self, dic):
        for (key, val) in dic.items():
            if key in self.loss_dic:
                self.loss_dic[key] += val
            else:
                self.loss_dic[key] = val
        return
    
    def average_loss(self, num_batches):
        for (key, val) in self.loss_dic.items():
            self.loss_dic[key] = val / num_batches
        return

    def add_result(self, dic):
        for key, val in dic.items():
            if key in self.result_dic:
                self.result_dic[key] = torch.cat([self.result_dic[key], val], dim=0)
            else:
                self.result_dic[key] = val
        return

    def write_loss(self, mode):
        for (k, v) in self.loss_dic.items():
            self.writer.add_scalar(mode + "/" + k, v, self.args.epoch)
            print(f"avg {k} loss: {v:>8f}")
        return

    def save_txt(self, keys=["x", "y", "recon", "rot"]):
        results = torch.cat([self.result_dic[key] for key in keys], dim=1).cpu().numpy()
        np.savetxt(os.path.join(self.result_path, f"{self.args.epoch}.txt"), results)
        return
    def save_fig(self, x_key="x", recon_key="recon", file_name=None, x_dim=0, recon_dim=0, s=0.01):
        if file_name is None:
            file_name = f"{self.args.epoch}.png"
        xy = self.result_dic[x_key][:, x_dim:x_dim+2].cpu().numpy()
        recon = self.result_dic[recon_key][:, recon_dim:recon_dim+2].cpu().numpy()
        colors = pos_to_color(xy)/255
        plt.scatter(recon[:, 0], recon[:, 1], s=s, c=colors)
        plt.savefig(os.path.join(self.result_path, file_name))
        if self.new:
            plt.scatter(xy[:, 0], xy[:, 1], s=s, c=colors)
            plt.savefig(os.path.join(self.result_path, "GT.png"))
            self.new = False
        return