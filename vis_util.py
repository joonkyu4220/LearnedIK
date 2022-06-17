import numpy as np
import torch
import matplotlib.pyplot as plt
eps = 0.000001
def make_colorwheel():
    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR
    
    colorwheel = np.zeros((ncols, 3))
    col = 0

    colorwheel[col:col+RY, 0] = 255
    colorwheel[col:col+RY, 1] = np.floor(255 * np.arange(0, RY) / RY)
    col += RY

    colorwheel[col:col+YG, 0] = 255 - np.floor(255 * np.arange(0, YG) / YG)
    colorwheel[col:col+YG, 1] = 255
    col += YG

    colorwheel[col:col+GC, 1] = 255
    colorwheel[col:col+GC, 2] = np.floor(255 * np.arange(0, GC) / GC)
    col += GC

    colorwheel[col:col+CB, 1] = 255 - np.floor(255 * np.arange(0, CB) / CB)
    colorwheel[col:col+CB, 2] = 255
    col += CB

    colorwheel[col:col+BM, 2] = 255
    colorwheel[col:col+BM, 0] = np.floor(255 * np.arange(0, BM) / BM)
    col += BM

    colorwheel[col:col+MR, 2] = 255 - np.floor(255 * np.arange(0, MR) / MR)
    colorwheel[col:col+MR, 0] = 255

    return colorwheel

def pos_to_color(xy):

    xy_norm = np.zeros_like(xy)
    xy_norm[:, 0] = xy[:, 0] / (np.max(np.abs(xy[:, 0])) + eps)
    xy_norm[:, 1] = xy[:, 1] / (np.max(np.abs(xy[:, 1])) + eps)

    rad = np.sqrt(np.sum(np.square(xy_norm), axis=1))
    a = np.arctan2(xy_norm[:, 1], xy_norm[:, 0]) / np.pi

    c = np.zeros((xy_norm.shape[0], 3))

    fk = (a + 1) / 2 * (ncols - 1)
    k0 = np.floor(fk).astype(np.int32)
    k1 = k0 + 1
    k1[k1 == ncols] = 0
    f = fk - k0

    for i in range(colorwheel.shape[1]):
        tmp = colorwheel[:, i]
        col0 = tmp[k0] / 255.0
        col1 = tmp[k1] / 255.0
        col = (1-f) * col0 + f * col1
        idx = (rad <= 1)
        col[idx] = 1 - rad[idx] * (1 - col[idx])
        col[~idx] = col[~idx] * 0.75
        c[:, i] = np.floor(255 * col)
    
    return c

colorwheel = make_colorwheel()
ncols = colorwheel.shape[0]

if __name__ == "__main__":
    SIZE = 1000
    xy = np.random.rand(SIZE, 2) * 2 - 1
    c = pos_to_color(xy)
    plt.scatter(xy[:, 0], xy[:, 1], s=10, c=c/255)
    plt.show()