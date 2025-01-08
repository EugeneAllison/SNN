#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import json
import time
import math
import copy
import joblib
import operator
import argparse
import itertools
import functools
import scipy.stats
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from functools import reduce
from collections import defaultdict
from sklearn.datasets import fetch_openml
from torchvision import datasets, transforms

sys.path.append(".")
# sys.path.append("/Users/zhangyongbo/Desktop/Systematically analyze aDFA/project_SNN/spiking_dfa_newtork (1)/")


from pyutils.figure import Figure
from pyutils.tqdm import tqdm, trange

import src.library.style
from src.library.numpy.func_loader import Fourier
from src.library.numpy.spiking_network import SpikingNetwork


parser = argparse.ArgumentParser()
# parser.add_argument('--root_dir', type=str, required=True)
parser.add_argument('--init_id', type=int, default=0)
parser.add_argument('--trial_num', type=int, default=1)
parser.add_argument('--net_dims', type=int, nargs="+", default=[784, 1000, 10])
parser.add_argument('--n_epochs', type=int, default=20)
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--lr', type=float, default=1)
# parser.add_argument('--bfunc', type=str, required=True)
parser.add_argument('--k', type=float, default=4) #check approx
parser.add_argument("--seed", type=float, default=5351)
parser.add_argument('--amp', type=float, default=0.01) #check approx


parser.add_argument('--radian', action="store_true")
parser.add_argument('--final_only', action="store_true")
parser.add_argument('--use_bp', action="store_true")

parser.add_argument('--T', type=float, default=100)
parser.add_argument('--T_th', type=float, default=20)
parser.add_argument('--dt', type=float, default=0.25)

# introduce the interface of spectral radius
parser.add_argument('--rho', type=float, default=0)

args = parser.parse_args()


class Fourier(object):
    def __init__(self, dim, seed=None, resolution=10001):
        self.dim = dim
        self.rnd = np.random.RandomState(seed)
        self.coefs = self.rnd.uniform(-1, 1, dim * 2)
        self.coefs /= np.abs(self.coefs).sum()
        self.resolution = resolution

    def __call__(self, t):
        return self._func(t) - self.m

    def _func(self, t):
        fs = np.pi * np.arange(1, self.dim + 1)
        ot = np.outer(t, fs)
        st = np.sin(ot).dot(self.coefs[: self.dim])
        ct = np.cos(ot).dot(self.coefs[self.dim :])
        out = st + ct
        return out.reshape(*t.shape)

    @property
    def m(self):
        if hasattr(self, "_m"):
            return self._m
        ts = np.linspace(-1, 1, self.resolution)
        ys = self._func(ts)
        self._m = np.min(ys)
        return self._m


# func = getattr(src.library.numpy.func_loader, args.bfunc)
func = Fourier(args.k, args.seed)
bfunc = lambda v: func(args.amp*v)


# 准备绘制数据
x = np.linspace(-100, 100, 10000)  # 定义输入范围，分为500个点
y = bfunc(x)  # 计算对应的输出值

# 确保保存图像的目录存在
save_path = "./fig_deBug"
if not os.path.exists(save_path):  # 如果路径不存在，则创建
    os.makedirs(save_path)

# 绘图
plt.figure(figsize=(8, 5))
plt.plot(x, y, label="bfunc", color="blue")
plt.title("Plot of bfunc", fontsize=14)
plt.xlabel("Input (v)", fontsize=12)
plt.ylabel("Output (bfunc(v))", fontsize=12)
plt.axhline(0, color="gray", linestyle="--", linewidth=0.8)  # 添加y=0的水平线
plt.legend()
plt.grid()

# 保存图像到指定路径
file_name = os.path.join(save_path, "bfunc_plot_5351(mine0.5).png")  # 定义保存的文件名
plt.savefig(file_name, dpi=300, bbox_inches="tight")  # 保存图像，设置高分辨率
