import matplotlib.pyplot as plt
import numpy as np
import src.library.numpy.func_loader
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
import pandas as pd
from src.library.numpy.func_loader import tanh
from src.library.numpy.func_loader import gaussian_optimize
from src.library.numpy.func_loader import gaussian_normalize
#from src.library.numpy.func_loader import Fourier
from src.library.numpy.func_loader import opto
from src.library.numpy.func_loader import approx
from src.library.numpy.func_loader import accurate
from src.library.numpy.func_loader import heaviside
from src.library.numpy.func_loader import acc_clip
from src.library.numpy.func_loader import sign
from src.library.numpy.func_loader import square
from src.library.numpy.func_loader import triangle
from src.library.numpy.func_loader import affine
from src.library.numpy.func_loader import sine
from functools import reduce
from collections import defaultdict
from sklearn.datasets import fetch_openml
from torchvision import datasets, transforms
# parser = argparse.ArgumentParser()
# parser.add_argument('--bfunc', type=str, required=True)
# parser.add_argument('--k', type=str, default=4)
# #parser.add_argument('--seed', type=str, default=0)
#
# args = parser.parse_args()
# func = getattr(src.library.numpy.func_loader, args.bfunc)
# bfunc = lambda v: func(v, args.k, args.seed)


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
        st = np.sin(ot).dot(self.coefs[:self.dim])
        ct = np.cos(ot).dot(self.coefs[self.dim:])
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

def record_parameter(path,path2):
    rng = [-0.6,-0.4,-0.2,0.0,0.2,0.4,0.6]
    x = np.linspace(-100,100,20001)
    for t in range(len(rng) - 1):
        lst=[]
        func_lst=[]
        func_lst2 = []
        for n in os.listdir(path + '/' + str(rng[t]) + '_' + str(rng[t + 1])):
            print(n)
            seed = int(n)
            bfunc = Fourier(4, seed)
            func = bfunc(0.01*x)
        #     para =list(para)
        #     #record parameters
        #     lst.append({seed:para})
        #     #record func
            func_lst.append(func)

        for n in os.listdir(path2 + '/' + str(rng[t]) + '_' + str(rng[t + 1])):
            print(n)
            seed = int(n)
            bfunc = Fourier(4, seed)
            func = bfunc(0.01 * x)
            func_lst2.append(func)
        #
        # with open(f'{path}/parameter'+str(rng[t]) + '_' + str(rng[t + 1])+'.json', mode="w") as f:
        #     json.dump(lst, f, indent=4)
        # plot figures of function
        fig = plt.figure(figsize=(7,5))
        for i in range(len(func_lst)):
            plt.plot(x, func_lst[i], color='#6495ED', linewidth=2)
            plt.plot(x, func_lst2[i], color='#F4A460', linewidth=2)

        ytick =[0.0,0.5,1.0,1.5]
        xtick = [-100,-50,0,50,100]
        ori_func = accurate(x)
        plt.plot(x, ori_func, color='#808080',linewidth=2)
        plt.ylim(0,1.5)
        plt.xticks(xtick)
        plt.yticks(ytick)
        plt.minorticks_on()
        plt.tick_params(which='major', width=1.5, length=5, direction='in')
        plt.tick_params(which='minor', direction='in')
        plt.ylabel('')
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.tight_layout()
        #plt.tick_params(labelsize=15)
        plt.savefig(path+'/function'+str(rng[t]) + '_' + str(rng[t + 1])+'.png', dpi=500)
        plt.show()


if __name__ == "__main__":
    x = np.linspace(-100, 100, 20001)
    record_parameter('/home/isi/zhang/SNN/src/script/Fourier/PRFS_Result/correlation_coefficient','/home/isi/zhang/SNN/src/script/Fourier_BP/trial_2' )

    # opto_MNIST = opto(0.15*x + 155*math.pi /180)
    # opto_FMNIST = opto(0.1*x + 160*math.pi/180)
    # Gau_MNIST = gaussian_optimize(x, a=1, b=0.4, c=13)
    # Gau_FMNIST = gaussian_optimize(x, a=1, b=0.1, c=9)
    # f = acc_clip(x)
    #
    # fig1 = plt.figure(figsize=(7,6))
    # plt.plot(x, f, label='f', color='#808080')
    # plt.plot(x, opto_MNIST,label='opto', color='#CD5C5C')
    # plt.plot(x, Gau_MNIST,label='Gaussian', color='#F4A460')
    # plt.ylabel('Output Value y', fontsize=15)
    # plt.xlabel('Input Value x', fontsize=15)
    # plt.title('MNIST')
    # plt.legend(loc='lower right', bbox_to_anchor=(0.9, 0.1))
    # plt.savefig('MNIST.png', dpi=500)
    # # plt.xticks(0.01, 0.26, 0.05)
    # plt.show()
    #
    # fig2 = plt.figure(figsize=(7,6))
    # plt.plot(x, f, label='f', color='#808080')
    # plt.plot(x, opto_FMNIST,label='opto', color='#CD5C5C')
    # plt.plot(x, Gau_FMNIST,label='Gaussian', color='#F4A460')
    # plt.ylabel('Output Value y', fontsize=15)
    # plt.xlabel('Input Value x', fontsize=15)
    # plt.title('Fashion MNIST')
    # plt.legend(loc='lower right', bbox_to_anchor=(0.9, 0.1))
    # plt.savefig('F_MNIST.png', dpi=500)
    # # plt.xticks(0.01, 0.26, 0.05)
    # plt.show()
