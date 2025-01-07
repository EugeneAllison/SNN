import matplotlib.pyplot as plt
import numpy as np
import sys

sys.path.append(".")
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

# from src.library.numpy.func_loader import gaussian_optimize
# from src.library.numpy.func_loader import gaussian_normalize

from src.library.numpy.func_loader import opto
from src.library.numpy.func_loader import acc_clip
from src.library.numpy.func_loader import approx
from src.library.numpy.func_loader import accurate
from functools import reduce
from collections import defaultdict
from sklearn.datasets import fetch_openml
from torchvision import datasets, transforms

# parser = argparse.ArgumentParser()
# parser.add_argument('--bfunc', type=str, required=True)
# parser.add_argument('--k', type=str, default=4)
# #parser.add_argument('--seed', type=str, default=0)


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



def Correlation(k, s, amp, phase):
    x = np.linspace(-100, 100, 20001)
    # x = np.arange(-1, 1, 0.01)
    g_func = acc_clip(x, clip_max=1.0)
    f_func = Fourier(k, s)
    f_func = f_func(amp * x + phase)
    up = np.sum((g_func - np.mean(g_func)) * (f_func - np.mean(f_func)))
    down = np.sqrt(np.sum(np.abs(g_func - np.mean(g_func)) ** 2)) * np.sqrt(
        np.sum(np.abs(f_func - np.mean(f_func)) ** 2)
    )
    nita = up / down
    return nita



def Correlation_test3(k, s, amp, phase):
    x = np.linspace(-100, 100, 20001)
    # x = np.arange(-1, 1, 0.01)
    g = acc_clip(x, clip_max=1.0)
    f = Fourier(amp * x + phase, k, s)
    return np.dot(f, g) / np.linalg.norm(f) / np.linalg.norm(g)



# def Correlation2(amp, phase):
#     x = np.linspace(-100,100,20001)
#     # x = np.arange(-1, 1, 0.01)
#     g_func = acc_clip(x,clip_max=1.0)
#     f_func = opto(amp*x + phase)
#     up = np.sum((g_func - np.mean(g_func))*(f_func - np.mean(f_func)))
#     down = np.sqrt(np.sum(np.abs(g_func - np.mean(g_func))**2)) * np.sqrt(np.sum(np.abs(f_func - np.mean(f_func))**2))
#     nita = up/down
#     return  nita
#
# def Correlation_test(amp, phase):
#     x = np.linspace(-100,100,20001)
#     # x = np.arange(-1, 1, 0.01)
#     g = acc_clip(x,clip_max=1.0)
#     f = opto(amp*x + phase)
#     return  np.dot(f,g) / np.linalg.norm(f) / np.linalg.norm(g)
#
## f: gaussian_optimize
# def Correlation_test2(a,b,c):
#     x = np.linspace(-100,100,20001)
#     # x = np.arange(-1, 1, 0.01)
#     g = acc_clip(x,clip_max=1.0)
#     f = gaussian_optimize(x,a,b,c)
#     return  np.dot(f,g) / np.linalg.norm(f) / np.linalg.norm(g)
#
## gaussian_optimize
# def Correlation3(a,b,c):
#     x = np.linspace(-100,100,20001)
#     #x = np.arange(-1, 1, 0.01)sine
#     #func = getattr(src.library.numpy.func_loader, func)
#     #func = lambda v: func(v, args.k, args.seed)
#     g_func = acc_clip(x, clip_max=1.0)
#     f_func = gaussian_optimize(x,a,b,c)
#     up = np.sum((g_func - np.mean(g_func))*(f_func - np.mean(f_func)))
#     down = np.sqrt(np.sum(np.abs(g_func - np.mean(g_func))**2)) * np.sqrt(np.sum(np.abs(f_func - np.mean(f_func))**2))
#     nita = up/down
#     return nita

if __name__ == "__main__":
    k = 4
    cor_lst001 = []  
    cor_lst01 = []
    cor_lst1 = []
    cor_lst0001 = []
    cor_sl = []
    lst_05 = []
    lst_04 = []
    lst_03 = []
    lst_02 = []
    lst_01 = []
    lst_00 = []
    # cor, func = Correlation(k, 0)
    for seed in range(0, 10000):  
        cor_001 = Correlation(k, seed, amp=0.01, phase=0)
        # if cor>=0.5:
        #     lst_05.append((seed, cor))
        # elif cor>=0.4:
        #     lst_04.append((seed, cor))
        # elif cor >= 0.3:
        #     lst_03.append((seed, cor))
        # elif cor >=0.2 :
        #     lst_02.append((seed,cor))
        # elif cor >=0.1:
        #     lst_01.append((seed, cor))
        # else:
        #     lst_00.append((seed, cor))
        cor_lst001.append(cor_001)
        # cor_sl.append((seed,cor))

    for i in range(0, 10000):
        cor_0001 = Correlation(k, i, amp=0.001, phase=0)
        cor_lst0001.append(cor_0001)

    for n in range(0, 10000):
        cor_01 = Correlation(k, n, amp=0.1, phase=0)
        cor_lst01.append(cor_01)

    for q in range(0, 10000):
        cor_1 = Correlation(k, q, amp=1, phase=0)
        cor_lst1.append(cor_1)
    # sorted_list = sorted(
    #     cor_sl,
    #     key=lambda t: t[1],
    #     reverse=True
    # )
    # print(sorted_list)
    tick = [-1.0, -0.5, 0, 0.5, 1.0]
    # fig1=plt.figure(1, figsize=(4,3))
    # plt.hist(cor_lst1, bins=50, range=(-1,1), density=True, color='#ADD8E6')
    # plt.tick_params(labelsize=20)
    # plt.xticks(tick)
    # fig1.savefig('distribution_1.png', dpi=500)
    ## ax1.set_title('amp = 1')
    ## ax1.set_ylabel('Density')
    
    # fig2 = plt.figure(1, figsize=(4, 3))
    # plt.hist(cor_lst01, bins=50, range=(-1,1), density=True, color='#ADD8E6')
    # plt.tick_params(labelsize=20)
    # plt.xticks(tick)
    # fig2.savefig('distribution_01.png', dpi=500)
    # ax2.set_title('amp = 0.1')

    # fig3 = plt.figure(1, figsize=(4, 3))
    # plt.hist(cor_lst001, bins=50, range=(-1,1), density=True, color='#ADD8E6')
    # plt.tick_params(labelsize=20)
    # plt.xticks(tick)
    # fig3.savefig('distribution_001.png', dpi=500)
    # ax3.set_title('amp = 0.01')
    # ax3.set_xlabel('Correlation Coefficient')
    # ax3.set_ylabel('Density')

    fig4 = plt.figure(1, figsize=(4, 3))
    plt.hist(cor_lst0001, bins=50, range=(-1, 1), density=True, color="#ADD8E6")
    plt.tick_params(labelsize=20)
    plt.xticks(tick)
    fig4.savefig("distribution_0001.png", dpi=500)
    # ax4.set_title('amp = 0.001')
    # ax4.set_xlabel('Correlation Coefficient')

    # fig.suptitle('Distribution of Correlation Coefficient', fontsize=18)

    plt.show()
    # print(lst_05)
    # print(lst_04)
    # print(lst_03)
    # print(lst_02)
    # print(lst_01)
    # print(lst_00)

    #
    # # opto_lst =[]
    # # for phase in range(0, 190, 15):
    # #     mid = np.abs(Correlation2(0.1, phase))
    # #     opto_lst.append(mid)
    #
    #
    # index = np.argsort(cor_lst)
    # sorted = sorted(cor_lst)
    # print(sorted, index)
    # # print(opto_lst)
    # # x = np.arange(0, 190, 15)
    # x = np.arange(0,200,1)
    # fig = plt.figure()
    # plt.plot(x, cor_lst, c='b')
    # plt.plot(x, sorted, c='r')
    # plt.show()
    # # x = np.arange(-1, 1, 0.01)
    # # plt.plot(x, func)
    # # plt.show()

    # cor_lst2 = []
    # func_lst= ["tanh", "dtanh", "opto", "dopto", "gaussian", "acc_clip", "approx", "sine", "cosine", "affine","heaviside", "triangle", "square", "sign", "linear"]
    # for i in func_lst:
    #     cor = Correlation3(i)
    #     cor_lst2.append((i,cor))
    # print(cor_lst2)
    #
    # cor_lst3 = []
    # cor_lst5 = []
    # #cor, func = Correlation(k, 0)
    # lst = [0.01, 0.1, 1.0, 10, 100]
    # for amp in lst:
    #     #func = lambda v: opto(amp*v + 150)
    #     cor = Correlation_test(amp, 150)
    #     cor_lst3.append(cor)
    #
    # print(cor_lst3)
    #
    # cor_lst4 = []
    # #cor, func = Correlation(k, 0)
    # lst = [0.01, 0.1, 1.0, 10, 100]
    # for c in lst:
    #     #func = lambda v: opto(amp*v + 150)
    #     cor = Correlation_test2(1.0, 0.4, c)
    #     cor_lst4.append(cor)
    #
    # print(cor_lst4)

    # color_bar = ['#9370DB', "#87CEFA", "#87CEFA", '#9370DB', "#87CEFA", "#87CEFA", "#87CEFA", '#9370DB', "#87CEFA",'#9370DB',"#87CEFA", "#87CEFA", "#87CEFA",'#9370DB','#9370DB']
    # plt.figure(figsize=(15, 15), dpi=500)
    # plt.bar(range(len(func_lst)), cor_lst, width=0.5, color=color_bar)
    # for q in range(len(cor_lst)):
    #     plt.text(q, cor_lst[q] + 0.02, "%.3f" % cor_lst[q], fontsize=10, horizontalalignment='center', fontweight='bold')
    # plt.xticks(range(len(func_lst)), func_lst, rotation=320, fontdict={ "size": 10}, fontweight='bold')
    # plt.ylim([-0.1, 1.1])
    # plt.ylabel('Correlation Coefficient', fontdict={"size": 20}, fontweight='bold')
    # plt.xlabel('Backward Function', fontdict={ "size": 20}, fontweight='bold')
    # plt.title('Application Scope of aDFA', fontdict={ "size": 23},
    #           fontweight='bold')
    # plt.show()

    # plot gaussian and opto
    # x = lst
    # plt.figure('f1', figsize=(6,3))
    # plt.plot(x, cor_lst4, '.-',color= '#FFA500')
    # plt.xscale('log')
    # plt.xticks(x)
    # plt.ylabel('$\eta$ between $f\prime$ and $g$')
    # plt.grid(linestyle='--')
    # plt.savefig('gaussian_cor.png', dpi=500)
    # plt.show()
    #
    # plt.figure('f2', figsize=(6,3))
    # plt.plot(x, cor_lst3, '.-',color= '#FFA500')
    # plt.xscale('log')
    # plt.xticks(x)
    # plt.ylabel('$\eta$ between $f\prime$ and $g$')
    # plt.grid(linestyle='--')
    # plt.savefig('opto_cor.png', dpi=500)
    # plt.show()
