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
from src.library.numpy.func_loader import gaussian_optimize
from src.library.numpy.func_loader import gaussian_normalize
from src.library.numpy.func_loader import Fourier
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
#
# args = parser.parse_args()
# func = getattr(src.library.numpy.func_loader, args.bfunc)
# bfunc = lambda v: func(v, args.k, args.seed)

def Correlation(k,s, amp, phase):
    x = np.linspace(-100,100,20001)
    #x = np.arange(-1, 1, 0.01)
    g_func = acc_clip(x, clip_max=1.0)
    f_func = Fourier(amp*x+phase,k,s)
    up = np.sum((g_func - np.mean(g_func))*(f_func - np.mean(f_func)))
    down = np.sqrt(np.sum(np.abs(g_func - np.mean(g_func))**2)) * np.sqrt(np.sum(np.abs(f_func - np.mean(f_func))**2))
    nita = up/down
    return  nita

def Correlation_test3(k,s, amp, phase):
    x = np.linspace(-100,100,20001)
    # x = np.arange(-1, 1, 0.01)
    g = acc_clip(x,clip_max=1.0)
    f = Fourier(amp*x+phase,k,s)
    return  np.dot(f,g) / np.linalg.norm(f) / np.linalg.norm(g)

def Correlation2(amp, phase, low, high):
    x = np.linspace(low, high ,20001)
    # x = np.arange(-1, 1, 0.01)
    g_func = acc_clip(x,clip_max=1.0)
    f_func = opto(amp*x + phase)
    up = np.sum((g_func - np.mean(g_func))*(f_func - np.mean(f_func)))
    down = np.sqrt(np.sum(np.abs(g_func - np.mean(g_func))**2)) * np.sqrt(np.sum(np.abs(f_func - np.mean(f_func))**2))
    nita = up/down
    return  nita

def Correlation2_ori(amp, phase):
    x = np.linspace(-100,100,20001)
    # x = np.arange(-1, 1, 0.01)
    g_func = acc_clip(x,clip_max=1.0)
    f_func = opto(amp*x + phase)
    up = np.sum((g_func - np.mean(g_func))*(f_func - np.mean(f_func)))
    down = np.sqrt(np.sum(np.abs(g_func - np.mean(g_func))**2)) * np.sqrt(np.sum(np.abs(f_func - np.mean(f_func))**2))
    nita = up/down
    return  nita
#
# def Correlation_test(amp, phase):
#     x = np.linspace(-100,100,20001)
#     # x = np.arange(-1, 1, 0.01)
#     g = acc_clip(x,clip_max=1.0)
#     f = opto(amp*x + phase)
#     return  np.dot(f,g) / np.linalg.norm(f) / np.linalg.norm(g)
#
# def Correlation_test2(a,b,c):
#     x = np.linspace(-100,100,20001)
#     # x = np.arange(-1, 1, 0.01)
#     g = acc_clip(x,clip_max=1.0)
#     f = gaussian_optimize(x,a,b,c)
#     return  np.dot(f,g) / np.linalg.norm(f) / np.linalg.norm(g)
#
def Correlation3(a,b,c,low, high):
    x = np.linspace(low,high,20001)
    #x = np.arange(-1, 1, 0.01)sine
    #func = getattr(src.library.numpy.func_loader, func)
    #func = lambda v: func(v, args.k, args.seed)
    g_func = acc_clip(x, clip_max=1.0)
    f_func = gaussian_optimize(x,a,b,c)
    up = np.sum((g_func - np.mean(g_func))*(f_func - np.mean(f_func)))
    down = np.sqrt(np.sum(np.abs(g_func - np.mean(g_func))**2)) * np.sqrt(np.sum(np.abs(f_func - np.mean(f_func))**2))
    nita = up/down
    return  nita


def Correlationf():
    x = np.linspace(-100,100,20001)
    #x = np.arange(-1, 1, 0.01)
    g_func = acc_clip(x, clip_max=1.0)
    f_func = accurate(x)
    up = np.sum((g_func - np.mean(g_func))*(f_func - np.mean(f_func)))
    down = np.sqrt(np.sum(np.abs(g_func - np.mean(g_func))**2)) * np.sqrt(np.sum(np.abs(f_func - np.mean(f_func))**2))
    nita = up/down
    return  nita
if __name__ == "__main__":
    k=4
    amp = [0.01, 0.1, 1.0, 10, 100]
    amp_dis = []
    amp_ori =[]
    for a in amp:
        #coe = Correlation2(a, 150)
        coe_ori = Correlation2_ori(a, 150)
        #amp_dis.append(coe)
        amp_ori.append(coe_ori)
    # amp_dis.append(Correlation2(0.01, 150, -15,30))
    # amp_dis.append(Correlation2(0.1, 150, -10, 20))
    # amp_dis.append(Correlation2(1, 150, -20, 20))
    # amp_dis.append(Correlation2(10, 150, -20, 30))
    # amp_dis.append(Correlation2(100, 150, -25, 35))
    # for a in amp:
    #     #coe = Correlation2(a, 150)
    #     coe_ori = Correlation3(1.0,0.4,a, -100 ,100)
    #     #amp_dis.append(coe)
    #     amp_ori.append(coe_ori)
    # # amp_dis.append(Correlation3(1.0,0.4,0.01, -10, 10))
    # # amp_dis.append(Correlation3(1.0, 0.4, 0.1, -10, 10))
    # # amp_dis.append(Correlation3(1.0, 0.4, 1, -15, 15))
    # # amp_dis.append(Correlation3(1.0, 0.4, 10, -20, 25))
    # # amp_dis.append(Correlation3(1.0, 0.4, 100, -20, 25))
    fig = plt.figure(figsize=(7,4))
    #plt.plot(amp, amp_dis,  '.-',color='#FF4500', label=' ')
    plt.plot(amp, amp_ori, '.-' ,color='#FF4500', label=' ')
    plt.xscale('log')
    plt.xticks(amp)
    plt.ylim((-0.1,0.6))
    #plt.legend(bbox_to_anchor=(0.7, 0.7), fontsize=18, frameon=False)
    #plt.grid( linestyle='--')
    plt.tick_params(labelsize=20)
    plt.savefig('correlation_opto.png', dpi=500)
    plt.show()
    #
    # cor_sl = []
    # lst_04_06 = []
    # lst_02_04 = []
    # lst_00_02 = []
    # lst_n02_00 = []
    # lst_n04_n02 = []
    # lst_n06_n04 = []
    #
    #
    # for seed in range(0, 10000):
    #     cor = Correlation(k, seed, amp=0.01, phase=0)
    #     if cor>=0.4:
    #         lst_04_06.append((seed, cor))
    #     elif cor>=0.2:
    #         lst_02_04.append((seed, cor))
    #     elif cor >= 0.0:
    #         lst_00_02.append((seed, cor))
    #     elif cor >= -0.2 :
    #         lst_n02_00.append((seed,cor))
    #     elif cor >= -0.4:
    #         lst_n04_n02.append((seed, cor))
    #     else:
    #         lst_n06_n04.append((seed, cor))


    #print(lst_04_06)
    #print(lst_02_04)
    #print(lst_00_02)
    #print(lst_n02_00)
    #print(lst_n04_n02)
    #print(lst_n06_n04)

