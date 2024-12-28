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

def Correlation2(amp, phase):
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
def Correlation3(a,b,c):
    x = np.linspace(-100,100,20001)
    #x = np.arange(-1, 1, 0.01)sine
    #func = getattr(src.library.numpy.func_loader, func)
    #func = lambda v: func(v, args.k, args.seed)
    g_func = acc_clip(x, clip_max=1.0)
    f_func = gaussian_optimize(x,a,b,c)
    up = np.sum((g_func - np.mean(g_func))*(f_func - np.mean(f_func)))
    down = np.sqrt(np.sum(np.abs(g_func - np.mean(g_func))**2)) * np.sqrt(np.sum(np.abs(f_func - np.mean(f_func))**2))
    nita = up/down
    return  nita

if __name__ == "__main__":

    cor_lst3 = []

    #cor, func = Correlation(k, 0)
    lst = [0.01, 0.1, 1.0, 10, 100]
    for amp in lst:
        #func = lambda v: opto(amp*v + 150)
        cor = Correlation2(amp, 150)
        cor_lst3.append(cor)
    #
    print(cor_lst3)
    #
    cor_lst4 = []
    #cor, func = Correlation(k, 0)
    lst = [0.01, 0.1, 1.0, 10, 100]
    for c in lst:
        #func = lambda v: opto(amp*v + 150)
        cor = Correlation3(1.0, 0.4, c)
        cor_lst4.append(cor)

    print(cor_lst4)


    #plot gaussian and opto
    x = lst
    plt.figure('f1', figsize=(6,3))
    plt.plot(x, cor_lst4, '.-',color= '#FFA500')
    plt.xscale('log')
    plt.xticks(x)
    plt.ylabel('$\eta$ between $f\prime$ and $g$')
    plt.grid(linestyle='--')
    plt.savefig('gaussian_cor.png', dpi=600)
    plt.show()

    plt.figure('f2', figsize=(6,3))
    plt.plot(x, cor_lst3, '.-',color= '#FFA500')
    plt.xscale('log')
    plt.xticks(x)
    plt.ylabel('$\eta$ between $f\prime$ and $g$')
    plt.grid(linestyle='--')
    plt.savefig('opto_cor.png', dpi=600)
    plt.show()