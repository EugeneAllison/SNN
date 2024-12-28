import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import json
import os
import sys
from src.script.Fourier.calculate_cc import Correlation
import seaborn as sns
from csv import reader
from pandas import read_csv
from matplotlib.pyplot import MultipleLocator
#sys.path.append('..')

def get_max(content):
    #acc_t = max(content['acc_t'])
    acc_e = max(content['acc_e'])
    return acc_e


# PATH2 = '../node_opto'
# file_list2 = os.listdir(PATH2)

def get_data(path):
    lst = [[],[],[],[],[],[]]
    lst_cor = [[],[],[],[],[],[]]
    rng = [-0.6,-0.4,-0.2,0.0,0.2,0.4,0.6]
    print(os.listdir(path))
    for t in range(len(rng)-1):
        print(os.listdir(path+'/'+str(rng[t])+'_'+str(rng[t+1])))
        for n in os.listdir(path+'/'+str(rng[t])+'_'+str(rng[t+1])):
#calculate correlation coefficient
            lst_cor[t].append(Correlation(k=4,s=int(n), amp=0.01, phase=0))
#get the accuracy
            with open(path+'/'+str(rng[t])+'_'+str(rng[t+1])+'/'+n+'/records.json') as f:
                content = json.loads(f.read())
                acc = get_max(content)
                lst[t].append(acc)

    return lst,lst_cor


def get_data_BP(path):
    lst = [[] for _ in range(6)]
    list_dir = os.listdir(path)
    for interval in list_dir:
        if interval== '0.4_0.6':
        #get the accuracy
            for i in os.listdir(path+'/'+interval):
                with open(path+'/'+ interval +'/'+i+'/records.json') as f:
                    content = json.loads(f.read())
                    acc = get_max(content)
                    lst[-1].append(acc)

        elif interval== '0.2_0.4':
        #get the accuracy
            for i in os.listdir(path+'/'+interval):
                with open(path+'/'+ interval +'/'+i+'/records.json') as f:
                    content = json.loads(f.read())
                    acc = get_max(content)
                    lst[-2].append(acc)

        elif interval== '0.0_0.2':
        #get the accuracy
            for i in os.listdir(path+'/'+interval):
                with open(path+'/'+ interval +'/'+i+'/records.json') as f:
                    content = json.loads(f.read())
                    acc = get_max(content)
                    lst[-3].append(acc)


        elif interval== '-0.2_0.0':
        #get the accuracy
            for i in os.listdir(path+'/'+interval):
                with open(path+'/'+ interval +'/'+i+'/records.json') as f:
                    content = json.loads(f.read())
                    acc = get_max(content)
                    lst[-4].append(acc)


        elif interval == '-0.4_-0.2':
            # get the accuracy
            for i in os.listdir(path + '/' + interval):
                with open(path + '/' + interval + '/' + i + '/records.json') as f:
                    content = json.loads(f.read())
                    acc = get_max(content)
                    lst[-5].append(acc)

        else:
            # get the accuracy
            for i in os.listdir(path + '/' + interval):
                with open(path + '/' + interval + '/' + i + '/records.json') as f:
                    content = json.loads(f.read())
                    acc = get_max(content)
                    lst[-6].append(acc)

    return lst

if __name__ == '__main__':

    #accuracy, correlation = get_data('PRFS_Result/correlation_coefficient')
    # x_axis = np.array(np.mean(correlation,-1))
    # print(x_axis,accuracy)
    filename = "view_search_random_fourier.csv"
    with open(filename, 'rt', encoding='UTF-8') as raw_data:
        readers = reader(raw_data, delimiter=',')
        x = list(readers)

    print(x)
    acc = x[0].index('accuracy')
    cor = x[0].index('correlation')
    acc_list =[[] for i in range(6)]
    x_axis =[[] for i in range(6)]
    for search in range(1,len(x)):
        accuracy = float(x[search][acc])
        cc = float(x[search][cor])
        if 0.4<cc<0.6:
            x_axis[-1].append(cc)
            acc_list[-1].append(accuracy)
        elif 0.2<cc<0.4:
            x_axis[-2].append(cc)
            acc_list[-2].append(accuracy)
        if 0.0<cc<0.2:
            x_axis[-3].append(cc)
            acc_list[-3].append(accuracy)
        if -0.2<cc<0.0:
            x_axis[-4].append(cc)
            acc_list[-4].append(accuracy)
        if -0.4<cc<-0.2:
            x_axis[-5].append(cc)
            acc_list[-5].append(accuracy)
        if -0.6<cc<-0.4:
            x_axis[-6].append(cc)
            acc_list[-6].append(accuracy)


    x_axis = np.array(np.mean(x_axis,-1))
    print(x_axis)
    acc_bp = get_data_BP('/home/isi/zhang/SNN/src/script/Fourier_BP/trial_2')
    x_bp = [-0.49719148, -0.28362246, -0.10907576,  0.11392425,  0.29413608,  0.49517264]


    fig, ax = plt.subplots(figsize=(9,8))

    violin = ax.violinplot(acc_list, x_axis, showmeans=True, widths=0.12)
    for patch in violin['bodies']:

        patch.set_facecolor('#6495ED')
        patch.set_edgecolor('#6495ED')

    violin['cmeans'].set_color('green')

    violin2 = ax.violinplot(acc_bp, x_bp, showmeans=True, widths=0.12)
    for patch in violin2['bodies']:

        patch.set_facecolor('#F4A460')
        patch.set_edgecolor('#F4A460')


    #plt.axhline(0.9786, ls="--", color='r', label=' ', linewidth=1)
    plt.axhline(0.9778, ls="--", color='#000000', label='', linewidth=2)
    plt.axhline(0.9675, ls="--", color='#CD5C5C', label='', linewidth=2)
    plt.xlim((-0.6,0.6))
    plt.ylim((0.5, 1))
    plt.xticks(np.arange(-0.6, 0.7, 0.2))
    plt.yticks(np.arange(0.5, 1.01, 0.1))
    # plt.xlabel(r'$\eta$', fontsize=13)
    # plt.ylabel('Accuracy',fontsize=13)
    plt.minorticks_on()
    plt.tick_params(which='major', width=1.5, length=5, direction='in')
    plt.tick_params(which='minor', direction='in')
    plt.ylabel('')
    plt.tight_layout()
    #plt.title('Correlation coefficient and performance',fontsize=15)
    plt.grid( linestyle='-')
    #plt.legend(bbox_to_anchor=(0.8, 0.4), fontsize=18, frameon=False)
    plt.tick_params(labelsize=20)
    fig.savefig('/home/isi/zhang/SNN/src/script/Fourier_BP/Fourier_cc_t2.png', dpi=500)
    plt.show()