# -*- coding: utf-8 -*-
"""
Created on Fri Jun  3 16:46:02 2022

@author: Qingsong 'Borges' Liu (qsliu1997@gmail.com) of Richards Lab at Northwestern University
"""

import cv2
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

#%% Functions
def reduce(samples,path_): # the main function calling other functions to convert the videos into a list of R_min of each frame
    path = os.getcwd() + path_
    data = {}
    
    for i in samples:
        #Find Rmin:
        if '_8000' in i:
            data['R0'] = getR0(i,path)
        else:
            name = i.split('_')[-3]    
            data[name] = getR(i,path)
    return data

def getR(i,path): # calculate the R_min in each frame in only the 19000fps videos
    R=[]
    thresh = getThre(i,path)
    data = getF(i,thresh,path)
    # data = getF_show(i,thresh,path)
    for i in data:
        R.append(calRmin(data[i]))
    return R
    
def getR0(i,path): # calculate the nozzel size R0 in only the 8000fps video
   data = getF_gray(i,path)
   thresh = np.min(data[0][0:20])+0.4*(np.max(data[0][0:20])-np.min(data[0][0:20]))
   data = getF(i,thresh,path)
   R = calRmin(data[0][0:20])
   return R

def getThre(i,path): # calculate the threshold value in each video, which will be used to define the edge
    data=getF_gray(i,path)
    l = len(data)
    return np.min(data[l-1])+0.4*(np.max(data[l-1])-np.min(data[l-1]))

def getF(i,thresh,path): # use the threshold value to find the edge of the liquid bridge in each frame
    f={}
    index = 0
    data=cv2.VideoCapture(os.path.join(path,i))
    
    while (data.isOpened()):
        ret, frame = data.read()
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            im_bw = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)[1]

            f[index]=im_bw
            index +=1 
        if ret==0:
            break
    return f

def calRmin(data): #find the R_min of each frame
    temp = np.gradient(data)[1]      
    x = np.arange(np.shape(temp)[0])
    y = np.arange(np.shape(temp)[1])
    xx,yy = np.meshgrid(x,y, indexing='ij')
    
    mask1 = data[:,0] == 0
    mask2 = data[:,-1] == 0
    
    mask1 = np.tile(mask1[:, np.newaxis], (1, yy.shape[1]))  
    mask2 = np.tile(mask2[:, np.newaxis], (1, yy.shape[1])) 
    
    mask = np.ma.masked_where(temp == 0,yy) & np.ma.masked_where(mask1, yy) & np.ma.masked_where(mask2, yy)
    ma = np.max(mask,axis = 1)
    mi = np.min(mask,axis = 1) 
    new = np.array(ma-mi)
    mask = new>1.9
    return np.min(ma-mi)  

def getF_gray(i,path): # obtain the black and white videos from orginal videos
    f={}
    index = 0
    data=cv2.VideoCapture(os.path.join(path,i))
    while (data.isOpened()):
        ret, frame = data.read()
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            f[index]=gray
            index +=1 
        if ret==0:
            break
    return f

#%% Main script

path_ = '\Data'
path=os.getcwd() + path_

temp = '1000_10cstar'
samples = [i for i in os.listdir(path) if temp in i and 'avi' in i]
print(samples)
data= reduce(samples,path_)

fig = plt.figure(dpi=300,figsize=(3,3))
data_output = {}
R0 = data['R0']

with pd.ExcelWriter(temp+'.xlsx') as writer:
    for i in data:
        if 'R0' not in i:
            data[i] = np.array(data[i])
                 
            x = np.linspace(0,len(data[i]),len(data[i]))/19
                              
            plt.plot(x, data[i]/R0,marker='o',linestyle='',
                     markersize=1,label=i)
            
plt.title(temp)
plt.tick_params(direction="in",which='both')
plt.rc('axes', labelsize=10)   
plt.legend(prop={'size': 7},loc = 'lower left',frameon=False )
plt.xlabel(r'$\rm t (ms)$')
plt.ylabel(r'$\rmR_{min}/R_0$')
plt.ylim([1e-2,1])
plt.yscale('log')  
           




