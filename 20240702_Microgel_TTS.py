# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 09:40:40 2024

@author: Matt Brucks
"""

#%% Import packages

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib as mpl
from matplotlib import cm
# from scipy.interpolate import InterpolatedUnivariateSpline as interpolate
import scipy.constants as sc
from mastercurves import MasterCurve
from mastercurves.transforms import Multiply, PowerLawAge
from matplotlib import rcParams
# import matplotlib.legend as mlegend
# from matplotlib.patches import Rectangle
import matplotlib.lines as mlines
rcParams['font.family'] = 'DejaVu Sans'

#%%Define Functions

def listfiles(pathname,ext):
    temp = []
    for file in os.listdir(pathname):
        if file.endswith(ext):
            temp.append(os.path.join(pathname,file))
    return temp

def figure_format(ax):
    ax.minorticks_off()
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(1.5)
    ax.tick_params(axis = 'both',
                   which = 'major',
                   direction = 'in',
                   length = 4,
                   width = 1.5,
                   labelsize = 14,
                   top = True,
                   right = True)
    plt.tick_params(axis = 'both',
                   which = 'minor',
                   direction = 'in',
                   length = 0,
                   width = 1.5,
                   labelsize = 14)
    
def Curve_Fitting_Function(x_vals, y_vals, function):
    error = {}
    popt, pcov = curve_fit(function, x_vals, y_vals, maxfev=100000) 
    x_vals_fitted = np.linspace(min(x_vals), max(x_vals), 50)
    y_vals_fitted = function(x_vals_fitted, *popt)
    # poly = np.polyfit(x_vals, y_vals, degree)
    # y_vals_fitted = poly[0]*x_vals**2 + poly[1]*x_vals + poly[2]
    residuals = []
    for i,val in enumerate(x_vals):
        resid = y_vals[i] - function(val, *popt)
        # resid = y_vals[i] - y_vals_fitted[i]
        residuals.append(resid)
    residuals = np.asarray(residuals)
    
    ss_res = np.sum(residuals**2) #residual sum of squares (ss_res)
    ss_tot = np.sum((y_vals-np.mean(y_vals))**2) #total sum of squares (ss_tot)
    r_squared = 1 - (ss_res / ss_tot)
    
    error['parameters'] = popt
    error['fitted x values'] = x_vals_fitted
    error['fitted y values'] = y_vals_fitted
    error['R squared'] = round(r_squared,3)
    
    return error
        
#%%Data Loading

pathname = os.path.abspath(os.getcwd())
filelist = listfiles(pathname, '.xls')

data_tts = {}
for file in filelist:
    key = os.path.basename(file).rstrip('.xls')
    data_tts[key] = pd.read_excel(file, sheet_name='Time sweep - 1', skiprows=[0,2])

 
#change start of angular freq
for file in data_tts.keys():
    sweep = 0
    sweep_list = []
    for i in range(len(data_tts[file]['Angular frequency'])-1):
        if(data_tts[file]['Angular frequency'][i] == 1 and data_tts[file]['Angular frequency'][i+1] != 1):
            sweep += 1
        sweep_list.append(sweep)
    sweep_list.append(sweep)
    data_tts[file]['Sweep number'] = sweep_list
    
# #%%Data Plotting Individual Steps
# # ylim for LSR: (3852.6895160901113, 42847.77256526243)
# # ylim for FLSR: (5117.065394750503, 91246.44029622881)

# cmap = cm.get_cmap('jet')

# for file in data_tts.keys():
#     if('check' not in file):
#         steps = list(set(data_tts[file]['Sweep number']))
        
#         for k in steps:
#             fig, ax = plt.subplots(dpi = 300, figsize = (4,4))
#             figure_format(ax)
#             ax.set_ylabel('$G^{\prime}, G^{\prime\prime}$ (Pa)', fontsize=16)
#             ax.set_xlabel('$\omega$ (rad/s)', fontsize=16)
#             ax.set_yscale('log')
#             ax.set_xscale('log')
            
#             mask = data_tts[file]['Sweep number'] == k
#             temp = np.mean(data_tts[file]['Temperature'][mask])
                            
#             freq = np.asarray(data_tts[file]['Angular frequency'])
#             Gp = np.asarray(data_tts[file]['Storage modulus'])
#             Gpp = np.asarray(data_tts[file]['Loss modulus'])
#             tan = np.asarray(data_tts[file]['Tan(delta)'])
            
#             # if(temp > 48): #masking geometry inertial effects at high temperatures
#             #     mask1 = data_tts[file]['Angular frequency'][mask] < 10
#             # else:
#             #     mask1 = data_tts[file]['Angular frequency'][mask] < 200
        
            
#             ax.set_title(f'{temp:.1f}')
#             ax.plot(freq[mask],
#                     Gp[mask],
#                     marker = 'o',
#                     ls = 'None',
#                     color = cmap(k/max(steps)),
#                     label = r'$G^{\prime}$'
#                     )
#             ax.plot(freq[mask],
#                     Gpp[mask],
#                     marker = 'o',
#                     mec = cmap(k/max(steps)),
#                     ls = 'None',
#                     fillstyle='none',
#                     label = r'$G^{\prime\prime}$'
#                     )
            
#             ax.set_yticks([1e3,1e4,1e5])

#             ax.legend(frameon = False, loc='lower right', fontsize=14)
    
# plt.show()
#%%Data plotting

cmap = cm.get_cmap('jet')

for file in data_tts.keys():
    fig, ax = plt.subplots(dpi = 300, figsize = (5,5))
    figure_format(ax)
    ax.set_ylabel('$G^{\prime}$ (Pa)', fontsize=16)
    ax.set_xlabel('$\omega$ (rad/s)', fontsize=16)
    ax.set_yscale('log')
    ax.set_xscale('log')

    fig, ax0 = plt.subplots(dpi = 300, figsize = (5,5))
    figure_format(ax0)
    ax0.set_ylabel('$G^{\prime\prime}$ (Pa)', fontsize=16)
    ax0.set_xlabel('$\omega$ (rad/s)', fontsize=16)
    ax0.set_yscale('log')
    ax0.set_xscale('log')   
    
    steps = list(set(data_tts[file]['Sweep number']))
    
    freq_array = []
    Gp_array = []
    Gpp_array = []
    temperatures = []
    
    for k in steps:
        mask = data_tts[file]['Sweep number'] == k
        temp = np.mean(data_tts[file]['Temperature'][mask])
        temperatures.append(temp)
                        
        freq = np.asarray(data_tts[file]['Angular frequency'])
        Gp = np.asarray(data_tts[file]['Storage modulus'])
        Gpp = np.asarray(data_tts[file]['Loss modulus'])
        tan = np.asarray(data_tts[file]['Tan(delta)'])
        
        # if(temp > 48): #masking geometry inertial effects at high temperatures
        #     mask1 = data_tts[file]['Angular frequency'][mask] < 10
        # else:
        #     mask1 = data_tts[file]['Angular frequency'][mask] < 200
        
        freq_array.append(np.log(freq[mask])) 
        Gp_array.append(np.log(Gp[mask]))
        Gpp_array.append(np.log(Gpp[mask]))
    
        ax.plot(freq[mask],
                Gp[mask],
                marker = 'o',
                ls = 'None',
                color = cmap(k/max(steps)),
                label = f'{temp:.1f}'
                )
        ax0.plot(freq[mask],
                Gpp[mask],
                marker = 'o',
                mec = cmap(k/max(steps)),
                ls = 'None',
                fillstyle = 'none',
                # label = f'{temp:.1f}'
                )
    
    ax.set_yticks([1e3,1e4,1e5])
    ax0.set_yticks([1e3,1e4,1e5])

    handles, labels = ax.get_legend_handles_labels()       
    ax.legend(handles[::2], labels[::2], frameon = False, title = 'T ($\degree$C)', ncol = 2)    

plt.show()
#%% Generate master Curve
T_ref = min(temperatures) #Reference Temperature at Max 

data_shift = {}

for file in data_tts.keys():
    
    steps = list(set(data_tts[file]['Sweep number']))
    
    freq_array = []
    Gp_array = []
    Gpp_array = []
    temperatures = []
    
    for k in steps:
        mask = data_tts[file]['Sweep number'] == k
        temp = np.mean(data_tts[file]['Temperature'][mask])
        temperatures.append(temp)
                        
        freq = np.asarray(data_tts[file]['Angular frequency'])
        Gp = np.asarray(data_tts[file]['Storage modulus'])
        Gpp = np.asarray(data_tts[file]['Loss modulus'])
        tan = np.asarray(data_tts[file]['Tan(delta)'])
        
        # if(temp > 48): #masking geometry inertial effects at high temperatures
        #     mask1 = data_tts[file]['Angular frequency'][mask] < 10
        # else:
        #     mask1 = data_tts[file]['Angular frequency'][mask] < 200
        
        freq_array.append(np.log(freq[mask])) 
        Gp_array.append(np.log(Gp[mask]))
        Gpp_array.append(np.log(Gpp[mask]))

    "Shifting Gp"
    mc = MasterCurve()
    mc.add_data(freq_array, Gp_array, temperatures)
    mc.add_htransform(Multiply(scale = 'log'))
    mc.superpose()
    a_Gp = mc.hparams[0]
    print(temperatures)
    print(a_Gp)
    mc.change_ref(T_ref)
    
    fig1, ax1, fig2, ax2, fig3, ax3 = mc.plot(log=True, colormap=cmap)
    ax1.set_ylabel('$G^{\prime}$ (Pa)')
    ax1.set_xlabel('$\omega$ (rad/s)')
    ax1.set_title(file)
    ax2.set_ylabel('$G^{\prime}$ (Pa)')
    ax2.set_xlabel('$\omega$ (rad/s)')
    ax2.set_title(file)
    ax3.set_ylabel('$G^{\prime}$ (Pa)')
    ax3.set_xlabel('$a_{T}\cdot\omega$ (rad/s)')
    ax3.set_title(file)
    
    for figure in [ax1, ax2, ax3]:
        figure_format(figure)
        mpl.rcParams['figure.dpi'] = 300
        mpl.rcParams['figure.figsize'] = (6,4)
        mpl.rcParams['lines.markeredgecolor'] = 'black'
        handles, labels = ax1.get_legend_handles_labels()
        ax1.legend(handles[::2], labels[::2], frameon = False, ncol = 2, title = 'T ($\degree$C)')
    
    
    "Shifting Gpp"
    mc = MasterCurve()
    mc.add_data(freq_array, Gpp_array, temperatures)
    mc.add_htransform(Multiply(scale = 'log'))
    mc.superpose()
    a_Gpp = mc.hparams[0]
    print(temperatures)
    print(a_Gpp)
    data_shift[file] = a_Gpp
    
    mc.change_ref(T_ref)
    
    fig4, ax4, fig5, ax5, fig6, ax6 = mc.plot(log=True, colormap=cmap)
    ax4.set_ylabel('$G^{\prime\prime}$ (Pa)')
    ax4.set_xlabel('$\omega$ (rad/s)')
    ax4.set_title(file)
    ax5.set_ylabel('$G^{\prime\prime}$ (Pa)')
    ax5.set_xlabel('$\omega$ (rad/s)')
    ax5.set_title(file)
    ax6.set_ylabel('$G^{\prime\prime}$ (Pa)')
    ax6.set_xlabel('$a_{T}\cdot\omega$ (rad/s)')
    ax6.set_title(file)
    
    for figure in [ax4, ax5, ax6]:
        figure_format(figure)
        mpl.rcParams['figure.dpi'] = 300
        mpl.rcParams['figure.figsize'] = (6,4)
        mpl.rcParams['lines.markeredgecolor'] = 'black'
        mpl.rcParams['lines.marker'] = '>'
        handles, labels = ax1.get_legend_handles_labels()
        figure.legend(handles[::2], labels[::2], frameon = False, ncol = 2, title = 'T ($\degree$C)')

plt.show()
#%% Plotting Gp and Gpp on Same curve

cmap = cm.get_cmap('jet')

for file in data_tts.keys():
    fig, ax1 = plt.subplots(dpi = 300, figsize = (3.75,3))
    figure_format(ax1)
    ax1.set_ylabel('$G^{\prime\prime}$ (Pa)', fontsize = 12)
    ax1.set_xlabel('$a_{T}\cdot\omega$ (rad/s)', fontsize = 12)
    ax1.set_yscale('log')
    ax1.set_xscale('log')
    # ax1.set_title(file)
    
    steps = list(set(data_tts[file]['Sweep number']))
    
    for i,k in enumerate(steps):
        mask = data_tts[file]['Sweep number'] == k
        temp = np.mean(data_tts[file]['Temperature'][mask])
                        
        freq = np.asarray(data_tts[file]['Angular frequency'])
        Gp = np.asarray(data_tts[file]['Storage modulus'])
        Gpp = np.asarray(data_tts[file]['Loss modulus'])
        tan = np.asarray(data_tts[file]['Tan(delta)'])
        
        if(temp > 48): #masking geometry inertial effects at high temperatures
            mask1 = data_tts[file]['Angular frequency'][mask] < 10
        else:
            mask1 = data_tts[file]['Angular frequency'][mask] < 200
        
        # ax1.plot(freq[mask]*data_shift[file][i],
        #         Gp[mask],
        #         marker = 'o',
        #         mec = 'black',
        #         ls = 'None',
        #         color = cmap(k/max(steps)),
        #         label = f'{temp:.1f}'
        #         )
            
        ax1.plot(freq[mask]*data_shift[file][i],
                Gpp[mask],
                marker = '>',
                mec = 'black',
                ls = 'None',
                color = cmap(k/max(steps)),
                # label = f'{temp:.1f}'
                )


    sm = plt.cm.ScalarMappable(cmap=cmap)
    sm.set_clim(vmin=min(temperatures), vmax=80)
    cb = plt.colorbar(sm, ax=plt.gca(), label = 'T ($\degree$C)')
    # cb = plt.colorbar()
    cb.outline.set_color('black')
    cb.outline.set_linewidth(1.5)
    ticklabs = cb.ax.get_yticklabels()
    print(ticklabs)
    # cb.ax.set_yticklabels(ticklabs, fontsize=12)
    
    # black_circle = mlines.Line2D([], [], 
    #                              color='black', 
    #                              marker='o', 
    #                              linestyle='None',
    #                              label='$G^{\prime}$')
    # black_triangle = mlines.Line2D([], [], 
    #                                color='black', 
    #                                marker='>', 
    #                                linestyle='None',
    #                                label='$G^{\prime\prime}$')
    
    # plt.legend(handles=[black_circle, black_triangle], frameon = False, fontsize = 12)
    
    # ax1.vlines(x = 8e-4, ymin = 1e-7, ymax = 1e-3, color = 'red', ls = '--')  

plt.show()

#%%Comparing to WLF Function

def wlf(T, C1, C2, Tr):
    factor = (C1*(T-Tr)/(C2 + (T-Tr)))
    return factor

temperatures = np.asarray(temperatures)
a_Gpp = np.asarray(a_Gpp)

fig, ax5 = plt.subplots(dpi = 300, figsize = (4,4))
ax5.set_ylabel('log($a_{T}$)', fontsize=14)
ax5.set_xlabel('1000 / T (K$^{-1}$)', fontsize=14)
# ax5.set_yscale('log')
ax5.plot(1000 / (273+temperatures),
         np.log(a_Gpp),
         marker = 'o',
         mec = 'black',
         ls = 'None')

ax5.set_ylim((-3, 0.5))
ax5.tick_params(direction='in')

mask2  = np.asarray(temperatures) > 35
Fitting = Curve_Fitting_Function(temperatures[mask2], np.log(a_Gpp[mask2]), wlf)
# # print(Fitting['parameters'])
# ax5.plot(np.linspace(35, 60),
#           np.exp(wlf(np.linspace(35,60), *Fitting['parameters'])),
#           # wlf(np.linspace(25,60), 5, -5, 25),
#           marker = 'None',
#           ls = '--',
#           color = 'black')

plt.show()

# %% Save a_T and T in excel file

data_new = pd.DataFrame()

data_new['T [C]'] = pd.Series(temperatures)
data_new['a_T'] = pd.Series(a_Gpp)

data_new.to_excel('LSR_aT_vs_T.xlsx')
