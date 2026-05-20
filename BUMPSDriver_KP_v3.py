# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 17:15:09 2025

@author: kushp
"""

import bumps.names as bmp
from bumps.fitters import fit
import numpy as np
import inspect
import matplotlib.pyplot as plt
import matplotlib.pyplot as mpl
import dill
import pandas as pd
import os
import warnings
from bumps.dream import views


class BUMPS_KP3():
    
    def __init__(self):
        try:
            import bumps
        except  ValueError:
            print('You must download bumps to continue')
        
        self.models=list()
        self.functions=list()
        self.parameters=list()
        self.constants=list()
        self.modelsettings={'function':[],
                            'lowmask':[],
                            'uppmask':[]}
        self.methodsettings=dict()
    
    
    def addfunctions(self, func_list):
        #automatically scrapes your fit function into the function list
        #functions can be in a list or added individually
        
        #checks if the input is a list/array that can be iterated on
        if hasattr(func_list, '__len__'):
            [self.functions.append(func_list[i]) for i in range(len(func_list))] #appends functions to list
        #if only one function was passed through, it is added to the list
        else:
            self.functions.append(func_list)
        
        
    def addparameters(self, param_list):
        #automattically scrapes parameters into the parameter list
        #parameters can be in a list or added individually
        #the list is a list of all the parameters being fit regardless of multiple functions that
        #may or may not have shared parameters
        
        #checks if the input is a list/array that can be iterated on
        if hasattr(param_list, '__len__'):
            [self.parameters.append(param_list[i]) for i in range(len(param_list))] #appends parameters to list
        #if only one parameter was passed through, it is added to the list
        else:
            self.parameters.append(param_list)
        
        #check for duplicates
        names = [par.name for par in self.parameters]
        if len(names) != len(set(names)):
            warnings.warn('There is a duplicated parameter in the parameter list')
        #note: duplicated parameters won't break the code
        #as the models are updated, duplicates will rewrite parameters of the same name
    
    def addconstants(self, constant_list):
        #automattically scrapes parameters into the parameter list
        #parameters can be in a list or added individually
        #the list is a list of all the parameters being fit regardless of multiple functions that
        #may or may not have shared parameters
        
        #checks if the input is a list/array that can be iterated on
        if hasattr(constant_list, '__len__'):
            [self.constants.append(constant_list[i]) for i in range(len(constant_list))] #appends parameters to list
        #if only one parameter was passed through, it is added to the list
        else:
            self.constants.append(constant_list)
        
        #check for duplicates
        names = [con.name for con in self.constants]
        if len(names) != len(set(names)):
            warnings.warn('There is a duplicated constant in the constant list')
        #note: duplicated parameters won't break the code
        #as the models are updated, duplicates will rewrite parameters of the same name
    
    
    def addmodel(self, x, y, dy = None, index = -1, xmask = [-np.inf, np.inf]):
        #will add models individually for each data set being fit
        #index should be set to the index of the function in the function list
        try:
            function = self.functions[index]
        except ValueError:
            print('Function list is empty')
            
        #will mask the data based on the x range indicated
        mask = (x>xmask[0]) & (x<xmask[1])
        x = x[mask]
        y = y[mask]
        try:
            dy = dy[mask] #if dy = None, this will be skipped
        except:
            pass
        
        
        self.models.append(bmp.Curve(function, x, y, dy)) #add the model to the model list
        
        #log the model in settings for potential reference
        self.modelsettings['function'].append(function)
        self.modelsettings['lowmask'].append(xmask[0])
        self.modelsettings['uppmask'].append(xmask[1])
        
        #this will set the parameters required for the function as attributes for the designated model
        details = inspect.signature(function)
        variablelist = self.parameters + self.constants
        for key in details.parameters:
            for var in variablelist:
                if (var.name == key):
                    setattr(self.models[-1], key, var)
    
    
    def setproblem(self, name = 'untitled'):
        #set the problem for fitting
        #this is to be done after all the models have been added
        #the name here will be used as the save name
        self.problem = bmp.FitProblem(self.models)
        self.name = name
    
    
    def fitproblem(self, method = 'dream', verbose = True, 
                                           store = None,
                                           alpha = 0.01,
                                           outliers = 'none',
                                           trim = False, 
                                           **kwargs):
        #fit the problem
        #kwargs needed for the fitting algorithm need to be entered (steps, samples, etc.)
        
        self.result = fit(self.problem,
                          method = method,
                          verbose = verbose,
                          store = store,
                          alpha = alpha,
                          outliers = outliers,
                          trim = trim,
                          **kwargs)
        
        #log the method settings
        self.methodsettings['method'] = method
        self.methodsettings.update(kwargs)
        self.methodsettings['finalchisq'] = self.finalchisq()
    
    
    def finalchisq(self):
        
        final_chisq = float(self.problem.chisq_str().split('(')[0])
        
        return final_chisq
    
    
    def getparameters(self):
        #will return a dictionary {p: [x, dx]}
        paramlist = [par.name for par in self.parameters]
        paramlist.sort()
        
        pard = {paramlist[i]: [self.result['x'][i], self.result['dx'][i]] for i in range(len(paramlist))}
        
        return pard
    
    
    def plotfit(self, scaling = 'linear', xlim = None, ylim = None, fig = None, ax = None,
                show = True, close = True):
        
        #Will plot the fits
        #input a str containing xlog and/or ylog to scaling for the appro scaling
        #input a tuple (lowlim, upplim) to xlim or ylim
        #input a fig and/or ax if plotting from source code
        #set show to False to not show output figure
        #set close to False if you want to continue to plot/manipulate figure
        #will output the figure and axes objects
        
        
        if fig:
            if ax:
                pass
            else:
                ax = fig.add_subplot()
                
        else:    
            fig, ax = plt.subplots(figsize = (3, 3), dpi = 300)
        
        colors = mpl.cm.tab20(np.linspace(0, 1, 20))
        
        for i, model in enumerate(self.models):
            
            x = model.x
            y = model.y
            dy = model.dy
            
            function = model.fn
            p = {}
            
            pard = self.getparameters()
            
            for constant in self.constants:
                p[constant.name] = constant.value
            
            details = inspect.signature(function)
            
            for key in pard:
                if key in details.parameters:
                    p[key] = pard[key][0]
            
            
            ax.errorbar(x, y, yerr = dy, ls = '', marker = 'o', color = colors[i%10*2 + 1], zorder = 0)
            
            if 'xlog' in scaling:
                xfit = np.logspace(np.log10(min(x)), np.log10(max(x)), 100)
            else:
                xfit = np.linspace(min(x), max(x), 100)
            
            ax.plot(xfit, function(xfit, **p), ls = '--', color = colors[i%10*2], zorder = 1)
        
        if 'xlog' in scaling:
            ax.set_xscale('log')
        if 'ylog' in scaling:
            ax.set_yscale('log')
        
        if xlim:
            ax.set_xlim(xlim)
        if ylim:
            ax.set_ylim(ylim)
        
        
        if show and close:
            plt.show()
        elif close:
            plt.close()
        else:
            pass
        
        return fig, ax
    
    
    def plotanalysis(self):
        #view the uncertainty analysis of the fit
        views.plot_all(self.result['state'])
    
    
    def savepickle(self, savepath = os.getcwd()):
        #pickle data
        if not os.path.exists(savepath):
            os.makedirs(savepath)
        
        with open(f'{savepath}\\{self.name}.pkl', 'wb') as file:
            file.write(dill.dumps(self))
        
    
    def loadpickle(self, loadpath):
        #load data from pickle
        with open(loadpath, 'rb') as file:
            self.__dict__.update(dill.loads(file.read()).__dict__)
    

    def savetoexcel(self, savepath = os.getcwd()):
        #save method settings and parameters to excel file
        settingdf = pd.DataFrame(data = self.methodsettings, index = [0])
        
        paramlist = [par.name for par in self.parameters]
        paramlist.sort()
        
        pard = {paramlist[i]: [self.result['x'][i], self.result['dx'][i]] for i in range(len(paramlist))}
        pard.update({con.name: [con.value, None] for con in self.constants})
        pardf = pd.DataFrame(pard, index = ['x', 'dx'])
        
        
        if not os.path.exists(savepath):
            os.makedirs(savepath)
        
        with pd.ExcelWriter(f'{savepath}\\{self.name}.xlsx') as writer:
            settingdf.to_excel(writer, sheet_name = 'settings')
            pardf.to_excel(writer, sheet_name = 'params')
        
    

















