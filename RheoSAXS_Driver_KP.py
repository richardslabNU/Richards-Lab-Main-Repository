# -*- coding: utf-8 -*-
"""
Created on Wed Jul  2 09:04:14 2025

@author: kushp
"""

import numpy as np
import pandas as pd
import os
import h5py
import hdf5plugin
import matplotlib.pyplot as plt
import matplotlib as mpl
from PIL import Image
from scipy.stats import binned_statistic, binned_statistic_2d
from scipy.integrate import simpson

from KP_pyplot_defaults import Set_RcParams, Custom_CM
Set_RcParams()
custom_cm = Custom_CM()


def interp_nan_rows(data):
    x = np.arange(data.shape[1])
    for i in range(data.shape[0]):
        row = data[i]
        mask = ~np.isnan(row)
        if np.sum(mask) >= 2:  # Need at least two points to interpolate
            data[i, ~mask] = np.interp(x[~mask], x[mask], row[mask])
    return data


class SAXS12ID():
    
    def __init__(self, path2data = os.getcwd()):
        self.path=path2data
        self.varpath = {
            'data': ['entry', 'data', 'data'],
            'xcenter':['entry','Metadata','Beam_x_pixel'],
            'ycenter':['entry','Metadata','Beam_y_pixel'],
            'SDD':['entry','Metadata','SDD'],
            'pixel_size':['entry','Metadata','pixel_size'],
            'Wavelength':['entry','Metadata','Wavelength'],
            'trans':['entry','Metadata','SAXS_phd']
            }
        self.isblank = False

    def get_info(self, file, subpath):
        
        subpath_len = len(subpath)
        
        x0 = file
        for i in range(subpath_len):
            x0 = x0[subpath[i]]
        x0 = x0[:]
            
        return x0
    
    
    def get_vars(self, varlist, path2file):
        
        file = h5py.File(path2file, 'r')
        
        retval = []
        
        for var in varlist:
            subpath = self.varpath[var]
            retval.append(self.get_info(file, subpath))
        
        file.close()

        return retval
    
    def set_blank(self, path2blank, masknonpos = False):
        self.path2blank = path2blank
        
        data, dtrans = self.get_vars(['data', 'trans'], self.path)
        blank, btrans = self.get_vars(['data', 'trans'], self.path2blank)
        
        self.subtracted_data = data - blank*dtrans/btrans
        self.S_poisson = np.sqrt(data + blank*dtrans/btrans)
        
        if masknonpos:
            self.submask = self.subtracted_data<=0
            self.subtracted_data = np.ma.masked_array(data = self.subtracted_data, mask = self.submask, fill_value = np.nan, dtype = float )
        
        
        
        if hasattr(self, 'mask'):
            self.mask_data(self.path2mask, flip=self.flip)
        
    
    def mask_data(self, path2mask, flip = True):
        
        self.path2mask = path2mask
        self.flip = flip
        
        if flip:
            self.mask = np.flip(~np.array(Image.open(path2mask)), axis=0)
        else:
            self.mask = np.array(Image.open(path2mask))
        
        if hasattr(self, 'subtracted_data'):
            self.masked_data = np.ma.masked_array(data = self.subtracted_data, mask = self.mask, fill_value = np.nan, dtype = float)
            if hasattr(self, 'submask'):
                self.mask = self.mask | self.submask
        else:
            self.masked_data = np.ma.masked_array(data = self.get_vars(['data'], self.path)[0], mask = self.mask, fill_value = np.nan, dtype = float)
        
    def create_maps(self):
        data, ycenter, xcenter, pixel_size, SDD, Wavelength = self.get_vars(['data', 'ycenter', 'xcenter',
                                                                             'pixel_size', 'SDD', 'Wavelength'], self.path)
        
        
        if not hasattr(self, 'xcenter'):
            self.xcenter = xcenter
        
        x = np.arange(np.shape(data)[1])
        y = np.arange(np.shape(data)[0])
        
        XX, YY = np.meshgrid(x, y)
        
        self.XX = XX
        self.YY = YY
        
        ycenter = np.shape(data)[0] - ycenter
        
        if not hasattr(self, 'ycenter'):
            self.ycenter = ycenter
        
        RR = np.sqrt((XX - self.xcenter)**2 + (YY - self.ycenter)**2)
        
        self.PP = np.arctan2(XX - self.xcenter, self.ycenter - YY)
        
        TT = np.arctan(RR*pixel_size/SDD)/2
        
        self.QQ = 4*np.pi/Wavelength * np.sin(TT)
        
        self.x_start = -self.QQ[int(np.round(self.ycenter)), :][0]
        self.x_end = self.QQ[int(np.round(self.ycenter)), :][-1]
        self.y_start = self.QQ[:, int(np.round(self.xcenter))][0]
        self.y_end = -self.QQ[:, int(np.round(self.xcenter))][-1]
        
        Qx = np.linspace(self.x_start, self.x_end, np.shape(data)[1])
        Qy = np.linspace(self.y_start, self.y_end, np.shape(data)[0])
        
        self.QxQx, self.QyQy = np.meshgrid(Qx, Qy)
        
        
        
    
    def plot_2Dimage(self,
                     figsize = (3,3),dpi=300,
                     linthresh = 0.1, vmin=1, vmax=1e4,
                     cmap = 'plasma',
                     xlim = (-0.1, 0.1), ylim = (-0.05, 0.15),
                     show = True):
        
        fig, ax = plt.subplots(figsize = figsize, dpi = dpi)
        norm = mpl.colors.SymLogNorm(linthresh=linthresh, vmin=vmin, vmax=vmax)
        
        if not hasattr(self, 'x_start'):
            self.create_maps()
        
        extent = [self.x_start, self.x_end, self.y_start, self.y_end]
        
        
        if hasattr(self, 'masked_data'):
            ax.imshow(self.masked_data, norm = norm, cmap = cmap, extent = extent, origin = 'lower')
        elif hasattr(self, 'subtracted_data'):
            ax.imshow(self.subtracted_data, norm = norm, cmap = cmap, extent = extent, origin = 'lower')
        else:
            ax.imshow(self.get_vars(['data'], self.path)[0], norm = norm, cmap = cmap, extent = extent, origin = 'lower')
        
        ax.set(xlim=xlim, ylim=ylim)
        
        # ax.set_xticklabels([])
        # ax.set_yticklabels([])

        # ax.tick_params(which = 'both', width = 0)
        
        if show:
            plt.show()
        
        return fig, ax
    
    def bin_1d(self, bins = [-2.6, 0, 100], phi = 0*np.pi, dphi = 0.05*np.pi):
        
        if not hasattr(self, 'PP'):
            self.create_maps()
        
        q_bins = np.logspace(bins[0], bins[1], bins[2])
        q_vals = np.array([(q_bins[i]+q_bins[i+1])/2 for i in range(len(q_bins)-1)])
        
        phicenter = phi
        philower = phicenter - dphi
        phiupper = phicenter + dphi
        
        phimask = ~((self.PP<phiupper) & (self.PP>philower))
        
        masked_QQ = np.ma.masked_array(data = self.QQ, mask = self.mask)
        phimasked_QQ = np.ma.masked_array(data = masked_QQ, mask = phimask)
        flatten_QQ = phimasked_QQ.compressed()

        phimasked_data = np.ma.masked_array(data = self.masked_data, mask = phimask)
        flatten_data = phimasked_data.compressed()

        i_binned = binned_statistic(flatten_QQ, flatten_data, statistic = np.nanmean, bins = q_bins)[0]
        
        return q_vals, i_binned
    
    def bin_2d(self,
               q_bins = np.linspace(0.01, 0.08, 100), p_bins = np.linspace(0, np.pi, 100)):
        
        if not hasattr(self, 'PP'):
            self.create_maps()
        
        phimask = ~((self.PP>-np.pi/2) & (self.PP<np.pi/2))
        
        masked_QQ = np.ma.masked_array(data = self.QQ, mask = self.mask)
        phimasked_QQ = np.ma.masked_array(data = masked_QQ, mask = phimask)
        flatten_QQ = phimasked_QQ.compressed()

        masked_PP = np.ma.masked_array(data = self.PP, mask = self.mask)
        phimasked_PP = np.ma.masked_array(data = masked_PP, mask = phimask)
        flatten_PP = phimasked_PP.compressed()
        
        phimasked_data = np.ma.masked_array(data = self.masked_data, mask = phimask)
        flatten_data = phimasked_data.compressed()
        
        
        i_binned2d = binned_statistic_2d(flatten_QQ, flatten_PP, flatten_data, statistic = np.nanmean, bins = [q_bins, p_bins])[0]

        q_vals = np.array([(q_bins[i]+q_bins[i+1])/2 for i in range(len(q_bins)-1)])
        p_vals = np.array([(p_bins[i]+p_bins[i+1])/2 for i in range(len(p_bins)-1)])
        
        return q_vals, p_vals, i_binned2d
    
    def AlignFact(self,
                  q_bins = np.linspace(0.001, 0.08, 100), 
                  p_bins = np.linspace(-np.pi/2 - np.pi/(100-1)/2, np.pi/2 + np.pi/(100-1)/2, 100)):
        
        q_pix, p_pix, temp = self.bin_2d(q_bins = q_bins, p_bins = p_bins)
        
        interp_nan_rows(temp)
        
        term1 = simpson(temp*np.cos(2*(p_pix)), x = p_pix, axis = 1)
        term2 = simpson(temp, x = p_pix, axis = 1)

        Af = term1/term2
        
        return q_pix, Af









