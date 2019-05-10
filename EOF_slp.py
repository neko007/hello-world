#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  9 12:17:47 2019

@author: zzz
"""
# %reset

import numpy as np
import matplotlib.pyplot as plt
import netCDF4 as nc
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

def EOF_PC(A, lon, lat, nEOF):
    len_time, len_lat, len_lon = A.shape
    A_Anomaly = A - np.mean(A,axis=0)
    A_Anomaly_re = A_Anomaly.reshape((len_time,len_lon*len_lat))
    x = A_Anomaly_re.T
    ss,v_Q = np.linalg.eig(np.dot(x.T,x))
    
    pc = np.zeros((nEOF,len_time))
    eof = np.zeros((nEOF,len_lat,len_lon))
    contribution = np.zeros(nEOF)
    
    for i in range(nEOF):
        contribution[i] = ss[i] / sum(ss)
        v_Q_max = v_Q[:,i]
        v = np.dot(x,v_Q_max)
        v_R = v / np.sqrt(ss[i])
        pc[i,:] = np.dot(v_R,x)
        eof[i,:,:] = v_R.reshape((len_lat,len_lon))
    return pc, eof, contribution

def draw_EOF_and_PC(eof,pc,lon,lat,contribution):
    # draw EOFi
    fig1 = plt.figure(figsize=[5,15],dpi=100)
    ax1 = []
    neof = eof.shape[0]
    for i in range(neof):
        ax1.append(fig1.add_subplot(neof,1,i+1,projection=ccrs.PlateCarree(central_longitude=180)))
        ax1[i].coastlines()
        p = ax1[i].contourf(lon,lat,eof[i,:,:]*10**2,10,transform=ccrs.PlateCarree(),cmap='coolwarm',extend='both')
        cb = plt.colorbar(p,pad=0.02)
        cb.set_label(r"$\times 10^{-2}$",y=1.1,labelpad=-10,rotation=0,fontsize=8)
        ax1[i].set_xticks([0, 60, 120, 180, 240, 300, 360], crs=ccrs.PlateCarree())
        ax1[i].set_yticks([-90, -60, -30, 0, 30, 60, 90], crs=ccrs.PlateCarree())
        lon_formatter = LongitudeFormatter(zero_direction_label=True)
        lat_formatter = LatitudeFormatter()
        ax1[i].xaxis.set_major_formatter(lon_formatter)
        ax1[i].yaxis.set_major_formatter(lat_formatter)   
        
        ax1[i].set_title('EOF'+str(i+1), loc='left', fontsize=10)
        ax1[i].set_title('{:.2f}%'.format(contribution[i]*100), loc='right', fontsize=10)
    fig1.savefig('EOF.png')
    # draw PCi
    len_time = pc.shape[1]
    time_list = np.arange(len_time)
    year_list = 1948 + time_list / 12
    fig2, ax2 = plt.subplots(neof,1,sharex='col')
    for i in range(neof):
        pc_pos = np.where(pc[i,:]<0,0,pc[i,:])
        pc_nag = np.where(pc[i,:]>0,0,pc[i,:])
        ax2[i].bar(year_list, pc_pos, width=0.1, color='r')
        ax2[i].bar(year_list, pc_nag, width=0.1, color='b')
        ax2[i].set_title('PC'+str(i+1), loc='left', fontsize=10)
        ax2[i].set_title('{:.2f}%'.format(contribution[i]*100), loc='right', fontsize=10)
        ax2[i].set_xlim([min(year_list), max(year_list)])
    fig2.savefig('PC.png')
    plt.show()

if __name__ == '__main__':
    # read datas
    file_name = 'slp.mon.mean.nc'
    f = nc.Dataset(file_name, 'r')
    lat = f.variables['lat'][:]
    lon = f.variables['lon'][:]
    slp = f.variables['slp'][:]
    time = f.variables['time'][:]
    pc, eof, contribution = EOF_PC(slp, lon, lat,3)
    draw_EOF_and_PC(eof, pc, lon, lat, contribution)


