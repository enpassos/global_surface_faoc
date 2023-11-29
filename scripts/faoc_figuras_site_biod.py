###########################################
### Código para a confecção das figuras ###
### do site da FAOC                     ###
### Autor(a): Elisa Passos              ###
### Data: 31/10/223                     ###
###########################################

import os
from datetime import datetime, timedelta
import numpy as np
import xarray as xr
from matplotlib.colors import LogNorm
import matplotlib as mpl
import matplotlib.pyplot as plt                   # Plotting library
import cartopy, cartopy.crs as ccrs               # Plot maps
import cartopy.feature as cfeature                # Common drawing and filtering operations
from matplotlib.scale import scale_factory
import scipy 
from netCDF4 import Dataset

path = os.getcwd()
os.chdir('../')
os.chdir(os.path.join(path,'downloads'))
arq = [x for x in os.listdir() if x.endswith('.nc')]

ds = xr.open_dataset(arq[0])
ano = int(ds['time'][0].dt.year.astype(str))
mes = int(ds['time'][0].dt.month.astype(str))

date = datetime(ano, mes, 1)
data_arq = date.strftime('%Y%m')
date_formatted = date.strftime('%Y-%m-%d')
data_title = date.strftime('%m/%Y')

print('Data Arq: ',data_arq)
print(' ')

#---------------------------------------------------------------------------------------------------------------------------
# LABOFIS / UERJ: Oceanography Products: CMEMS in Robinson Projection
# Author: Elisa Passos
# TSM + Vel superficial
#---------------------------------------------------------------------------------------------------------------------------

print('FAOC / UERJ: Oceanography Products: CMEMS in Robinson Projection')
print('Author: Elisa Passos')
print(' ')
print('Figuras:')
print('TSM + Vel superficial')

# Open the file using the NetCDF4 library
file = xr.open_dataset("./cmems_temp.nc")

# Extract the Sea Surface Temperature
data = file.variables['thetao'][0,0,:,:] #*file.variables['thetao'].scale_factor) + file.variables['thetao'].add_offset

# Extract the coordinates
lat = file.variables['latitude'][:]
lon  = file.variables['longitude'][:]
mlon, mlat = np.meshgrid(lon, lat)

# Extract velocity components
file = Dataset("./cmems_vel.nc")
uvel = file.variables['uo'][0,0,:,:] #(file.variables['uo'][ind_phy,0,:,:]*file.variables['uo'].scale_factor)
uvel = np.where(uvel<=-1000,np.nan,uvel)
vvel = file.variables['vo'][0,0,:,:] #(file.variables['vo'][ind_phy,0,:,:]*file.variables['vo'].scale_factor)
vvel = np.where(vvel<=-1000,np.nan,vvel)


# Choose the plot size (width x height, in inches)
fig = plt.figure(figsize=(19.20, 10.80), facecolor='white')

# Use the Mercator projection in cartopy
ax = plt.axes(projection=ccrs.Robinson()) #central_longitude=0.0

# Method used for "global" plots
ax.set_global()

# Add coastlines, borders and gridlines
ax.add_feature(cfeature.LAND, color='gray', edgecolor='k', zorder=2) # adding land mask
ax.coastlines(resolution='50m', color='black', linewidth=0.8, zorder=3)
ax.add_feature(cfeature.BORDERS, edgecolor='black', linewidth=0.5, zorder=2)
gl = ax.gridlines(crs=ccrs.PlateCarree(), color='white', alpha=1.0, linestyle='--',\
                linewidth=0.25, xlocs=np.arange(-180, 180, 30), ylocs=np.arange(-90, 90, 10), draw_labels=True)

gl.top_labels = False
gl.right_labels = False

# Ploting the data temperature
img = plt.pcolor(mlon[::10,::10], mlat[::10,::10], data[::10,::10], vmin=-2, vmax=35, cmap='inferno', transform=ccrs.PlateCarree(), zorder=0)

# velocity
skip = 10
img2 = plt.streamplot(mlon[::skip,::skip], mlat[::skip,::skip], uvel[::skip,::skip], vvel[::skip,::skip], density=5.5, color='k', linewidth=0.7, arrowsize=0.5, arrowstyle='-|>', transform=ccrs.PlateCarree(), zorder=0)

# Adding a colorbar
plt.colorbar(img, label='Temperatura da Superfície do Mar (°C)', extend='both',\
                          orientation='horizontal', pad=0.05, fraction=0.05, shrink=0.7)

# Escrevendo os países
txt01 = ax.annotate("Brasil", xy=(-50,-10), xycoords=ccrs.PlateCarree()._as_mpl_transform(ax), color='black', size=12, clip_on=True, annotation_clip=True, horizontalalignment='center', verticalalignment='center', transform=ccrs.PlateCarree(), zorder=2)
txt02 = ax.annotate("EUA", xy=(-105,40), xycoords=ccrs.PlateCarree()._as_mpl_transform(ax), color='black', size=12, clip_on=True, annotation_clip=True, horizontalalignment='center', verticalalignment='center', transform=ccrs.PlateCarree(), zorder=2)
txt03 = ax.annotate("Canadá", xy=(-115,60), xycoords=ccrs.PlateCarree()._as_mpl_transform(ax), color='black', size=12, clip_on=True, annotation_clip=True, horizontalalignment='center', verticalalignment='center', transform=ccrs.PlateCarree(), zorder=2)
txt04 = ax.annotate("Rússia", xy=(90,60), xycoords=ccrs.PlateCarree()._as_mpl_transform(ax), color='black', size=12, clip_on=True, annotation_clip=True, horizontalalignment='center', verticalalignment='center', transform=ccrs.PlateCarree(), zorder=2)
txt05 = ax.annotate("China", xy=(105,35), xycoords=ccrs.PlateCarree()._as_mpl_transform(ax), color='black', size=12, clip_on=True, annotation_clip=True, horizontalalignment='center', verticalalignment='center', transform=ccrs.PlateCarree(), zorder=2)


# Add a bottom note
plt.figtext(0.5, 0.01, "Fonte dos dados: CMEMS - Correntes marinhas e Temperatura da superfície do mar em "+data_title+".", ha="center", fontsize=16)#, bbox={"facecolor":"white", "alpha":1, "pad":5})

# Save the image
plt.savefig('../Output/cmems_tsm+vel_'+data_arq+'.png')


#---------------------------------------------------------------------------------------------------------------------------
# LABOFIS / UERJ: Oceanography Products: CMEMS in Robinson Projection
# Author: Elisa Passos
# Chl + Vel superficial
#---------------------------------------------------------------------------------------------------------------------------

print('Chl + Vel superficial')


file = xr.open_dataset("./cmems_bio.nc")

# Extract the Chl
data = file.chl[0,0,:,:] 
lat_chl = file.variables['latitude'][:]
lon_chl  = file.variables['longitude'][:]
mlon_bio, mlat_bio = np.meshgrid(lon_chl, lat_chl)

# Choose the plot size (width x height, in inches)
fig = plt.figure(figsize=(19.20, 10.80), facecolor='white')

# Use the Mercator projection in cartopy
ax = plt.axes(projection=ccrs.Robinson()) #central_longitude=0.0

# Method used for "global" plots
ax.set_global()

# Add coastlines, borders and gridlines
ax.add_feature(cfeature.LAND, color='gray', edgecolor='k', zorder=100) # adding land mask
ax.coastlines(resolution='50m', color='black', linewidth=0.8, zorder=105)
ax.add_feature(cfeature.BORDERS, edgecolor='black', linewidth=0.5, zorder=110)
gl = ax.gridlines(crs=ccrs.PlateCarree(), color='white', alpha=1.0, linestyle='--',\
                linewidth=0.25, xlocs=np.arange(-180, 180, 30), ylocs=np.arange(-90, 90, 10), draw_labels=True)

gl.top_labels = False
gl.right_labels = False

# Ploting the data temperature
img = plt.pcolor(mlon_bio[::2,::2], mlat_bio[::2,::2], data[::2,::2], norm=LogNorm(vmin=data.min(), vmax=data.max()), cmap='YlGn', transform=ccrs.PlateCarree())

# velocity
img2 = ax.streamplot(mlon[::10,::10], mlat[::10,::10], uvel[::10,::10], vvel[::10,::10], density=5.5, color='k', linewidth=0.7, arrowsize=0.5, arrowstyle='-|>', transform=ccrs.PlateCarree())


# Adding a colorbar
plt.colorbar(img, label='Concentração de Clorofila da Superfície do Mar ($mg \cdot m^{-3}$)', extend='both',\
                          orientation='horizontal', pad=0.05, fraction=0.05, shrink=0.7)

# Escrevendo os países
txt01 = ax.annotate("Brasil", xy=(-50,-10), xycoords=ccrs.PlateCarree()._as_mpl_transform(ax), color='black', size=12, clip_on=True, annotation_clip=True, horizontalalignment='center', verticalalignment='center', transform=ccrs.PlateCarree(), zorder=115)
txt02 = ax.annotate("EUA", xy=(-105,40), xycoords=ccrs.PlateCarree()._as_mpl_transform(ax), color='black', size=12, clip_on=True, annotation_clip=True, horizontalalignment='center', verticalalignment='center', transform=ccrs.PlateCarree(), zorder=115)
txt03 = ax.annotate("Canadá", xy=(-115,60), xycoords=ccrs.PlateCarree()._as_mpl_transform(ax), color='black', size=12, clip_on=True, annotation_clip=True, horizontalalignment='center', verticalalignment='center', transform=ccrs.PlateCarree(), zorder=115)
txt04 = ax.annotate("Rússia", xy=(90,60), xycoords=ccrs.PlateCarree()._as_mpl_transform(ax), color='black', size=12, clip_on=True, annotation_clip=True, horizontalalignment='center', verticalalignment='center', transform=ccrs.PlateCarree(), zorder=115)
txt05 = ax.annotate("China", xy=(105,35), xycoords=ccrs.PlateCarree()._as_mpl_transform(ax), color='black', size=12, clip_on=True, annotation_clip=True, horizontalalignment='center', verticalalignment='center', transform=ccrs.PlateCarree(), zorder=115)

# Add a bottom note
plt.figtext(0.5, 0.01, "Fonte dos dados: CMEMS - Correntes marinhas e Concentração de Clorofila da superfície do mar em "+data_title+".", ha="center", fontsize=16)#, bbox={"facecolor":"white", "alpha":1, "pad":5})

# Save the image
plt.savefig('../Output/cmems_chl+vel_'+data_arq+'.png', transparent=False)


#---------------------------------------------------------------------------------------------------------------------------
# LABOFIS / UERJ: Oceanography Products: CMEMS in Robinson Projection
# Author: Elisa Passos
# O2 + Vel superficial
#---------------------------------------------------------------------------------------------------------------------------
print('O2  + Vel superficial')

# Extract the O2
data = file.o2[0,0,:,:] 

# Choose the plot size (width x height, in inches)
fig = plt.figure(figsize=(19.20, 10.80), facecolor='white')

# Use the Mercator projection in cartopy
ax = plt.axes(projection=ccrs.Robinson()) #central_longitude=0.0

# Method used for "global" plots
ax.set_global()

# Add coastlines, borders and gridlines
ax.add_feature(cfeature.LAND, color='gray', edgecolor='k', zorder=100) # adding land mask
ax.coastlines(resolution='50m', color='black', linewidth=0.8, zorder=105)
ax.add_feature(cfeature.BORDERS, edgecolor='black', linewidth=0.5, zorder=110)
gl = ax.gridlines(crs=ccrs.PlateCarree(), color='white', alpha=1.0, linestyle='--',\
                linewidth=0.25, xlocs=np.arange(-180, 180, 30), ylocs=np.arange(-90, 90, 10), draw_labels=True)

gl.top_labels = False
gl.right_labels = False

# Ploting the data temperature
img = plt.pcolor(mlon_bio[::2,::2], mlat_bio[::2,::2], data[::2,::2], vmin=200, vmax=400, cmap='YlGnBu', transform=ccrs.PlateCarree())

# velocity
img2 = ax.streamplot(mlon[::10,::10], mlat[::10,::10], uvel[::10,::10], vvel[::10,::10], density=5.5, color='k', linewidth=0.7, arrowsize=0.5, arrowstyle='-|>', transform=ccrs.PlateCarree())

# Adding a colorbar
plt.colorbar(img, label='Oxigênio dissolvido da Superfície do Mar ($\mu mol \cdot L^{-1}$)', extend='both',\
                          orientation='horizontal', pad=0.05, fraction=0.05, shrink=0.7)

# Escrevendo os países
txt01 = ax.annotate("Brasil", xy=(-50,-10), xycoords=ccrs.PlateCarree()._as_mpl_transform(ax), color='black', size=12, clip_on=True, annotation_clip=True, horizontalalignment='center', verticalalignment='center', transform=ccrs.PlateCarree(), zorder=115)
txt02 = ax.annotate("EUA", xy=(-105,40), xycoords=ccrs.PlateCarree()._as_mpl_transform(ax), color='black', size=12, clip_on=True, annotation_clip=True, horizontalalignment='center', verticalalignment='center', transform=ccrs.PlateCarree(), zorder=115)
txt03 = ax.annotate("Canadá", xy=(-115,60), xycoords=ccrs.PlateCarree()._as_mpl_transform(ax), color='black', size=12, clip_on=True, annotation_clip=True, horizontalalignment='center', verticalalignment='center', transform=ccrs.PlateCarree(), zorder=115)
txt04 = ax.annotate("Rússia", xy=(90,60), xycoords=ccrs.PlateCarree()._as_mpl_transform(ax), color='black', size=12, clip_on=True, annotation_clip=True, horizontalalignment='center', verticalalignment='center', transform=ccrs.PlateCarree(), zorder=115)
txt05 = ax.annotate("China", xy=(105,35), xycoords=ccrs.PlateCarree()._as_mpl_transform(ax), color='black', size=12, clip_on=True, annotation_clip=True, horizontalalignment='center', verticalalignment='center', transform=ccrs.PlateCarree(), zorder=115)

# Add a bottom note
plt.figtext(0.5, 0.01, "Fonte dos dados: CMEMS - Correntes marinhas e Oxigênio dissolvido da superfície do mar em "+data_title+".", ha="center", fontsize=16)#, bbox={"facecolor":"white", "alpha":1, "pad":5})

# Save the image
plt.savefig('../Output/cmems_o2+vel_'+data_arq+'.png')


#---------------------------------------------------------------------------------------------------------------------------
# LABOFIS / UERJ: Oceanography Products: CMEMS in Robinson Projection
# Author: Elisa Passos
# NO3 + Vel superficial
#---------------------------------------------------------------------------------------------------------------------------
print('NO3 + Vel superficial')

# Extract the NO3
data = file.no3[0,0,:,:]

# Choose the plot size (width x height, in inches)
fig = plt.figure(figsize=(19.20, 10.80), facecolor='white')

# Use the Mercator projection in cartopy
ax = plt.axes(projection=ccrs.Robinson()) #central_longitude=0.0

# Method used for "global" plots
ax.set_global()

# Add coastlines, borders and gridlines
ax.add_feature(cfeature.LAND, color='gray', edgecolor='k', zorder=100) # adding land mask
ax.coastlines(resolution='50m', color='black', linewidth=0.8, zorder=105)
ax.add_feature(cfeature.BORDERS, edgecolor='black', linewidth=0.5, zorder=110)
gl = ax.gridlines(crs=ccrs.PlateCarree(), color='white', alpha=1.0, linestyle='--',\
                linewidth=0.25, xlocs=np.arange(-180, 180, 30), ylocs=np.arange(-90, 90, 10), draw_labels=True)

gl.top_labels = False
gl.right_labels = False

# Ploting the data temperature
img = plt.pcolor(mlon_bio[::2,::2], mlat_bio[::2,::2], data[::2,::2], vmin=0, vmax=30, cmap='Spectral_r', transform=ccrs.PlateCarree())

# velocity
img2 = ax.streamplot(mlon[::10,::10], mlat[::10,::10], uvel[::10,::10], vvel[::10,::10], density=5.5, color='k', linewidth=0.7, arrowsize=0.5, arrowstyle='-|>', transform=ccrs.PlateCarree())

# Adding a colorbar
plt.colorbar(img, label='Nitrato da Superfície do Mar ($\mu mol \cdot L^{-1}$)', extend='max',\
                          orientation='horizontal', pad=0.05, fraction=0.05, shrink=0.7)

# Escrevendo os países
txt01 = ax.annotate("Brasil", xy=(-50,-10), xycoords=ccrs.PlateCarree()._as_mpl_transform(ax), color='black', size=12, clip_on=True, annotation_clip=True, horizontalalignment='center', verticalalignment='center', transform=ccrs.PlateCarree(), zorder=115)
txt02 = ax.annotate("EUA", xy=(-105,40), xycoords=ccrs.PlateCarree()._as_mpl_transform(ax), color='black', size=12, clip_on=True, annotation_clip=True, horizontalalignment='center', verticalalignment='center', transform=ccrs.PlateCarree(), zorder=115)
txt03 = ax.annotate("Canadá", xy=(-115,60), xycoords=ccrs.PlateCarree()._as_mpl_transform(ax), color='black', size=12, clip_on=True, annotation_clip=True, horizontalalignment='center', verticalalignment='center', transform=ccrs.PlateCarree(), zorder=115)
txt04 = ax.annotate("Rússia", xy=(90,60), xycoords=ccrs.PlateCarree()._as_mpl_transform(ax), color='black', size=12, clip_on=True, annotation_clip=True, horizontalalignment='center', verticalalignment='center', transform=ccrs.PlateCarree(), zorder=115)
txt05 = ax.annotate("China", xy=(105,35), xycoords=ccrs.PlateCarree()._as_mpl_transform(ax), color='black', size=12, clip_on=True, annotation_clip=True, horizontalalignment='center', verticalalignment='center', transform=ccrs.PlateCarree(), zorder=115)

# Add a bottom note
plt.figtext(0.5, 0.01, "Fonte dos dados: CMEMS - Correntes marinhas e Nitrato da superfície do mar em "+data_title+".", ha="center", fontsize=16)#, bbox={"facecolor":"white", "alpha":1, "pad":5})

# Save the image
plt.savefig('../Output/cmems_no3+vel_'+data_arq+'.png')



#---------------------------------------------------------------------------------------------------------------------------
# LABOFIS / UERJ: Oceanography Products: CMEMS in Robinson Projection
# Author: Elisa Passos
# pH + Vel superficial
#---------------------------------------------------------------------------------------------------------------------------
print('pH  + Vel superficial')

# Extract the pH
data = file.ph[0,0,:,:] 

# Colorbar
cmap = mpl.cm.rainbow
# bounds = [1, 7.5, 7.9, 8, 8.1, 8.2, 8.3, 8.5, 9, 14]
bounds = [1, 7.8, 7.9, 7.95, 8, 8.025, 8.05, 8.075, 8.1, 8.125, 8.15, 8.175, 8.2, 8.25, 8.3, 8.4, 14]
norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

# Choose the plot size (width x height, in inches)
fig = plt.figure(figsize=(19.20, 10.80), facecolor='white')

# Use the Mercator projection in cartopy
ax = plt.axes(projection=ccrs.Robinson()) #central_longitude=0.0

# Method used for "global" plots
ax.set_global()

# Add coastlines, borders and gridlines
ax.add_feature(cfeature.LAND, color='gray', edgecolor='k', zorder=100) # adding land mask
ax.coastlines(resolution='50m', color='black', linewidth=0.8, zorder=105)
ax.add_feature(cfeature.BORDERS, edgecolor='black', linewidth=0.5, zorder=110)
gl = ax.gridlines(crs=ccrs.PlateCarree(), color='white', alpha=1.0, linestyle='--',\
                linewidth=0.25, xlocs=np.arange(-180, 180, 30), ylocs=np.arange(-90, 90, 10), draw_labels=True)

gl.top_labels = False
gl.right_labels = False

# Ploting the data temperature
img = plt.pcolor(mlon_bio[::2,::2], mlat_bio[::2,::2], data[::2,::2], cmap=cmap, norm=norm, transform=ccrs.PlateCarree())

# velocity
img2 = ax.streamplot(mlon[::10,::10], mlat[::10,::10], uvel[::10,::10], vvel[::10,::10], density=5.5, color='k', linewidth=0.7, arrowsize=0.5, arrowstyle='-|>', transform=ccrs.PlateCarree())

# Adding a colorbar

plt.colorbar(img, label='pH da Superfície do Mar', ticks=bounds, extend='neither',\
             orientation='horizontal', pad=0.05, fraction=0.05, shrink=0.7)

# Escrevendo os países
txt01 = ax.annotate("Brasil", xy=(-50,-10), xycoords=ccrs.PlateCarree()._as_mpl_transform(ax), color='black', size=12, clip_on=True, annotation_clip=True, horizontalalignment='center', verticalalignment='center', transform=ccrs.PlateCarree(), zorder=115)
txt02 = ax.annotate("EUA", xy=(-105,40), xycoords=ccrs.PlateCarree()._as_mpl_transform(ax), color='black', size=12, clip_on=True, annotation_clip=True, horizontalalignment='center', verticalalignment='center', transform=ccrs.PlateCarree(), zorder=115)
txt03 = ax.annotate("Canadá", xy=(-115,60), xycoords=ccrs.PlateCarree()._as_mpl_transform(ax), color='black', size=12, clip_on=True, annotation_clip=True, horizontalalignment='center', verticalalignment='center', transform=ccrs.PlateCarree(), zorder=115)
txt04 = ax.annotate("Rússia", xy=(90,60), xycoords=ccrs.PlateCarree()._as_mpl_transform(ax), color='black', size=12, clip_on=True, annotation_clip=True, horizontalalignment='center', verticalalignment='center', transform=ccrs.PlateCarree(), zorder=115)
txt05 = ax.annotate("China", xy=(105,35), xycoords=ccrs.PlateCarree()._as_mpl_transform(ax), color='black', size=12, clip_on=True, annotation_clip=True, horizontalalignment='center', verticalalignment='center', transform=ccrs.PlateCarree(), zorder=115)

# Add a bottom note
plt.figtext(0.5, 0.01, "Fonte dos dados: CMEMS - Correntes marinhas e pH da superfície do mar em "+data_title+".", ha="center", fontsize=16)#, bbox={"facecolor":"white", "alpha":1, "pad":5})

# Save the image
plt.savefig('../Output/cmems_ph+vel_'+data_arq+'.png')

# --------------------------------------------------------------------------------------------------------------------------
# LABOFIS / UERJ: Oceanography Products: CMEMS in Robinson Projection
# Author: Elisa Passos
# Si + Vel superficial
# --------------------------------------------------------------------------------------------------------------------------
print('Si  + Vel superficial')

# Extract the Si
data = file.si[0,0,:,:] 

# Choose the plot size (width x height, in inches)
fig = plt.figure(figsize=(19.20, 10.80), facecolor='white')

# Use the Mercator projection in cartopy
ax = plt.axes(projection=ccrs.Robinson()) #central_longitude=0.0

# Method used for "global" plots
ax.set_global()

# Add coastlines, borders and gridlines
ax.add_feature(cfeature.LAND, color='gray', edgecolor='k', zorder=100) # adding land mask
ax.coastlines(resolution='50m', color='black', linewidth=0.8, zorder=105)
ax.add_feature(cfeature.BORDERS, edgecolor='black', linewidth=0.5, zorder=110)
gl = ax.gridlines(crs=ccrs.PlateCarree(), color='white', alpha=1.0, linestyle='--',\
                linewidth=0.25, xlocs=np.arange(-180, 180, 30), ylocs=np.arange(-90, 90, 10), draw_labels=True)

gl.top_labels = False
gl.right_labels = False

# Ploting the data temperature
img = plt.pcolor(mlon_bio[::2,::2], mlat_bio[::2,::2], data[::2,::2], vmin=0, vmax=80, cmap='nipy_spectral', transform=ccrs.PlateCarree())

# velocity
img2 = ax.streamplot(mlon[::10,::10], mlat[::10,::10], uvel[::10,::10], vvel[::10,::10], density=5.5, color='w', linewidth=0.7, arrowsize=0.5, arrowstyle='-|>', transform=ccrs.PlateCarree())

# Adding a colorbar
plt.colorbar(img, label='Silicato da Superfície do Mar ($\mu mol \cdot L^{-1}$)', extend='max',\
                          orientation='horizontal', pad=0.05, fraction=0.05, shrink=0.7)

# Escrevendo os países
txt01 = ax.annotate("Brasil", xy=(-50,-10), xycoords=ccrs.PlateCarree()._as_mpl_transform(ax), color='black', size=12, clip_on=True, annotation_clip=True, horizontalalignment='center', verticalalignment='center', transform=ccrs.PlateCarree(), zorder=115)
txt02 = ax.annotate("EUA", xy=(-105,40), xycoords=ccrs.PlateCarree()._as_mpl_transform(ax), color='black', size=12, clip_on=True, annotation_clip=True, horizontalalignment='center', verticalalignment='center', transform=ccrs.PlateCarree(), zorder=115)
txt03 = ax.annotate("Canadá", xy=(-115,60), xycoords=ccrs.PlateCarree()._as_mpl_transform(ax), color='black', size=12, clip_on=True, annotation_clip=True, horizontalalignment='center', verticalalignment='center', transform=ccrs.PlateCarree(), zorder=115)
txt04 = ax.annotate("Rússia", xy=(90,60), xycoords=ccrs.PlateCarree()._as_mpl_transform(ax), color='black', size=12, clip_on=True, annotation_clip=True, horizontalalignment='center', verticalalignment='center', transform=ccrs.PlateCarree(), zorder=115)
txt05 = ax.annotate("China", xy=(105,35), xycoords=ccrs.PlateCarree()._as_mpl_transform(ax), color='black', size=12, clip_on=True, annotation_clip=True, horizontalalignment='center', verticalalignment='center', transform=ccrs.PlateCarree(), zorder=115)

# Add a bottom note
plt.figtext(0.5, 0.01, "Fonte dos dados: CMEMS - Correntes marinhas e Silicato da superfície do mar em "+data_title+".", ha="center", fontsize=16)#, bbox={"facecolor":"white", "alpha":1, "pad":5})

# Save the image
plt.savefig('../Output/cmems_si+vel_'+data_arq+'.png')


#---------------------------------------------------------------------------------------------------------------------------
# LABOFIS / UERJ: Oceanography Products: CMEMS in Robinson Projection
# Author: Elisa Passos
# PO4 + Vel superficial
#---------------------------------------------------------------------------------------------------------------------------
print('PO4 + Vel superficial')

# Extract the PO4
data = file.po4[0,0,:,:] #*file.variables['thetao'].scale_factor) + file.variables['thetao'].add_offset

# Choose the plot size (width x height, in inches)
fig = plt.figure(figsize=(19.20, 10.80), facecolor='white')

# Use the Mercator projection in cartopy
ax = plt.axes(projection=ccrs.Robinson()) #central_longitude=0.0

# Method used for "global" plots
ax.set_global()

# Add coastlines, borders and gridlines
ax.add_feature(cfeature.LAND, color='gray', edgecolor='k', zorder=100) # adding land mask
ax.coastlines(resolution='50m', color='black', linewidth=0.8, zorder=105)
ax.add_feature(cfeature.BORDERS, edgecolor='black', linewidth=0.5, zorder=110)
gl = ax.gridlines(crs=ccrs.PlateCarree(), color='white', alpha=1.0, linestyle='--',\
                linewidth=0.25, xlocs=np.arange(-180, 180, 30), ylocs=np.arange(-90, 90, 10), draw_labels=True)

gl.top_labels = False
gl.right_labels = False

# Ploting the data temperature
img = plt.pcolor(mlon_bio[::2,::2], mlat_bio[::2,::2], data[::2,::2], vmin=0, vmax=2.5, cmap='rainbow', transform=ccrs.PlateCarree())

# velocity
img2 = ax.streamplot(mlon[::10,::10], mlat[::10,::10], uvel[::10,::10], vvel[::10,::10], density=5.5, color='k', linewidth=0.7, arrowsize=0.5, arrowstyle='-|>', transform=ccrs.PlateCarree())

# Adding a colorbar
plt.colorbar(img, label='Fosfato da Superfície do Mar ($\mu mol \cdot L^{-1}$)', extend='max',\
                          orientation='horizontal', pad=0.05, fraction=0.05, shrink=0.7)

# Escrevendo os países
txt01 = ax.annotate("Brasil", xy=(-50,-10), xycoords=ccrs.PlateCarree()._as_mpl_transform(ax), color='black', size=12, clip_on=True, annotation_clip=True, horizontalalignment='center', verticalalignment='center', transform=ccrs.PlateCarree(), zorder=115)
txt02 = ax.annotate("EUA", xy=(-105,40), xycoords=ccrs.PlateCarree()._as_mpl_transform(ax), color='black', size=12, clip_on=True, annotation_clip=True, horizontalalignment='center', verticalalignment='center', transform=ccrs.PlateCarree(), zorder=115)
txt03 = ax.annotate("Canadá", xy=(-115,60), xycoords=ccrs.PlateCarree()._as_mpl_transform(ax), color='black', size=12, clip_on=True, annotation_clip=True, horizontalalignment='center', verticalalignment='center', transform=ccrs.PlateCarree(), zorder=115)
txt04 = ax.annotate("Rússia", xy=(90,60), xycoords=ccrs.PlateCarree()._as_mpl_transform(ax), color='black', size=12, clip_on=True, annotation_clip=True, horizontalalignment='center', verticalalignment='center', transform=ccrs.PlateCarree(), zorder=115)
txt05 = ax.annotate("China", xy=(105,35), xycoords=ccrs.PlateCarree()._as_mpl_transform(ax), color='black', size=12, clip_on=True, annotation_clip=True, horizontalalignment='center', verticalalignment='center', transform=ccrs.PlateCarree(), zorder=115)

# Add a bottom note
plt.figtext(0.5, 0.01, "Fonte dos dados: CMEMS - Correntes marinhas e Fosfato da superfície do mar em "+data_title+".", ha="center", fontsize=16)#, bbox={"facecolor":"white", "alpha":1, "pad":5})

# Save the image
plt.savefig('../Output/cmems_po4+vel_'+data_arq+'.png')


#---------------------------------------------------------------------------------------------------------------------------
# LABOFIS / UERJ: Oceanography Products: CMEMS in Robinson Projection
# Author: Elisa Passos
# Fe + Vel superficial
#---------------------------------------------------------------------------------------------------------------------------
print('Fe  + Vel superficial')

# Extract the Fe
data = file.fe[0,0,:,:] 
data = data*1000

# Choose the plot size (width x height, in inches)
fig = plt.figure(figsize=(19.20, 10.80), facecolor='white')

# Use the Mercator projection in cartopy
ax = plt.axes(projection=ccrs.Robinson()) #central_longitude=0.0

# Method used for "global" plots
ax.set_global()

# Add coastlines, borders and gridlines
ax.add_feature(cfeature.LAND, color='gray', edgecolor='k', zorder=100) # adding land mask
ax.coastlines(resolution='50m', color='black', linewidth=0.8, zorder=105)
ax.add_feature(cfeature.BORDERS, edgecolor='black', linewidth=0.5, zorder=110)
gl = ax.gridlines(crs=ccrs.PlateCarree(), color='white', alpha=1.0, linestyle='--',\
                linewidth=0.25, xlocs=np.arange(-180, 180, 30), ylocs=np.arange(-90, 90, 10), draw_labels=True)

gl.top_labels = False
gl.right_labels = False

# Ploting the data temperature
img = plt.pcolor(mlon_bio[::2,::2], mlat_bio[::2,::2], data[::2,::2], vmin=0, vmax=1.2, cmap='gist_rainbow_r', transform=ccrs.PlateCarree())

# velocity
img2 = ax.streamplot(mlon[::10,::10], mlat[::10,::10], uvel[::10,::10], vvel[::10,::10], density=5.5, color='k', linewidth=0.7, arrowsize=0.5, arrowstyle='-|>', transform=ccrs.PlateCarree())

# Adding a colorbar
plt.colorbar(img, label='Ferro dissolvido da Superfície do Mar ($nmol \cdot L^{-1}$)', extend='max',\
                          orientation='horizontal', pad=0.05, fraction=0.05, shrink=0.7)

# Escrevendo os países
txt01 = ax.annotate("Brasil", xy=(-50,-10), xycoords=ccrs.PlateCarree()._as_mpl_transform(ax), color='black', size=12, clip_on=True, annotation_clip=True, horizontalalignment='center', verticalalignment='center', transform=ccrs.PlateCarree(), zorder=115)
txt02 = ax.annotate("EUA", xy=(-105,40), xycoords=ccrs.PlateCarree()._as_mpl_transform(ax), color='black', size=12, clip_on=True, annotation_clip=True, horizontalalignment='center', verticalalignment='center', transform=ccrs.PlateCarree(), zorder=115)
txt03 = ax.annotate("Canadá", xy=(-115,60), xycoords=ccrs.PlateCarree()._as_mpl_transform(ax), color='black', size=12, clip_on=True, annotation_clip=True, horizontalalignment='center', verticalalignment='center', transform=ccrs.PlateCarree(), zorder=115)
txt04 = ax.annotate("Rússia", xy=(90,60), xycoords=ccrs.PlateCarree()._as_mpl_transform(ax), color='black', size=12, clip_on=True, annotation_clip=True, horizontalalignment='center', verticalalignment='center', transform=ccrs.PlateCarree(), zorder=115)
txt05 = ax.annotate("China", xy=(105,35), xycoords=ccrs.PlateCarree()._as_mpl_transform(ax), color='black', size=12, clip_on=True, annotation_clip=True, horizontalalignment='center', verticalalignment='center', transform=ccrs.PlateCarree(), zorder=115)

# Add a bottom note
plt.figtext(0.5, 0.01, "Fonte dos dados: CMEMS - Correntes marinhas e Ferro dissolvido da superfície do mar em "+data_title+".", ha="center", fontsize=16)#, bbox={"facecolor":"white", "alpha":1, "pad":5})

# Save the image
plt.savefig('../Output/cmems_fe+vel_'+data_arq+'.png')


