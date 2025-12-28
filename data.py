"""
Created on Thu Dec 11 22:00:00 2025

@author: Jules Malavieille
"""

############################################################################
# L'objectif de ce script est de construire le tableau de données initial : 
# de récuperer les variables séléctionné pour chaque occurence
############################################################################

import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
import xarray as xr
from tqdm import trange


df = pd.read_csv("Data/Occurences1980_2025.csv", sep=";", encoding="latin1")

df["ok_coord"] = df["decimalLatitude"].notna() & df["decimalLongitude"].notna()
df_valid = df[df["ok_coord"]]

df_valid = df_valid[(df_valid["year"] > 2000) | ((df_valid["year"] == 2000) & (df_valid["month"] >= 12))]
df_valid = df_valid.reset_index(drop=True)

temp2000_2010 = xr.open_dataset("Variables/temp2010_2000.nc", decode_times=True)
temp2010_2025 = xr.open_dataset("Variables/temp2025_2010.nc", decode_times=True)

sal2000_2010 = xr.open_dataset("Variables/sal2010_2000.nc", decode_times=True)
sal2010_2025 = xr.open_dataset("Variables/sal2025_2010.nc", decode_times=True)

E2000_2010 = xr.open_dataset("Variables/courantE2010_2000.nc", decode_times=True)
E2010_2025 = xr.open_dataset("Variables/courantE2025_2010.nc", decode_times=True)

N2000_2010 = xr.open_dataset("Variables/courantN2010_2000.nc", decode_times=True)
N2010_2025 = xr.open_dataset("Variables/courantN2010_2025.nc", decode_times=True)

wave2000_2013 = xr.open_dataset("Variables/wave2013_2000.nc", decode_times=True)
wave2013_2025 = xr.open_dataset("Variables/wave2025_2013.nc", decode_times=True)

termo2000_2010 = xr.open_dataset("Variables/termo2010_2000.nc", decode_times=True)
termo2010_2025 = xr.open_dataset("Variables/termo2025_2010.nc", decode_times=True)

bathy = xr.open_dataset("Variables/bathy.nc", decode_times=True)
OC = xr.open_dataset("Variables/O2carbon2000_2025.nc", decode_times=True)

data = pd.DataFrame(columns=["Date", "Lat", "Long", "Temp", "Sal", "Bathy", "Waves", "Current_speed", "Current_dir", "Thermo", "Prim_prod", "O2_conc", "Species"])

data["Lat"] = df_valid["decimalLatitude"]
data["Long"] = df_valid["decimalLongitude"]
data["Species"] = df_valid["ScientificName"]

data["Date"] = pd.to_datetime(dict(year=df_valid.year, month=df_valid.month, day=df_valid.day), errors="coerce")
data = data.dropna(subset=["Date"]).reset_index(drop=True)  # On enlève les dates invalides

Temp = np.empty(len(data))
Sal = np.empty(len(data))
Bathy = np.empty(len(data))
Waves = np.empty(len(data))
Current_speed = np.empty(len(data))
Current_dir = np.empty(len(data))
Thermo = np.empty(len(data))
Prim_prod = np.empty(len(data))
O2_conc = np.empty(len(data))

threshold = np.datetime64("2010-11-25")
dif = 0.02 # degres 
for i in trange(data.shape[0], desc="Complétion du dataframe"):
    lat = data["Lat"].iloc[i]
    long = data["Long"].iloc[i]
    date = np.datetime64(data["Date"].iloc[i])
    
    if date < threshold:
        #####
        t = temp2000_2010["bottomT"].sel(latitude=lat, longitude=long, time=date, method="nearest").item()
        s = sal2000_2010["so"].sel(latitude=lat, longitude=long, time=date, method="nearest").item()
        th = termo2000_2010["mlotst"].sel(latitude=lat, longitude=long, time=date, method="nearest").item()
        ######
        
        u = E2000_2010["uo"].sel(latitude=lat, longitude=long, time=date, method="nearest").item()
        v = N2000_2010["vo"].sel(latitude=lat, longitude=long, time=date, method="nearest").item()

    if date >= threshold:
        #####
        t = temp2010_2025["bottomT"].sel(latitude=lat, longitude=long, time=date, method="nearest").item()
        s = sal2010_2025["so"].sel(latitude=lat, longitude=long, time=date, method="nearest").item()
        th = termo2010_2025["mlotst"].sel(latitude=lat, longitude=long, time=date, method="nearest").item()
        ######
        
        u = E2010_2025["uo"].sel(latitude=lat, longitude=long, time=date, method="nearest").item()
        v = N2010_2025["vo"].sel(latitude=lat, longitude=long, time=date, method="nearest").item()
    
    if date > np.datetime64('2013-10-31T21:00:00.000000000'):
        #####
        w = wave2013_2025["VHM0"].sel(latitude=lat, longitude=long, time=date, method="nearest").item()
        #####
        
    if date <= np.datetime64('2013-10-31T21:00:00.000000000'):
        #####
        w = wave2000_2013["VHM0"].sel(latitude=lat, longitude=long, time=date, method="nearest").item()
        #####
        
    ######
    sp = np.sqrt(u**2 + v**2) # Vitesse du vent
    ######
    
    d_rad = np.arctan2(v,u)
    d_deg = np.degrees(d_rad)
    
    #####
    d = (270 - d_deg) % 360  # D'ou vient le vent convention météo
    #####
    
    #####
    b = bathy["deptho"].sel(latitude=lat, longitude=long, method="nearest").item()
    #####
    
    #####
    prim = OC["nppv"].sel(latitude=lat, longitude=long, time=date, method="nearest").item()
    #####
    
    #####
    o2 = OC["o2"].sel(latitude=lat, longitude=long, time=date, method="nearest").item()
    #####
    
    Temp[i] = t
    Sal[i] = s
    Bathy[i] = b
    Waves[i] = w
    Current_speed[i] = sp
    Current_dir[i] = d
    Thermo[i] = th
    Prim_prod[i] = prim
    O2_conc[i] = o2
    
    
data["Temp"] = Temp
data["Sal"] = Sal
data["Bathy"] = Bathy
data["Waves"] = Waves
data["Current_speed"] = Current_speed
data["Current_dir"] = Current_dir
data["Thermo"] = Thermo
data["Prim_prod"] = Prim_prod
data["O2_conc"] = O2_conc

#data.to_csv("Data/Data_occurences_clean.csv")

    


