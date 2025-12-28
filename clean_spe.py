"""
Created on Fri Dec 12 15:49:36 2025

@author: Jules Malavieille
"""

#############################################
# Objectifs : analyse statistiques des donnés
#############################################

import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
import cartopy.crs as ccrs
import cartopy.feature as cfeature

data = pd.read_csv("Data/Data_occurences_clean.csv", sep=",")
data = data.drop(columns=["Unnamed: 0"])

data["Species"] = (
    data["Species"]
    .astype(str)
    .str.strip()
    .str.replace(r"[(),]", " ", regex=True)
    .str.split()
    .str[:2]
    .str.join(" "))

# S'asurer qu'il n'y ai pas de valeur aberrante
for col in ["Temp", "Sal", "Bathy", "Waves", "Current_speed", "Current_dir", "Thermo", "Prim_prod", "O2_conc"]:
    print(col, data[col].min(), data[col].max())
    

n = data["Species"].unique()

# De ce plot on regroupe les espèces "mal nommé"
species_map = {"Mugil cephalus":"Mugil cephalus", "Mugil curema":"Mugil cephalus", "Mugil spec.":"Mugil cephalus", "Mugil capurrii":"Mugil cephalus", "Mugil Linnaeus":"Mugil cephalus", 
               "Symphodus tinca":"Symphodus tinca", "Symphodus Crenilabrus":"Symphodus tinca", 
               "Sparus aurata":"Sparus aurata", "Sparus auratus":"Sparus aurata", 
               "Dentex dentex":"Dentex dentex", "Dentex Dentex":"Dentex dentex",
               "Sphyraena sphyraena":"Sphyraena sphyraena", "Sphyraena vulgaris":"Sphyraena sphyraena",
               "Scomber scombrus":"Scomber scombrus", "Scomber japonicus":"Scomber scombrus", "Scomber colias":"Scomber scombrus", "Scomber":"Scomber scombrus",
               "Pomatomus saltatrix":"Pomatomus saltatrix", "Pomatomus saltator":"Pomatomus saltatrix",
               "Lichia amia":"Lichia amia",
               "Seriola dumerili":"Seriola dumerili",
               "Labrus viridis":"Labrus viridis",
               "Phycis phycis":"Phycis phycis",
               "Diplodus puntazzo":"Diplodus puntazzo",
               "Diplodus sargus": "Diplodus sargus",
               "Diplodus vulgaris":"Diplodus vulgaris",
               "Diplodus cervinus":"Diplodus cervinus",
               "Lithognathus mormyrus":"Lithognathus mormyrus",
               "Sarpa salpa":"Sarpa salpa",
               "Sarda sarda":"Sarda sarda",
               "Scorpaena scrofa":"Scorpaena scrofa",
               "Pagellus erythrinus":"Pagellus erythrinus",
               "Solea solea":"Solea solea",
               "Dicentrarchus labrax":"Dicentrarchus labrax",
               "Trachurus trachurus":"Trachurus trachurus",
               "Mullus surmuletus":"Mullus surmuletus"}
    
data["Species"] = data["Species"].map(species_map).fillna(data["Species"])
data = data.dropna()

data.to_csv("Data/data_occurences_speclean.csv")

#################################
# Plot général de tous les points
#################################
# fig = plt.figure(figsize=(13, 15))
# ax = plt.axes(projection=ccrs.PlateCarree())

# ax.add_feature(cfeature.LAND, facecolor="lightgrey")
# ax.add_feature(cfeature.COASTLINE)
# ax.add_feature(cfeature.BORDERS, linestyle=':')

# ax.scatter(data["Long"], data["Lat"], s=1, alpha=0.25, transform=ccrs.PlateCarree()) # Plot le nuage de point
# ax.set_extent([-15, 14, 35, 60])   # Long_min_max, Lat_min_max
# ax.set_xlabel("Longitude")
# ax.set_ylabel("Latitude")
# ax.set_title("Occurrences (Med-Atl FR-ES, 2000–2025)")
# plt.show()











