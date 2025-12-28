"""
Created on Sun Dec 14 14:55:13 2025

@author: Jules Malavieille
"""

###############################################################################################################
# Utiliser les données de présence absence pour générer un modèle de probabilité d'occurence pour chaque espèce
###############################################################################################################

import numpy as np
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score
from joblib import dump
from tqdm import trange


var = ["Temp", "Sal", "Bathy", "Waves", "Current_speed", "Thermo", "Prim_prod", "O2_conc", "Jour_sin", "Jour_cos", "Dir_sin", "Dir_cos"]

species = ["Mugil cephalus", 
                "Symphodus tinca", 
                "Sparus aurata", 
                "Dentex dentex",
                "Sphyraena sphyraena",
                "Scomber scombrus",
                "Pomatomus saltatrix",
                "Lichia amia",
                "Seriola dumerili",
                "Labrus viridis",
                "Phycis phycis",
                "Diplodus puntazzo",
                "Diplodus sargus",
                "Diplodus vulgaris",
                "Diplodus cervinus",
                "Lithognathus mormyrus",
                "Sarpa salpa",
                "Sarda sarda",
                "Scorpaena scrofa",
                "Pagellus erythrinus",
                "Solea solea",
                "Dicentrarchus labrax",
                "Trachurus trachurus",
                "Mullus surmuletus"]

for i in trange(len(species)):
    spe = species[i]
    data = np.genfromtxt("Especes/data_pres_abs_"+spe+".csv", delimiter=",", skip_header=True)
    
    variables = data[:, 1:-1]
    target = data[:, -1]
    
    X_train, X_test, Y_train, Y_test = train_test_split(variables, target, test_size=0.1, shuffle=True)
    
    ###############
    # Random Forest
    ###############
    params = {"n_estimators":[100, 200, 300], "max_depth":[None, 5, 10], "min_samples_split":[2,5,7], "min_samples_leaf":[1, 3, 6]}
    RF_param = RandomForestClassifier(random_state=42)
    grid = GridSearchCV(RF_param, params, cv=5, scoring="roc_auc")
    grid.fit(X_train, Y_train)
    # Paramètres sont trouvé dans grid.best_params_ : n_est = 200; max_depth = None; min_split = 5; min_leaf = 1
    
    ne = grid.best_params_["n_estimators"]
    msl = grid.best_params_["min_samples_leaf"]
    mss = grid.best_params_["min_samples_split"]
    md = grid.best_params_["max_depth"]
    
    RF = RandomForestClassifier(random_state=42, max_depth=md, min_samples_leaf=msl, min_samples_split=mss, n_estimators=ne)
    RF.fit(X_train, Y_train)
    Y_proba = RF.predict_proba(X_test)[:, 1]
    AUC = roc_auc_score(Y_test, Y_proba)
    print("-------------------------------------")
    print(spe)
    print("L'AUC de la Random Forest sur les données test :", AUC)
    print("L'accuracy de la Random Forest sur les données test :", RF.score(X_test, Y_test))
    print("-------------------------------------")
    print()
    
    safe_spe = spe.replace(" ", "_")
    dump({"model":RF, "features":var}, "Modele/model_"+safe_spe+".joblib")
    
    # Avoir la proba d'occurence pour un ensemble de paramètres:
    # Temp, Sal, Bathy, waves, Current speed, Thermo, Jour sin, Jour cos, Dir sin, Dir cos
    # X_fact = [[15, 38, 40, 0.1, 0.01, 10, -0.668064, -0.744104, -0.00943354, -0.999956]]
    # X_prob = RF.predict_proba(X_fact)[0,1]
    
    # print("La probabilité qu'un "+spe+" soit dans avec les paramètres environmentaux spécifié est de", X_prob*100,"%")



