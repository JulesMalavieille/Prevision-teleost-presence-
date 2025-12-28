"""
Created on Wed Dec 17 23:01:13 2025

@author: Jules Malavieille
"""

import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
from joblib import load


def ask_float(prompt, min_val=None, max_val=None, allow_empty=False):
    while True:
        s = input(prompt).strip().replace(",", ".")
        if allow_empty and s == "":
            return None
        try:
            x = float(s)
        except ValueError:
            print("Veuillez entrer un nombre (ex: 12.5).")
            continue
        if min_val is not None and x < min_val:
            print(f"Valeur trop basse (minimum possible {min_val}).")
            continue
        if max_val is not None and x > max_val:
            print(f"Valeur trop haute (maximum possible {max_val}).")
            continue
        return x


def ask_date(prompt):
    while True:
        s = input(prompt).strip()
        try:
            return pd.to_datetime(s, dayfirst=True)
        except Exception:
            print("Date invalide. Format attendu: AAAA-MM-JJ (ex: 2025-08-17).")
    

species = ["Mugil_cephalus", 
                "Sparus_aurata", 
                "Dentex_dentex",
                "Sphyraena_sphyraena",
                "Scomber_scombrus",
                "Lichia_amia",
                "Seriola_dumerili",
                "Labrus_viridis",
                "Phycis_phycis",
                "Diplodus_cervinus",
                "Lithognathus_mormyrus",
                "Sarpa_salpa",
                "Sarda_sarda",
                "Scorpaena_scrofa",
                "Pagellus_erythrinus",
                "Solea_solea",
                "Dicentrarchus_labrax",
                "Trachurus_trachurus",
                "Mullus_surmuletus"]

vernaculaire = {
    "Mugil_cephalus": "Mulet",
    "Symphodus_tinca": "Crénilabre paon",
    "Sparus_aurata": "Daurade royale",
    "Dentex_dentex": "Denté",
    "Sphyraena_sphyraena": "Barracuda européen",
    "Scomber_scombrus": "Maquereau commun",
    "Pomatomus_saltatrix": "Tassergal",
    "Lichia_amia": "Liche amie",
    "Seriola_dumerili": "Sériole couronnée",
    "Labrus_viridis": "Labre vert",
    "Phycis_phycis": "Mostelle",
    "Diplodus_puntazzo": "Sar à museau pointu",
    "Diplodus_sargus": "Sar commun",
    "Diplodus_vulgaris": "Sar à tête noir",
    "Diplodus_cervinus": "Sar tambour",
    "Lithognathus_mormyrus": "Marbré",
    "Sarpa_salpa": "Saupe",
    "Sarda_sarda": "Bonite à dos rayé",
    "Scorpaena_scrofa": "Rascasse rouge",
    "Pagellus_erythrinus": "Pageot commun",
    "Solea_solea": "Sole commune",
    "Dicentrarchus_labrax": "Loup (bar)",
    "Trachurus_trachurus": "Chinchard",
    "Mullus_surmuletus": "Rouget"}

jour = ask_date("Quel jour est-il (AAAA-MM-JJ) ? : ")
temp = ask_float("Quel est la température de l'eau (°C) ? : ", min_val=0, max_val=35)
sal = ask_float("Quel est la salinité de l'eau (PSU) ? : ", min_val=30, max_val=45)
bathy = ask_float("A quel profondeur maximale allez-vous (m) ? : ", min_val=0, max_val=1000)
waves = ask_float("Quel est la hauteur de vague (m) ? : ", min_val=0, max_val=30)
speed = ask_float("Quel est la vitesse du courant (m/s) ? : ", min_val=0, max_val=3)
direction = ask_float("De quel direction vient le courant (en azimut : 0-360) ? : ", min_val=0, max_val=360)
thermo = ask_float("Quel est la profondeur de la couche de mélange (m) ? : ", min_val=0, max_val=300)

warnings = []

if waves > 3 or speed > 1:
    warnings.append("Mer extrêmement agité, prudence.")
    
if waves > 5:
    warnings.append("Hauteur de vagues extrêmes, incohérent en Méditérranée")

if speed > 2:
    warnings.append("Vitesse de courant très élevé, incohérent en Méditérranée.")

if thermo > 50:
    warnings.append("Profondeur de thermocline très grande, incohérent en Méditérranée.")


jour = pd.to_datetime(jour)
njour = jour.dayofyear

jour_sin = np.sin(2 * np.pi * njour / 365)
jour_cos = np.cos(2 * np.pi * njour / 365)

tetha = np.deg2rad(direction)
dir_sin = np.sin(tetha)
dir_cos = np.cos(tetha)

x_test = [[temp, sal, bathy, waves, speed, thermo, jour_sin, jour_cos, dir_sin, dir_cos]]
val_prev = []
for i in range(len(species)):
    sp = species[i]
    model = load("Modele/model_"+sp+".joblib")
    rf = model["model"]
    y_pred = rf.predict_proba(x_test)[0,1]
    val_prev.append((y_pred, sp))


val_prev.sort(reverse=True)#, key=lambda x: x[0])

if warnings:
    print("\n" + "="*78)
    print("CONTRÔLES DE COHÉRENCE")
    for w in warnings:
        print(" - " + w)
    print("="*78 + "\n")

    
# ====== Affichage contexte ======
print("\n" + "="*78)
print(f"Prévision de présence — Date: {jour.date()} (jour {njour}/365)")
print(f"Température={temp}°C | Salinité={sal} | Profondeur max={bathy}m | Vagues={waves}m")
print(f"Vitesse du courant={speed}m/s | Direction du courant={direction}° | Profondeur de la thermocline={thermo}m")
print("="*78)
    

# ====== Rendu terminal ======
print("Prévision des probabilités de présence : ")
print("-"*78)
for p, sp in val_prev:
    s = sp.replace("_", " ")
    fr = vernaculaire.get(sp, "—")
    print(f"{100*p:7.2f}%  {s:<25}  {fr}")
print("-"*78)


# ====== Affichage warning =======
print("\n" + "="*78)
print("INFORMATION IMPORTANTE")
print("Une probabilité élevée indique des conditions favorables")
print("à la présence de l'espèce, mais n'assure pas son observation.")
print("En effet, les déplacements des poissons, la visibilité et votre présence")
print("peuvent fortement influencer l'observabilité des organismes.")
print("="*78 + "\n")















