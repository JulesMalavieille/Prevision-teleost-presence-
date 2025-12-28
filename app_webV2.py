"""
Created on Fri Dec 19 13:25:07 2025

@author: Jules Malavieille
"""

#################################################
# Créer une appli web pour le modèle de prévision
#################################################

import math
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path
from joblib import load
import traceback
import copernicusmarine as cm
import folium
from streamlit_folium import st_folium

st.set_page_config(page_title="Prévision d'occurrence", layout="centered")
st.title("Prévision de présence d'espèces de poisson")
st.caption("En entrant des conditions environnementales, le modèle rend la probabilité de présence par espèce dans ces conditions. "
           "Ces probabilités sont calculées à partir d'un modèle de random forest entrainé sur 200 000 points d'occurences d'individus"
           " récolté par des observateurs de pêche et des plongeurs en Méditérranée et en Atlantique"
           " sur les côtes Française et Espagnole. Les données proviennent du site GBIF.")

species = [
    "Mugil_cephalus", "Sparus_aurata","Dentex_dentex","Sphyraena_sphyraena","Scomber_scombrus",
    "Seriola_dumerili","Labrus_viridis","Phycis_phycis","Diplodus_puntazzo",
    "Diplodus_sargus","Diplodus_vulgaris","Diplodus_cervinus","Lithognathus_mormyrus","Sarpa_salpa","Sarda_sarda",
    "Scorpaena_scrofa","Pagellus_erythrinus","Solea_solea","Dicentrarchus_labrax","Trachurus_trachurus","Mullus_surmuletus"]

vernaculaire = {
    "Mugil_cephalus": "Mulet",
    "Symphodus_tinca": "Crénilabre paon",
    "Sparus_aurata": "Daurade royale",
    "Dentex_dentex": "Denté",
    "Sphyraena_sphyraena": "Barracuda européen",
    "Scomber_scombrus": "Maquereau commun",
    "Pomatomus_saltatrix": "Tassergal",
    "Seriola_dumerili": "Sériole couronnée",
    "Labrus_viridis": "Labre vert",
    "Phycis_phycis": "Mostelle",
    "Diplodus_puntazzo": "Sar becofino",
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

D = pd.read_csv("Data/penalisation.csv")
D = D.drop(columns=["Unnamed: 0"])

D["Espèce"] = D["Espèce"].str.replace(" ", "_", regex=False)

not_penalised = ["Mugil_cephalus", "Symphodus_tinca", "Diplodus_puntazzo", "Diplodus_sargus", "Diplodus_vulgaris", "Sarpa_salpa"]

MODELS_DIR = Path("Modele")

@st.cache_resource
def load_models():
    models = {}
    for sp in species:
        path = MODELS_DIR / f"model_{sp}.joblib"
        if path.exists():
            obj = load(path)
            rf = obj["model"] if isinstance(obj, dict) and "model" in obj else obj
            models[sp] = rf
    return models

models = load_models()

# NOM DATASET CMEMS
DATASET_PHY = "cmems_mod_glo_phy_anfc_0.083deg_PT1H-m"  # temp/sal/u/v
DATASET_COX = "cmems_mod_glo_bgc-bio_anfc_0.25deg_P1D-m" #C/02
DATASET_THERM = "cmems_mod_glo_phy_anfc_0.083deg_P1D-m"  # thermo
DATASET_BATHY = "cmems_mod_glo_phy_anfc_0.083deg_static"  # bathy
DATASET_WAV = "cmems_mod_glo_wav_anfc_0.083deg_PT3H-i"  # vagues (Hs)

V_TEMP, V_SAL, V_U, V_V = "thetao", "so", "uo", "vo"
PRIM, O2 = "nppv", "o2"
THERM = "mlotst"
BATHY = "deptho"
V_WAV = "VHM0"  

DEFAULT_POINT = (43.2145, 5.4366)  # Sugiton 
DEFAULT_PARAMS = dict(temp=26.0, sal=38.0, bathy=15, waves=0.3, speed=0.05, direction=180.0, thermo=10.0, prim=100, o2=100)

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_cmems(day, lat, lon):
    """Retourne un dict de paramètres depuis CMEMS au point (lat,lon) pour la date day.
    Lève une exception si indisponible."""
    day = pd.to_datetime(day)

    # petite bbox pour robustesse (grille)
    d = 0.05
    minlon, maxlon = lon - d, lon + d
    minlat, maxlat = lat - d, lat + d

    df_phy = cm.read_dataframe(
        dataset_id=DATASET_PHY,
        variables=[V_TEMP, V_SAL, V_U, V_V],
        minimum_longitude=minlon, maximum_longitude=maxlon,
        minimum_latitude=minlat, maximum_latitude=maxlat,
        start_datetime=str(day),
        end_datetime=str(day + pd.Timedelta(hours=1)),
        coordinates_selection_method="nearest",
        disable_progress_bar=True)
    
    if df_phy.empty:
        raise ValueError("CMEMS PHY: aucune donnée trouvée pour ce point/date.")
        
    df_cox = cm.read_dataframe(
        dataset_id=DATASET_COX,
        variables=[PRIM, O2],
        minimum_longitude=minlon, maximum_longitude=maxlon,
        minimum_latitude=minlat, maximum_latitude=maxlat,
        start_datetime=str(day),
        end_datetime=str(day + pd.Timedelta(hours=1)),
        coordinates_selection_method="nearest",
        disable_progress_bar=True)
    
    if df_cox.empty:
        raise ValueError("CMEMS PHY: aucune donnée trouvée pour ce point/date.")
    
    df_therm = cm.read_dataframe(
        dataset_id=DATASET_THERM,
        variables=[THERM],
        minimum_longitude=minlon, maximum_longitude=maxlon,
        minimum_latitude=minlat, maximum_latitude=maxlat,
        start_datetime=str(day),
        end_datetime=str(day + pd.Timedelta(hours=24)),
        coordinates_selection_method="nearest",
        disable_progress_bar=True)
    
    if df_therm.empty:
        raise ValueError("CMEMS THERM: aucune donnée trouvée pour ce point/date.")
    
    row_phy = df_phy.iloc[0]
    row_cox = df_cox.iloc[0]
    row_th  = df_therm.iloc[0]
    temp = float(row_phy[V_TEMP])
    sal = float(row_phy[V_SAL])
    u = float(row_phy[V_U])
    v = float(row_phy[V_V])
    therm = float(row_th[THERM])
    prim = float(row_cox[PRIM])
    o2 = float(row_cox[O2])

    speed = math.sqrt(u*u + v*v)

    # convention météorologique (d'où vient le courant)
    # azimut : 0=N, 90=E (avec u Est, v Nord)
    dir_to = (math.degrees(math.atan2(v, u)))
    direction = (270 - dir_to) % 360.0

    df_wav = cm.read_dataframe(
        dataset_id=DATASET_WAV,
        variables=[V_WAV],
        minimum_longitude=minlon, maximum_longitude=maxlon,
        minimum_latitude=minlat, maximum_latitude=maxlat,
        start_datetime=str(day),
        end_datetime=str(day + pd.Timedelta(hours=3)),
        coordinates_selection_method="nearest",
        disable_progress_bar=True)
    
    if df_wav.empty:
        raise ValueError("CMEMS WAV: aucune donnée trouvée pour ce point/date.")

    waves = float(df_wav.iloc[0][V_WAV])
    
    df_bathy = cm.read_dataframe(
        dataset_id=DATASET_BATHY,
        variables=[BATHY],
        minimum_longitude=minlon, maximum_longitude=maxlon,
        minimum_latitude=minlat, maximum_latitude=maxlat,
        coordinates_selection_method="nearest",
        disable_progress_bar=True)
    
    if df_bathy.empty:
        raise ValueError("CMEMS BATHY: aucune donnée trouvée pour ce point.")
    
    bathy = float(df_bathy.iloc[0][BATHY])

    return dict(temp=temp, sal=sal, bathy=bathy, waves=waves, speed=speed, thermo=therm, prim=prim, o2=o2, direction=direction)



def init_state():
    """Initie les conditions environementales avec le remplissage automatique 
    au point par défaut (Sugiton) et la date d'aujourd'hui"""
    if "lat" not in st.session_state:
        st.session_state.lat, st.session_state.lon = DEFAULT_POINT

    if "day" not in st.session_state:
        st.session_state.day = pd.Timestamp.today().date()

    # paramètres courants (ceux affichés dans les widgets)
    for k, v in DEFAULT_PARAMS.items():
        st.session_state.setdefault(k, v)

    # messages UI
    st.session_state.setdefault("autofill_status", "")
    st.session_state.setdefault("autofill_ok", None)

def autofill_from_cmems(reset_on_fail=False):
    """Essaye de remplir depuis CMEMS. 
    - reset_on_fail=True : si échec au tout premier remplissage, lève une erreur.
    - reset_on_fail=False : si échec après changement de point, maintient des valeurs précédentes et message d'erreur."""
    try:
        vals = fetch_cmems(st.session_state.day, st.session_state.lat, st.session_state.lon)
        st.session_state.temp = float(vals["temp"])
        st.session_state.sal = float(vals["sal"])
        st.session_state.bathy = float(vals["bathy"])
        st.session_state.waves = float(vals["waves"])
        st.session_state.speed = float(vals["speed"])
        st.session_state.thermo = float(vals["thermo"])
        st.session_state.prim = float(vals["prim"])
        st.session_state.o2 = float(vals["o2"])
        st.session_state.direction = float(vals["direction"])

        st.session_state.autofill_ok = True
        st.session_state.autofill_status = "Remplissage automatique CMEMS réussi."
    except Exception as e:
        if reset_on_fail:
            for k, v in DEFAULT_PARAMS.items():
                st.session_state[k] = v
    
        st.session_state.autofill_ok = False
        st.session_state.autofill_status = (
            "Remplissage automatique CMEMS impossible.\n"
            f"Erreur: {type(e).__name__}: {e}")
        
        st.session_state.autofill_status = f"... {type(e).__name__}: {e}"
        st.session_state.autofill_trace = traceback.format_exc()
        if "autofill_trace" in st.session_state and st.session_state.autofill_ok is False:
            st.code(st.session_state.autofill_trace)

# Initie le remplissage par le CMEMS
init_state()

# 1) Premier chargement : on tente CMEMS aujourd’hui au point par défaut
if "did_first_autofill" not in st.session_state:
    autofill_from_cmems(reset_on_fail=True)
    st.session_state.did_first_autofill = True


# Callback : appelé quand lat/lon changent
def on_position_change():
    autofill_from_cmems(reset_on_fail=False)
    
# Changer lat/long, si marche pas -> point par défaut Sugiton
if "lat" not in st.session_state:
    st.session_state.lat = 43.2145
if "lon" not in st.session_state:
    st.session_state.lon = 5.4366
if "show_map" not in st.session_state:
    st.session_state.show_map = False

if "pending_lat" in st.session_state and "pending_lon" in st.session_state:
    st.session_state.lat = st.session_state.pop("pending_lat")
    st.session_state.lon = st.session_state.pop("pending_lon")
    on_position_change()  
    
def toggle_map():
    st.session_state.show_map = not st.session_state.show_map

# Initie la barre d'utilitaire (changer position et paramètres CMEMS)
with st.sidebar:
    st.header("Localisation & date")
    st.date_input("Date", key="day", on_change=on_position_change)

    st.number_input("Latitude", key="lat", format="%.6f", on_change=on_position_change)
    st.number_input("Longitude", key="lon", format="%.6f", on_change=on_position_change)
    
    # bouton 
    st.button("Choisir un point sur la carte", on_click=toggle_map)
    
    # carte
    if st.session_state.show_map:
        st.caption("Clique sur la carte pour définir Latitude / Longitude.")
    
        # Carte centrée sur la position actuelle
        m = folium.Map(
            location=[st.session_state.lat, st.session_state.lon],
            zoom_start=9,
            tiles="CartoDB dark_matter"  # ou "OpenStreetMap"
        )
    
        # Marqueur sur la position actuelle
        folium.Marker(
            [st.session_state.lat, st.session_state.lon],
            tooltip="Position actuelle",
        ).add_to(m)
    
        # Affiche la carte et récupère les interactions
        out = st_folium(m, height=400, width=None)
    
        # Si l'utilisateur clique, on récupère lat/lon
        if out and out.get("last_clicked"):
            clicked = out["last_clicked"]
            st.session_state.pending_lat = float(clicked["lat"])
            st.session_state.pending_lon = float(clicked["lng"])
            st.session_state.show_map = False
            st.rerun()
    
            # Refermer la carte après clic
            st.session_state.show_map = False
    
            # Déclenche autofill CMEMS
            on_position_change()
    
            st.success(f"Point sélectionné : lat={st.session_state.lat:.6f}, lon={st.session_state.lon:.6f}")
            st.rerun()

    # Paramètres environementaux (autorempli puis modifiable manuellement)
    st.divider()
    st.header("Paramètres physiques")
    st.number_input("Température (°C)", key="temp", min_value=0.0, max_value=35.0, step=0.1)
    st.number_input("Salinité (PSU)", key="sal", min_value=30.0, max_value=45.0, step=0.1)
    st.number_input("Profondeur max (m)", key="bathy", min_value=0.0, max_value=1000.0, step=1.0)
    st.number_input("Hauteur de vague (m)", key="waves", min_value=0.0, max_value=15.0, step=0.1)
    st.number_input("Vitesse du courant (m/s)", key="speed", min_value=0.0, max_value=5.0, step=0.01)
    st.number_input("Direction d'où vient le courant (°)", key="direction", min_value=0.0, max_value=360.0, step=1.0)
    st.number_input("Profondeur couche de mélange (m)", key="thermo", min_value=0.0, max_value=300.0, step=1.0)
    st.number_input("Production primaire net (mg/m3/jour)", key="prim", min_value=0.0, max_value=500.0, step=1.0)
    st.number_input("Concentration en O2 (mmol/m3)", key="o2", min_value=50.0, max_value=400.0, step=1.0)
    

    show_top = st.slider("Afficher top N", 5, 25)


# Test de remplissage après chaque remplissage 
if st.session_state.autofill_ok is True:
    st.success(st.session_state.autofill_status)
elif st.session_state.autofill_ok is False:
    st.warning(st.session_state.autofill_status)
else:
    st.info("Remplissage CMEMS non tenté.")

# Passage en du jour et de la direction en variable interprétable par le modèle 
jour = pd.to_datetime(st.session_state.day)
njour = jour.dayofyear
jour_sin = np.sin(2*np.pi*njour/365.0)
jour_cos = np.cos(2*np.pi*njour/365.0)

theta = np.deg2rad(st.session_state.direction)
dir_sin = np.sin(theta)
dir_cos = np.cos(theta)

# Création du datapoint 
X = np.array([
    st.session_state.temp,
    st.session_state.sal,
    st.session_state.bathy,
    st.session_state.waves,
    st.session_state.speed,
    st.session_state.thermo,
    st.session_state.prim,
    st.session_state.o2,
    jour_sin, jour_cos, dir_sin, dir_cos
], dtype=float).reshape(1, -1)


# Prediction de la présence d'espèce au condition env. du datapoint par les modèles 
rows = []
alpha = 0.5
for sp, rf in models.items():
    p = float(rf.predict_proba(X)[0, 1])
    
    if sp in not_penalised:
        p_corr = p
    
    else:
        d = D.loc[D["Espèce"]==sp, "Di"].iloc[0]
        p_corr = p*(alpha+(1-alpha)*d)
        
    rows.append({
        "Proba (%)": 100*p_corr,
        "Espèce": sp.replace("_", " "),
        "Vernaculaire": vernaculaire.get(sp, "—")
    })

# Affichage des résultats 
df = pd.DataFrame(rows).sort_values("Proba (%)", ascending=False).reset_index(drop=True)

st.subheader("Résultats")
st.dataframe(df.head(show_top), use_container_width=True)

st.bar_chart(df.set_index("Espèce")["Proba (%)"].head(show_top))

# Message warning
st.caption(
    "ATTENTION : "
    "Une probabilité élevée indique des conditions de présence favorables, mais ne garantit pas l'observation."
    " En effet, les déplacements des poissons, la visibilité et votre présence peuvent fortement influencer l'observabilité.")

st.caption(
    "Les données utilisées proviennent d’observations réalisées par des pêcheurs et des plongeurs. "
    "Elles peuvent donc présenter des biais d’échantillonnage (conditions de sortie, accessibilité, "
    "comportement humain, etc). Les probabilités affichées doivent donc être interprétées avec prudence.")

# Téléchargement des résultats 
st.download_button(
    "Télécharger les résultats (CSV)",
    df.to_csv(index=False).encode("utf-8"),
    file_name=f"occurrence_{jour.date()}.csv",
    mime="text/csv")



