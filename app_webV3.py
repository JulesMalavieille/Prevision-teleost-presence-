"""
Created on Sun Dec 21 11:59:15 2025

@author: Jules Malavieille
"""

#######################################################################################################
# Appli Streamlit : cartes de probabilité de présence (RF) + CMEMS + masquage terre rapide (clip bbox)
#######################################################################################################

import math
import traceback
import streamlit as st
import numpy as np
import pandas as pd
from datetime import date
from pathlib import Path
from scipy.interpolate import griddata
import pydeck as pdk
import copernicusmarine as cm
from joblib import load
import geopandas as gpd
from shapely.geometry import box

# ======================================================================================
# Streamlit config
# ======================================================================================
st.set_page_config(page_title="Cartes probabilité poissons", layout="wide")
st.title("Cartes de densité de probabilité – Poissons")

# ======================================================================================
# CMEMS datasets / variables
# ======================================================================================
DATASET_PHY = "cmems_mod_glo_phy_anfc_0.083deg_PT1H-m"  # temp/sal/u/v
DATASET_COX = "cmems_mod_glo_bgc-bio_anfc_0.25deg_P1D-m" #C/02
DATASET_THERM = "cmems_mod_glo_phy_anfc_0.083deg_P1D-m"  # thermo
DATASET_BATHY = "cmems_mod_glo_phy_anfc_0.083deg_static"  # bathy
DATASET_WAV = "cmems_mod_glo_wav_anfc_0.083deg_PT3H-i"  # vagues (Hs)
# Variables : adapte selon tes datasets (à confirmer via cm.describe)
V_TEMP, V_SAL, V_U, V_V = "thetao", "so", "uo", "vo"
PRIM, O2 = "nppv", "o2"
THERM = "mlotst"
BATHY = "deptho"
V_WAV = "VHM0"  

# ======================================================================================
# Initialisation des outils (modèle et trait de côte)
# ======================================================================================
DEFAULT_POINT = (43.2145, 5.4366)  # Sugiton
MODELS_DIR = Path("Modele")

LAND_PATH = Path("Data/ne_10m_land/ne_10m_land.shp")  # NaturalEarth land polygons (10m)

# ======================================================================================
# Espèces séléctionné 
# ======================================================================================
SPECIES_CODES = [
    "Mugil_cephalus","Symphodus_tinca", "Sparus_aurata","Dentex_dentex","Sphyraena_sphyraena","Scomber_scombrus",
    "Seriola_dumerili","Labrus_viridis","Phycis_phycis","Diplodus_puntazzo",
    "Diplodus_sargus","Diplodus_vulgaris","Diplodus_cervinus","Lithognathus_mormyrus","Sarpa_salpa","Sarda_sarda",
    "Scorpaena_scrofa","Pagellus_erythrinus","Solea_solea","Dicentrarchus_labrax","Trachurus_trachurus","Mullus_surmuletus"]

VERNACULAIRE = {
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

# ======================================================================================
# Paramètres de pénalisation et espèce non-ciblé  
# ======================================================================================
D = pd.read_csv("Data/penalisation.csv")
D = D.drop(columns=["Unnamed: 0"])

D["Espèce"] = D["Espèce"].str.replace(" ", "_", regex=False)

not_penalised = ["Mugil_cephalus", "Symphodus_tinca", "Diplodus_puntazzo", "Diplodus_sargus", "Diplodus_vulgaris", "Sarpa_salpa"]

# ======================================================================================
# Récupération des modèles de prévision
# ======================================================================================
@st.cache_resource
def load_models():
    models = {}
    missing = []
    for sp in SPECIES_CODES:
        path = MODELS_DIR / f"model_{sp}.joblib"
        if path.exists():
            obj = load(path)
            rf = obj["model"] if isinstance(obj, dict) and "model" in obj else obj
            models[sp] = rf
        else:
            missing.append(sp)
    return models, missing

MODELS, MISSING = load_models()

# ======================================================================================
# Télécherger le trait de côte, générer un mask qui ne séléctionne que la mer
# ======================================================================================
@st.cache_resource
def load_land_10m():
    land = gpd.read_file(LAND_PATH).to_crs(epsg=4326)
    land = land[land.geometry.notna() & (~land.geometry.is_empty)].copy()
    # force construction sindex une fois
    _ = land.sindex
    return land


def _bbox_key(bbox, ndigits=4):
    # clé stable (0.0001° ~ 11 m). Mets 3 si tu veux plus de cache.
    minlat, maxlat, minlon, maxlon = bbox
    return (round(minlat, ndigits), round(maxlat, ndigits),
            round(minlon, ndigits), round(maxlon, ndigits))


@st.cache_data(show_spinner=False)
def land_clip_for_bbox(bbox_key):
    land = load_land_10m()
    minlat, maxlat, minlon, maxlon = bbox_key
    bbox_poly = box(minlon, minlat, maxlon, maxlat)

    # 1) préfiltre hyper rapide via sindex
    idx = list(land.sindex.query(bbox_poly, predicate="intersects"))
    if not idx:
        return land.iloc[0:0].copy()  # empty GeoDataFrame

    land_local = land.iloc[idx].copy()

    # 2) recoupe géométriquement au bbox (optionnel mais ça accélère ensuite)
    # Ça réduit la complexité des polygones sur ta zone
    land_local["geometry"] = land_local.geometry.intersection(bbox_poly)
    land_local = land_local[land_local.geometry.notna() & (~land_local.geometry.is_empty)].copy()

    # rebuild sindex local
    _ = land_local.sindex
    return land_local

def mask_sea_points(grid_df: pd.DataFrame, bbox) -> pd.DataFrame:
    bbox_key = _bbox_key(bbox, ndigits=4)
    land_local = land_clip_for_bbox(bbox_key)
    if land_local.empty:
        return grid_df.reset_index(drop=True)

    # points -> GeoDataFrame
    pts = gpd.GeoDataFrame(
        grid_df.copy(),
        geometry=gpd.points_from_xy(grid_df["lon"], grid_df["lat"]),
        crs="EPSG:4326")

    # Join spatial : quelles points tombent dans un polygone land ?
    # predicate="within" (ou "intersects" si tu veux virer aussi les points sur la côte)
    hits = gpd.sjoin(pts, land_local[["geometry"]], how="left", predicate="within")

    # index_right non-null => point sur terre
    sea = hits[hits["index_right"].isna()].drop(columns=["geometry", "index_right"]).reset_index(drop=True)
    return pd.DataFrame(sea)

# ======================================================================================
# Grille autour du point séléctionné 
# ======================================================================================
@st.cache_data
def make_grid(lat0, lon0, radius_km=20, step_m=500):
    step_deg_lat = step_m / 111_000
    step_deg_lon = step_m / (111_000 * np.cos(np.deg2rad(lat0)))

    r_deg_lat = (radius_km * 1000) / 111_000
    r_deg_lon = (radius_km * 1000) / (111_000 * np.cos(np.deg2rad(lat0)))

    lats = np.arange(lat0 - r_deg_lat, lat0 + r_deg_lat + step_deg_lat, step_deg_lat)
    lons = np.arange(lon0 - r_deg_lon, lon0 + r_deg_lon + step_deg_lon, step_deg_lon)

    Lon, Lat = np.meshgrid(lons, lats)
    grid_df = pd.DataFrame({"lat": Lat.ravel(), "lon": Lon.ravel()})
    shape = Lat.shape
    bbox = (float(Lat.min()), float(Lat.max()), float(Lon.min()), float(Lon.max()))
    return grid_df, shape, bbox

# ======================================================================================
# Paramètres environementaux sur toute la grille 
# ======================================================================================
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_cmems_field(day, minlat, maxlat, minlon, maxlon):
    day = pd.to_datetime(day)

    df_phy = cm.read_dataframe(
        dataset_id=DATASET_PHY,
        variables=[V_TEMP, V_SAL, V_U, V_V],
        minimum_longitude=minlon, maximum_longitude=maxlon,
        minimum_latitude=minlat, maximum_latitude=maxlat,
        start_datetime=str(day),
        end_datetime=str(day + pd.Timedelta(hours=1)),
        disable_progress_bar=True)
    
    if df_phy.empty:
        raise ValueError("CMEMS PHY: champ vide sur bbox/date.")
        
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
        disable_progress_bar=True)
    
    if df_therm.empty:
        raise ValueError("CMEMS THERM: champ vide sur bbox/date.")

    df_wav = cm.read_dataframe(
        dataset_id=DATASET_WAV,
        variables=[V_WAV],
        minimum_longitude=minlon, maximum_longitude=maxlon,
        minimum_latitude=minlat, maximum_latitude=maxlat,
        start_datetime=str(day),
        end_datetime=str(day + pd.Timedelta(hours=3)),
        disable_progress_bar=True)
    
    if df_wav.empty:
        raise ValueError("CMEMS WAV: champ vide sur bbox/date.")

    df_bathy = cm.read_dataframe(
        dataset_id=DATASET_BATHY,
        variables=[BATHY],
        minimum_longitude=minlon, maximum_longitude=maxlon,
        minimum_latitude=minlat, maximum_latitude=maxlat,
        disable_progress_bar=True)
    
    if df_bathy.empty:
        raise ValueError("CMEMS BATHY: champ vide sur bbox.")
    return df_phy, df_cox, df_therm, df_wav, df_bathy

# ======================================================================================
# Interpole les paramètres environementaux sur la grille pour un rendu continu
# ======================================================================================
def ensure_coords_columns(df: pd.DataFrame) -> pd.DataFrame:
    try:
        if not isinstance(df.index, pd.RangeIndex):
            df = df.reset_index()
    except Exception:
        pass
    return df

def find_lat_lon_cols(df: pd.DataFrame):
    lat_candidates = ["lat", "latitude", "LATITUDE", "Latitude", "y", "nav_lat"]
    lon_candidates = ["lon", "longitude", "LONGITUDE", "Longitude", "x", "nav_lon"]
    latc = next((c for c in lat_candidates if c in df.columns), None)
    lonc = next((c for c in lon_candidates if c in df.columns), None)
    if latc is None or lonc is None:
        raise KeyError(
            "Coords introuvables après reset_index(). "
            f"Colonnes dispo: {list(df.columns)[:40]} ... | "
            f"Index names: {getattr(df.index, 'names', None)}")
        
    return latc, lonc

def griddata_with_fallback(pts, values, q):
    out = griddata(pts, values, q, method="linear")
    if np.isnan(out).any():
        out2 = griddata(pts, values, q, method="nearest")
        out = np.where(np.isnan(out), out2, out)
    return out

@st.cache_data(ttl=3600, show_spinner=False)
def interpolate_features_on_grid(day, grid_df, bbox):
    minlat, maxlat, minlon, maxlon = bbox
    df_phy, df_cox, df_therm, df_wav, df_bathy = fetch_cmems_field(day, minlat, maxlat, minlon, maxlon)

    df_phy   = ensure_coords_columns(df_phy)
    df_cox   = ensure_coords_columns(df_cox)
    df_therm = ensure_coords_columns(df_therm)
    df_wav   = ensure_coords_columns(df_wav)
    df_bathy = ensure_coords_columns(df_bathy)

    q = np.column_stack([grid_df["lon"].to_numpy(), grid_df["lat"].to_numpy()])

    # PHY
    latc, lonc = find_lat_lon_cols(df_phy)
    pts = np.column_stack([df_phy[lonc].to_numpy(), df_phy[latc].to_numpy()])
    temp = griddata_with_fallback(pts, df_phy[V_TEMP].to_numpy(), q)
    sal  = griddata_with_fallback(pts, df_phy[V_SAL].to_numpy(),  q)
    u    = griddata_with_fallback(pts, df_phy[V_U].to_numpy(),    q)
    v    = griddata_with_fallback(pts, df_phy[V_V].to_numpy(),    q)
    
    # COX
    latc, lonc = find_lat_lon_cols(df_cox)
    pts = np.column_stack([df_cox[lonc].to_numpy(), df_cox[latc].to_numpy()])
    prim = griddata_with_fallback(pts, df_cox[PRIM].to_numpy(), q)
    o2  = griddata_with_fallback(pts, df_cox[O2].to_numpy(),  q)

    # THERMO
    latc_t, lonc_t = find_lat_lon_cols(df_therm)
    pts_t = np.column_stack([df_therm[lonc_t].to_numpy(), df_therm[latc_t].to_numpy()])
    therm = griddata_with_fallback(pts_t, df_therm[THERM].to_numpy(), q)

    # WAV
    latc_w, lonc_w = find_lat_lon_cols(df_wav)
    pts_w = np.column_stack([df_wav[lonc_w].to_numpy(), df_wav[latc_w].to_numpy()])
    waves = griddata_with_fallback(pts_w, df_wav[V_WAV].to_numpy(), q)

    # BATHY
    latc_b, lonc_b = find_lat_lon_cols(df_bathy)
    pts_b = np.column_stack([df_bathy[lonc_b].to_numpy(), df_bathy[latc_b].to_numpy()])
    bathy = griddata_with_fallback(pts_b, df_bathy[BATHY].to_numpy(), q)

    # direction -> sin/cos
    speed = np.sqrt(u*u + v*v)
    dir_to = np.degrees(np.arctan2(v, u))
    direction = (270 - dir_to) % 360.0
    theta = np.deg2rad(direction)
    dir_sin = np.sin(theta)
    dir_cos = np.cos(theta)

    # date -> features cycliques
    day_ts = pd.to_datetime(day)
    njour = int(day_ts.dayofyear)
    jour_sin = np.sin(2*np.pi*njour/365.0)
    jour_cos = np.cos(2*np.pi*njour/365.0)

    X = pd.DataFrame({
        "temp": temp,
        "sal": sal,
        "bathy": bathy,
        "waves": waves,
        "speed": speed,
        "thermo": therm,
        "prim": prim,
        "o2": o2,
        "jour_sin": jour_sin,
        "jour_cos": jour_cos,
        "dir_sin": dir_sin,
        "dir_cos": dir_cos})

    return X.bfill().ffill()

def get_grid_features(day, lat0, lon0, radius_km, step_m):
    """Permet de récuperer les paramètres et de ne conserver que la mer"""
    grid_df, shape, bbox = make_grid(lat0, lon0, radius_km, step_m)

    # IMPORTANT: masque terre AVANT interpolation (pour éviter valeurs "terre" interpolées)
    grid_df = mask_sea_points(grid_df, bbox)

    X = interpolate_features_on_grid(day, grid_df, bbox)
    return grid_df, X, shape


# ======================================================================================
# Prédiction avec pénalisation des espèces concerné 
# ======================================================================================
ALPHA = 0.5  #terme de pénalisation de la pêche

def _get_di(sp_code: str) -> float:
    """Récupère Di de chaque espèce"""
    if D is None or D.empty:
        return 1.0

    # accepte "Di" ou "D_i" (au cas où)
    col = "Di" if "Di" in D.columns else ("D_i" if "D_i" in D.columns else None)
    if col is None:
        return 1.0

    row = D.loc[D["Espèce"] == sp_code, col]
    if row.empty:
        return 1.0
    return float(row.iloc[0])

def penalty_factor(sp_code: str, alpha: float = ALPHA) -> float:
    """Calcul la prédiction pénalisé"""
    if sp_code in not_penalised:
        return 1.0

    d = _get_di(sp_code)
    # clamp sécurité
    d = float(np.clip(d, 0.0, 1.0))
    return float(alpha + (1.0 - alpha) * d)

@st.cache_data(ttl=3600, show_spinner=False)
def predict_species_map(day, lat0, lon0, radius_km, step_m, sp_code):
    """Applique la prédiction pénalisé sur toute la grille pour une espèce"""
    if sp_code not in MODELS:
        raise ValueError(f"Modèle introuvable pour {sp_code}.")
    model = MODELS[sp_code]

    grid_df, X, _shape = get_grid_features(day, lat0, lon0, radius_km, step_m)

    p = model.predict_proba(X)[:, 1].astype(float)
    f = penalty_factor(sp_code)
    p_corr = np.clip(p * f, 0.0, 1.0)

    return grid_df.assign(p=p_corr)

@st.cache_data(ttl=3600, show_spinner=False)
def predict_general_map(day, lat0, lon0, radius_km, step_m, agg="Somme", threshold=0.5):
    """Applique la prédiction pénalisé sur toute la grille pour toute les espèces"""
    grid_df, X, _shape = get_grid_features(day, lat0, lon0, radius_km, step_m)

    avail = list(MODELS.keys())
    if len(avail) == 0:
        raise ValueError("Aucun modèle chargé.")

    # Probabilités brutes (n_points, n_species)
    P = np.column_stack([MODELS[s].predict_proba(X)[:, 1] for s in avail]).astype(float)

    # Applique la même pénalisation espèce par espèce
    factors = np.array([penalty_factor(s) for s in avail], dtype=float)  # (n_species,)
    P_corr = np.clip(P * factors[None, :], 0.0, 1.0)

    # Agrégation sur les probas pénalisées (cohérence)
    if agg == "Somme":
        p = P_corr.sum(axis=1)
    elif agg == "Max":
        p = P_corr.max(axis=1)
    elif agg == "# espèces > seuil":
        p = (P_corr > float(threshold)).sum(axis=1)
    else:
        raise ValueError("Agrégation inconnue.")

    return grid_df.assign(p=p)


# ======================================================================================
# Interface utilisateur 
# ======================================================================================
if "center_lat" not in st.session_state:
    st.session_state.center_lat = float(DEFAULT_POINT[0])
if "center_lon" not in st.session_state:
    st.session_state.center_lon = float(DEFAULT_POINT[1])
if "day" not in st.session_state:
    st.session_state.day = date.today()

predefined = {
    "Calanques de Marseille/Cassis": (43.204, 5.434),
    "Côte Bleue": (43.325, 5.1754), 
    "Ciotat": (43.161, 5.611),
    "Toulon":(43.051, 6.001)}

with st.sidebar:
    st.header("Paramètres")

    mode = st.radio("Point", ["Prédéfini", "Manuel"], index=0)
    if mode == "Prédéfini":
        place = st.selectbox("Lieu", list(predefined.keys()))
        st.session_state.center_lat = float(predefined[place][0])
        st.session_state.center_lon = float(predefined[place][1])

    st.number_input("Latitude", key="center_lat", format="%.6f")
    st.number_input("Longitude", key="center_lon", format="%.6f")
    st.date_input("Date", key="day")

    radius_km = st.slider("Rayon (km)", 5, 60, 20)
    step_m = st.select_slider("Résolution (m)", options=[200, 300, 500, 800, 1000], value=500)

    view = st.selectbox("Carte", ["Par espèce", "Générale"])

    if view == "Par espèce":
        opts = [(VERNACULAIRE.get(s, s), s) for s in MODELS.keys()]
        opts = sorted(opts, key=lambda x: x[0])
        label_to_code = {lab: code for lab, code in opts}
        sp_label = st.selectbox("Espèce", [lab for lab, _ in opts])
        sp_code = label_to_code[sp_label]
    else:
        agg = st.selectbox("Agrégation", ["Somme", "Max", "# espèces > seuil"])
        threshold = st.slider("Seuil", 0.0, 1.0, 0.5, 0.05)

    if MISSING:
        st.caption(f"Modèles manquants: {len(MISSING)}")

# ======================================================================================
# Résultats (HeatmapLayer)
# ======================================================================================
lat0 = float(st.session_state.center_lat)
lon0 = float(st.session_state.center_lon)
day = st.session_state.day

try:
    if view == "Par espèce":
        with st.spinner("Calcul carte espèce (CMEMS + interpolation + RF)..."):
            grid_df = predict_species_map(day, lat0, lon0, radius_km, step_m, sp_code)
        title = f"Probabilité – {VERNACULAIRE.get(sp_code, sp_code)}"
    else:
        with st.spinner("Calcul carte générale (CMEMS + interpolation + agrégation)..."):
            grid_df = predict_general_map(day, lat0, lon0, radius_km, step_m, agg=agg, threshold=threshold)
        title = f"Carte générale – {agg}"

    st.subheader(title)

    grid_df["lat"] = grid_df["lat"].astype(float)
    grid_df["lon"] = grid_df["lon"].astype(float)
    grid_df["p"] = pd.to_numeric(grid_df["p"], errors="coerce")
    grid_df = grid_df.dropna(subset=["lat", "lon", "p"]).reset_index(drop=True)

    layer = pdk.Layer("HeatmapLayer",
                        data=grid_df,
                        get_position=["lon", "lat"],
                        get_weight="p",
                        radiusPixels=int(np.clip(step_m / 10, 10, 60)),
                        threshold=0.04)


    view_state = pdk.ViewState(latitude=lat0, longitude=lon0, zoom=11, pitch=0, bearing=0)
    deck = pdk.Deck(
        layers=[layer],
        initial_view_state=view_state,
        map_style="mapbox://styles/mapbox/light-v9",
        tooltip={"html": "<b>p</b>: {p}<br/><b>lat</b>: {lat}<br/><b>lon</b>: {lon}"})

    st.pydeck_chart(deck, use_container_width=True)
    
    # st.markdown(
    #     """
    #     <div style="
    #         width:300px;
    #         padding:10px 14px;
    #         border-radius:10px;
    #         background:rgba(255,255,255,0.9);
    #         box-shadow:0 2px 6px rgba(0,0,0,0.25);
    #         position:relative;
    #         z-index:999;
    #     ">
    #       <div style="font-size:13px; font-weight:600; margin-bottom:6px;">
    #         Probabilité de présence
    #       </div>
    #       <div style="
    #           height:14px;
    #           border-radius:8px;
    #           background: linear-gradient(to right,
    #             rgb(68,1,84),
    #             rgb(71,44,122),
    #             rgb(59,81,139),
    #             rgb(44,113,142),
    #             rgb(33,144,141),
    #             rgb(39,173,129),
    #             rgb(92,200,99),
    #             rgb(170,220,50),
    #             rgb(253,231,37)
    #           );
    #       "></div>
    #       <div style="
    #           display:flex;
    #           justify-content:space-between;
    #           font-size:12px;
    #           margin-top:4px;
    #       ">
    #         <span>0</span><span>0.5</span><span>1</span>
    #       </div>
    #     </div>
    #     """,
    #     unsafe_allow_html=True)

    with st.expander("Statistiques"):
        st.write("N points:", len(grid_df))
        st.write(grid_df["p"].describe())

except Exception as e:
    st.error(f"Erreur: {type(e).__name__}: {e}")
    st.code(traceback.format_exc())
