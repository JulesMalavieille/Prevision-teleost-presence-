"""
Created on Fri Dec 12 18:07:44 2025

@author: Jules Malavieille
"""

#############
#Analyse stat
#############

import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from sklearn import preprocessing
from sklearn.cluster import KMeans
from matplotlib.lines import Line2D
from scipy.stats import chi2
from tqdm import trange


def generate_pseudo_absences_gauss(
    muZ, sigZ, invsigZ, chi_en,
    A, muX, sigX,
    xmin, xmax,
    n_pa, batch=5000, max_iter=200):
    
    kept = []
    total = 0

    for _ in trange(max_iter):
        Z_sim = np.random.multivariate_normal(muZ, sigZ, size=batch)
        D2_sim = mahalanobis_sq(Z_sim, muZ, invsigZ)

        # hors enveloppe (pseudo-absence)
        Z_out = Z_sim[D2_sim > chi_en]

        # inverse ACP -> espace variables
        X_out = ACP_inv(Z_out, A, muX, sigX)

        # filtre bornes par variable
        ok = np.all((X_out >= xmin) & (X_out <= xmax), axis=1)
        X_ok = X_out[ok]

        kept.append(X_ok)
        total += X_ok.shape[0]
        if total >= n_pa:
            break

    if total < n_pa:
        raise RuntimeError(f"Pas assez de pseudo-absences valides: {total}/{n_pa}. Augmente batch/max_iter ou détends les bornes.")

    return np.vstack(kept)[:n_pa]


def generate_pseudo_absences_gauss(
    muZ, sigZ, invsigZ, chi_en,
    A, muX, sigX,
    xmin, xmax,
    n_pa, batch=5000, max_iter=200):
    
    kept = []
    total = 0

    for _ in trange(max_iter):
        Z_sim = np.random.multivariate_normal(muZ, sigZ, size=batch)
        D2_sim = mahalanobis_sq(Z_sim, muZ, invsigZ)

        # hors enveloppe (pseudo-absence)
        Z_out = Z_sim[D2_sim > chi_en]
        

        # inverse ACP -> espace variables
        X_out = ACP_inv(Z_out, A, muX, sigX)

        # filtre bornes par variable
        ok = np.all((X_out >= xmin) & (X_out <= xmax), axis=1)
        X_ok = X_out[ok]

        kept.append(X_ok)
        total += X_ok.shape[0]
        if total >= n_pa:
            break

    if total < n_pa:
        raise RuntimeError(f"Pas assez de pseudo-absences valides: {total}/{n_pa}. Augmente batch/max_iter ou détends les bornes.")

    return np.vstack(kept)[:n_pa]


def mahalanobis_sq(X, mu, Sigma_inv):
    Xc = X - mu
    return np.einsum("ij,jk,ik->i", Xc, Sigma_inv, Xc)


def ACP(X, p, name_var, figure=True):  

    def plot_circle(r):
       theta = np.linspace(0, 2*np.pi, 100)
       x = r*np.cos(theta)
       y = r*np.sin(theta)
       return x, y
   
    n = X.shape[0]
    ind_moy = 1/n * np.dot(X.T, np.ones(n))
    X_c = X - ind_moy
    V = 1/n * np.dot(X_c.T, X_c)
    
    M = np.zeros([V.shape[0], V.shape[1]])
    for i in range(V.shape[0]):
        for j in range(V.shape[1]):
            if i == j:
                M[i,j] = 1/ V[i,j]
    
    X_cr = np.dot(X_c, M**(1/2))
    R = 1/n * np.dot(X_cr.T, X_cr)
    val, vec = np.linalg.eigh(R)
    
    idx = np.argsort(val)[::-1]   
    val = val[idx]
    vec = vec[:, idx]

    Z = np.dot(X_cr, vec[:,:p])
    A = vec[:,:p]
    
    if figure==True:
    
        """Visualisation de la spectralisation des variables"""
        tot_val = np.sum(val)
        hist = val/tot_val
        axe = np.arange(0, p, 1)
        
        plt.figure(1)
        plt.bar(axe, hist)
        plt.xlabel(axe)
        
        """Projection des composantes principales sur le cercle de corrélation"""
        plt.figure(2)
        x, y = plot_circle(1)
        for i in range(vec.shape[0]):
            a = vec[i, 0]*np.sqrt(val[0])
            b = vec[i, 1]*np.sqrt(val[1])
            plt.quiver(0, 0, a, b, angles="xy", scale_units="xy", scale=1)
            plt.text(a, b, name_var[i], color='black', fontsize=12)
    
        plt.plot(x, y)
        plt.plot([-1, 1], [0, 0], color="black")
        plt.plot([0, 0], [-1, 1], color="black")
        plt.xlabel("PC1 ("+str(round(hist[0]*100, 2))+"%)")
        plt.ylabel("PC2 ("+str(round(hist[1]*100, 2))+"%)")
        plt.title("ACP cercle de corrélation des variables")
        
        """Projection des individus sur l'espace des 2 premières composantes principales""" 
        plt.figure(3)
        plt.plot(Z[:,0], Z[:,1], ".")
        plt.plot([-100, 100], [0, 0], color="black")
        plt.plot([0, 0], [-100, 100], color="black")
        plt.xlim(-5, 9)
        plt.ylim(-6, 6)
        plt.xlabel("PC1 ("+str(round(hist[0]*100, 2))+"%)")
        plt.ylabel("PC2 ("+str(round(hist[1]*100, 2))+"%)")
        plt.title("ACP individus contaminant étang de berre")
    
    return A, Z


def ACP_inv(Z, A, muX, sigX):
    X_std = Z @ A.T
    X = X_std * sigX + muX
    return X 
    


data = pd.read_csv("Data/Data_occurences_speclean.csv", sep=",")
data = data.drop(columns=["Unnamed: 0"])

data["Bathy"] = np.abs(data["Bathy"])
data["Date"] = pd.to_datetime(data["Date"])
data["Jour"] = data["Date"].dt.dayofyear
data["Jour_sin"] = np.sin(2 * np.pi * data["Jour"] / 365)
data["Jour_cos"] = np.cos(2 * np.pi * data["Jour"] / 365)
data = data.drop(columns=["Jour"])

tetha = np.deg2rad(data["Current_dir"])
data["Dir_sin"] = np.sin(tetha)
data["Dir_cos"] = np.cos(tetha)

pre_ACP = data.drop(columns=["Date", "Lat", "Long", "Current_dir", "Species"])

p = pre_ACP.shape[1]
var = ["Temp", "Sal", "Bathy", "Waves", "Current_speed", "Thermo", "Prim_prod", "O2_conc", "Jour_sin", "Jour_cos", "Dir_sin", "Dir_cos"]
var_ACP, ind_ACP = ACP(pre_ACP, p, var, figure=False)

muX = pre_ACP.mean(axis=0)
sigX = pre_ACP.std(axis=0, ddof=0)
muX = np.asarray(muX, dtype=float).reshape(1, -1)   # (1,p)
sigX = np.asarray(sigX, dtype=float).reshape(1, -1) # (1,p)



################################################
# Plot par espèce sur carte plus récupération Di
################################################
n = data["Species"].unique()
N = pd.DataFrame(columns=["Espèce", "Nombre", "Nombre corrigé", "Di"])
val = []
for i in range(len(n)):
    d = data[data["Species"] == n[i]]
    val.append(len(d))
    
    # fig = plt.figure(figsize=(13, 9))
    # ax = plt.axes(projection=ccrs.PlateCarree())
    # ax.add_feature(cfeature.LAND, facecolor="lightgrey")
    # ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
    
    # ax.scatter(d["Long"], d["Lat"], s=2, alpha=0.35, transform=ccrs.PlateCarree())
    # ax.set_extent([-12, 14, 35, 52], crs=ccrs.PlateCarree())
    # ax.set_title(n[i])
    # plt.show()

N["Espèce"] = n
N["Nombre"] = val

Ni = []
Di = []
pech = [0.578, 0.874, 0.635, 0.943, 0.411, 0.475, 0.085, 0.06, 0.039, 0, 0.326, 0, 0, 0, 0, 0.071, 0, 0, 0, 0, 0, 0, 0, 0.441]
w = 0.2
for i in range(len(n)):
    N_corr = w*(pech[i]*N["Nombre"].iloc[i]) + N["Nombre"].iloc[i]*(1-pech[i])
    di = np.log(N_corr)/np.log(18608)
    Ni.append(N_corr)
    Di.append(di)
    
N["Nombre corrigé"] = Ni
N["Di"] = Di

data["Species"] = (
    data["Species"]
    .astype(str)
    .str.strip()
    .str.replace(r"[(),]", " ", regex=True)
    .str.split()
    .str[:2]
    .str.join(" "))
###############################
# Plot ACP espèce par espèce 2D
###############################
# n = data["Species"].unique()
# plt.figure()
# for sp in n:
#     idx = data["Species"] == sp
#     X = ind_ACP[idx]
#     plt.scatter(X[:,0], X[:,2], s=2, label=sp)
#     plt.legend(markerscale=3, fontsize=6)
#     plt.pause(2)

# plt.xlabel("PC1")
# plt.xlabel("PC2")


###############################
# Plot ACP espèce par espèce 3D
###############################
# n = data["Species"].unique()
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# for sp in n:
#     idx = data["Species"] == sp
#     X = ind_ACP[idx]
#     ax.scatter(X[::20,0], X[::20,1], X[::20,2], s=2, label=sp)
#     ax.legend(markerscale=3, fontsize=6)
#     plt.pause(2)

# ax.set_xlabel("PC1")
# ax.set_ylabel("PC2")
# ax.set_zlabel("PC3")


##############################
# Etude clustering par K-means
##############################
# X = pre_ACP.copy()
# X_scaled = preprocessing.scale(X)

# # wcss = []
# # nb_cluster = range(1, 10)
# # for i in range(1,10):
# #     kmeans = KMeans(i)
# #     kmeans.fit(X_scaled)
# #     wcss.append(kmeans.inertia_)

# # plt.figure()
# # plt.plot(nb_cluster, wcss)
# # plt.xlabel("Nombre de cluster")
# # plt.ylabel("Inertie intraclasse")
# # plt.title("Elbow-method")
# # Elbow method = 4 classes 

# kmeans = KMeans(4, random_state=42)
# kmeans.fit(X_scaled)

# cluster = kmeans.fit_predict(X_scaled)
# cluster = cluster.reshape(-1,1)

# clusters = np.concatenate((ind_ACP, cluster), axis=1)
# fig = plt.figure()
# ax = fig.add_subplot(projection="3d")
# ax.scatter(clusters[::20,0], clusters[::20,1], clusters[::20,2], c=clusters[::20,-1], cmap="rainbow")
# ax.set_xlabel("Axe 1")
# ax.set_ylabel("Axe 2")
# ax.set_zlabel("Axe 3")
# ax.set_title("Kmeans représenté sur le premier cube ACP")

# #######################################
# # Plot par espèce sur carte avec classe 
# #######################################
# cluster_labels = {
#     0: "Mixte - peu de courant",
#     1: "Côtier - peu de courant",
#     2: "Mixte - courant Est fort",
#     3: "Hauturier – courant N fort"}

# cluster_colors = {
#     0: "violet",
#     1: "blue",
#     2: "green",
#     3: "red"}

# n = data["Species"].unique()
# data["Cluster"] = cluster
# for i in range(len(n)):
#     d = data[data["Species"] == n[i]].copy()
#     d["color"] = d["Cluster"].map(cluster_colors)
    
#     fig = plt.figure(figsize=(13, 9))
#     ax = plt.axes(projection=ccrs.PlateCarree())
#     ax.add_feature(cfeature.LAND, facecolor="lightgrey")
#     ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
    
#     ax.scatter(
#         d["Long"], d["Lat"],
#         c=d["color"],
#         s=2, alpha=0.35,
#         transform=ccrs.PlateCarree())

#     present_clusters = sorted(d["Cluster"].unique())
#     handles = [Line2D([0], [0], marker='o', color='w',
#                 markerfacecolor=cluster_colors[k], markersize=6,
#                 label=cluster_labels[k]) for k in cluster_labels]
#     ax.legend(handles=handles, fontsize=7, loc="lower left", frameon=True)

#     ax.set_extent([-12, 14, 35, 52], crs=ccrs.PlateCarree())
#     ax.set_title(n[i])
#     plt.show()


bounds = {
  "Temp": (0, 35),
  "Sal": (0, 42),
  "Bathy": (0, 5000),        
  "Waves": (0, 20),
  "Current_speed": (0, 1),
  "Thermo": (0, 500),
  "Prim_prod": (0, 500),
  "O2_conc":(100, 400),       
  "Jour_sin": (-1, 1),
  "Jour_cos": (-1, 1),
  "Dir_sin": (-1, 1),
  "Dir_cos": (-1, 1),}

xmin = np.array([bounds[v][0] for v in var], dtype=float)
xmax = np.array([bounds[v][1] for v in var], dtype=float)

#################################
# Génération des pseudos-absences
#################################
# species = data["Species"].to_numpy()
# sp = np.unique(species)
# for specie in sp:
#     mask = (species == specie)
#     Z = ind_ACP[mask, :]
    
#     p = Z.shape[1]
#     muZ = np.mean(Z, axis=0)
#     sigZ = np.cov(Z, rowvar=False)
#     sigZ += 1e-8 * np.eye(p)
#     sig_inv = np.linalg.inv(sigZ)
    
#     chi_en = chi2.ppf(0.75, df=p)
    
#     n_pa = Z.shape[0]  
    
#     X_pa = generate_pseudo_absences_gauss(
#         muZ=muZ, sigZ=sigZ, invsigZ=sig_inv, chi_en=chi_en,
#         A=var_ACP[:,:p], muX=muX, sigX=sigX,
#         xmin=xmin, xmax=xmax,
#         n_pa=n_pa, batch=20*n_pa)

#     # X_pres : variables réelles des présences (directement depuis data)
#     X_pres = pre_ACP.loc[mask, var].to_numpy().astype(float)
    
#     X_final = np.vstack([X_pres, X_pa])
#     y_final = np.hstack([np.ones(X_pres.shape[0]), np.zeros(X_pa.shape[0])]).reshape(-1,1)
    
#     df_sp = pd.DataFrame(X_final, columns=var)
#     df_sp["Presence"] = y_final
    
    
#     df_sp.to_csv("Especes/data_pres_abs_" + specie +".csv")
    
    
###########################################################
# Test de fonctionnement de la génération de pseudo-absence
###########################################################
# species = data["Species"].to_numpy()
# sp = np.unique(species)
# for specie in sp:
#     mask = (species == specie)
#     Z = ind_ACP[mask, :2]
    
#     p = Z.shape[1]
#     muZ = np.mean(Z, axis=0)
#     sigZ = np.cov(Z, rowvar=False)
#     sigZ += 1e-8 * np.eye(p)
#     sig_inv = np.linalg.inv(sigZ)
    
#     chi_en = chi2.ppf(0.75, df=p)
    
#     D2 = mahalanobis_sq(Z, muZ, sig_inv)
#     inside = D2 <= chi_en
#     Z_seuil = Z[inside]
    
#     n_pa = Z.shape[0]  
    
#     # Échantillonnage gaussien large
#     Z_sim = np.random.multivariate_normal(muZ, sigZ, size=20 * n_pa)
    
#     # Distance de Mahalanobis
#     D2_sim = mahalanobis_sq(Z_sim, muZ, sig_inv)
    
#     # Pseudo-absences = hors enveloppe
#     Z_pa = Z_sim[D2_sim > chi_en][:n_pa]

#     plt.figure()
#     plt.scatter(Z[::10,0], Z[::10,1], s=2, color="blue", label="Présence")
#     plt.scatter(Z_pa[::10,0], Z_pa[::10,1], s=2, color="red", label="Absence")
#     plt.legend()
    






















    
    
    