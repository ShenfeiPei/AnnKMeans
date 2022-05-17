import time
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

from alias_copyi_AnnKMeans.Public import Funs, Mfuns
from alias_copyi_AnnKMeans.AKM import AKM

name = "Epileptic"
data = pd.read_csv(f"data/{name}.csv", header=None)
X = data.to_numpy()
X = X.astype(np.float64)

cen = pd.read_csv(f"data/{name}2.csv", header=None)
cen = cen.to_numpy()
c_true = cen.shape[0]
Cens = cen.reshape(1, cen.shape[0], cen.shape[1])

# AKM
mod = AKM(X, c_true, debug=True)
mod.opt(Cens, ITER=100)
y_pred = mod.y_pre[0]
n_iter = mod.n_iter_[0]
# times = mod.time_arr
cal_num_dist = mod.cal_num_dist[0]
dist_num_arr = mod.dist_num_arr[0]
print(y_pred[:10])
print(n_iter)
print(Mfuns.kmeans_obj(X, y_pred))
print(cal_num_dist)
print(dist_num_arr[:n_iter])
print(np.sum(dist_num_arr[:n_iter]))

# KMeans
mod = KMeans(n_clusters=c_true, init=Cens[0], n_init=1).fit(X)
n_iter = mod.n_iter_
y_pred = mod.labels_
print(y_pred[:10])
print(n_iter)
print(Mfuns.kmeans_obj(X, y_pred))
