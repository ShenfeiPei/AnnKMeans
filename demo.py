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
Y = mod.y_pre
n_iter = mod.n_iter_
times = mod.time_arr
cal_num_dist = mod.cal_num_dist

print(n_iter)
print(cal_num_dist)
print(times)

# KMeans
t_start = time.time()
mod = KMeans(n_clusters=c_true, init=Cens[0]).fit(X)
t_end = time.time()
n_iter = mod.n_iter_
print(t_end - t_start)