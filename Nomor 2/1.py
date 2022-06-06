import pandas as pd
import numpy as np
import matplotlib.pyplot as mp

from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

# membaca file csv
data = pd.read_csv("gempa.csv")

# memilih atribut yang diperlukan
data_x = data.drop(["Cluster", "Nomor_KK", "Nama_KK", "Alamat_Asli"], axis = 1)
# print(data_x)

# merubah data di file menjadi array
x_array = np.array(data_x)
# print(x_array)

# proses normalisasi
scaler = MinMaxScaler()
x_scaled = scaler.fit_transform(x_array)
# print(x_scaled)

a = int(input("Masukkan Jumlah Cluster yang anda inginkan:"))

# memasukkan array ke algoritma k-means
kmeans = KMeans(n_clusters=a, random_state=123)
kmeans.fit(x_scaled)

# menampilkan data cluster di console
for i in range(0,19):
    print("Nomor KK:"+str(data.values[i,0])+", Nama KK:"+str(data.values[i,1])+", Alamat:"+str(data.values[i,4])+", Cluster:"+str(kmeans.labels_[i]))
    print("-----------------------------------------------------------")

# menampilkan visualisasi datanya
data["kluster"] = kmeans.labels_
output = mp.scatter(x_scaled[:,0], x_scaled[:,1], s = 100, c = data.kluster, marker = "o", alpha = 1, )
centers = kmeans.cluster_centers_
mp.scatter(centers[:,0], centers[:,1], c='red', s=200, alpha=1, marker="s")
mp.title("Hasil Clustering K-Means")
mp.colorbar(output)
mp.show()
