import numpy as np
from sklearn.cluster import MeanShift
import matplotlib.pyplot as plt
#importer le style d'affichage
from matplotlib import style
style.use("ggplot")
#importer le dataset
from sklearn.datasets import make_blobs
#créer le datset: définir les centres initiaux et créer des clusters en fonction des centres définis
centers = [[3,3,3],[4,5,5],[3,10,10]]
X, _ = make_blobs(n_samples = 700, centers = centers, cluster_std = 0.5)
#affichage du dataset initial
plt.scatter(X[:,0],X[:,1])
plt.show()
#lancer l'apprentissage
ms = MeanShift()
ms.fit(X)
#Affichage des centres des clusters
labels = ms.labels_
cluster_centers = ms.cluster_centers_
print(cluster_centers)
#Affichage des clusters
n_clusters_ = len(np.unique(labels))
print("Estimated clusters:", n_clusters_)
colors = 10*['r.','g.','b.','c.','k.','y.','m.']
for i in range(len(X)):
	plt.plot(X[i][0], X[i][1], colors[labels[i]], markersize = 3)
	plt.scatter(cluster_centers[:,0],cluster_centers[:,1],marker=".",color='k', s=20, linewidths = 5, zorder=10)
plt.show()