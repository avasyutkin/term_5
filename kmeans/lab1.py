"""
from sklearn.datasets.samples_generator import make_blobs
import matplotlib.pyplot as plt
X, y = make_blobs(n_samples=1000, n_features=2, centers=[(20,20), (4,4)], cluster_std=2.0)
print(X.shape)
plt.figure()
plt.scatter(X[:, 0], X[:, 1], s=10, cmap='rainbow') #c=y
plt.title("Data")
plt.xlabel("x1")
plt.ylabel("x2")
plt.show()

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=5, init='k-means++')
y_pred = kmeans.fit_predict(X)
print(kmeans.cluster_centers_)
print(kmeans.labels_)

plt.figure()
plt.scatter(X[:, 0], X[:, 1], s=10, cmap='rainbow', c=kmeans.labels_) #c=y
plt.title("Data")
plt.xlabel("x1")
plt.ylabel("x2")
plt.show()

import numpy as np
X_new = np.array([[0,0], [10,10], [0,10], [10,0]])
print(kmeans.predict(X_new))
print(kmeans.transform(X_new))
##надо масштабировать
print(kmeans.score(X))
print(kmeans.inertia_)
kmeans = KMeans(n_clusters=5, init='k-means++', n_init=20)
y_pred = kmeans.fit_predict(X)
print(kmeans.score(X))
print(kmeans.inertia_)

from mlxtend.plotting import plot_decision_regions
centers = kmeans.cluster_centers_
plot_decision_regions(X, y_pred, clf=kmeans)
plt.scatter(centers[:, 0], centers[:, 1], c='white', edgecolor="black", s=20)
plt.title("Clusters")
plt.xlabel('x1')
plt.ylabel('x2')
plt.legend(loc='upper left')
plt.show()

from scipy.spatial import Voronoi, voronoi_plot_2d
vrn = Voronoi(kmeans.cluster_centers_)
voronoi_plot_2d(vrn)
plt.scatter(X[:, 0], X[:, 1], s=10, c=y_pred)
plt.title("Clusters")
plt.xlabel('x1')
plt.ylabel('x2')
plt.show()
"""



import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import pandas as pd
from sklearn import preprocessing

data = pd.read_csv("colors_1.csv")
kmeans = KMeans(n_clusters=15)
predicted_class = kmeans.fit_predict(data)
print(predicted_class)

data = data.assign(label=predicted_class)
data1 = preprocessing.scale(data)

print(kmeans.cluster_centers_)

with open('color.html', 'w') as colors:
    colors.write(f'''
    <body>
    <table>''')
    i = 0
    for rows in kmeans.cluster_centers_:
        colors.write('<tr>')
        colors.write(f'<td style="background-color: RGB({int(kmeans.cluster_centers_[i][0])}, {int(kmeans.cluster_centers_[i][1])}, {int(kmeans.cluster_centers_[i][2])})">{"сред"}</td>')
        for index, col in data[data['label'] == i].iterrows():
            colors.write(f'<td style="background-color: RGB({col["r"]}, {col["g"]}, {col["b"]})">{index}</td>')
        colors.write('</tr>')
        i = i + 1
    colors.write(f'''</table>
    </body>''')

