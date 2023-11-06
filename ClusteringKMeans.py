import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

class ClusteringKMeans:
    def __init__(self, k: int, max_iteraciones: int = 100):
        self.k = k
        self.max_iteraciones = max_iteraciones
        self.centroids = None

    def distancia(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))

    def clasifica(self, datos: pd.DataFrame):
        self.centroids = datos.sample(n=self.k).values
        iteracion = 0

        while iteracion < self.max_iteraciones:
            clusters = {i: [] for i in range(self.k)}
            for _, row in datos.iterrows():
                distancia_minima = np.inf
                cluster_idx = None
                for i, centroid in enumerate(self.centroids):
                    d = self.distancia(row.values, centroid)
                    if d < distancia_minima:
                        distancia_minima = d
                        cluster_idx = i
                clusters[cluster_idx].append(row.values)

            nuevos_centroids = []
            for i in clusters:
                if clusters[i]:
                    nuevos_centroids.append(np.mean(clusters[i], axis=0))
                else:
                    nuevos_centroids.append(self.centroids[i])
            nuevos_centroids = np.array(nuevos_centroids)

            if np.all(self.centroids == nuevos_centroids):
                break
            self.centroids = nuevos_centroids

            iteracion += 1

        return clusters

class ClusteringKMeansSKLearn:
    def __init__(self, k: int, max_iteraciones: int = 100):
        self.k = k
        self.max_iteraciones = max_iteraciones
        self.kmeans = None

    def clasifica(self, datos: pd.DataFrame):
        # Inicializar y entrenar el modelo K-Means
        self.kmeans = KMeans(n_clusters=self.k, max_iter=self.max_iteraciones)
        self.kmeans.fit(datos)

        # Obtener las etiquetas de clúster para cada punto de datos
        labels = self.kmeans.labels_

        # Crear un diccionario para almacenar los puntos de datos en cada clúster
        clusters = {i: [] for i in range(self.k)}
        for i, label in enumerate(labels):
            clusters[label].append(datos.values[i])

        return clusters

if __name__ == '__main__':

    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt

    from Datos import Datos

    kmeans = ClusteringKMeans(k=3)

    """np.random.seed(42)
    data1 = np.random.randn(50, 2)
    data2 = np.random.randn(50, 2)
    data3 = np.random.randn(50, 2)

    data = np.concatenate([data1, data2, data3])
    df = pd.DataFrame(data, columns=["x", "y"])

    clusters = kmeans.clasifica(df)"""

    dataset = Datos('./datasets/iris.csv')
    clusters = kmeans.clasifica(dataset.datos.iloc[:,:-1])

    for i in clusters:
        print(i, "\n", clusters[i], "\n")

    colors = ['red', 'blue', 'green']
    for i, color in enumerate(colors):
        if clusters[i]:
            cluster_data = np.array(clusters[i])
            plt.scatter(cluster_data[:, 2], cluster_data[:, 3], c=color, s=50)

    centroid_data = np.array(kmeans.centroids)
    plt.scatter(centroid_data[:, 2], centroid_data[:, 3], c='black', s=200, marker='X')
    plt.show()