import numpy as np
import pandas as pd

class KMeans:
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

            #TODO: pillar una comprobacion que no sea tan exacta
            if np.all(self.centroids == nuevos_centroids):
                break
            self.centroids = nuevos_centroids

            iteracion += 1

        return clusters
    

if __name__ == '__main__':

    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt

    np.random.seed(42)
    data1 = np.random.randn(50, 2)
    data2 = np.random.randn(50, 2) + [5, 5]
    data3 = np.random.randn(50, 2) + [10, -5]

    data = np.concatenate([data1, data2, data3])
    df = pd.DataFrame(data, columns=["x", "y"])

    kmeans = KMeans(k=3)
    clusters = kmeans.clasifica(df)

    colors = ['red', 'blue', 'green']
    for i, color in enumerate(colors):
        if clusters[i]:
            cluster_data = np.array(clusters[i])
            plt.scatter(cluster_data[:, 0], cluster_data[:, 1], c=color, s=50)

    centroid_data = np.array(kmeans.centroids)
    plt.scatter(centroid_data[:, 0], centroid_data[:, 1], c='black', s=200, marker='X')
    plt.show()