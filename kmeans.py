# kmeans.py
import numpy as np
import sklearn.datasets as datasets

class KMeans:
    def __init__(self, k, init_method="kmeans++"):
        self.data = None
        self.k = k
        self.init_method = init_method
        self.assignment = None
        self.centers = None
        self.converged = False

    def initialize(self):
        self.data, _ = datasets.make_blobs(n_samples=500, centers=self.k, cluster_std=1)
        self.assignment = [-1 for _ in range(len(self.data))]
        self.centers = None
        self.converged = False
        if self.init_method == "random":
            self.centers = self.random_init()
        elif self.init_method == "farthest":
            self.centers = self.farthest_first_init()
        elif self.init_method == "kmeans++":
            self.centers = self.kmeans_plus_plus_init()

    def random_init(self):
        return self.data[np.random.choice(len(self.data), size=self.k, replace=False)]

    def farthest_first_init(self):
        centers = [self.data[np.random.choice(len(self.data))]]
        for _ in range(1, self.k):
            distances = np.min(np.array([np.linalg.norm(self.data - center, axis=1) for center in centers]), axis=0)
            next_center = self.data[np.argmax(distances)]
            centers.append(next_center)
        return np.array(centers)

    def kmeans_plus_plus_init(self):
        centers = [self.data[np.random.choice(len(self.data))]]
        for _ in range(1, self.k):
            distances = np.min(np.array([np.linalg.norm(self.data - center, axis=1) for center in centers]), axis=0) ** 2
            probabilities = distances / np.sum(distances)
            new_center = self.data[np.random.choice(len(self.data), p=probabilities)]
            centers.append(new_center)
        return np.array(centers)

    def step(self):
        if self.converged:
            return
        self.assign_clusters()
        new_centers = self.compute_centers()
        if np.array_equal(self.centers, new_centers):
            self.converged = True
        self.centers = new_centers

    def assign_clusters(self):
        for i in range(len(self.data)):
            distances = [np.linalg.norm(self.data[i] - center) for center in self.centers]
            self.assignment[i] = np.argmin(distances)

    def compute_centers(self):
        return np.array([self.data[np.array(self.assignment) == i].mean(axis=0) if np.any(np.array(self.assignment) == i) else center for i, center in enumerate(self.centers)])

    def to_dict(self):
        return {
            'data': self.data.tolist(),
            'centers': self.centers.tolist(),
            'assignments': self.assignment,
            'converged': self.converged
        }
