import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from kmeans import KMeans  # Assuming the KMeans class is imported from the kmeans.py

class KMeansVisualizer:
    def __init__(self, kmeans):
        self.kmeans = kmeans
        self.fig, self.ax = plt.subplots()
        self.scatter = None
        self.scatter_centroids = None
        self.step_button = None
        self.init_button = None
        self.manual_button = None

    def plot(self):
        self.ax.clear()
        # Plot data points, color by current cluster assignment
        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'orange', 'pink']
        assignments = self.kmeans.assignment
        if self.scatter:
            self.scatter.remove()
        self.scatter = self.ax.scatter(self.kmeans.data[:, 0], self.kmeans.data[:, 1], 
                                       c=[colors[a] for a in assignments], s=20, label="Data Points")

        # Plot centroids
        if self.kmeans.centers is not None:
            if self.scatter_centroids:
                self.scatter_centroids.remove()
            self.scatter_centroids = self.ax.scatter(self.kmeans.centers[:, 0], self.kmeans.centers[:, 1], 
                                                     c='black', marker='x', s=100, label="Centroids")
        self.ax.legend()
        self.ax.set_title('KMeans Clustering')
        plt.draw()

    def on_step(self, event):
        # Perform one step of KMeans and plot the result
        self.kmeans.step()
        self.plot()
        if self.kmeans.converged:
            print("KMeans has converged!")

    def on_init(self, event):
        # Reinitialize centroids and plot the result
        self.kmeans.initialize()
        self.plot()

    def on_manual_select(self, event):
        # Manually select centroids and re-assign clusters
        manual_centroids = np.array(plt.ginput(self.kmeans.k, timeout=-1))  # User clicks to select centroids
        self.kmeans.centers = manual_centroids
        self.kmeans.converged = False
        self.kmeans.step()
        self.plot()

    def setup_buttons(self):
        # Add buttons for step, initialize, and manual selection
        ax_step = plt.axes([0.7, 0.05, 0.1, 0.075])
        ax_init = plt.axes([0.5, 0.05, 0.1, 0.075])
        ax_manual = plt.axes([0.3, 0.05, 0.1, 0.075])

        self.step_button = Button(ax_step, 'Step')
        self.init_button = Button(ax_init, 'Initialize')
        self.manual_button = Button(ax_manual, 'Manual Select')

        self.step_button.on_clicked(self.on_step)
        self.init_button.on_clicked(self.on_init)
        self.manual_button.on_clicked(self.on_manual_select)

    def visualize(self):
        self.setup_buttons()
        self.plot()
        plt.show()

