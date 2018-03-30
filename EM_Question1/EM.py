import math
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
import numpy as np

from sklearn import datasets

# k = 3
# prob = np.random.rand(k, M)

pi = 22 / 7

colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']


class ExpectationMaximization:

    def __init__(self, data, num_clusters, show_plots):
        self._data = data
        self._which_cluster = []
        self._show_plots = show_plots
        self._num_features = np.shape(self._data)[0]
        self._num_data_points = np.shape(self._data)[1]
        self._num_clusters = int(num_clusters)
        self._prob = np.array([[0, 0], [0, 0]])
        self._old_prob = self._prob

    def do_em(self):
        self._prob = self.normalize(np.random.rand(self._num_clusters, self._num_data_points))
        muc = 10 * np.random.rand(self._num_features, self._num_clusters)
        print(muc)
        pic = np.random.rand(self._num_clusters, 1)
        sum_pic = np.sum(pic)
        for i in range(len(pic)):
            pic[i] = pic[i]/sum_pic
        print("THis is pic", pic)
        sigma = np.random.rand(self._num_clusters, self._num_features, self._num_features)
        print(sigma)
        for i in range(self._num_clusters):
            sigma[i, :, :] = 100*datasets.make_spd_matrix(self._num_features)

        while True:

            self._prob = self.expectation(muc, sigma, pic)
            pic, muc, sigma = self.maximization()
            self._old_prob = self._prob

            for b in range(self._num_clusters):
                for a in range(self._num_data_points):
                    if np.abs(self._prob[b, a] - self._old_prob[b, a]) < 0.000001:
                        flag = 1

            if flag == 1:
                print("Values converged")
                print(self._prob)
                break

            if self.prob_fitness_calc() > 50:
                if self._show_plots:
                    self.hard_cluster()
                    self.show_em_plots()
                break

    def normalize(self, matrix):
        for column in range(self._num_data_points):
            norm = sum(matrix[:, column])
            matrix[:, column] /= norm
        return matrix

    def log_liklihood(self):
        log_liklihood = np.log(np.sum((self._prob * self._pic), axis=0))
        log_liklihood = np.sum(log_liklihood)
        print(log_liklihood)

    def maximization(self):
        pic = np.sum(self._prob, axis=1) / self._num_data_points
        sum_pic = np.sum(pic)
        for i in range(len(pic)):
            pic[i] = pic[i]/sum_pic
        print("Pic", pic)
        mu = np.zeros([self._num_features, self._num_clusters])
        sig = np.zeros([self._num_clusters, self._num_features, self._num_features])

        for i in range(self._num_clusters):
            for j in range(self._num_data_points):
                mu[:, i] += self._prob[i, j] * self._data[:, j]

        muc = mu / self._num_data_points
        print("MUC", muc)

        for i in range(self._num_clusters):
            for j in range(self._num_data_points):
                A = np.matrix(self._data[:, j] - muc[:, i])
                val = (A.T * self._prob[i, j] * A)
                sig[i, :, :] += val

        sigma = sig / self._num_data_points

        for mat in sigma:
            if np.linalg.det(mat) == 0:
                pass

        return pic, muc, sigma

    def expectation(self, mu, sigma, pic):
        for i in range(self._num_clusters):
            for j in range(self._num_data_points):
                var = multivariate_normal(mu[:, i], sigma[i, :, :])
                self._prob[i, j] = var.pdf(self._data[:, j])

        for i in range(self._num_clusters):
            self._prob[i, :] *= pic[i]

        # self._prob = pic * self._prob
        self._prob = self.normalize(self._prob)
        print(self._prob)
        return self._prob

    def hard_cluster(self):
        self._which_cluster = np.amax(self._prob, axis=1)

    def show_em_plots(self):
        list_of_points = [[] for _ in range(self._num_clusters)]

        for points, which in zip(self._data.T, self._which_cluster):
            list_of_points[which].append(points)

        fig = plt.figure()

        ax = fig.add_subplot(111)

        cluster_index = 0

        for cluster in list_of_points:
            name = 'Cluster ' + cluster_index
            ax.scatter(x=cluster[0], y=cluster[1], c=colors[cluster_index], label=name)
            cluster_index += 1

        plt.legend()
        plt.show()

    def prob_fitness_calc(self):
        prob_diff = np.abs(self._old_prob - self._prob)

        return prob_diff.mean()
