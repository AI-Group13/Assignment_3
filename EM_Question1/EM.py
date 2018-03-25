import math

import matplotlib.pyplot as plt
import numpy as np

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
        # muc = 30 * np.random.rand(2, 3)
        # pic = np.random.rand(3, 1)
        # sigma = 100 * np.random.rand(3, 2, 2)
        while True:

            # self._old_prob = self._prob
            pic, muc, sigma = self.em_maximization()
            self.expectation(muc, sigma, pic)

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

    def norm_pdf_multivariate(self, mu, sigma):
        # for error referencing
        result = -99

        size = self._num_features
        for i in range(self._num_clusters):
            for j in range(self._num_data_points):
                if (size, self._num_clusters) == mu.shape and (size, size) == sigma[i, :, :].shape:
                    det = np.linalg.det(sigma[i, :, :])
                    if det == 0:
                        raise NameError("The covariance matrix can't be singular")

                    norm_const = 1.0 / (np.power((2 * pi), float(size) / 2) * np.power(det, 1.0 / 2))

                    x_mu = np.matrix(self._data[:, j] - mu[:, i])
                    ainv = np.matrix(np.linalg.inv(sigma[i, :, :]))
                    result = np.power(math.e, -0.5 * (x_mu * ainv * x_mu.T))
                    result = norm_const * result
                    self._prob[i, j] = result

                else:
                    raise NameError("The dimensions of the input don't match")

        return self._prob

    def em_maximization(self):
        pic = np.sum(self._prob, axis=1) / self._num_data_points
        print(pic)
        mu = np.zeros([self._num_features, self._num_clusters])
        sig = np.zeros([self._num_clusters, self._num_features, self._num_features])

        for i in range(self._num_clusters):
            for j in range(self._num_data_points):
                mu[:, i] += self._prob[i, j] * self._data[:, j]

        muc = mu / self._num_data_points

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
        self._prob = self.norm_pdf_multivariate(mu, sigma)

        for i in range(self._num_features):
            self._prob[i, :] *= pic[i]

        # self._prob = pic * self._prob
        self._prob = self.normalize(self._prob)
        print(self._prob)

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
