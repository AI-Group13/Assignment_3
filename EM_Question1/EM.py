import numpy as np


# k = 3
# prob = np.random.rand(k, M)


class ExpectationMaximization:

    def __init__(self, data):
        self._data = data

        self._num_features = np.shape(self._data)[0]
        self._num_data_points = np.shape(self._data)[1]

        self._num_clusters = 0

        self._prob = 0

    def calculate_clusters(self, num_clusters=0):
        if num_clusters is not 0:
            self._prob = self.normalize(np.random.rand(num_clusters, self._num_data_points))

            shouldExit = False

            while not shouldExit:
                pass

    def normalize(self, prob):

        for column in range(self._num_data_points):
            norm = sum(prob[:, column])
            prob[:, column] /= norm

        return prob

    def meancov2prob(self, param, prob):

        pass

    def estimation(self, mu, prob):

        pass

    def maximization(self):
        pi = np.mean(prob, axis=1)
        muc = np.sum((prob * data), axis=1) / M
        cov = np.sum((prob * (data - muc).T * (data - muc)))

    # print(N)
