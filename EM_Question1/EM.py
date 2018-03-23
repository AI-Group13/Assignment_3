import numpy as np
import math

# k = 3
# prob = np.random.rand(k, M)

pi = 22/7

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

    def norm_pdf_multivariate(self, data, mu, sigma):
        size = len(self._data)
        for i in range(self._num_clusters):
            for j in range(self._num_data_points):
                if size == len(mu) and (size, size) == sigma[i].shape:
                    det = np.linalg.det(sigma[i])
                    if det == 0:
                        raise NameError("The covariance matrix can't be singular")

                    norm_const = 1.0/(np.power((2*pi), float(size)/2) * np.power(det, 1.0/2))

                    x_mu = data[:, j] - mu[i]
                    ainv = np.linalg.inv(sigma[i])
                    result = np.power(math.e, -0.5 * (x_mu.T * ainv * x_mu))
                    result = norm_const * result

                else:
                    raise NameError("The dimensions of the input don't match")
        prob[:, j] =
        return result

    def maximization(self):
        pic = np.mean(self._prob, axis=1)

        for i in range(self._num_clusters):
            for j in range(self._num_data_points):
                muc = np.sum((self._prob[i, j] * self._data[:, j])) / self._num_data_points
                sigma = np.sum((self._prob[i, j] * (self._data[:, j] - muc[i]).T * (self._data[:, j] - muc[i])))

        return pic, muc, sigma


    def expectation(self, mu, prob, data, sigma, pic):
        result = norm_pdf_multivariate(self., mu, sigma, prob)
        prob = pic * prob
        prob = normalize(prob)




