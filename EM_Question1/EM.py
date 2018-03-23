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

    def normalize(self, prob):
        for column in range(self._num_data_points):
            norm = sum(prob[:, column])
            prob[:, column] /= norm
        return prob

    def calculate_clusters(self, num_clusters=0):
        if num_clusters is not 0:
            self._prob = self.normalize(np.random.rand(num_clusters, self._num_data_points))

            shouldExit = False

            while not shouldExit:
                pass

    def norm_pdf_multivariate(self, k, j, data, mu, sigma):
        size = len(self._data)
        if size == len(mu) and (size, size) == sigma[k].shape:
            det = np.linalg.det(sigma[k])
            if det == 0:
                raise NameError("The covariance matrix can't be singular")

            norm_const = 1.0/(np.power((2*pi), float(size)/2) * np.power(det, 1.0/2))

            x_mu = data[:, j] - mu[k]
            ainv = np.linalg.inv(sigma[k])
            result = np.power(math.e, -0.5 * (x_mu * ainv * x_mu.T))
            result = norm_const * result
            return result

        else:
            raise NameError("The dimensions of the input don't match")

    def maximization(self):
        pic = np.mean(self._prob, axis=1)

        for i in range(self._num_clusters):
            for j in range(self._num_data_points):
                muc = np.sum((self._prob[i, j] * self._data[:, j])) / self._num_data_points
                sigma = np.sum((self._prob[i, j] * (self._data[:, j] - muc[i]).T * (self._data[:, j] - muc[i])))

        return pic, muc, sigma


    def expectation(self, mu, prob, data, sigma, pic):
        for a in range(self._num_clusters):
            for b in range(self._num_data_points):
                result = norm_pdf_multivariate(a, b, data, mu, sigma)
                prob[a, b] = result

        prob = pic * prob
        prob = normalize(prob)




