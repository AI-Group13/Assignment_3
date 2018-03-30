import numpy as np
import matplotlib.pyplot as plt
import random
import time


class ExpectMaxmize():


    def __init__(self, data, num_clusters):

        self.given_data = data
        self.mu = None
        self.cov = None
        self.clusResp = None
        self.ClusData = None
        self.DataClus = None
        self.numClus = num_clusters
        self.numFea = None


    def initMean_Cov(self):

        print (np.amax(self.given_data.shape[0]) - np.amin(self.given_data.shape[0]))
        # ran1 = [ np.amax(self.given_data.shape[i]) - np.amin(self.given_data.shape[i]) for i in range(np.shape(self.given_data[1])) ]
        # print (ran1)
        self.mu =0


        self.cov = 0

        self.given_data = 5