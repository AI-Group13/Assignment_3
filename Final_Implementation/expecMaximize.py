import numpy as np
import matplotlib.pyplot as plt
import random
import time
import matplotlib.cm as cm

class ExpectMaxmize():

    def __init__(self, data, num_clusters):

        self.given_data = data
        self.mu = []
        self.cov = None
        self.ClusResp = []
        self.ClusData = []
        self.DataClus = []
        self.numClus = int(num_clusters)
        self.numFea = None
        self.LLHD_array = []
        self.temp_LLHD = 1
        self.current_LLHD = 0
        self.tolerance = 0.01
        self.cache_mu = []
        self.cache_var = []
        self.cache_probabilities = []
        self.restart_val = 0

    def initMean_Cov_ClusResp(self):

        self.numFea = np.shape(self.given_data)[1]
        self.mu = []
        ran = [np.random.choice(self.given_data[:, i]) for i in range(self.given_data.shape[1])]
        for p in range(self.numClus):
            self.mu.append([ np.random.randint(0,np.abs(int(a))+5) for a in ran])
        self.mu = np.asarray(self.mu)
        self.first_mu = self.mu
        # Printing the initialized values for mean matrix of size: number of clusters X num of features

        # print ("Initialized Value of mean")
        # print ("Shape of Mean Matrix", np.shape(self.mu))
        # print (self.mu)

        # Printing the initialized values for the covariance matrix of size: number of clusters X num of features X num of features

        co_var = np.abs((np.random.choice(self.given_data[:,0]) + np.random.choice(self.given_data[:,1])) / 2)
        self.cov = np.asarray([co_var * np.identity(np.shape(self.given_data)[1]) for i in range(self.numClus)])
        self.first_mu = self.cov

        # print ("Initialized Value of the Covariance Matrix")
        # print ("Shape of Covariance Matrix", np.shape(self.cov))
        # print (self.cov)

        # Printing the initialized value for the Cluster Probability or Responsibility of size: num of clusters X 1

        self.ClusResp = (np.ones(self.numClus)/self.numClus)

        # print ("Initialized Value of Cluster Responsibility or Probability")
        # print (np.shape(self.ClusResp))
        # print (self.ClusResp)

        # print ("\n\n ------------------------------------------- \n\n")

    def calc_prob(self, meanMat, covariance, datapoints):

        # prob = (1/((2*np.pi*np.linalg.det(covariance))**0.5))*np.exp(-(datapoints - meanMat)**2/np.linalg.det(covariance))

        v1 	= np.linalg.solve(covariance,(datapoints - meanMat).T).T
        v2 	= (datapoints - meanMat)
        prob = np.exp(-0.5*np.sum(v2*v1,axis=1))/(((2*np.pi**datapoints.shape[1])*np.linalg.det(covariance))**0.5)

        # print ("\n\nCalculated Probability \n", prob, "\n\n")

        return prob

    # The Expectation Step Function

    def Expectation(self):

        # print ("Expectation Step\n\n")
        self.DataClus = []
        for ii in range(self.numClus):
            temp1 = self.calc_prob(self.mu[ii], self.cov[ii], self.given_data)
            self.DataClus.append(temp1)
        self.DataClus = np.asarray(self.DataClus)
        self.ClusData = self.DataClus.T*self.ClusResp
        self.ClusData = self.ClusData/np.sum(self.ClusData, axis=1)[:,np.newaxis]



    # The Maximization Step Function

    def Maximization(self):

        # print ("\n\nMaximization Step\n\n")

        normalized_ClusData = np.sum(self.ClusData, axis=0)
        # print(normalized_ClusData)

        # Updating the value of cluster probability or responsibility
        self.ClusResp = normalized_ClusData / np.shape(self.given_data)[0]
        # print ("Updated Cluster Probability \n", self.ClusResp, "\n\n" )

        # Updating the Value of means
        mean_num = np.dot(self.ClusData.T, self.given_data)
        self.mu = mean_num / normalized_ClusData[:, np.newaxis]
        # print ("Updated Mean Matrix \n", self.mu, "\n\n")

        # Updating the covariance matrix

        # # ttt = np.dot(((self.ClusData[:, 0].T*(self.given_data - self.mu[0, :]).T)), (self.given_data - self.mu[0, :]))
        # print ("temp", ttt)
        self.cov = [ np.dot(((self.ClusData[:, i].T*(self.given_data - self.mu[i, :]).T)), (self.given_data - self.mu[i, :]))/normalized_ClusData[i] for i in range(self.numClus)]
        # print("Updated Covariance Matrix \n", self.cov, "\n\n")
        self.cache_mu.append(self.mu)
        self.cache_var.append(self.cov)
        self.cache_probabilities.append(self.ClusResp)

    def calc_LLHD(self):

        # print ("Calculating the loglikelihood \n")
        temp = np.sum(np.log(np.sum((self.ClusResp)*(self.DataClus).T,axis=1)))
        return temp

    def plot_performance(self):

        plt.plot(self.LLHD_array)
        plt.xlabel("Number of iterations")
        plt.ylabel("Log likelihood Value")
        plt.show()

    def BIC(self):

        BIC_Calc = -2 * self.LLHD_array[-1] + ((self.given_data.shape[1] * (self.given_data.shape[1] + 1) / 2 +
                                            self.given_data.shape[1]) * self.numClus + self.numClus - 1) * np.log(self.given_data.shape[0])


        return BIC_Calc

    def plot_scatter(self):

        cluster_selection = np.argmax(self.ClusData, axis=1)
        colorsnn = cm.rainbow(np.linspace(0, 1, self.numClus))

        labels = ['Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster 4', 'Cluster 5', 'Cluster 6', 'Cluster 7',
                  'Cluster 8']

        for i in range(np.shape(self.given_data)[0]):
            for k in range(self.numClus):
                if cluster_selection[i] == k:
                    plt.scatter(self.given_data[i,0], self.given_data[i,1],color = colorsnn[k])

        plt.title("Cluster visualization (Each colors are individual clusters)")
        plt.xlabel("Attribute 1")
        plt.ylabel("Attribute 2")
        plt.show()


    def do_em(self):

        # Initializing the complete process
        self.initMean_Cov_ClusResp()

        LLHD_diff = np.abs(self.temp_LLHD - self.current_LLHD)

        #print ("Loglikelihood difference: ", LLHD_diff, "\n\n")

        iter_count = 0

        while (LLHD_diff > self.tolerance):
            #print ("Running the complete EM process \n\n")

            # Running the expectation step
            self.Expectation()

            # Running the maximization step
            self.Maximization()

            # Storing the previous value of Log likelihood in the temporary variable that checks the condition for convergence
            self.temp_LLHD = self.current_LLHD

            # Calculating the new loglikelihood and appending it to the array
            self.current_LLHD = self.calc_LLHD()
            print ("Current Log likelihood", self.current_LLHD, "\n")

            self.LLHD_array.append(self.current_LLHD)

            # Counting the number of iterations
            iter_count +=1
            #print ("Iteration Number: %d" %(iter_count), "\n")

            LLHD_diff = np.abs(self.temp_LLHD - self.current_LLHD)
            # print("Loglikelihood difference: ", LLHD_diff, "\n\n")

        print ("Final Values of the Cluster Centers are  \n")
        print("Final likelihood", self.current_LLHD, "\n\n")
        print ("Final Mean Matrix \n", self.mu, "\n\n")
        print ("Final CoVariance Matrix \n", self.cov, "\n\n" )

