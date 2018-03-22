import matplotlib.pyplot as plt
import numpy as np
from numpy import genfromtxt

data = genfromtxt('data.csv', delimiter=',')
k = 3
data = data.T
N = np.shape(data)[0]
M = np.shape(data)[1]
prob = np.random.rand(k,M)

def init():
	for i in range(M):
		norm = sum(prob[:,i])
		prob[:,i] = prob[:,i]/norm

def meancov2prob(param, prob):
	
def estimation(param, prob):
	meancov2prob()
	init()

def maximization():
	pi = np.mean(prob,axis=1)
	muc = np.sum((prob*data), axis=1)/M
	cov = np.sum((prob*(data-muc).T * (data - muc)))


# print(N)

