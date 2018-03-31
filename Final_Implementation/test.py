import csv
import numpy as np
import matplotlib.pyplot as plt
linalg = np.linalg

with open('new_data.csv','w') as new_file:
    csv_writer = csv.writer(new_file, delimiter=',')
    
    ds1 = np.random.multivariate_normal([1,1], [[0.3, 0.2],[0.2, 0.2]], 1000)
    ds2 = np.random.multivariate_normal([3,2], [[0.5, 0.1],[0.1, 0.2]],1000)

    for data in ds1:
        string = [str(data[0]) ,str(data[1])]
        csv_writer.writerow(string)

    for data in ds2:
        string = [str(data[0]) ,str(data[1])]
        csv_writer.writerow(string)

N = 1000
mean = [1,1]
cov = [[0.3, 0.2],[0.2, 0.2]]
data = np.random.multivariate_normal(mean, cov, N)
mean2 = [3,2]
cov2 = [[0.5, 0.1],[0.1, 0.2]]
data2 = np.random.multivariate_normal(mean2, cov2, N)
plt.scatter(data2[:,0], data2[:,1], c='green')    
plt.scatter(data[:,0], data[:,1], c='yellow')
plt.show()
