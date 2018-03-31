import argparse
from numpy import genfromtxt
import numpy as np
from matplotlib import pyplot as plt
import warnings
import sys

from expecMaximize import *

if not sys.warnoptions:
    warnings.simplefilter("ignore")

if __name__ == '__main__':

    parser = argparse.ArgumentParser('EM Calculations')

    parser.add_argument('data_file', type=str, help='The input file to read the EM points from')
    parser.add_argument('num_clusters', help='The number of Clusters to generate')

    parser.add_argument('-p', '--plot', action='store_true', help='Flag for showing generated scatterplot')

    args = vars(parser.parse_args())

    data = genfromtxt(args['data_file'], delimiter=',')
    print(data)
    show_plot = args['plot']

    #em = ExpectMaxmize(data, 5)
    #em.do_em_noBIC()

    if args['num_clusters'] == 'X':
        BIC_list = []
        i = 1
        plot_x = list(range(i, 20))

        best_mu = []
        best_var = []
        best_prob = []

        while True:
            em = ExpectMaxmize(data, i)
            em.do_em()
            BIC = em.BIC()
            BIC_list.append(BIC)
            best_mu.append(em.cache_mu[-1])
            best_var.append(em.cache_var[-1])
            best_prob.append(em.cache_probabilities[-1])

            if (len(BIC_list) > 2 and BIC_list[-2] < BIC_list[-1]):
                print(BIC_list)
                print("\n\nBest mean: ",best_prob[-2],"\n\n")
                print("Best Variance: ", np.asarray(best_var[-2]), "\n\n")
                print("Best Probabilities: ", best_prob[-2], "\n\n")

                plt.plot(plot_x[:i], BIC_list)
                plt.title("BIC value plot")
                plt.xlabel("No. of clusters")
                plt.ylabel("BIC value")
                plt.show()
                break
            print(i, "\n")
            i += 1

    else:


        em = ExpectMaxmize(data, args['num_clusters'])
        # Defining restart value
        LogLik_cache = []
        while em.restart_val < 6:
            print("Random Restart no: ",em.restart_val+1, "\n")
            em.do_em()
            LogLik_cache.append(em.LLHD_array[-1])
            em.restart_val += 1
        em.plot_performance()
        em.plot_scatter()
        Aggregate_logLik = np.sum(LogLik_cache) / len(LogLik_cache)

        print(" \n \nAverage Log likelihood value after multiple random restarts", Aggregate_logLik, "\n")