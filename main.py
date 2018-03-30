import argparse
from numpy import genfromtxt

from expecMaximize import *

if __name__ == '__main__':

    parser = argparse.ArgumentParser('EM Calculations')

    parser.add_argument('data_file', type=str, help='The input file to read the EM points from')
    parser.add_argument('num_clusters', help='The number of Clusters to generate')

    parser.add_argument('-p', '--plot', action='store_true', help='Flag for showing generated scatterplot')

    args = vars(parser.parse_args())

    data = genfromtxt(args['data_file'], delimiter=',')
    print(data)
    show_plot = args['plot']

    em = ExpectMaxmize(data, args['num_clusters'])
    em.initMean_Cov()

