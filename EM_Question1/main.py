import argparse

from numpy import genfromtxt

if __name__ == '__main__':
    parser = argparse.ArgumentParser('EM Calculations')
    parser.add_argument('data_file', metavar='FILE', type=str, help='The input file to read the EM points from')
    parser.add_argument('num_clusters', metavar='CLUSTERS', help='The number of Clusters to generate')

    args = vars(parser.parse_args())

    data = genfromtxt(args['data_file'], delimiter=',').T

    if args['num_clusters'] == 'X':
        # do EM Question 2
        pass
    else:
        # do EM Question 1
        pass
