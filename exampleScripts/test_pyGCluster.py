#!/usr/bin/env python

"""
Testscript to demonstrate functionality of pyGCluster

A synthetic dataset is used to check the correct installation of pyGCluster.
This dataset contains 10 ratios (Gene 0-9) which were randomly sampled between 
39.5 and 40.5 in 0.1 steps with a low standard deviation (randomly sampled 
between 0.1 and 1) and 90 ratios (Gene 10-99) which were randomly sampled 
between 3 and 7 in 0.1 steps with a high standard deviation (randomly sampled 
between 0.1 and 5)

5000 iterations are performed and the presence of the most frequent cluster is
checked.

This cluster should contain the Genes 0 to 9.

Usage::

    ./test_pyGCluster.py


When the iteration has finished (this should normally take not longer than 20
seconds), the script asks if you want to stop the iteration process or continue::

    iter_max reached. See convergence plot. Stopping re-sampling if not defined 
    otherwise ...
    ... plot of convergence finished. 
    See plot in "../exampleFiles/functionalityCheck/convergence_plot.pdf".

    Enter how many iterations you would like to continue. 
    (Has to be a multiple of iterstep = 5000)
    (enter "0" to stop resampling.)
    (enter "-1" to resample until iter_max (= 5000) is reached.)
    Enter a number ...

Please enter 0 and hit enter (The script will stop and the test will finish).

The results are saved into the folder functionalityCheck.

Additionally expression maps and expression profiles are plotted.

"""

from __future__ import print_function
import sys
import os
import multiprocessing
import pyGCluster # dependencies are NumPy, SciPy, optionally fastcluster and rpy2
import csv
import unittest

WORKING_DIRECTORY = 'functionalityCheck'
PICKLE_FILENAME   = 'pyGCluster_resampled.pkl'

class TestpyGCluster(unittest.TestCase):

    def setUp(self):
        self.testSet = set( [ 'Gene_{0}'.format(n) for n in range(10) ] )

    def test_syntheticData(self):
        data = {}
        for line in csv.DictReader(open('../exampleFiles/syntheticDataset.csv','r')):
            gene = line['Gene']
            if gene not in data.keys():
                data[gene] = {}
            for n in range(8):
                data[ gene ][ str(n) ] = ( float(line['{0}_mean'.format(n)]) , float(line['{0}_std'.format(n)])   )

        if not os.path.exists( WORKING_DIRECTORY ):
            os.mkdir( WORKING_DIRECTORY )
        else:
            print( '''
[ INFO ] Result folder already {0} exists.
[ INFO ] Old results might not be overwritten and lead to confusion ;)
[ INFO ] Check file creation date!'''.format( WORKING_DIRECTORY ) )



        # if os.path.exists( os.path.join( WORKING_DIRECTORY, PICKLE_FILENAME ) ):
        #     _ = input('Pickle already')
        # os.remove( )

        print( '[ INFO ] The results of the example script are saved into folder {0}.'.format( WORKING_DIRECTORY ) )

        TestCluster = pyGCluster.Cluster( data = data, working_directory = WORKING_DIRECTORY, verbosity_level = 2 )

        distance_metrices = [ 'euclidean' ]
        linkage_methods = [ 'complete' ]

        # cpus_2_use = 1
        # if multiprocessing.cpu_count() < cpus_2_use:
        #     cpus_2_use = multiprocessing.cpu_count()
        # print()
        TestCluster.resample(
                              distances = distance_metrices,
                              linkages = linkage_methods,
                              iter_max = 5000,
                              pickle_filename = PICKLE_FILENAME,
                              # cpus_2_use = cpus_2_use,
                              iter_till_the_end = True
        )

        mostfreq = TestCluster._get_most_frequent_clusters( top_X_clusters = 1)

        realIdsSet = set()
        for cluster in mostfreq:
            for index in cluster:
                realIdsSet.add( TestCluster[ 'Identifiers' ][ index ] )

        self.assertEqual( realIdsSet , self.testSet )

        TestCluster.build_nodemap( min_cluster_size = 4, threshold_4_the_lowest_max_freq = 0.01)
        TestCluster.draw_community_expression_maps( min_value_4_expression_map = -40, max_value_4_expression_map = 40 , color_gradient = 'Spectral')
        TestCluster.draw_expression_profiles( min_value_4_expression_map = -40, max_value_4_expression_map = 40 )

if __name__ == '__main__':
    #invoke the freeze_support funtion for windows based systems
    try:
        sys.getwindowsversion()
        multiprocessing.freeze_support()
    except:
        pass
    unittest.main()
