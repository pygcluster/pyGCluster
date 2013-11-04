#!/usr/bin/env python

"""
Testscript to demonstrate functionality of pyGCluster

This script imports the data of Hoehner et al. (2013) and executes pyGCluster 
with 250,000 iterations of resampling. pyGCluster will evoke 4 threads 
(if possible), which each require approx. 1.5GB RAM. Please make sure you have 
enough RAM available (4 threads in all require approx. 6GB RAM).
Duration will be approx. 2 hours to complete 250,000 iterations on 4 threads.

Usage::

  ./basicClusterHoehnerExampleData.py <pathToExampleFile>


If this script is executed in folder pyGCluster/exampleScripts, the command would be::

  ./basicClusterHoehnerExampleData.py ../exampleFiles/hoehner_dataset.csv

The results are saved in ".../pyGCluster/exampleScripts/hoehner_example_run/".

"""


from __future__ import print_function
import sys
import os
import csv
import multiprocessing
import pyGCluster # dependencies are NumPy, SciPy, optionally fastcluster and rpy2


def main():
  pyGCluster_dir = os.path.split( sys.argv[ 0 ] )[ 0 ]
  ## parse data
  data = dict()
  with open( sys.argv[ 1 ] ) as fin:
    reader = csv.DictReader( fin, delimiter = ',' )
    conditions = set()
    for row in reader:
      if not conditions:
        conditions = set( [ _.split( '__' )[ 0 ] for _ in row.keys() ] ) - set( [ 'identifier' ] )
      data[ row[ 'identifier' ] ] = dict()
      for condition in conditions:
        mean = float( row[ '{0}__MEAN'.format( condition ) ] )
        std = float( row[ '{0}__STD'.format( condition ) ] )
        data[ row[ 'identifier' ] ][ condition ] = ( mean, std )

  working_dir = os.path.join( pyGCluster_dir, 'hoehner_example_run/' )
  if not os.path.exists( working_dir ):
      os.mkdir( working_dir )
  print( '[ INFO ] ... the results of the example script are saved in "{0}".\n'.format( working_dir ) )

  cpus_2_use = 4
  if multiprocessing.cpu_count() < cpus_2_use:
      print( '[ INFO ] 4 threads are not available -> re-sampling is performed with only {0} thread(s) (this increases calculation time approx. proportional).'.format( multiprocessing.cpu_count() ) )
      cpus_2_use = multiprocessing.cpu_count()
   
  cluster = pyGCluster.Cluster( data = data, working_directory = working_dir, verbosity_level = 2 )

  print( "[ INFO ] pyGCluster will evoke 4 threads (if possible), which each require approx. 1.5GB RAM. Please make sure you have enough RAM available (4 threads in all require approx. 6GB RAM)." )
  print( "[ INFO ] It will take approx. 2 hours to complete 250,000 iterations on 4 threads." )
  
  cluster.do_it_all( 
                    distances = [ 'euclidean', 'correlation' ], 
                    linkages = [ 'complete', 'average', 'ward' ], 
                    iter_max = 10000, 
                    cpus_2_use = cpus_2_use, 
                    min_value_4_expression_map = -3, 
                    max_value_4_expression_map = 3,
                    threshold_4_the_lowest_max_freq = 0.005
  )

if __name__ == '__main__':
    if len(sys.argv) <= 1:
       print(__doc__)
       exit()
    #invoke the freeze_support funtion for windows based systems
    try:
        sys.getwindowsversion()
        multiprocessing.freeze_support()
    except:
        pass   

    main()    

