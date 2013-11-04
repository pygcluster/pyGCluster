#!/usr/bin/env python

"""
Testscript to demonstrate functionality of pyGCluster

This script imports the data of Hoehner et al. (2013) and executes pyGCluster 
with 250,000 iterations of resampling. pyGCluster will evoke 4 threads 
(if possible), which each require approx. 1.5GB RAM. Please make sure you have 
enough RAM available (4 threads in all require approx. 6GB RAM).
Duration will be approx. 2 hours to complete 250,000 iterations on 4 threads.

Usage::

  ./advancedClusterHoehnerExampleData.py <pathToExampleFile>


If this script is executed in folder pyGCluster/exampleScripts, the command would be::

  ./advancedClusterHoehnerExampleData.py ../exampleFiles/hoehner_dataset.csv

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
  print( '[ INFO ] ... the results of the example script are saved in "{0}".\n'.format( working_dir ) )
  ## initialize pyGCluster
  if not os.path.exists( working_dir ):
      os.mkdir( working_dir )
  TestCluster = pyGCluster.Cluster( data = data, working_directory = os.path.join( pyGCluster_dir, 'hoehner_example_run/' ), verbosity_level = 2 )
  print( "[ INFO ] pyGCluster will evoke 4 threads (if possible), which each require approx. 1.5GB RAM. Please make sure you have enough RAM available (4 threads in all require approx. 6GB RAM)." )
  print( "[ INFO ] It will take approx. 2 hours to complete 250,000 iterations on 4 threads." )
  ## start the re-sampling process ... if 4 threads are available, this may take X hours and Y GB RAM.
  distance_metrices = [ 'correlation', 'euclidean' ]
  linkage_methods = [ 'complete', 'average', 'ward' ]
  print( '[ INFO ] performing re-sampling ...' )
  cpus_2_use = 4
  if multiprocessing.cpu_count() < cpus_2_use:
      print( '[ INFO ] 4 threads are not available -> re-sampling is performed with only {0} thread(s) (this increases calculation time approx. proportional).'.format( multiprocessing.cpu_count() ) )
      cpus_2_use = multiprocessing.cpu_count()
  print()
  TestCluster.resample( distances = distance_metrices, linkages = linkage_methods, iter_max = 250000, pickle_filename = 'example.pkl', cpus_2_use = cpus_2_use )
  # after re-sampling, the results are saved in a file given by "pickle_filename"

  ## plot a heat map showing the frequencies among the distance-linkage combinations (DLCs) of the first 33 clusters:
  TestCluster.plot_clusterfreqs( min_cluster_size = 4, top_X_clusters = 33 )

  ## create and plot communities
  TestCluster.build_nodemap( min_cluster_size = 4, threshold_4_the_lowest_max_freq = 0.005 ) # create communities from a set of the 1 promille (or more) clusters
  TestCluster.write_dot( filename = 'hoehner_1promilleclusters_minsize4.dot', min_value_4_expression_map = -3, max_value_4_expression_map = 3, color_gradient = '1337' ) # creates DOT file of the node map showing the cluster composition of the communities
  
  TestCluster.draw_community_expression_maps( min_value_4_expression_map = -3, max_value_4_expression_map = 3, color_gradient = '1337' ) # draw a heat map showing the protein composition of each community
  TestCluster.draw_expression_profiles( min_value_4_expression_map = -3, max_value_4_expression_map = 3 ) # draw a plot showing the expression patterns of the proteins (with standard deviation) inside each community

  ## save to be able to continue analysis at a later timepoint
  TestCluster.save( filename = 'example_1promille_communities.pkl' )
  #TestCluster.load( 'example_1percent_communities.pkl' )

  # create CSV containing the protein composition of communities
  # => two cols: community ID -> identifier
  with open( os.path.join( TestCluster[ 'Working directory' ], 'community2protein.csv' ), 'w' ) as fout:
    writer = csv.DictWriter( fout, fieldnames = [ 'community ID', 'identifier' ] )
    writer.writeheader()
    _max_level = max( [ _communityID[ 1 ] for _communityID in TestCluster[ 'Communities' ] ] )
    for cluster in TestCluster._get_levelX_clusters( level = _max_level ):
      _communityID = ( cluster, _max_level )
      for protein_index in TestCluster[ 'Communities' ][ _communityID ][ 'index 2 obCoFreq dict' ]:
        protein = TestCluster[ 'Identifiers' ][ protein_index ]
        name = '{0}-{1}'.format( TestCluster[ 'Communities' ][ _communityID ][ 'cluster ID' ], _max_level )
        writer.writerow( { 'community ID' : name, 'identifier' : protein } )
  print( '[ INFO ] test script successfully executed.' )





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

