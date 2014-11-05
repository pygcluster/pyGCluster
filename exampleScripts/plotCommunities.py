#!/usr/bin/env python2.7
'''
Plot communities of a given pyGCluster pkl file. All output will be
directed into the folder, wehre the input data is located

The top_X_clusters option can be used to use the top X clusters for community
determination.
The threshold_4_the_lowest_max_freq option define the threshold for the maximum
frequency of the clusters which should be incoporated into the community determination.

Default values are:

    -threshold_4_the_lowest_max_freq=0.005

Usage::

    ./plotCommunities.py <pathTopyGClusterPickle>

optional::

    ./plotCommunities.py <pathTopyGClusterPickle> <threshold_4_the_lowest_max_freq=0.005>
    OR <top_X_clusters=100>


'''

from __future__ import print_function
import sys, os
import pyGCluster



def main():
    threshold_4_the_lowest_max_freq = 0.005
    top_X_clusters = 0
    for n in sys.argv[1:]:
        if "threshold_4_the_lowest_max_freq" in n:
            threshold_4_the_lowest_max_freq = float(n.split("=")[1])
        elif "top_X_clusters" in n:
            top_X_clusters = int(n.split("=")[1])
            threshold_4_the_lowest_max_freq = 0.0

    cluster = pyGCluster.Cluster()
    cluster.load(sys.argv[1])
    cluster['Working directory'] = os.path.dirname(sys.argv[1])
    cluster.build_nodemap( min_cluster_size = 4, top_X_clusters = top_X_clusters, threshold_4_the_lowest_max_freq = threshold_4_the_lowest_max_freq )
    print( cluster.keys() )
    cluster.draw_community_expression_maps( min_value_4_expression_map = -3, max_value_4_expression_map = 3)
    cluster.draw_expression_profiles( min_value_4_expression_map = -3, max_value_4_expression_map = 3 )

if __name__ == '__main__':
    if len(sys.argv) <= 1:
       print(__doc__)
       exit()
    main()
