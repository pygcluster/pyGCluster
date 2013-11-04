#!/usr/bin/env python2.7
'''
Get some information from a pyGCluster pkl object

Usage::

    ./getpyGPickleInfo.py <MERGED_pyGCluster_pkl_object>
'''
from __future__ import print_function
import sys, os
import pyGCluster


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print( __doc__ )
        exit(1)

    cluster = pyGCluster.Cluster()
    cluster.load( sys.argv[1] )

    cluster.info()
   