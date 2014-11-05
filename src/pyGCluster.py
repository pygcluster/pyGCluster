#!/usr/bin/env python2.7
"""
pyGCluster is a clustering algorithm focusing on noise injection for subsequent cluster validation.
By requesting identical cluster identity, the reproducibility of a large amount of clusters
obtained with agglomerative hierarchical clustering (AHC) is assessed.
Furthermore, a multitude of different distance-linkage combinations (DLCs) are evaluated.
Finally, associations of highly reproducible clusters, called communities, are created.
Graphical representation of the results as node maps and expression maps is implemented.

The pyGCluster module contains the main class :py:class:`pyGCluster.Cluster` and some functions
    | :py:func:`pyGCluster.create_default_alphabet`
    | :py:func:`pyGCluster.resampling_multiprocess`
    | :py:func:`pyGCluster.seekAndDestry`
    | :py:func:`pyGCluster.yield_noisejected_dataset`

"""
#
# pyGCluster
#
# Copyright (C) D. Jaeger and C. Fufezan
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
#
from __future__ import print_function
import sys, os
from collections import defaultdict as ddict
from collections import OrderedDict
import math
import time
import random
import subprocess
import string
import codecs
import bisect
import multiprocessing
import itertools
if sys.version_info[0] == 3:
    import pickle
else: # 2k, explicitely import cPickle
    import cPickle as pickle

def yield_noisejected_dataset(data, iterations):
    '''
    Generator yielding a re-sampled dataset with each iteration.
    A re-sampled dataset is created by re-sampling each data point
    from the normal distribution given by its associated mean and standard deviation value.
    See the example in Supplementary Material in pyGCluster's publication for how to define an own noise-function (e.g. uniform noise).

    :param data: dictionary ( OrderedDict! ) holding the data to be re-sampled.
    :type data: collections.OrderedDict()
    :param iterations: the number of re-sampled datasets this generator will yield.
    :type iterations: int

    :rtype: none
    '''
    import numpy
    # the check that no condition is missing in arg: data is made prior, in Cluster.__init__()
    # this is required, because only equally shaped arrays can be clustered!
    # otherwise, 'ValueError: setting an array element with a sequence.'
    Random                     = numpy.random.RandomState() # get instance for new seed!
    n_conditions               = len( data[ sorted( data.keys() )[ 0 ] ] )
    simulated_dataset          = numpy.zeros( ( len( data ), n_conditions ) )
    for i in range( iterations ):
        for row_index, identifier in enumerate( data ):
            for col_index, (condition, data_tuple) in enumerate( data[ identifier ].items() ):
                mean, sd  = data_tuple
                new_ratio = Random.normal( mean, sd )
                simulated_dataset[ row_index ][ col_index ] = new_ratio
        yield simulated_dataset
    return

def create_default_alphabet():
    '''
    Returns the default alphabet which is used to save clusters in a lesser memory-intense form:
    instead of saving e.g. a cluster containing identifiers with indices of 1,20,30 as "1,20,30", the indices are converted to a baseX system -> "1,k,u".

    The default alphabet that is returned is:
        >>> string.printable.replace( ',', '' )

    :rtype: string
    '''
    return string.printable.replace( ',', '' )

def seekAndDestry(processes):
    '''
    Any multiprocesses given by processes are terminated.

    :param processes: list containing multiprocess.Process()
    :type processes: list

    :rtype: none

    '''
    for p in processes:
        if p.is_alive():
            p.terminate()
    return

def resampling_multiprocess(
                                DataQ                                    = None,
                                data                                     = None,
                                iterations                               = 5000,
                                alphabet                                 = None,
                                dlc                                      = None,
                                min_cluster_size                         = 4,
                                min_cluster_freq_2_retain                  = 0.001,
                                function_2_generate_noise_injected_datasets = None
                            ):
    '''
    This is the function that is called for each multiprocesses that is evoked internally in pyGCluster during the re-sampling routine.
    Agglomerative hierarchical clustering is performed for each distance-linkage combination (DLC) on each of iteration datasets.
    Clusters from each hierarchical tree are extracted, and their counts are saved in a temporary cluster-count matrix.
    After *iterations* iterations, clusters are filtered according to min_cluster_freq_2_retain.
    These clusters, together with their respective counts among all DLCs, are returned.
    The return value is a list containing tuples with two elements: cluster (string) and counts ( one dimensional np.array )

    :param DataQ: data queue which is used to pipe the re-sampling results back to pyGCluster.
    :type DataQ: multiprocessing.Queue()
    :param data: dictionary ( OrderedDict! ) holding the data to be clustered -> passed through to the noise-function.
    :type data: collections.OrderedDict()
    :param iterations: the number of iterations this multiprocess is going to perform.
    :type iterations: int
    :param alphabet: in order to save memory, the indices describing a cluster are converted to a specific alphabet (rather than decimal system).
    :type alphabet: string
    :param dlc: list of the distance-linkage combinations that are going to be evaluated.
    :type dlc: list
    :param min_cluster_size: minimum size of a cluster to be considered in the re-sampling routine (smaller clusters are discarded)
    :type min_cluster_size: int
    :param min_cluster_freq_2_retain: once all iterations are performed, clusters are filtered according to 50% (because typically forwarded from pyGCluster) of this threshold.
    :type min_cluster_freq_2_retain: float
    :param function_2_generate_noise_injected_datasets: function to generate re-sampled datasets.
    :type function_2_generate_noise_injected_datasets: function

    :rtype: list
    '''
    import numpy
    import scipy.spatial.distance as ssd
    imported_from_scipy = False
    try:
        from fastcluster import linkage as ahc
    except ImportError:
        try:
            from scipy.cluster.hierarchy import linkage as ahc
            imported_from_scipy = True
        except ImportError:
            print('You do require either "fastcluster" or "scipy"!')

    if DataQ == None or data == None:
        print( '[ ERROR ] need a Data-Queune and a data object! Returning ...' )
        return
    if alphabet == None:
        alphabet = create_default_alphabet()
    assert ',' not in alphabet, '[ ERROR ] the alphabet must not contain a comma (",")!'
    if dlc == None:
        dlc = [ 'euclidean-average' ]  # NOTE maybe better have all as default ! :)
    if function_2_generate_noise_injected_datasets == None:
        function_2_generate_noise_injected_datasets = yield_noisejected_dataset
    n_objects   = len( data.keys() )
    n_dlc       = len( dlc )
    metrices                       = set( [ combo.split( '-' )[ 0 ] for combo in dlc ] )

    # build lookup-dict to convert index into baseX system, given by alphabet
    baseX       = len( alphabet )
    index2baseX = { 0 : '0' }
    for index in range( 1, n_objects ):
        old_index = index
        digits = [] # modified ref: http://stackoverflow.com/questions/2267362/convert-integer-to-a-string-in-a-given-numeric-base-in-python
        while index:
            digits.append( alphabet[ index % baseX ] )
            index = int( round( index / baseX ) )
        digits.reverse()
        converted_index = ''.join( digits )
        index2baseX[ old_index ] = converted_index

    # build initial template of 'clusters'-dict (which is needed to extract clusters from the hierarchical tree)
    clusters_template = { ID : [ index2baseX[ ID ] ] for ID in range( n_objects ) }
    # initialize temporary cluster-count matrix and the other necessary objects to fill it
    tmpstruct_clustercount_monitor = {}
    tmpstruct_clustercount_monitor[ 'Cluster counts' ]                  = numpy.zeros( ( 10 ** 6, n_dlc ), dtype = numpy.uint32 )
    tmpstruct_clustercount_monitor[ 'Cluster 2 clusterID' ]             = {}
    tmpstruct_clustercount_monitor[ 'Distance-linkage combinations' ]   = dlc
    tmpstruct_clustercount_monitor[ 'Cluster sieve' ]                   = set()
    tmpstruct_clustercount_monitor[ 'Discarded IDs' ]                   = set()
    # get simulated datasets
    for simulated_dataset in function_2_generate_noise_injected_datasets( data, iterations ):
        # calculate distance matrices:
        metric2condenseddist = {}
        if not imported_from_scipy:
            for metric in metrices:
                metric2condenseddist[ metric ] = ssd.pdist( simulated_dataset, metric = metric )
        # perform AHC:
        for dlc_index, combo in enumerate( dlc ):
            metric, linkage = combo.split( '-' )
            '''
            linkage matrix example:
            original data:
            [[1,2,3],
             [3,2,1],
             [1,3,5]]

            Linkage matrix representing AHC with euclidean distance and ward linkage:
            [[ 0.        ,  2.        ,  2.23606798,  2.        ],      CLUSTER ID 3
             [ 1.        ,  3.        ,  4.2031734 ,  3.        ]]      CLUSTER ID 4
               ^ child1     ^ child2     ^ distance   ^ cluster size
            Hence, element 0 and 2 were merged into cluster with ID = 3 (size = 2),
            then element 1 and cluster 3 are merged into the root cluster with ID = 4 (size = 3).
            '''
            # perform AHC
            if imported_from_scipy:
                linkage_matrix = ahc( simulated_dataset, method = linkage, metric = metric )
            else:
                linkage_matrix = ahc( metric2condenseddist[ metric ], method = linkage, preserve_input = True )
            # reconstruct clusters from the linkage matrix
            clusters = {} # key = clusterID, value = cluster-indices
            clusters.update( clusters_template )
            clusterID_linkagematrix = n_objects - 1
            for childID_1, childID_2, dist, size in linkage_matrix:
                clusterID_linkagematrix             += 1
                cluster_linkagematrix               = sorted( clusters[ childID_1 ] + clusters[ childID_2 ] )
                clusters[ clusterID_linkagematrix ] = cluster_linkagematrix
                if len( cluster_linkagematrix ) < min_cluster_size:
                    continue
                cluster = ','.join( cluster_linkagematrix )
                # insert cluster into tmpstruct_clustercount_monitor and update it:
                # but add only if its count > 1 (determined via the 'Cluster sieve'):
                add = False
                if cluster in tmpstruct_clustercount_monitor[ 'Cluster 2 clusterID' ]:
                    clusterID = tmpstruct_clustercount_monitor[ 'Cluster 2 clusterID' ][ cluster ]
                    add       = True
                else:
                    if cluster in tmpstruct_clustercount_monitor[ 'Cluster sieve' ]:
                        if tmpstruct_clustercount_monitor[ 'Discarded IDs' ]:
                            try:
                                clusterID = tmpstruct_clustercount_monitor[ 'Discarded IDs' ].pop()
                            except KeyError: # KeyError: 'pop from an empty set' = set is empty
                                clusterID = len( tmpstruct_clustercount_monitor[ 'Cluster 2 clusterID' ] )
                        else:
                            clusterID = len( tmpstruct_clustercount_monitor[ 'Cluster 2 clusterID' ] )
                        tmpstruct_clustercount_monitor[ 'Cluster 2 clusterID' ][ cluster ] = clusterID
                        add = True
                    else:
                        tmpstruct_clustercount_monitor[ 'Cluster sieve' ].add( cluster )
                        add = False
                if add:
                    # increase count by 1
                    # if new cluster, add 10 ** 5 new rows
                    try:
                        tmpstruct_clustercount_monitor[ 'Cluster counts' ][ clusterID ][ dlc_index ] += 1
                    except IndexError:
                        tmpstruct_clustercount_monitor[ 'Cluster counts' ] = numpy.concatenate(
                            ( tmpstruct_clustercount_monitor['Cluster counts'],
                             numpy.zeros( ( 10 ** 5, n_dlc ), dtype = numpy.uint32 )
                            )
                        )
                        tmpstruct_clustercount_monitor[ 'Cluster counts' ][ clusterID ][ dlc_index ] += 1 # increase count by 1
            del clusters
        del metric2condenseddist
    del simulated_dataset

    # only transfer clusters equal or above 50% of 'min_cluster_freq_2_retain' threshold to pyGCluster:
    min_count = int( min_cluster_freq_2_retain * iterations * 0.5 )
    clusterIDs2retain = set( numpy.nonzero( tmpstruct_clustercount_monitor[ 'Cluster counts' ] >= min_count )[ 0 ] )
    cluster_counts_list = []
    for cluster, clusterID in tmpstruct_clustercount_monitor[ 'Cluster 2 clusterID' ].items():
        if clusterID in clusterIDs2retain:
            counts = tmpstruct_clustercount_monitor[ 'Cluster counts' ][ clusterID ]
            cluster_counts_list.append( (cluster, counts) )
    del tmpstruct_clustercount_monitor
    DataQ.put( cluster_counts_list )
    del cluster_counts_list
    return

class Cluster(dict):
    '''
    The pyGCluster class

    :param working_directory: directory in which all results are written (requires write-permission!).
    :type working_directory: string
    :param verbosity_level: either 0, 1 or 2.
    :type verbosity_level: int
    :param data: Dictionary containing the data which is to be clustered.
    :type data: dict

    In order to work with the default noise-injection function as well as plot
    expression maps correctly, the data-dict **has** to have the following
    structure.

    Example:

        >>> data = {
        ...            Identifier1 : {
        ...                            condition1 :  ( mean11, sd11 ),
        ...                            condition2 :  ( mean12, sd12 ),
        ...                            condition3 :  ( mean13, sd13 ),
        ...             },
        ...            Identifier2 : {
        ...                            condition2 :  ( mean22, sd22 ),
        ...                            condition3 :  ( mean23, sd23 ),
        ...                            condition3 :  ( mean13, sd13 ),
        ...             },
        ... }
        >>> import pyGCluster
        >>> ClusterClass = pyGCluster.Cluster(data=data, verbosity_level=1, working_directory=...)

    .. note ::
        If any condition for an identifier in the "nested_data_dict"-dict is missing,
        this entry is discarded, i.e. not imported into the Cluster Class.
        This is because pyGCluster does not implement any missing value estimation.
        One possible solution is to replace missing values by a mean value and a standard
        deviation that is representative for the complete data range in the given condition.

    pyGCluster inherits from the regular Python Dictionary object.
    Hence, the attributes of pyGCluster can be accessed as Python Dictionary keys.

    A selection of the most important attributes / keys are:

        >>> # general
        >>> ClusterClass[ 'Working directory' ]
        ...     # this is the directory where all pyGCluster results
        ...     # (pickle objects, expression maps, node map, ...) are saved into.
        /Users/Shared/moClusterDirectory
        >>> # original data ca be accessed via
        >>> ClusterClass[ 'Data' ]
        ...     # this collections.OrderedDict contains the data that has been
        ...     # or will be clustered (see also below).
        ... plenty of data ;)
        >>> ClusterClass[ 'Conditions' ]
        ...     # sorted list of all conditions that are defined in the "Data"-dictionary
        [ 'condition1', 'condition2', 'condition3' ]
        >>> ClusterClass[ 'Identifiers' ]
        ...     # sorted tuple of all identifiers, i.e. ClusterClass[ 'Data' ].keys()
        ( 'Identifier1', 'Identifier2' , ... 'IdentifierN' )
        >>> # re-sampling paramerters
        >>> ClusterClass[ 'Iterations' ]
        ...     # the number of datasets that were clustered.
        1000000
        >>> ClusterClass[ 'Cluster 2 clusterID' ]
        ...     # dictionary with clusters as keys, and their respective row index
        ...     # in the "Cluster count"-matrix (= clusterID) as values.
        { ... }
        >>> ClusterClass[ 'Cluster counts' ]
        ...     # numpy.uint32 matrix holding the counts for each
        ...     # distance-linkage combination of the clusters.
        >>> ClusterClass[ 'Distance-linkage combinations' ]
        ...     # sorted list containing the distance-linkage combinations
        ...     # that were evaluted in the re-sampling routine.
        >>> # Communities
        >>> ClusterClass[ 'Communities' ]
        ...     # see function pyGCluster.Cluster.build_nodemap for further information.
        >>> # Visualization
        >>> ClusterClass[ 'Additional labels' ]
        ...     # dictionary with an identifier of the "Data"-dict as key,
        ...     # and a list of additional information (e.g. annotation, GO terms) as value.
        {
            'Identifier1' :
                        ['Photosynthesis related' , 'zeroFactor: 12.31' ],
            'Identifier2' : [ ... ] ,
             ...
        }
        >>> ClusterClass[ 'for IO skip clusters bigger than' ]
        ...     # Default = 100. Since some clusters are really large
        ...     # (with sizes close to the root (the cluster holding all objects)),
        ...     # clusters with more objects than this value
        ...     # are not plotted as expression maps or expression profile plots.


    pyGCluster offers the possibility to save the analysis (e.g. after re-sampling)
    via :py:func:`pyGCluster.Cluster.save` , and continue
    via :py:func:`pyGCluster.Cluster.load`
    Initializes  pyGCluster.Cluster class

    Classically, users start the multiprocessing clustering routine with multiple
    distance linkage combinations via the :py:func:`pyGCluster.Cluster.do_it_all`
    function. This function allows to update the pyGCluster class with all user
    parameters before it calls :py:func:`pyGCluster.Cluster.resample`.
    The main advantage in calling :py:func:`pyGCluster.Cluster.do_it_all` is
    that all general plotting functions are called afterwards as well, these are:

        | :py:func:`pyGCluster.Cluster.plot_clusterfreqs`
        | :py:func:`pyGCluster.Cluster.build_nodemap`
        | :py:func:`pyGCluster.Cluster.write_dot`
        | :py:func:`pyGCluster.Cluster.draw_community_expression_maps`

    If one choses, one can manually update the parameters (setting the key, value
    pairs in pyGCluster) and then evoke :py:func:`pyGCluster.Cluster.resample`
    with the appropriate parameters. This useful if certain memory intensive
    distance-linkage combinations are to be clustered on a specific computer.

    .. note ::
       Cluster Class can be initilized empty and filled using :py:func:`pyGCluster.Cluster.load`



    '''
    def __init__(self, data = None, working_directory = None, verbosity_level = 1):
        self.delete_resampling_results() # initializes important variables
        if working_directory == None:
            working_directory = os.getcwd()
        self[ 'Working directory' ]             = working_directory
        self[ 'for IO skip clusters bigger than' ]   = 100
        self[ 'Version' ]                       = (0, 7, 1)
        self[ 'Verbosity level' ]               = verbosity_level
        self[ 'Additional labels' ]             = {}   # will be used as dict in draw functions, i.e. ids
        self[ 'Data' ]                          = None
        self[ 'Heat map']                       = {
                                        'Params':   {   'title'         : 'pyGCluster expression map',
                                                        'font family'    : 'Helvetica',
                                                        'font size'      : 14 ,
                                                        'rBox width'     : 40,
                                                        'rBox height'    : 20,
                                                        'left border'    : 10,
                                                        'top border'     : 70, # will be adjusted depending on the labels :)
                                                        'text spacing'   : 2,
                                                        'text width'     : 2000,
                                                        'separator width': 7,
                                                        'min'            : None,
                                                        'max'            : None,
                                                        'legend filename': 'legend.svg',
                                                        'heat map filename' : 'expression_map.svg',
                                                        'default color'  : [255, 255, 255],
                                                        'color gradient' : 'default',
                                        },
                                        'Color Gradients' : {
                                                    'default'       : [(-1, (255,40,255)), (-0.40,(255,40,40)), (-0.05,(40,40,40)), (0,(0,0,0)), (+0.05,(40,40,40)), (+0.40,(40,255,40)), (+1,(255,255,40)) ],
                                                    'Daniel'       : [(-1, (255,0,0)), (-0.01, (0,0,255)), (0, (0,0,0)), (0.01, (255,255,0)), (0.5, (0,255,0)), (1, (0,255,255))],
                                                    'barplot'      : [(-1, (  0,0,0)), (0, (0,0,0)), (0.0000001, (255,255,0)), (0.2, (255,0,0)), (1, (120,120,120))],
                                                    '1337'         : [(-1, (255,0,0)), (-0.5,(255,0,255)), (-0.02,(77,77,77)), (0,(0,0,0)) ,(+0.02,(77,77,77)), (+0.5,(255,255,0)), (+1,(0,255,0)) ],
                                                    'BrBG'         : [(-1, (166, 97, 26)),  (-0.5, (223, 194, 125)), (0, (245, 245, 245)), (+0.5, (128, 205, 193)), (+1, (1, 133, 113)) ],
                                                    'PiYG'         : [(-1, (208, 28, 139)), (-0.5, (241, 182, 218)), (0, (247, 247, 247)), (+0.5, (184, 225, 134)), (+1, (77, 172, 38)) ],
                                                    'PRGn'         : [(-1, (123, 50, 148)), (-0.5, (194, 165, 207)), (0, (247, 247, 247)), (+0.5, (166, 219, 160)), (+1, (0, 136, 55)) ],
                                                    'PuOr'         : [(-1, (230, 97, 1)),   (-0.5, (253, 184, 99)),  (0, (247, 247, 247)), (+0.5, (178, 171, 210)), (+1, (94, 60, 153)) ],
                                                    'RdBu'         : [(-1, (202, 0, 32)),   (-0.5, (244, 165, 130)), (0, (247, 247, 247)), (+0.5, (146, 197, 222)), (+1, (5, 113, 176)), ],
                                                    'RdGy'         : [(-1, (202, 0, 32)),   (-0.5, (244, 165, 130)), (0, (255, 255, 255)), (+0.5, (186, 186, 186)), (+1, (64, 64, 64)), ],
                                                    'RdYlBu'       : [(-1, (215, 25, 28)),  (-0.5, (253, 174, 97)),  (0, (255, 255, 191)), (+0.5, (171, 217, 233)), (+1, (44, 123, 182)), ],
                                                    'RdYlGn'       : [(-1, (215, 25, 28)),  (-0.5, (253, 174, 97)),  (0, (255, 255, 191)), (+0.5, (166, 217, 106)), (+1, (26, 150, 65)), ],
                                                    'Spectral'     : [(-1, (215, 25, 28)),  (-0.5, (253, 174, 97)),  (0, (255, 255, 191)), (+0.5, (171, 221, 164)), (+1, (43, 131, 186)), ],
                                        },
                                        'SVG box styles'  : {
            'modern' : '''
            <g id="rowPos{0}_conPos{1}">
                <title>{ratio}&#177;{std} - [{x0}.{y0} w:{width} h:{height}</title>
                <rect x="{x0}" y="{y0}" width="{width}" height="{height}" style="fill:rgb({r},{g},{b});fill-opacity:0.2;stroke:white;stroke-width:1;" title="{ratio}&#177;{std}" />
                <path d = "M {x0} {y0} L {x3} {y0} L {x2} {y1} L {x1} {y1} L {x1} {y2} L {x0} {y3} L {x0} {y0}" style="fill:rgb({r},{g},{b});stroke:black;stroke-width:1;stroke-opacity:0.0;"/>
                <path d = "M {x2} {y2} L {x1} {y2} L {x2} {y1} L {x2} {y2}" style="fill:rgb({r},{g},{b});stroke:black;stroke-width:1;stroke-opacity:0.0;"/>
                <path d = "M {x1} {y1} L {x1} {y2} L {x2} {y1} L {x1} {y1}" style="fill:rgb({r},{g},{b}); fill-opacity:0.7; stroke:red;stroke-width:1;stroke-opacity:0.0;"/>
            </g>''',

            'fusion' : '''
            <g id="rowPos{0}_conPos{1}">
                <title>{ratio}&#177;{std} - [{x0}.{y0} w:{width} h:{height}</title>
                <rect x="{x0}" y="{y0}" width="{width}" height="{height}" style="fill:rgb({r},{g},{b});fill-opacity:0.7;stroke:white;stroke-width:1;" title="{ratio}&#177;{std}" />
                <path d = "M {x0} {y0} L {x3} {y0} L {x2} {y1} L {x1} {y1} L {x1} {y2} L {x0} {y3} L {x0} {y0}" style="fill:rgb({r},{g},{b});stroke:black;stroke-width:1;stroke-opacity:0.0;"/>
                <path d = "M {x2} {y2} L {x1} {y2} L {x2} {y1} L {x2} {y2}" style="fill:rgb({r},{g},{b});stroke:black;stroke-width:1;stroke-opacity:0.0;"/>
                <path d = "M {x1} {y1} L {x1} {y2} L {x2} {y1} L {x1} {y1}" style="fill:rgb({r},{g},{b}); fill-opacity:0.7; stroke:red;stroke-width:1;stroke-opacity:0.0;"/>
                <rect x="{x1}" y="{y1}" width="{widthNew}" height="{heightNew}" style="fill:None;stroke:black;stroke-width:1;" title="{ratio}&#177;{std}" />

            </g>''',

            'classic' : '''
                <g id="rowPos{0}_conPos{1}">
                    <title>{ratio}&#177;{std} - [{x0}.{y0} w:{width} h:{height}</title>
                    <rect x="{x0}" y="{y0}" width="{width}" height="{height}" style="fill:rgb({r},{g},{b});stroke:white;stroke-width:1;" title="{ratio}&#177;{std}" />
                    <rect x="{x1}" y="{y1}" width="{widthNew}" height="{heightNew}" style="fill:None;stroke:black;stroke-width:1;" title="{ratio}&#177;{std}" />
                </g>''',
            }
        }

        # check if data is valid, i.e. contains a value for each condition
        data_as_ordered_dict = OrderedDict()
        if data != None:
            conditions = set()
            # determine number of different conditions:
            for identifier in data.keys():
                for condition in data[ identifier ].keys():
                    conditions.add(condition)
            for identifier in data.keys():
                # discard entry if any condition is missing:
                missing_conditions = conditions - set( data[ identifier ].keys() )
                if len(missing_conditions) > 0:
                    del data[identifier]
            for identifier in sorted( data ):
                data_as_ordered_dict[ identifier ] = OrderedDict()
                for condition in sorted( data[ identifier ] ):
                    data_as_ordered_dict[ identifier ][ condition ] = data[ identifier ][ condition ]
            self[ 'Conditions' ]  = sorted( conditions )
            self[ 'Data' ]        = data_as_ordered_dict
            self[ 'Identifiers' ] = tuple( sorted( data ) )
            self[ 'Root size' ] = len( data )
            self[ 'Root' ] = tuple( range( self[ 'Root size' ] ) )
            if not self.check_if_data_is_log2_transformed():
                self._print( '[ WARNING ] there are NO ratios < 0! Is the data log2 transformed?', file=sys.stderr, verbosity_level = 1 )
            s = 'pyGCluster initialized with {0} objects among {1} different conditions.'
            self._print( s.format( len( data.keys() ), len( conditions ) ), verbosity_level = 1 )
        return

    def draw_expression_map( self, identifiers = None, data = None, conditions = None, additional_labels = None, min_value_4_expression_map = None, max_value_4_expression_map = None, expression_map_filename = None, legend_filename = None, color_gradient = None , box_style = 'classic' ):
        '''
        Draws expression map as SVG

        :param min_value_4_expression_map: lower bound for color coding of values in the expression map. Remember that log2-values are expected, i.e. this value should be < 0!
        :type min_value_4_expression_map: float
        :param max_value_4_expression_map: upper bound for color coding of values in the expression map.
        :type max_value_4_expression_map: float
        :param color_gradient: name of the color gradient used for plotting the expression map. Currently supported are default, Daniel, barplot, 1337, BrBG, PiYG, PRGn, PuOr, RdBu, RdGy, RdYlBu, RdYlGn and Spectral
        :type color_gradient: string
        :param expression_map_filename: file name for expression map. .svg will be added if required.
        :type expression_map_filename: string
        :param legend_filename: file name for legend .svg will be added if required.
        :type legend_filename: string
        :param box_style: the way the relative standard deviation is visualized in the expression map. Currently supported are 'modern', 'fusion' or 'classic'.
        :type box_style: string
        :param additional_labels: dictionary, where additional labels can be defined which will be added in the expression map plots to the gene/protein names
        :type additional_labels: dict

        :rtype: none

        Data has to be a nested dict in the following format:
            >>>  data =   {
            ...         fastaID1 : {
            ...                 cond1 : ( mean, sd ) , cond2 : ( mean, sd ), ...
            ...         }
            ...         fastaID2 : {
            ...                 cond1 : ( mean, sd ) , cond2 : ( mean, sd ), ...
            ...         }
            ...  }

        optional and, if needed, data will be extracted from
            | self[ 'Data' ]
            | self[ 'Identifiers' ]
            | self[ 'Conditions' ]



        '''


        if additional_labels == None:
            additional_labels = {}
        if conditions == None:
            conditions = set()
            for identifier in data.keys():
                conditions |= set( data[ identifier ].keys() )
            conditions = sorted( list( conditions ) )
        if identifiers == None:
            if type(data) == type(OrderedDict()):
                identifiers = list( data.keys() )
            else:
                identifiers = sorted(list( data.keys() ))
        #
        # Updating self[ 'Additional labels' ]
        #
        if additional_labels != None:
            for identifier in additional_labels.keys():
                if identifier not in self[ 'Additional labels' ].keys():
                    self[ 'Additional labels' ][ identifier ] = []
                # self[ 'Additional labels' ][ identifier ] += additional_labels[ identifier ]
        #
        # Updating min/max if required
        #
        if max_value_4_expression_map != None:
            self[ 'Heat map'][ 'Params' ][ 'max' ] = max_value_4_expression_map
        if min_value_4_expression_map != None:
            self[ 'Heat map'][ 'Params' ][ 'min' ] = min_value_4_expression_map
        #
        # determine range id needed
        #
        if self[ 'Heat map'][ 'Params' ][ 'min' ] == None or self[ 'Heat map'][ 'Params' ][ 'max' ] == None:
            allValues = []
            for identifier in data.keys():
                for condition in data[ identifier ].keys():
                    allValues.append( data[ identifier ][ condition][0] )
            if self[ 'Heat map' ][ 'Params' ][ 'min' ] == None:
                self[ 'Heat map' ][ 'Params' ][ 'min' ] = math.floor( min( allValues ) )
            if self[ 'Heat map' ][ 'Params' ][ 'max' ] == None:
                self[ 'Heat map' ][ 'Params' ][ 'max' ] = math.ceil( max( allValues ) )
        #
        # setting default color gradient if match is found
        #
        if color_gradient != None:
            if color_gradient not in self[ 'Heat map' ][ 'Color Gradients' ].keys():
                print('Do not know color gradient {0}, falling back to default'.format( color_gradient ), file = sys.stderr)
                color_gradient = 'default'
            self[ 'Heat map' ][ 'Params' ][ 'color gradient' ] = color_gradient
        #
        #
        #
        if expression_map_filename != None:
            self[ 'Heat map'][ 'Params' ][ 'heat map filename' ] = expression_map_filename
        if legend_filename != None:
            self[ 'Heat map'][ 'Params' ][ 'legend filename' ] = legend_filename

        self[ 'Heat map'][ 'Params' ][ 'expression profile filename' ] = self[ 'Heat map'][ 'Params' ][ 'heat map filename' ]+'_expP.svg'
        for filename in ['heat map filename', 'legend filename', 'expression profile filename']:
            if '.svg' not in self[ 'Heat map'][ 'Params' ][ filename ]:
                self[ 'Heat map'][ 'Params' ][ filename ] += '.svg'
        #
        # recalculate topBorder
        #
        for pos, line in enumerate( conditions ):
            lineHeight = len( line ) * self[ 'Heat map'][ 'Params' ]['font size']
            if lineHeight > self[ 'Heat map'][ 'Params' ][ 'top border' ]:
                self[ 'Heat map'][ 'Params' ][ 'top border' ] = lineHeight

        #
        #
        #
        expProf = {}
        assert type(identifiers) == type( [] ) , 'require a list of identifiers!'
        # self._draw_expression_map_legend()
        svgOut               = codecs.open( os.path.join( self[ 'Working directory' ], self[ 'Heat map' ][ 'Params' ]['heat map filename'] ), 'w', 'utf-8')
        svgWidth             = len( conditions ) * self[ 'Heat map'][ 'Params' ][ 'rBox width' ] + self[ 'Heat map'][ 'Params' ]['left border'] + self[ 'Heat map'][ 'Params' ]['text width']
        svgHeight            = len( identifiers ) * self[ 'Heat map'][ 'Params' ][ 'rBox height' ] + self[ 'Heat map'][ 'Params' ]['top border']
        number_of_separators = 0

        print("""<svg
        xmlns="http://www.w3.org/2000/svg"
        version="1.1"
        preserveAspectRatio="xMinYMin meet"
        width="{0}"
        height="{1}"
        font-size="{font size}px"
        font-family="{font family}"
        fill="black"
        text-anchor="beginning"
        baseline-alignment="middle"
        >
        <title>{title}</title>
        """.format(
                svgWidth,
                svgHeight,
                **self[ 'Heat map'][ 'Params' ]
            ),
        file = svgOut
        )
        #
        # write top legend
        #
        for condPos, condition in enumerate( conditions ):
            x = int(self[ 'Heat map'][ 'Params' ][ 'left border' ] + (condPos) * self[ 'Heat map'][ 'Params' ]['rBox width'] + self[ 'Heat map'][ 'Params' ]['rBox width'] / 2.0 )
            y = int(self[ 'Heat map'][ 'Params' ][ 'top border' ] - self[ 'Heat map'][ 'Params' ]['text spacing'] )
            print(  unicode(
                            '            <text x="{0}" y="{1}" text-anchor="left" transform="rotate(-90, {0}, {1})">{2}</text>'.format(
                            x,
                            y,
                            condition
                            ),
                    errors = 'replace'
                    ),
                    file = svgOut
            )
        for rowPos, identifier in enumerate( identifiers ):
            adjustedRowPos = rowPos - number_of_separators
            if identifier == '_placeholder_':
                shapeDict = self._HM_calcShapeAndColor(
                                                        x                    = 0,
                                                        y                    = adjustedRowPos,
                                                        ratio                = 0,
                                                        std                  = 0,
                                                        number_of_separators = number_of_separators,
                )
                shapeDict['x1_separator'] = shapeDict['x0']
                shapeDict['x2_separator'] = shapeDict['x0'] + ( self[ 'Heat map'][ 'Params' ]['rBox width']  * len( conditions ))
                print( unicode('''
                            <line x1="{x1_separator}" y1="{y0}" x2="{x2_separator}" y2="{y0}" style="stroke:rgb{0};stroke-width:{1}"/>
                            '''.format(
                                    self[ 'Heat map'][ 'Params' ]['default color'],
                                    self[ 'Heat map'][ 'Params' ]['separator width'],
                                    **shapeDict
                                ),
                        errors = 'replace'
                        ),
                    file = svgOut
                )
                number_of_separators += 1
            else:
                expProf[ identifier ] = [ [] ]
                for conPos, condition in enumerate( conditions ):
                    try:
                        ratio, std = data[ identifier ][ condition ]
                        insertion_point = int( len( expProf[ identifier ][ -1 ] ) / 2 )
                        # first entry in profile
                        expProf[ identifier ][ -1 ].insert( insertion_point,  ratio - std )
                        expProf[ identifier ][ -1 ].insert( insertion_point,  ratio + std )

                    except:
                        ratio, std = None, None
                        expProf[ identifier ].append( [] )

                    shapeDict = self._HM_calcShapeAndColor(
                                                            x                    = conPos,
                                                            y                    = adjustedRowPos,
                                                            ratio                = ratio,
                                                            std                  = std,
                                                            number_of_separators = number_of_separators,
                    )

                    print( unicode( self['Heat map']['SVG box styles'][ box_style ].format(
                                        rowPos,
                                        conPos,
                                        **shapeDict
                                        ),
                                    errors = 'replace'
                                    ),
                        file = svgOut
                    )

                #

                shapeDict['x_text']           = (conPos + 1  ) * self[ 'Heat map'][ 'Params' ]['rBox width'] + self[ 'Heat map'][ 'Params' ]['left border'] + self[ 'Heat map'][ 'Params' ]['text spacing']
                shapeDict['y_text']           = (adjustedRowPos + 0.77) * self[ 'Heat map'][ 'Params' ]['rBox height'] + self[ 'Heat map'][ 'Params' ]['top border'] + (self[ 'Heat map'][ 'Params' ]['separator width'] * number_of_separators)
                shapeDict['text']             = '{0} '.format( identifier )
                if identifier in additional_labels.keys():
                    shapeDict['text'] +=  ' '.join(additional_labels[ identifier ])

                if identifier in self[ 'Additional labels' ].keys():
                    shapeDict['text'] +=  ' '.join( self[ 'Additional labels' ][ identifier ])

                print( unicode('''
            <g id="Text rowPos{0}_conPos{1}">
                <title>{ratio}&#177;{std}</title>
                <text xml:space='preserve' x="{x_text}" y="{y_text}">{text}</text>
            </g>'''.format(
                                    rowPos,
                                    conPos,
                                    **shapeDict
                                    ),
                                errors = 'replace'
                                ),
                    file = svgOut
                )

        # eof
        print("</svg>", file = svgOut )
        svgOut.close()
        #
        # Drawing legend
        #
        svgLegendOut         = codecs.open( os.path.join( self[ 'Working directory' ], self[ 'Heat map' ][ 'Params' ]['legend filename'] ), 'w', 'utf-8')
        svgWidth             = len( conditions ) * self[ 'Heat map'][ 'Params' ][ 'rBox width' ] + self[ 'Heat map'][ 'Params' ]['left border'] + self[ 'Heat map'][ 'Params' ]['text width']
        svgHeight            = 11 * self[ 'Heat map'][ 'Params' ][ 'rBox height' ] + self[ 'Heat map'][ 'Params' ]['top border']
        number_of_separators = 0

        print("""<svg
        xmlns="http://www.w3.org/2000/svg"
        version="1.1"
        preserveAspectRatio="xMinYMin meet"
        width="{0}"
        height="{1}"
        font-size="{font size}px"
        font-family="{font family}"
        fill="black"
        text-anchor="beginning"
        baseline-alignment="middle"
        >
        <title>Legend</title>
        <text x="{2}" y="{3}" text-anchor="left" transform="rotate(-90, {2}, {3})">ratio</text>
        <text x="{4}" y="{3}" text-anchor="left" transform="rotate(-90, {4}, {3})">rel. std</text>
        """.format(
                svgWidth,
                svgHeight,
                int(self[ 'Heat map'][ 'Params' ][ 'left border' ] + 2 * self[ 'Heat map'][ 'Params' ]['rBox width'] + self[ 'Heat map'][ 'Params' ]['rBox width'] / 2.0 ),
                int(self[ 'Heat map'][ 'Params' ][ 'top border' ] - self[ 'Heat map'][ 'Params' ]['text spacing'] ) - 10,
                int(self[ 'Heat map'][ 'Params' ][ 'left border' ] + 3 * self[ 'Heat map'][ 'Params' ]['rBox width'] + self[ 'Heat map'][ 'Params' ]['rBox width'] / 2.0 ),
                **self[ 'Heat map'][ 'Params' ]
            ),
        file = svgLegendOut
        )
        positive_step_size = self[ 'Heat map' ]['Params'][ 'max' ] / 5.0
        negative_step_size = self[ 'Heat map' ]['Params'][ 'min' ] / 5.0
        number_of_separators = 0
        for y in range(0,11):
            _ = 5 - y
            if _ >= 0:
                ratio = positive_step_size * _
            else:
                ratio = negative_step_size * -1 * _

            shapeDict = self._HM_calcShapeAndColor(
                        x       = 2,
                        y       = y,
                        ratio   = ratio,
                        std     = 0.0
            )
            print( unicode( self['Heat map']['SVG box styles'][ box_style ].format(
                                        y,
                                        2,
                                        **shapeDict
                                        ),
                                    errors = 'replace'
                                    ),
                        file = svgLegendOut
            )

            std = y * 0.1
            shapeDict = self._HM_calcShapeAndColor(
                        x       = 3,
                        y       = y,
                        ratio   = 1.0,
                        std     = std
            )
            shapeDict['r'] = 147
            shapeDict['g'] = 147
            shapeDict['b'] = 147
            print( unicode(self['Heat map']['SVG box styles'][ box_style ].format(
                                        y,
                                        3,
                                        **shapeDict
                                        ),
                                    errors = 'replace'
                                    ),
                        file = svgLegendOut
            )
            shapeDict['x_text_left']           = self[ 'Heat map'][ 'Params' ]['rBox width'] + self[ 'Heat map'][ 'Params' ]['left border'] + self[ 'Heat map'][ 'Params' ]['text spacing']
            shapeDict['x_text_right']          = 4 * self[ 'Heat map'][ 'Params' ]['rBox width'] + self[ 'Heat map'][ 'Params' ]['left border'] + self[ 'Heat map'][ 'Params' ]['text spacing']
            shapeDict['y_text_left']           = (y + 0.77) * self[ 'Heat map'][ 'Params' ]['rBox height'] + self[ 'Heat map'][ 'Params' ]['top border'] + (self[ 'Heat map'][ 'Params' ]['separator width'] * number_of_separators)
            shapeDict['text_left']             = '{0:3.2f}'.format( ratio )
            shapeDict['text_right']            = '{0:2.1f}'.format( std )

            print( unicode('''
                    <g id="Legend {0}">
                        <title>{ratio}&#177;{std}</title>
                        <text xml:space='preserve' x="{x_text_left}" y="{y_text_left}">{text_left}</text>
                        <text xml:space='preserve' x="{x_text_right}" y="{y_text_left}">{text_right}</text>
                    </g>'''.format(
                                    y,
                                    **shapeDict
                                    ),
                                errors = 'replace'
                                ),
                    file = svgLegendOut
            )
        print("</svg>", file = svgLegendOut )
        svgLegendOut.close()
        return

    def _HM_calcShapeAndColor(self, x = None, y = None, ratio = None, std = None, number_of_separators = 0):
        '''
        Internal function to determine shape and color of expression map entries
        '''
        shapeDict = {}

        shapeDict['ratio']                             = ratio
        shapeDict['std']                               = std
        shapeDict['r'], shapeDict['g'], shapeDict['b'] = self._HM_visualizeColor( ratio )
        shapeDict['x0']                                 = int(self[ 'Heat map'][ 'Params' ]['left border'] + self[ 'Heat map'][ 'Params' ]['rBox width']  * x)
        shapeDict['y0']                                 = int(self[ 'Heat map'][ 'Params' ]['top border']  + self[ 'Heat map'][ 'Params' ]['rBox height'] * y)
        shapeDict['width']                             = self[ 'Heat map'][ 'Params' ]['rBox width']
        shapeDict['height']                            = self[ 'Heat map'][ 'Params' ]['rBox height']

        if std != None or (std == None and ratio == None): # or std != 0.0:
            if std == None:
                # ratio and sd for this entry are None, this will lead to white box
                stdAsPercentOfRatio = 0

            else:
                if ratio == 0.0:
                    ratio += 0.01
                stdAsPercentOfRatio = abs( std / float( ratio ) )
                if stdAsPercentOfRatio > 1:
                    stdAsPercentOfRatio = 1

            shapeDict['widthNew']  = int(round( (1 - stdAsPercentOfRatio) * self[ 'Heat map'][ 'Params' ]['rBox width']  ))
            shapeDict['heightNew'] = int(round( (1 - stdAsPercentOfRatio) * self[ 'Heat map'][ 'Params' ]['rBox height'] ))
            shapeDict['x1']      = int(shapeDict['x0']  +  0.5 * (self[ 'Heat map'][ 'Params' ]['rBox width'] - shapeDict['widthNew']))
            shapeDict['y1']      = int(shapeDict['y0']  +  0.5 * (self[ 'Heat map'][ 'Params' ]['rBox height'] - shapeDict['heightNew']))

            shapeDict['y0']                += self[ 'Heat map'][ 'Params' ]['separator width'] * number_of_separators
            shapeDict['y1']             += self[ 'Heat map'][ 'Params' ]['separator width'] * number_of_separators
            shapeDict['height_half']      = shapeDict['height'] / 2.0
            shapeDict['y3']       = shapeDict['y0'] + shapeDict['height']
            shapeDict['x3']        = shapeDict['x0'] + shapeDict['width']
            shapeDict['y2'] = shapeDict['y1'] + shapeDict['heightNew']
            shapeDict['x2']  = shapeDict['x1'] + shapeDict['widthNew']
        return shapeDict

    def _HM_visualizeColor( self, ratio ):
        '''
        determine color for expression map values
        '''
        ##
        color = self[ 'Heat map'][ 'Params' ][ 'default color' ][:]
        colorGradient = self[ 'Heat map' ][ 'Color Gradients' ][  self[ 'Heat map' ]['Params']['color gradient']  ]
        if ratio != None:
            if ratio >= 0:
                scaling = self[ 'Heat map' ]['Params'][ 'max' ] / float( colorGradient[-1][0] )
            else:
                scaling = self[ 'Heat map' ]['Params'][ 'min' ] / float( colorGradient[0][0] )

            scaled_ratio = ratio / scaling
            idx = bisect.bisect(  colorGradient, ( scaled_ratio, )  )

            if idx == 0:
                color =  colorGradient[0][1]
            elif idx == len( colorGradient):
                color =  colorGradient[-1][1]
            else:
                # linear interpolation ... between idx-1 & idx
                dX = ( scaled_ratio -  colorGradient[ idx - 1 ][ 0 ] ) / (  colorGradient[ idx ][ 0 ] - colorGradient[ idx - 1 ][ 0 ] )
                for color_chanel in range(3):
                    d_ = dX * ( colorGradient[ idx ][ 1 ][ color_chanel ] -  colorGradient[ idx - 1 ][ 1 ][ color_chanel ])
                    if abs( d_ ) <= sys.float_info.epsilon :
                        color[ color_chanel ] = int(round( colorGradient[idx - 1][ 1 ][ color_chanel ]))
                    else:
                        color[ color_chanel ] = int(round( colorGradient[idx - 1][ 1 ][ color_chanel ] + d_))
        return color

    def draw_expression_map_for_cluster(self, clusterID = None, cluster = None, filename = None, min_value_4_expression_map = None, max_value_4_expression_map = None, color_gradient = 'default', box_style = 'classic' ):
        '''
        Plots an expression map for a given cluster.
        Either the parameter "clusterID" or "cluster" can be defined.
        This function is useful to plot a user-defined cluster, e.g. knowledge-based cluster (TCA-cluster, Glycolysis-cluster ...). In this case, the parameter "cluster" should be defined.

        :param clusterID: ID of a cluster (those are obtained e.g. from the plot of cluster frequencies or the node map)
        :type clusterID: int
        :param cluster: tuple containing the indices of the objects describing a cluster.
        :type cluster: tuple
        :param filename: name of the SVG file for the expression map.
        :type filename: string

        The following parameters are passed to :py:func:`pyGCluster.Cluster.draw_expression_map`:

        :param min_value_4_expression_map: lower bound for color coding of values in the expression map. Remember that log2-values are expected, i.e. this value should be < 0!
        :type min_value_4_expression_map: float
        :param max_value_4_expression_map: upper bound for color coding of values in the expression map.
        :type max_value_4_expression_map: float
        :param color_gradient: name of the color gradient used for plotting the expression map. Currently supported are default, Daniel, barplot, 1337, BrBG, PiYG, PRGn, PuOr, RdBu, RdGy, RdYlBu, RdYlGn and Spectral
        :type color_gradient: string
        :param box_style: name of box style used in SVG. Currently supported are classic, modern, fusion.
        :type box_style: string


        :rtype: none
        '''
        # check if function call was valid:
        if clusterID == None and cluster == None:
            self._print( '[ ERROR ] call function "draw_expression_map_for_cluster" with either a clusterID or a cluster.', verbosity_level = 0 )
            return
        elif clusterID != None and cluster != None:
            self._print( '[ ERROR ] call function "draw_expression_map_for_cluster" with either a clusterID or a cluster.', verbosity_level = 0 )
            return
        # if clusterID is given, get the corresponding cluster:
        elif clusterID != None:
            for c, cID in self[ 'Cluster 2 clusterID' ].iteritems():
                if cID == clusterID:
                    break
            cluster = c

        # determine hm_filename:
        if filename == None:
            filename = '{0}.svg'.format( self[ 'Cluster 2 clusterID' ][ cluster ] )
        hm_filename = os.path.join( self[ 'Working directory' ], filename )
        # prepare for drawing of expression map ...
        identifiers = []
        data = {}
        additional_labels = {}
        try:
            cFreq, cFreqDict = self.frequencies( cluster = cluster )
        except KeyError:
            cFreq = 0.0
        for index in cluster:
            identifier = self[ 'Identifiers' ][ index ]
            identifiers.append( identifier )
            data[ identifier ] = {}
            for condition in self[ 'Conditions' ]:
                data[ identifier ][ condition ] = self[ 'Data' ][ identifier ][ condition ]
            additional_labels[ identifier ] = [ '{0:3.4f}'.format( cFreq ) ]
        self.draw_expression_map(
            identifiers       = identifiers,
            data              = data,
            conditions        = self[ 'Conditions' ],
            additional_labels = additional_labels,
            min_value_4_expression_map              = min_value_4_expression_map,
            max_value_4_expression_map              = max_value_4_expression_map,
            expression_map_filename = hm_filename,
            legend_filename   = None,
            color_gradient    = color_gradient,
            box_style         = box_style
        )
        self._print( '... expression map saved as "{0}".'.format( hm_filename ), verbosity_level = 1 )
        return

    def draw_expression_map_for_community_cluster(self, name, min_value_4_expression_map = None, max_value_4_expression_map = None, color_gradient = '1337', sub_folder = None, min_obcofreq_2_plot = None, box_style = 'classic'):
        '''
        Plots the expression map for a given "community cluster":
        Any cluster in the community node map is internally represented as a tuple with two elements:
        "cluster" and "level". Those objects are stored as keys in self[ 'Communities' ],
        from where they may be extracted and fed into this function.

        :param name: "community cluster" -> best obtain from self[ 'Communities' ].keys()
        :type name: tuple
        :param min_obcofreq_2_plot: minimum obCoFreq of an cluster's object to be shown in the expression map.
        :type min_obcofreq_2_plot: float

        The following parameters are passed to :py:func:`pyGCluster.Cluster.draw_expression_map`:

        :param min_value_4_expression_map: lower bound for color coding of values in the expression map. Remember that log2-values are expected, i.e. this value should be < 0!
        :type min_value_4_expression_map: float
        :param max_value_4_expression_map: upper bound for color coding of values in the expression map.
        :type max_value_4_expression_map: float
        :param color_gradient: name of the color gradient used for plotting the expression map. Currently supported are default, Daniel, barplot, 1337, BrBG, PiYG, PRGn, PuOr, RdBu, RdGy, RdYlBu, RdYlGn and Spectral
        :type color_gradient: string
        :param box_style: name of box style used in SVG. Currently supported are classic, modern, fusion.
        :type box_style: string
        :param sub_folder: if specified, the expression map is saved in this folder, rather than in pyGCluster's working directory.
        :type sub_folder: string

        :rtype: none
        '''
        identifiers = []
        data = {}
        additional_labels = {}
        for index in self[ 'Communities' ][ name ][ 'index 2 obCoFreq dict' ]:
            identifier = None
            if index > 0:
                normalized_obCoFreq = self[ 'Communities' ][ name ][ 'index 2 obCoFreq dict' ][ index ]
                if normalized_obCoFreq < min_obcofreq_2_plot:
                    continue
                identifier = self[ 'Identifiers' ][ index ]
                identifiers.append( identifier )
                data[ identifier ] = {}
                for condition in self[ 'Conditions' ]:
                    data[ identifier ][ condition ] = self[ 'Data' ][ identifier ][ condition ]
                additional_labels[ identifier ] = [ '{0:3.4f}'.format( normalized_obCoFreq ) ]
            else:
                identifiers.append( '_placeholder_' )
        hm_filename = '{0}-{1}.svg'.format( self[ 'Communities' ][ name ][ 'cluster ID' ], name[ 1 ] )
        if sub_folder != None:
            if not os.path.exists( os.path.join(  self[ 'Working directory' ], sub_folder ) ):
                os.mkdir( os.path.join(  self[ 'Working directory' ], sub_folder ) )
            hm_filename = os.path.join( sub_folder , hm_filename )

        self.draw_expression_map(
            identifiers       = identifiers,
            data              = data,
            conditions        = self[ 'Conditions' ],
            additional_labels = additional_labels,
            min_value_4_expression_map              = min_value_4_expression_map,
            max_value_4_expression_map              = max_value_4_expression_map,
            expression_map_filename = hm_filename,
            legend_filename   = None,
            color_gradient    = color_gradient,
            box_style         = box_style
        )
        return

    def draw_community_expression_maps(self, min_value_4_expression_map = None, max_value_4_expression_map = None, color_gradient = 'default', box_style = 'classic', conditions= None, additional_labels=None):
        '''
        Plots the expression map for each community showing its object composition.

        The following parameters are passed to :py:func:`pyGCluster.Cluster.draw_expression_map`:

        :param min_value_4_expression_map: lower bound for color coding of values in the expression map. Remember that log2-values are expected, i.e. this value should be < 0!
        :type min_value_4_expression_map: float
        :param max_value_4_expression_map: upper bound for color coding of values in the expression map.
        :type max_value_4_expression_map: float
        :param color_gradient: name of the color gradient used for plotting the expression map. Currently supported are default, Daniel, barplot, 1337, BrBG, PiYG, PRGn, PuOr, RdBu, RdGy, RdYlBu, RdYlGn and Spectral
        :type color_gradient: string
        :param box_style: name of box style used in SVG. Currently supported are classic, modern, fusion.
        :type box_style: string
        :param additional_labels: dict with additional labels, k = identified and v = list of additional labels.
        :type additional_labels: dict

        :rtype: none
        '''
        if conditions == None:
            conditions = self[ 'Conditions' ]
        max_level = max( [ name[ 1 ] for name in self[ 'Communities' ] ] )
        for cluster in self._get_levelX_clusters( level = max_level ):
            name = ( cluster, max_level )
            if len( cluster ) > self[ 'for IO skip clusters bigger than' ]:
                continue
            identifiers = []
            data = {}
            internal_additional_labels = {}
            for index in self[ 'Communities' ][ name ][ 'index 2 obCoFreq dict' ]:
                identifier = None
                if index > 0:
                    identifier = self[ 'Identifiers' ][ index ]
                    identifiers.append( identifier )
                    data[ identifier ] = {}
                    for condition in self[ 'Conditions' ]:
                        data[ identifier ][ condition ] = self[ 'Data' ][ identifier ][ condition ]
                    internal_additional_labels[ identifier ] = [ '{0:4.2f}'.format( self[ 'Communities' ][ name ][ 'index 2 obCoFreq dict' ][ index ] ) ]
                else:
                    identifiers.append( '_placeholder_' )
            if additional_labels != None:
                for k in internal_additional_labels.keys():
                    if k in additional_labels.keys():
                        internal_additional_labels[ k ] += additional_labels[ k ]

            hm_filename = '{0}-{1}.svg'.format( self[ 'Communities' ][ name ][ 'cluster ID' ], name[ 1 ] )
            self.draw_expression_map(
                identifiers       = identifiers,
                data              = data,
                conditions        = conditions,
                additional_labels = internal_additional_labels,
                min_value_4_expression_map              = min_value_4_expression_map,
                max_value_4_expression_map              = max_value_4_expression_map,
                expression_map_filename = hm_filename,
                legend_filename   = None,
                color_gradient    = color_gradient,
                box_style         = box_style
            )
        self._print( '... community expression maps saved in "{0}"'.format( self[ 'Working directory' ] ), verbosity_level = 1 )
        return

    def delete_resampling_results(self):
        '''
        Resets all variables holding any result of the re-sampling process.
        This includes the convergence determination as well as the community structure.
        Does not delete the data that is intended to be clustered.

        :rtype: None
        '''
        self[ 'Cluster 2 clusterID' ]                                     = {}
        self[ 'Cluster counts' ]                                          = None
        self[ 'Distances' ]                                               = []
        self[ 'Linkages' ]                                                = []
        self[ 'Distance-linkage combinations' ]                           = []
        self[ 'Iterations' ]                                              = 0

        self[ 'Convergence determination - params' ]                      = {}
        self[ 'Convergence determination - iteration 2 n_mostfreq' ]      = {}
        self[ 'Convergence determination - first detected at iteration' ] = 0

        self[ 'Communities' ]                                             = {}

        self[ 'Function parameters' ]                                     = {}
        return

    def check_if_data_is_log2_transformed(self):
        '''
        Simple check if any value of the data_tuples (i.e. any mean) is below zero.
        Below zero indicates that the input data was log2 transformed.

        :rtype: boolean
        '''
        for identifier in self[ 'Data' ].keys():
            for condition, data_tuple in self[ 'Data' ][ identifier ].items():
                for value in data_tuple:
                    if value < 0:
                        return True
        return False

    def __add__(self, other):
        '''
        Adds re-sampling results of a pyGCluster instance into another one.
        If the clustered data differs among those two instances, the other instance is NOT added.
        If the distance-linkage combinations among those two instances differ, the other instance is NOT added.

        :param other: the pyGCluster instance that is to be added to self.
        :type other: pyGCluster instance

        :rtype: None
        '''
        import numpy
        assert self[ 'Data' ] == other[ 'Data' ], '[ ERROR ] pyGCluster-instances with different clustered data cannot be merged!'
        assert sorted( self[ 'Distance-linkage combinations' ] ) == sorted( other[ 'Distance-linkage combinations' ] ), '[ ERROR ] pyGCluster-instances with a different distance-linkage combinations cannot be merged!'
        self[ 'Iterations' ] += other[ 'Iterations' ]
        if self[ 'Cluster counts' ] == None:
            self[ 'Cluster counts' ] = numpy.zeros( ( 10 ** 4, len( self[ 'Distance-linkage combinations' ] ) ), dtype = numpy.uint32 )
        otherDLC2selfDLC = {}
        for other_dlc_index, dlc in enumerate( other[ 'Distance-linkage combinations' ] ):
            self_dlc_index = self[ 'Distance-linkage combinations' ].index( dlc )
            otherDLC2selfDLC[ other_dlc_index ] = self_dlc_index
        # merge clusters from other into self
        for cluster, other_clusterID in other[ 'Cluster 2 clusterID' ].iteritems():
            if cluster not in self[ 'Cluster 2 clusterID' ]:
                self[ 'Cluster 2 clusterID' ][ cluster ] = len( self[ 'Cluster 2 clusterID' ] ) # new cluster found, assign index
            self_clusterID = self[ 'Cluster 2 clusterID' ][ cluster ]
            for other_dlc_index, self_dlc_index in otherDLC2selfDLC.items():
                try:
                    self[ 'Cluster counts' ][ self_clusterID ]
                except IndexError:
                    self[ 'Cluster counts' ] = numpy.concatenate(
                        (
                            self[ 'Cluster counts' ],
                            numpy.zeros( ( 10 ** 4, len( self[ 'Distance-linkage combinations' ] ) ), dtype = numpy.uint32 )
                        )
                    ) # add rows at bottom
                self[ 'Cluster counts' ][ self_clusterID ][ self_dlc_index ] += other[ 'Cluster counts' ][ other_clusterID ][ other_dlc_index ]
        return

    def resample(self, distances, linkages, function_2_generate_noise_injected_datasets = None, min_cluster_size = 4, alphabet = None, force_plotting = False, min_cluster_freq_2_retain = 0.001, pickle_filename = 'pyGCluster_resampled.pkl', cpus_2_use = None, iter_tol = 0.01 / 100000, iter_step = 5000, iter_max = 250000, iter_top_P = 0.001, iter_window = 50000, iter_till_the_end = False):
        '''
        Routine for the assessment of cluster reproducibility (re-sampling routine).
        To this, a high number of noise-injected datasets are created, which are subsequently clustered by AHC.
        Those are created via :py:func:`pyGCluster.function_2_generate_noise_injected_datasets` (default = usage of Gaussian distributions).
        Each 'simulated' dataset is then subjected to AHC x times, where x equals the number of distance-linkage combinations that come from all possible combinations of "distances" and "linkages".
        In order to speed up the re-sampling routine, it is distributed to multiple threads, if cpus_2_use > 1.

        The re-sampling routine stops once either convergence (see below) is detected or iter_max iterations have been performed.
        Eventually, only clusters with a maximum frequency of at least min_cluster_freq_2_retain are stored; all others are discarded.

        In order to visually inspect convergence, a convergence plot is created.
        For more information about the convergence estimation, see Supplementary Material of pyGCluster's publication.

        For a complete list of possible
        Distance matrix calculations
        see: http://docs.scipy.org/doc/scipy/reference/spatial.distance.html
        or Linkage methods
        see: http://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html

        .. note ::
            If memory is of concern (e.g. for a large dataset, > 5000 objects), cpus_2_use should be kept low.

        :param distances: list of distance metrices, given as strings, e.g. [ 'correlation', 'euclidean' ]
        :type distances: list
        :param linkages: list of distance metrices, given as strings, e.g. [ 'average', 'complete', 'ward' ]
        :type linkages: list
        :param function_2_generate_noise_injected_datasets: function to generate noise-injected datasets. If None (default), Gaussian distributions are used.
        :type function_2_generate_noise_injected_datasets: function
        :param min_cluster_size: minimum size of a cluster, so that it is included in the assessment of cluster reproducibilities.
        :type min_cluster_size: int
        :param alphabet: alphabet used to convert decimal indices to characters to save memory. Defaults to string.printable, without ','.
        :type alphabet: string

        .. note ::
            If alphabet contains ',', this character is removed from alphabet, because the indices comprising a cluster are saved comma-seperated.

        :param force_plotting: the convergence plot is created after each iter_step iteration (otherwise only when convergence is detected).
        :type force_plotting: boolean
        :param min_cluster_freq_2_retain: ]0, 1[ minimum frequency of a cluster (only the maximum of the dlc-frequencies matters here) it has to exhibit to be stored in pyGCluster once all iterations are finished.
        :type min_cluster_freq_2_retain: float
        :param cpus_2_use: number of threads that are evoked in the re-sampling routine.
        :type cpus_2_use: int
        :param iter_max: maximum number of re-sampling iterations.
        :type iter_max: int

        Convergence determination:

        :param iter_tol: ]0, 1e-3[ value for the threshold of the median of normalized slopes, in order to declare convergence.
        :type iter_tol: float
        :param iter_step: number of iterations each multiprocess performs and simultaneously the interval in which to check for convergence.
        :type iter_step: int
        :param iter_top_P: ]0, 1[ for the convergence estmation, the amount of most frequent clusters is examined. This is the threshold for the minimum frequency of a cluster to be included.
        :type iter_top_P: float
        :param iter_window: size of the sliding window in iterations. The median is obtained from normalized slopes inside this window - *should be a multiple of iter_step*
        :type iter_window: int
        :param iter_till_the_end: if set to True, the convergence determination is switched off; hence, re-sampling is performed until iter_max is reached.
        :type iter_till_the_end: boolean

        :rtype: None
        '''
        self[ 'Function parameters' ][ self.resample.__name__ ] = { k : v for k, v in locals().items() if k != 'self' }
        import numpy
        import scipy.cluster.hierarchy as sch
        import scipy.spatial.distance as ssd

        if function_2_generate_noise_injected_datasets == None:
            function_2_generate_noise_injected_datasets = yield_noisejected_dataset
        if alphabet == None:
            alphabet = string.printable
        alphabet = alphabet.replace( ',', '' )

        ## create distance-linkage combinations (dlc)
        self[ 'Distances' ] = distances
        self[ 'Linkages' ] = linkages
        # check if all distance metrices are valid:
        invalid_dists = set( self[ 'Distances' ] ) - set( dir( ssd ) )
        if invalid_dists:
            s = '[ WARNING ] invalid distance metrices! "{0}" are not in "scipy.spatial.distance".'
            self._print( s.format( ', '.join( invalid_dists ) ), verbosity_level = 0 )
        # check if all linkage methods are valid:
        invalid_linkages = set( self[ 'Linkages' ] ) - sch._cpy_linkage_methods
        if invalid_linkages:
            s = '[ WARNING ] invalid linkage methods! "{0}" are not in "scipy.cluster.hierarchy".'
            self._print( s.format( ', '.join( invalid_linkages ) ), verbosity_level = 0 )
        # get all possible distance-linkage combinations:
        self[ 'Distance-linkage combinations' ] = []
        for distance in self[ 'Distances' ]:
            for linkage in self[ 'Linkages' ]:
                if distance != 'euclidean' and linkage in sch._cpy_euclid_methods:
                    continue
                self[ 'Distance-linkage combinations' ].append( '{0}-{1}'.format( distance, linkage ) )
        n_dlc = len( self[ 'Distance-linkage combinations' ] )
        self._print( '{0} distance-linkage combinations are evaluated.'.format( n_dlc ), verbosity_level = 2 )
        self._print( '... those are: {0}.'.format( ', '.join( self[ 'Distance-linkage combinations' ] ) ), verbosity_level = 2 )

        ## check if permission to write:
        if pickle_filename:
            try:
                with open( os.path.join( self[ 'Working directory' ], 'tmp.txt' ), 'w' ) as fout:
                    pass
                os.remove( os.path.join( self[ 'Working directory' ], 'tmp.txt' ) )
            except IOError:
                s = '[ WARNING ] You do not have permission, or folder does not exist!\n\tresults in "{0}" are NOT pickled!'
                self._print( s.format( self[ 'Working directory' ] ), verbosity_level = 0 )
                pickle_filename = False

        ## check if a pickle file with the same name is already in the "Working directory"
        ## this indicates that clustering is likely to be continued:
        if pickle_filename in os.listdir( self[ 'Working directory' ] ):
            self._print( 'Pickle file with the same name detected! Pickle "{0}" will be loaded and clustering continued ...'.format( pickle_filename ), verbosity_level = 0 )
            loaded = Cluster()
            loaded.load( os.path.join( self[ 'Working directory' ], pickle_filename ) )
            self + loaded

        ## create tmp_struct to store the cluster counts:
        tmpstruct_clustercounts = {}
        tmpstruct_clustercounts[ 'Cluster counts' ]         = numpy.zeros( ( 10 ** 5, n_dlc ), dtype = numpy.uint32 )
        tmpstruct_clustercounts[ 'Cluster 2 clusterID' ]    = {}
        tmpstruct_clustercounts[ 'Discarded IDs' ]          = set()

        ## initialize variables for the convergence determination
        self[ 'Convergence determination - params' ] = {}
        self[ 'Convergence determination - params' ][ 'iter_step' ]     = iter_step
        self[ 'Convergence determination - params' ][ 'iter_top_P' ]     = iter_top_P
        self[ 'Convergence determination - params' ][ 'iter_tol' ]      = iter_tol
        self[ 'Convergence determination - params' ][ 'iter_window' ]   = iter_window
        self[ 'Convergence determination - params' ][ 'iter_max' ]      = iter_max
        if iter_window % iter_step:
            s = '[ WARNING ] iter_window = {0} is NOT a multiple of iter_step = {1}. Better re-call with a multiple of iter_step!'
            self._print( s.format( iter_window, iter_step ), verbosity_level = 1 )

        ## prepare for multiprocesses:
        if cpus_2_use != None:
            if cpus_2_use > multiprocessing.cpu_count():
                s = '[ WARNING ] You requested to perform re-sampling on {0} threads, but only {1} available -> better re-call with "cpus_2_use = {1}"!'
                self._print( s.format( cpus_2_use, multiprocessing.cpu_count() ), verbosity_level = 0 )
            n_multiprocesses = cpus_2_use
        else:
            n_multiprocesses = multiprocessing.cpu_count()
        DataQ = multiprocessing.Queue()
        kwargs4multiprocess = {}
        kwargs4multiprocess[ 'DataQ' ]                      = DataQ
        kwargs4multiprocess[ 'data' ]                       = self[ 'Data' ]
        kwargs4multiprocess[ 'iterations' ]                 = iter_step
        kwargs4multiprocess[ 'alphabet' ]                   = alphabet
        kwargs4multiprocess[ 'dlc' ]                        = self[ 'Distance-linkage combinations' ]
        kwargs4multiprocess[ 'min_cluster_size' ]           = min_cluster_size
        kwargs4multiprocess[ 'min_cluster_freq_2_retain' ]    = min_cluster_freq_2_retain
        kwargs4multiprocess[ 'function_2_generate_noise_injected_datasets' ] = function_2_generate_noise_injected_datasets

        #this does not work on windows... we have to check that
        try:
            sys.getwindowsversion()
        except:
            os.nice( 10 )
        min_count = int( min_cluster_freq_2_retain * kwargs4multiprocess[ 'iterations' ] * 0.5 )
        if min_count < 2:
            s = '[ WARNING ] params "min_cluster_freq_2_retain" = {0} and "iter_step" = {1}, hence min_count = {2}\n\t-> huge accumulation of unstable clusters in pyGCluster!'
            self._print( s.format( min_cluster_freq_2_retain, kwargs4multiprocess[ 'iterations' ], min_count ), verbosity_level = 1 )
        # check if multiprocess are valid:
        self._print( 'checking if multiprocesses are functioning ...', end = ' ',  verbosity_level = 2 )
        try:
            tmp_kwargs4multiprocess = { k : v for k, v in kwargs4multiprocess.items() }
            tmp_kwargs4multiprocess[ 'iterations' ] = 1
            p = multiprocessing.Process( target = resampling_multiprocess, kwargs = tmp_kwargs4multiprocess )
            p.start()
            del tmp_kwargs4multiprocess
        except:
            self._print( '[ ERROR ] Failed to launch multi-processes!',  file = sys.stderr, verbosity_level = 0 )
            seekAndDestry( [ p ] )
            raise
        try:
            DataQ.get()
            p.join()
        except:
            self._print( '[ ERROR ] Failed to collect multi-processes!',  file = sys.stderr, verbosity_level = 0 )
            seekAndDestry( [ p ] )
            raise
        self._print( 'success!', verbosity_level = 2 )

        ## other stuff:
        self[ 'Convergence determination - iteration 2 n_mostfreq' ] = {}
        iteration = 0
        converged = False
        ask2continue = False
        iter_to_continue = False

        ### now comes the actual re-sampling routine :)
        while not converged:
            # prevent exceeding iter_max:
            if iter_max < iteration + n_multiprocesses * iter_step and not iter_to_continue:
                n_multiprocesses_tmp = int( math.ceil( float( iter_max - iteration ) / iter_step ) )
                s = 'Continuing another {0} (# processes) * {1} (iter_step) iterations would exceed iter_max (= {2}). Hence, # processes are lowered to {3} so that {4} iterations have been totally performed.'
                self._print( s.format( n_multiprocesses, iter_step, iter_max, n_multiprocesses_tmp, iteration + n_multiprocesses_tmp * iter_step ), verbosity_level = 2 )
                n_multiprocesses = n_multiprocesses_tmp
            # Launching multi-processes
            processes = []
            for i in range( n_multiprocesses ):
                p = multiprocessing.Process( target = resampling_multiprocess, kwargs = kwargs4multiprocess )
                p.start()
                processes.append( p )
                time.sleep( random.random() ) # to increase randomness!

            # Collecting Process outputs and transfer cluster-counts into 'tmpstruct_clustercounts'
            for i in range( n_multiprocesses ):
                cluster_counts_list = DataQ.get()
                iteration += kwargs4multiprocess[ 'iterations' ]
                self._print( "Clustering. Resampling data : iteration {0: >7}/{1}".format( iteration, iter_max ), end = '\r', file = sys.stderr, verbosity_level = 1 )
                for cluster, counts in cluster_counts_list:
                    # get cluster ID:
                    if cluster in tmpstruct_clustercounts[ 'Cluster 2 clusterID' ]:
                        clusterID = tmpstruct_clustercounts[ 'Cluster 2 clusterID' ][ cluster ]
                    else:
                        # if available, get a discarded ID and assign this ID to the cluster:
                        if tmpstruct_clustercounts[ 'Discarded IDs' ]:
                            try:
                                clusterID = tmpstruct_clustercounts[ 'Discarded IDs' ].pop()
                            except KeyError: # KeyError: 'pop from an empty set' = set is empty
                                clusterID = len( tmpstruct_clustercounts[ 'Cluster 2 clusterID' ] )
                        else:
                            clusterID = len( tmpstruct_clustercounts[ 'Cluster 2 clusterID' ] )
                        tmpstruct_clustercounts[ 'Cluster 2 clusterID' ][ cluster ] = clusterID
                    # update counts:
                    try:
                        tmpstruct_clustercounts[ 'Cluster counts' ][ clusterID ] += counts
                    except IndexError:
                        tmpstruct_clustercounts[ 'Cluster counts' ] = numpy.concatenate(
                            ( tmpstruct_clustercounts[ 'Cluster counts' ],
                              numpy.zeros( ( 10 ** 5, n_dlc ), dtype = numpy.uint32 )
                            )
                        )
                        tmpstruct_clustercounts[ 'Cluster counts' ][ clusterID ] += counts
                # determine most frequent clusters:
                min_count = iteration * iter_top_P
                mostfreqIDs = numpy.unique( numpy.nonzero( tmpstruct_clustercounts[ 'Cluster counts' ] >= min_count )[ 0 ] )
                self[ 'Convergence determination - iteration 2 n_mostfreq' ][ iteration ] = len( mostfreqIDs )
                del mostfreqIDs
                # check if converged:
                if iter_till_the_end == False:
                    converged = self.check4convergence()
                if converged or force_plotting:
                    self.convergence_plot()
            del cluster_counts_list
            # terminate processes:
            for p in processes:
                p.join()
            # once all processes finished iter_step clusterings, perform a purging step:
            # discard all clusters with a maximum count of the threshold:
            min_required_count = int( min_cluster_freq_2_retain * 0.5 * ( kwargs4multiprocess[ 'iterations' ] * n_multiprocesses ) )
            self._print('\nDiscarding {0}-count-clusters ...'.format( min_required_count ), end = ' ', file = sys.stderr, verbosity_level = 2)
            max_counts = numpy.amax( tmpstruct_clustercounts[ 'Cluster counts' ], axis = 1 ) # get max count for each cluster
            IDs2discard = set( numpy.nonzero( max_counts == 1 )[ 0 ] )
            del max_counts
            # reset counts:
            for ID in IDs2discard:
                tmpstruct_clustercounts[ 'Cluster counts' ][ ID ][ : ] = 0
            # delete those clusters which were attributed the discarded clusterIDs
            clusters2discard = [ c for c, cID in tmpstruct_clustercounts[ 'Cluster 2 clusterID' ].iteritems() if cID in IDs2discard ]
            for cluster in clusters2discard:
                del tmpstruct_clustercounts[ 'Cluster 2 clusterID' ][ cluster ]
                del cluster
            del clusters2discard
            self._print( '{0} discarded.'.format( len( IDs2discard ) ), file = sys.stderr, verbosity_level = 2 )
            tmpstruct_clustercounts[ 'Discarded IDs' ] = IDs2discard
            del IDs2discard

            if converged and iteration < iter_max and not iter_till_the_end:
                ask2continue = True
            elif iteration >= iter_max:
                self._print( '\niter_max reached. See convergence plot. Stopping re-sampling if not defined otherwise ...', verbosity_level = 1 )
                converged = True
                self.convergence_plot()
                ask2continue = True
            # ask if user wants to continue with the re-sampling process:
            if ask2continue and self[ 'Verbosity level' ] > 0:
                self._print( '\nEnter how many iterations you would like to continue. (Has to be a multiple of iterstep = {0})'.format( iter_step ), verbosity_level = 1 )
                self._print( '(enter "0" to stop resampling.)', verbosity_level = 1 )
                self._print( '(enter "-1" to resample until iter_max (= {0}) is reached.)'.format( iter_max ), verbosity_level = 1 )
                while True:
                    answer = raw_input( 'Enter a number ...' )
                    try:
                        iter_to_continue = int( answer )
                        break
                    except:
                        self._print( 'INT conversion failed. Please try again!', verbosity_level = 1 )
                converged = False
                if iter_to_continue == 0:
                    converged = True
                elif iter_to_continue == -1:
                    iter_till_the_end = True
                    ask2continue = False
                    if iteration == iter_max:
                        converged = True
                else:
                    iter_to_continue = int( iter_step * round(iter_to_continue / float(iter_step)) )
                    if iter_to_continue < iter_step:
                        iter_to_continue = iter_step
                    iter_to_continue = int( math.ceil( float( iter_to_continue ) / n_multiprocesses ) )
                    self._print( 'Resampling will continue another {0} iterations.'.format( iter_to_continue * n_multiprocesses ), verbosity_level = 1 )
                    kwargs4multiprocess[ 'iterations' ] = iter_to_continue

        # final filtering: store only clusters in pyGCluster whose max_frequencies are above min_cluster_freq_2_retain (default 0.001):
        min_count = iteration * min_cluster_freq_2_retain
        clusterIDs2retain = set( numpy.nonzero( tmpstruct_clustercounts[ 'Cluster counts' ] >= min_count )[0] )
        self._print( '{0} clusters above threshold of {1}. '.format( len( clusterIDs2retain ), min_cluster_freq_2_retain ), verbosity_level = 2 )
        self[ 'Cluster counts' ] = numpy.zeros( ( len( clusterIDs2retain ), n_dlc ), dtype = numpy.uint32 )
        baseX = len( alphabet )
        tmp = {}
        tmp[ 'Iterations' ] = iteration
        tmp[ 'Cluster 2 clusterID' ] = {}
        tmp[ 'Cluster counts' ] = tmpstruct_clustercounts[ 'Cluster counts' ]
        tmp[ 'Distance-linkage combinations' ] = self[ 'Distance-linkage combinations' ]
        tmp[ 'Data' ] = self[ 'Data' ]
        for cluster, clusterID in tmpstruct_clustercounts[ 'Cluster 2 clusterID' ].iteritems():
            if clusterID in clusterIDs2retain:
                final_cluster = []
                # map cluster back to decimal indices:
                for baseXstring in cluster.split( ',' ):
                    index = 0
                    for i, digit in enumerate( baseXstring[ ::-1 ] ):
                        index += alphabet.find( digit ) * baseX ** i
                    final_cluster.append( index )
                final_cluster.sort()
                final_cluster = tuple( final_cluster )
                tmp[ 'Cluster 2 clusterID' ][ final_cluster ] = clusterID
        self.__add__( tmp )
        # pickle results:
        if pickle_filename:
            self.save( pickle_filename )
        s = 're-sampling routine for {0} iterations finished. {1} clusters were obtained.'
        self._print( s.format( iteration, len(clusterIDs2retain) ), verbosity_level = 1 )
        return

    def _get_normalized_slope(self, y2, y1, iter_step):
        '''
        Calculates the normalized slope between two 2D-coordinates:
        i.e. ( y2 / y1 ) - (1.0) / iter_step,
        where y = amount of most frequent clusters at a certain iteration,
        and iter_step = x2 - x1.

        :param y2: the y-coordinate of the second point.
        :type y2: float
        :param y1: the y-coordinate of the first point.
        :type y1: float
        :param iter_step: the difference between the x-coordinates of the two points, i.e. x2 - x1.
        :type iter_step: float

        rtype: float
        '''
        numerator   = float( y2 ) / float( y1 ) - 1.0
        norm_slope  = numerator / float( iter_step )
        return norm_slope

    def check4convergence(self):
        '''
        Checks if the re-sampling routine may be terminated, because the number of most frequent clusters remains almost constant.
        This is done by examining a plot of the amount of most frequent clusters vs. the number of iterations.
        Convergence is declared once the median normalized slope in a given window of iterations is equal or below "iter_tol".
        For further information see Supplementary Material of the corresponding publication.

        :rtype: boolean
        '''
        converged          = False
        sorted_iter2nfreqs = sorted( self[ 'Convergence determination - iteration 2 n_mostfreq' ].items() )
        iter_step          = self[ 'Convergence determination - params' ][ 'iter_step' ]
        iter_window        = self[ 'Convergence determination - params' ][ 'iter_window' ]
        iter_tol           = self[ 'Convergence determination - params' ][ 'iter_tol' ]
        # determine normalized slope:
        norm_slopes        = []
        for i, ( iteration, n_mostfreq ) in enumerate( sorted_iter2nfreqs ):
            if i == 0:
                continue
            n_mostfreq_before = sorted_iter2nfreqs[ i - 1 ][ 1 ]
            norm_slope = self._get_normalized_slope( y2 = n_mostfreq, y1 = n_mostfreq_before, iter_step = iter_step )
            norm_slopes.append( norm_slope )
        # determine convergence - is the median of normalized slopes in iter_window iterations <= iter_tol?
        n_slopes = int( round( float( iter_window ) / iter_step ) ) # prepare for sliding window
        for i in range( len( norm_slopes ) - n_slopes + 1 ):
            iteration                = iter_step + iter_step * n_slopes + i * iter_step
            slopes_in_sliding_window = norm_slopes[ i : i + n_slopes ]
            median_slope             = self.median( slopes_in_sliding_window )
            if -iter_tol <= median_slope <= iter_tol:
                converged = True
                self._print( '\npotentially converged. Check convergence plot!', file = sys.stderr, verbosity_level = 2 )
                self[ 'Convergence determination - first detected at iteration' ] = iteration
                break
        return converged

    def convergence_plot(self, filename = 'convergence_plot.pdf'):
        '''
        Creates a two-sided PDF file containing the full picture of the convergence plot, as well as a zoom of it.
        The convergence plot illustrates the development of the amount of most frequent clusters vs. the number of iterations.
        The dotted line in this plots represents the normalized slope, which is used for internal convergence determination.

        If rpy2 cannot be imported, a CSV file is created instead.

        :param filename: the filename of the PDF (or CSV) file.
        :type filename: string

        :rtype: none
        '''
        try:
            from rpy2.robjects import IntVector, FloatVector, StrVector
            from rpy2.robjects.packages import importr
            graphics = importr( 'graphics' )
            grdevices = importr( 'grDevices' )
        except ImportError:
            filename = filename.replace( '.pdf', '.csv' )
            with open( os.path.join( self[ 'Working directory' ], filename ), 'w' ) as fout:
                print( 'iteration,amount of most frequent clusters', file = fout )
                for iteration, n_mostfreq in self[ 'Convergence determination - iteration 2 n_mostfreq' ].items():
                    print( '{0},{1}'.format( iteration, n_mostfreq ), file = fout )
            self._print( '[ INFO ] Since rpy2 could not be imported, a CSV file instead of a PDF plot of convergence was created. See in "{0}".'.format( os.path.join( self[ 'Working directory' ], filename ) ), file = sys.stderr, verbosity_level = 1 )
            return

        def _add_lines( points2connect, lty = 1, color = 'black' ):
            for i, ( x, y ) in enumerate( points2connect ):
                if i == 0:
                    continue
                x_before, y_before = points2connect[ i - 1 ]
                graphics.lines( IntVector( [ x_before, x ] ),
                                FloatVector( [ y_before, y ] ),
                                lty = lty,
                                col = color
                )

        iter_step           = self[ 'Convergence determination - params' ][ 'iter_step' ]
        iter_window         = self[ 'Convergence determination - params' ][ 'iter_window' ]
        iter_tol            = self[ 'Convergence determination - params' ][ 'iter_tol' ]
        iteration2mostfreq  = self[ 'Convergence determination - iteration 2 n_mostfreq' ]
        sorted_iter2mostfreq = sorted( iteration2mostfreq.items() )

        # plot convergence curve:
        grdevices.pdf( file = os.path.join( self[ 'Working directory' ], filename ), width = 12, height = 12 )
        for tag in [ 'full', 'zoom' ]:
            points = sorted_iter2mostfreq
            Ys = [ y for x, y in points ]
            if tag == 'full':
                ylim = ( min( Ys ), max( Ys ) )
                title = '#most_freq (left y-axis) and normalized_slope (= (current / before - 1.0) / iter_step) (right y-axis)'
            elif tag == 'zoom':
                ylim = ( min( Ys ), min( Ys ) * 1.075 )
                title = 'ZOOM'
            subtitle = 'iter_top_P = {0}, iter_step = {1}, iter_tol = {2}, iter_window = {4}, iter_max = {3}'
            subtitle = subtitle.format(
                self[ 'Convergence determination - params' ][ 'iter_top_P' ],
                iter_step,
                iter_tol,
                self[ 'Convergence determination - params' ][ 'iter_max' ],
                iter_window
            )
            graphics.plot(
                IntVector( [ x for x, y in points ] ),
                IntVector( Ys ),
                main = title,
                sub  = subtitle,
                xlab = 'iteration', xaxt = 'n',
                ylab = 'len(most_freq)', ylim = IntVector( ylim ),
                col  = 'black',
                pch  = 16
            )
            _add_lines( points, lty = 1, color = 'black' )
            x_axis_ticks = tuple( range( iter_step, max( iteration2mostfreq.keys() ) + 1, iter_step ) )
            graphics.axis(1, at = IntVector( x_axis_ticks ), labels = [ '{0}k'.format( tick / 1000 ) for tick in x_axis_ticks ], las = 2, **{ 'cex.axis' : 0.75 } )
            graphics.axis(3, at = IntVector( x_axis_ticks ), labels = StrVector( [ '' for tick in x_axis_ticks ] ) )
            graphics.legend(
                'bottomleft',
                legend = StrVector( [ '#most_freq', 'normalized_slope' ] ),
                lty    = IntVector( [1, 2] ),
                pch    = IntVector( [16, 1] ),
                bty    = 'n'
            )

            # add second plot = normalized_slope-plot:
            graphics.par( new = True )
            critical_interval = ( -iter_tol, iter_tol )
            try:
                firstConvergedAtIter = self[ 'Convergence determination - first detected at iteration' ]
            except KeyError:
                self.check4convergence()
                firstConvergedAtIter = self[ 'Convergence determination - first detected at iteration' ]
            iter2normslope = [ ( iter_step, -1.0 ) ]
            for i, (iteration, n_mostfreq) in enumerate( sorted_iter2mostfreq[ 1: ] ):
                iteration_before, n_mostfreq_before = sorted_iter2mostfreq[ i ] # iteration2mostfreq[ iteration - iter_step ]
                norm_slope = self._get_normalized_slope( y2 = n_mostfreq, y1 = n_mostfreq_before, iter_step = iteration - iteration_before )
                iter2normslope.append( ( iteration, norm_slope ) )
            points = iter2normslope
            Ys = [ y for x, y in points ]
            ylim = ( critical_interval[ 0 ] * 20, critical_interval[ 1 ] * 20)
            graphics.plot(
                IntVector( [ x for x, y in points ] ),
                FloatVector( Ys ),
                main = '',
                xlab = '',
                xaxt = 'n',
                ylab = '',
                yaxt = 'n',
                ylim = FloatVector( ylim ),
                pch  = 1
            )
            _add_lines( points, lty = 2, color = 'black' )
            graphics.abline( v = firstConvergedAtIter, lty = 1, col = 'blue' )
            graphics.lines( IntVector( [ firstConvergedAtIter - iter_window, firstConvergedAtIter ] ), FloatVector( [ 0, 0 ] ), col = 'darkgreen' )
            graphics.text( firstConvergedAtIter, 0, str( firstConvergedAtIter / 1000 ), col = 'blue' )
            graphics.abline( h = critical_interval[ 0 ], lty = 3, col = 'darkgreen' )
            graphics.abline( h = critical_interval[ 1 ], lty = 3, col = 'darkgreen' )
            graphics.axis(4, FloatVector( [ ylim[ 0 ], ylim[ 0 ] / 20. * 10, 0, ylim[ 1 ] / 20. * 10, ylim[ 1 ] ] ) )

        grdevices.dev_off()
        self._print( '... plot of convergence finished. See plot in "{0}".'.format( os.path.join( self[ 'Working directory' ], filename ) ), file = sys.stderr, verbosity_level = 2 )
        return

    def plot_clusterfreqs(self, min_cluster_size = 4, top_X_clusters = 0, threshold_4_the_lowest_max_freq = 0.01):
        '''
        Plot the frequencies of each cluster as a expression map:
        which cluster was found by which distance-linkage combination, and with what frequency?
        The plot's filename is prefixed by 'clusterFreqsMap', followed by the values of the parameters.
        E.g. 'clusterFreqsMap_minSize4_top0clusters_top10promille.svg'.
        Clusters are sorted by size.

        :param min_cluster_size: only clusters with a size equal or greater than min_cluster_size appear in the plot of the cluster freqs.
        :type min_cluster_size: int
        :param threshold_4_the_lowest_max_freq: ]0, 1[ Clusters must have a maximum frequency of at least threshold_4_the_lowest_max_freq to appear in the plot.
        :type threshold_4_the_lowest_max_freq: float
        :param top_X_clusters: Plot of the top X clusters in the sorted list (by freq) of clusters having a maximum cluster frequency of at least threshold_4_the_lowest_max_freq (clusterfreq-plot is still sorted by size).
        :type top_X_clusters: int

        .. note ::
            if top_X_clusters is set to zero ( 0 ), this filter is switched off (switched off by default).

        :rtype: None
        '''
        self[ 'Function parameters' ][ self.plot_clusterfreqs.__name__ ] = { k : v for k, v in locals().items() if k != 'self' }
        allClusters_sortedByLength_l = sorted( self._get_most_frequent_clusters(min_cluster_size = min_cluster_size, top_X_clusters = top_X_clusters, threshold_4_the_lowest_max_freq = threshold_4_the_lowest_max_freq), key = len, reverse = True )
        identifiers = []
        data = {}
        freqs = set()
        for cluster in allClusters_sortedByLength_l:
            identifier = 'Cluster ID: {0}, size: {1}'.format( self[ 'Cluster 2 clusterID' ][ cluster ], len( cluster ) )
            identifiers.append( identifier )
            data[ identifier ] = {}
            cFreq, cFreqDict = self.frequencies( cluster = cluster )
            for dlc, frequency in sorted( cFreqDict.items() ):
                data[ identifier ][ dlc ] = ( frequency, sys.float_info.epsilon )
                freqs.add( round( frequency, 2) )
        hm_filename = 'clusterFreqsMap_minSize{0}_top{1}clusters_top{2:.0f}promille'.format( min_cluster_size, top_X_clusters, threshold_4_the_lowest_max_freq * 1000 )
        # max_value_4_expression_map = sorted( freqs )[ -3 ] # since root cluster has a freq of 1.0, position -1 is always 1.0 (and -2 close to 1.0 (root, too)!)
        self.draw_expression_map(
            identifiers       = identifiers,
            data              = data,
            conditions        = sorted( cFreqDict.keys() ),
            additional_labels = {},
            # min_value_4_expression_map              = 0.0,
            max_value_4_expression_map              = max( freqs ),
            expression_map_filename = hm_filename+'.svg',
            legend_filename   = hm_filename+'_legend.svg',
            color_gradient    = 'barplot'
        )
        self._print( '... clusterfreqs_expressionmap saved as: "{0}"'.format( hm_filename+'.svg' ), verbosity_level = 1 )
        return

    def _get_most_frequent_clusters(self, min_cluster_size = 4, top_X_clusters = 0, threshold_4_the_lowest_max_freq = 0.01):
        '''
        Gets the most frequent clusters. Filters either according to a frequency-threshold or gets the top X clusters.

        .. note ::
            Each cluster has attributed X counts, or frequencies, where X = len( Distance-linkage combinations ).
            For determination of most frequent clusters, only the max( X frequencies ) matters.
            Hence, a single frequency above threshold_4_the_lowest_max_freq is sufficient to include that cluster.

        :param min_cluster_size: only clusters bigger or equal than threshold are considered; e.g. 4
        :type min_cluster_size: int

        :param threshold_4_the_lowest_max_freq: ]0, 1[ get all clusters with a max frequency above threshold, e.g. 0.01 => 1%-clusters
        :type threshold_4_the_lowest_max_freq: float

        :param top_X_clusters: get the top X clusters in the sorted list (by freq) of clusters having a maximum cluster frequency of at least threshold_4_the_lowest_max_freq.
        :type top_X_clusters: int

        .. note ::
           top_X_clusters = 0 means that this filter is switched off (switched off by default).

        :rtype: List of the most frequent clusters in ARBITRARY order.
        '''
        import numpy
        threshold_4_the_lowest_max_freq   = float( threshold_4_the_lowest_max_freq )
        top_X_clusters       = int( top_X_clusters )

        topP_count = self[ 'Iterations' ] * threshold_4_the_lowest_max_freq


        most_freq = []
        max_counts = numpy.amax( self[ 'Cluster counts' ], axis = 1 ) # get max count for each cluster
        if not top_X_clusters:
            mostfreqIDs = set( numpy.nonzero( max_counts >= topP_count )[ 0 ] )
            for cluster, clusterID in self[ 'Cluster 2 clusterID' ].iteritems():
                if len( cluster ) >= min_cluster_size:
                    if clusterID in mostfreqIDs:
                        most_freq.append( cluster )
        else: # top_X_clusters filter is requested:
            cID_mask = [ cID for c, cID in self[ 'Cluster 2 clusterID' ].iteritems() if len( c ) < min_cluster_size ]
            clusterIDs2retain = []
            for cID, _ in enumerate( max_counts >= topP_count ):
                if _:
                    if cID in cID_mask:
                        continue
                    clusterIDs2retain.append( ( max_counts[ cID ], cID ) )
            clusterIDs2retain.sort( reverse = True )
            topX_clusterIDs = set( [ cID for count, cID in clusterIDs2retain[ : top_X_clusters ] ] )
            for cluster, clusterID in self[ 'Cluster 2 clusterID' ].iteritems():
                if clusterID in topX_clusterIDs:
                    most_freq.append(cluster)

        s = '{0} clusters are found for a threshold for {1} and a min cluster len of {2}.'
        self._print( s.format( len( most_freq ), threshold_4_the_lowest_max_freq, min_cluster_size ), verbosity_level = 1 )
        return most_freq

    def plot_nodetree(self, tree_filename = 'tree.dot'):
        '''
        plot the dendrogram for the clustering of the most_frequent_clusters.
            - node label = nodeID internally used for self['Nodemap'] (not the same as clusterID!).
            - node border color is white if the node is a close2root-cluster (i.e. larger than self[ 'for IO skip clusters bigger than' ] ).
            - edge label = distance between parent and children.
            - edge - color codes:
                - black   = default; highlights child which is not a most_frequent_cluster but was created during formation of the dendrogram.
                - green   = children are connected with the root.
                - red     = highlights child which is a most_frequent_cluster.
                - yellow  = most_frequent_cluster is directly connected with the root.

        :param tree_filename: name of the Graphviz DOT file containing the dendrogram of the AHC of most frequent clusters. Best given with ".dot"-extension!
        :type tree_filename: string

        :rtype: none
        '''
        with open( os.path.join( self[ 'Working directory' ], tree_filename ), 'w' ) as fout:
            print( 'digraph "pyGCluster nodemap_of_the_clustering_of_most_freq_clusters" {', file = fout )
            node2tag = {}
            # draw nodes:
            for tag, node in enumerate( self[ 'Nodemap - binary tree' ] ):
                color = 'black'
                try:
                    label = str( self[ 'Cluster 2 clusterID' ][ tuple( sorted( set( node ) ) ) ] )
                except KeyError:
                    label = '-1'
                label = 'size={0}, id={1}'.format( len( set( node ) ), label )
                if self[ 'Root size' ] > len( set( node ) ) > self[ 'for IO skip clusters bigger than' ]:
                    color = 'white'
                print( '"{0}" [label="{1}", color = "{2}"];'.format( tag, label, color ), file = fout )
                node2tag[ node ] = tag
            # insert connecting arrows:
            for tag, parent in enumerate( self[ 'Nodemap - binary tree' ] ):
                is_root_node = False
                if len( set( parent ) ) == self[ 'Root size' ]:
                    is_root_node = True
                for child in self[ 'Nodemap - binary tree' ][ parent ][ 'children' ]:
                    color = 'black'
                    if len( self[ 'Nodemap - binary tree' ][ child ][ 'children' ] ) == 0:
                        color = 'red'
                    if is_root_node:
                        if color == 'red':
                            color = 'yellow'
                        else:
                            color = 'green'
                    print( '"{0}" -> "{1}" [color="{2}"];'.format( tag, node2tag[ child ], color ), file = fout )
            print( '}', file = fout )
        # plot tree:
        try:
            input_file  = '{0}'.format( os.path.join( self[ 'Working directory' ], tree_filename ) )
            output_file = '{0}'.format( os.path.join( self[ 'Working directory' ], '{0}.pdf'.format( tree_filename[ :-4 ] ) ) )
            subprocess.Popen( [ 'dot', '-Tpdf', input_file, '-o', output_file ] ).communicate()
        except:
            self._print( '[ INFO ] plotting via "dot -Tpdf ..." of the binary cluster-tree failed; only DOT file created.', verbosity_level = 1 )
        return

    def calculate_distance_matrix(self, clusters, min_overlap = 0.25):
        '''
        Calculates the specifically developed distance matrix for the AHC of clusters:
            (1) Clusters sharing *not* the minimum overlap are attributed a distance of "self[ 'Root size' ]" (i.e. len( self[ 'Data' ] ) ).
            (2) Clusters are attributed a distance of "self[ 'Root size' ] - 1" to the root cluster.
            (3) Clusters sharing the minimum overlap are attributed a distance of "size of the larger of the two clusters minus size of the overlap".

        The overlap betweeen a pair of clusters is relative, i.e. defined as the size of the overlap divided by the size of the larger of the two clusters.

        The resulting condensed distance matrix in not returned, but rather stored in self[ 'Nodemap - condensed distance matrix' ].

        :param clusters: the most frequent clusters whose "distance" is to be determined.
        :type clusters: list of clusters. Clusters are represented as tuples consisting of their object's indices.
        :param min_overlap: ]0, 1[ threshold value to determine if the distance between two clusters is calculated according to (1) or (3).
        :type min_overlap: float

        :rtype: none
        '''
        self._print( 'calculating distance matrix for {0} clusters ...'.format( len( clusters ) ) , end = ' ', verbosity_level = 2 )
        condensed_dist_matrix = []
        a, b = 1, 1
        clusters = [ set( c ) for c in clusters ]
        for clusterI, clusterJ in itertools.combinations( clusters, 2 ):
            if len( clusterI ) == self[ 'Root size' ] or len( clusterJ ) == self[ 'Root size' ]:
                dist = a * self[ 'Root size' ] - b
            else:
                overlap = clusterI & clusterJ
                n_overlap = float( len( overlap ) )
                n_sizeI = float( len( clusterI ) )
                n_sizeJ = float( len( clusterJ ) )
                if n_sizeI > n_sizeJ:
                    max_size = n_sizeI
                    min_size = n_sizeJ
                else:
                    max_size = n_sizeJ
                    min_size = n_sizeI
                if float( n_overlap ) / float( max_size ) < min_overlap:
                    dist = a * self[ 'Root size' ]
                else:
                    dist = a * max_size - b * n_overlap
            condensed_dist_matrix.append( dist )
        self[ 'Nodemap - condensed distance matrix' ] = condensed_dist_matrix
        self._print( 'done.', verbosity_level = 2 )
        return

    def _get_levelX_clusters(self, level):
        '''
        Returns a list of all clusters that are present on a specific level in the node map.
        Each level corresponds to an iteration in the community construction.

        :param level: [0, max community-iterations] sets the level (or iteration) from which the clusters are to be returned.
        :type level: int

        :rtype: list
        '''
        cluster_list = []
        for name in self[ 'Communities' ]:
            cluster, current_level = name
            if current_level == level:
                cluster_list.append( cluster )
        return sorted( cluster_list )

    def build_nodemap(self, min_cluster_size = 4, top_X_clusters = 0, threshold_4_the_lowest_max_freq = 0.01, starting_min_overlap = 0.1, increasing_min_overlap = 0.05):
        '''
        Construction of communities from a set of most_frequent_cluster.
        This set is obtained via :py:func:`pyGCluster.Cluster._get_most_frequent_clusters`, to which the first three parameters are passed.
        These clusters are then subjected to AHC with complete linkage.
        The distance matrix is calculated via :py:func:`pyGCluster.Cluster.calculate_distance_matrix`.
        The combination of complete linkage and the distance matrix assures that all clusters in a community exhibit at least the "starting_min_overlap" to each other.
        From the resulting cluster tree, a "first draft" of communities is obtained.
        These "first" communities are then themselves considered as clusters, and subjected to AHC again, until the community assignment of clusters remains constant.
        By this, clusters are inserted into a target community, which initially did not overlap with each cluster inside the target community,
        but do overlap if the clusters in the target community are combined into a single cluster.
        By this, the degree of stringency is reduced; the clusters fit into a community in a broader sense.
        For further information on the community construction, see the publication of pyGCluster.

        Internal structure of communities:
            >>> name = ( cluster, level )
            ...         # internal name of the community.
            ...         # The first element in the tuple ("cluster") contains the indices
            ...         # of the objects that comprise a community.
            ...         # The second element gives the level,
            ...         # or iteration when the community was formed.
            >>> self[ 'Communities' ][ name ][ 'children' ]
            ...         # list containing the clusters that build the community.
            >>> self[ 'Communities' ][ name ][ '# of nodes merged into community' ]
            ...         # the number of clusters that build the community.
            >>> self[ 'Communities' ][ name ][ 'index 2 obCoFreq dict' ]
            ...         # an OrderedDict in which each index is assigned its obCoFreq.
            ...         # Negative indices correspond to "placeholders",
            ...         # which are required for the insertion of black lines into expression maps.
            ...         # Black lines in expression maps seperate the individual clusters
            ...         # that form a community, sorted by when
            ...         # they were inserted into the community.
            >>> self[ 'Communities' ][ name ][ 'highest obCoFreq' ]
            ...         # the highest obCoFreq encountered in a community.
            >>> self[ 'Communities' ][ name ][ 'cluster ID' ]
            ...         # the ID of the cluster containing the object with the highest obCoFreq.

        Of the following parameters, the first three are passed to :py:func:`pyGCluster.Cluster._get_most_frequent_clusters`:

        :param min_cluster_size: clusters smaller than this threshold are not considered for the community construction.
        :type min_cluster_size: int
        :param top_X_clusters: form communities from the top X clusters sorted by their maximum frequency.
        :type top_X_clusters: int
        :param threshold_4_the_lowest_max_freq: [0, 1[ form communities from clusters whose maximum frequency is at least this value.
        :type threshold_4_the_lowest_max_freq: float

        :param starting_min_overlap: ]0, 1[ minimum required relative overlap between clusters so that they are assigned the same community. The relative overlap is defined as the size of the overlap between two clusters, divided by the size of the larger cluster.
        :type starting_min_overlap: float

        :param increasing_min_overlap: defines the increase of the required overlap between communities
        :type increasing_min_overlap: float

        :rtype: none
        '''
        self[ 'Function parameters' ][ self.build_nodemap.__name__ ] = { k : v for k, v in locals().items() if k != 'self' }
        import scipy.spatial.distance as ssd
        imported_from_scipy = False
        try:
            from fastcluster import linkage as ahc
        except ImportError:
            try:
                from scipy.cluster.hierarchy import linkage as ahc
                imported_from_scipy = True
            except ImportError:
                self._print( '[ ERROR ] You do require either "fastcluster" or "scipy" for the construction of communities.', verbosity_level = 0 )

        # The algorithm is as follows:
        # Starting from the top, all descendants of any cluster that is smaller than the root are determined.
        # Those descendants form a community.
        def communities_by_ahc(cluster_list, min_overlap):
            # calculate distance matrix
            self.calculate_distance_matrix( clusters = cluster_list, min_overlap = min_overlap )
            # perform AHC
            self._print( 'performing AHC for {0} clusters ...'.format( len( cluster_list ) ), end = ' ', verbosity_level = 2 )
            # avoid scipy crash when only 2 objects are subjected to AHC:
            if len( self[ 'Nodemap - condensed distance matrix' ] ) == 1 and len( cluster_list ) == 2:
                self[ 'Nodemap - linkage matrix' ] = [ [ 0, 1, -99, len( set( cluster_list[ 0 ] + cluster_list[ 1 ] ) ) ] ]
            else:
                if imported_from_scipy:
                    self[ 'Nodemap - linkage matrix' ]  = ahc( self[ 'Nodemap - condensed distance matrix' ], method = 'complete' )
                else:
                    self[ 'Nodemap - linkage matrix' ]  = ahc( self[ 'Nodemap - condensed distance matrix' ], method = 'complete', preserve_input = True )
            self._print( 'done.', verbosity_level = 2 )
            # parse clusters
            self._print( 'parsing clusters ...', end = ' ', verbosity_level = 2 )
            clusters    = {} # required to reconstruct the clusters from the linkage matrix
            nodemap     = {} # each node = value is a dict with two keys: 'parent' -> parent cluster (as tuple), 'children' -> set of child clusters (tuples)
            for i, cluster in enumerate( cluster_list ):
                clusters[ i ] = cluster
                nodemap[ cluster ] = { 'children' : [], 'parent' : None }
            parentID = len( cluster_list ) - 1
            for childID_1, childID_2, distance, size in self[ 'Nodemap - linkage matrix' ]:
                parentID += 1
                child1 = clusters[ childID_1 ]
                child2 = clusters[ childID_2 ]
                parent = child1 + child2
                clusters[ parentID ] = parent
                nodemap[ child1 ][ 'parent' ] = parent
                nodemap[ child2 ][ 'parent' ] = parent
                nodemap[ parent ] = { 'children' : [ child1, child2 ], 'parent' : None }
            self[ 'Nodemap - binary tree' ] = nodemap
            self._print( 'done.', verbosity_level = 2 )

            # recursive function 2 find communities:
            def get_communities( node , community_list = None ):
                if community_list == None:
                    community_list = []
                for child in nodemap[ node ][ 'children' ]:
                    if len( set( child ) ) == self[ 'Root size' ]:
                        community_list = get_communities( child,  community_list = community_list )
                    else:
                        community_list.append( self._get_descendants_in_binary_tree( node = child ) + [ child ] )
                return community_list

            # get root_node = top node of the tree:
            for node in nodemap:
                if nodemap[ node ][ 'parent' ] == None:
                    root_node = node
                    break
            community_list = get_communities( node = root_node, community_list = None )

            clusters_combined_into_communities = []
            for community in community_list:
                endnodes = []
                for cluster in community:
                    if cluster in cluster_list:
                        endnodes.append( cluster )
                clusters_combined_into_communities.append( endnodes )
            return clusters_combined_into_communities

        def update_communities(level, clusters_combined_into_communities):
            for community in clusters_combined_into_communities:
                # find cluster with highest freq
                community_obCoFreq2cluster_list = []
                community_indices = set()
                for cluster in community:
                    highest_obCoFreq = self[ 'Communities' ][ ( cluster, level ) ][ 'highest obCoFreq' ]
                    community_obCoFreq2cluster_list.append( ( highest_obCoFreq, cluster ) )
                    community_indices |= set( cluster )
                community_obCoFreq2cluster_list.sort( reverse = True )
                first_cluster = community_obCoFreq2cluster_list[ 0 ][ 1 ]
                name = ( tuple( sorted( community_indices ) ), level + 1 )
                if name in self[ 'Communities' ]:
                    current_highest_obCoFreq = community_obCoFreq2cluster_list[ 0 ][ 0 ]
                    if current_highest_obCoFreq > self[ 'Communities' ][ name ][ 'highest obCoFreq' ]:
                        self[ 'Communities' ][ name ][ 'cluster ID' ] = self[ 'Communities' ][ ( first_cluster, level ) ][ 'cluster ID' ]
                    community_obCoFreq2cluster_list.insert( 0, None ) # assure that the first cluster is also properly inserted
                else:
                    self[ 'Communities' ][ name ] = {}
                    self[ 'Communities' ][ name ][ 'children' ] = [ first_cluster ]
                    self[ 'Communities' ][ name ][ 'index 2 obCoFreq dict' ] = self[ 'Communities' ][ ( first_cluster, level ) ][ 'index 2 obCoFreq dict' ].copy()
                    self[ 'Communities' ][ name ][ 'cluster ID' ] = self[ 'Communities' ][ ( first_cluster, level ) ][ 'cluster ID' ]
                    self[ 'Communities' ][ name ][ 'highest obCoFreq' ] = None
                    self[ 'Communities' ][ name ][ '# of nodes merged into community' ] = self[ 'Communities' ][ ( first_cluster, level ) ][ '# of nodes merged into community' ]
                self[ 'Communities' ][ name ][ '# of nodes merged into community' ] += len( community_obCoFreq2cluster_list ) - 1.
                # insert children and update obCoFreq-Dict:
                for _, cluster in community_obCoFreq2cluster_list[ 1 : ]:
                    self[ 'Communities' ][ name ][ 'children' ].append( cluster )
                    placeholder_added = False
                    for index in cluster:
                        obCoFreq = self[ 'Communities' ][ ( cluster, level ) ][ 'index 2 obCoFreq dict' ][ index ]
                        if index in self[ 'Communities' ][ name ][ 'index 2 obCoFreq dict' ]:
                            self[ 'Communities' ][ name ][ 'index 2 obCoFreq dict' ][ index ] += obCoFreq
                        else:
                            if not placeholder_added:
                                placeholder = len( self[ 'Communities' ][ name ][ 'index 2 obCoFreq dict' ] ) * -1
                                self[ 'Communities' ][ name ][ 'index 2 obCoFreq dict' ][ placeholder ] = -99
                                placeholder_added = True
                            self[ 'Communities' ][ name ][ 'index 2 obCoFreq dict' ][ index ] = obCoFreq
                max_freq = max( self[ 'Communities' ][ name ][ 'index 2 obCoFreq dict' ].values() )
                self[ 'Communities' ][ name ][ 'highest obCoFreq' ] = max_freq
            return

        def init_cluster2community0_level():
            most_frequent_clusters = self._get_most_frequent_clusters( min_cluster_size = min_cluster_size, top_X_clusters = top_X_clusters, threshold_4_the_lowest_max_freq = threshold_4_the_lowest_max_freq )
            level = 0
            self[ 'Communities' ] = {}
            for cluster in sorted( most_frequent_clusters ):
                index2obCoFreq = OrderedDict()
                cFreq, cFreqDict = self.frequencies( cluster = cluster )
                for index in cluster:
                    index2obCoFreq[ index ] = cFreq
                max_freq = cFreq # max_freq = max( index2obCoFreq.values() ) = cFreq, because the indices are only from a single cluster at level 0
                name = ( cluster, level )
                self[ 'Communities' ][ name ]                                           = {}
                self[ 'Communities' ][ name ][ 'children' ]                             = []
                self[ 'Communities' ][ name ][ 'index 2 obCoFreq dict' ]                = index2obCoFreq
                self[ 'Communities' ][ name ][ 'highest obCoFreq' ]                     = max_freq
                self[ 'Communities' ][ name ][ 'cluster ID' ]                           = self[ 'Cluster 2 clusterID' ][ cluster ]
                self[ 'Communities' ][ name ][ '# of nodes merged into community' ]     = 1.
            return

        min_overlap = starting_min_overlap
        init_cluster2community0_level()
        level              = 0
        community_snapshot = None
        while True:
            cluster_list                       = self._get_levelX_clusters( level = level )
            clusters_combined_into_communities = communities_by_ahc( cluster_list, min_overlap )
            if community_snapshot == sorted( clusters_combined_into_communities ) or min_overlap >= 1.0:
                break
            self.plot_nodetree( 'AHCofClusters_binaryTree_iteration{0}.dot'.format(level) )
            update_communities( level = level, clusters_combined_into_communities = clusters_combined_into_communities )
            community_snapshot = sorted( clusters_combined_into_communities )
            min_overlap       += increasing_min_overlap
            level += 1
        return

    def _get_descendants_in_binary_tree(self, node, children = None):
        '''
        Recursively determines the descendants of a given node in an AHC tree.

        :param node: tuple describing uniquely a node in the AHC tree. It resembles a pyGCluster-cluster, but may contain the same index several times, e.g. if (1,2) and (1,3) are merged into (1,1,2,3).
        :type node: tuple
        :param children: all descendants determined so far. Should equal None for the first call.
        :type children: list

        :rtype: list
        '''
        if children == None:
            children = []
        if len( self[ 'Nodemap - binary tree' ][ node ][ 'children' ] ) > 0:
            for child in self[ 'Nodemap - binary tree' ][ node ][ 'children' ]:
                children.append( child )
                self._get_descendants_in_binary_tree( node = child, children = children )
        return children

    def _get_descendants_in_community_tree(self, parent_name, children = None):
        '''
        Recursively determines the descendants of a given node in the community tree.
        In contrast to :py:func:`pyGCluster.Cluster._get_descendants_in_binary_tree` , the community tree is not a binary tree;
        and "parent_name" differs from the "node"-parameter of the former (see below).

        :param parent_name: tuple with two elements: ( cluster, level ). Here, cluster is a pyGCluster-cluster, i.e. a tuple containing each index describing a cluster only once.
        :type parent_name: tuple
        :param children: all descendants determined so far. Should equal None for the first call.
        :type children: list

        :rtype: list
        '''
        if children == None:
            children = []
        if len( self[ 'Communities' ][ parent_name ][ 'children' ] ) > 0:
            parent, level = parent_name
            for child in self[ 'Communities' ][ parent_name ][ 'children' ]:
                child_name = ( child, level - 1 )
                children.append( child_name )
                self._get_descendants_in_community_tree( parent_name = child_name, children = children )
        return children

    def create_rainbow_colors( self, n_colors = 10):
        '''
        Returns a list of rainbow colors. Colors are expressed as hexcodes of RGB values.

        :param n_colors: number of rainbow colors.
        :type n_colors: int

        :rtype: list
        '''
        import colorsys
        colors = []
        for i in range( n_colors ):
            # i has to be [0.0, 1.0[
            i /= float( n_colors )
            rgb = [ int(value) for value in colorsys.hsv_to_rgb(i, 1, 255) ]
            hexcode = '#'
            for _ in rgb:
                _hex = hex(_)[2:]
                if len(_hex) == 1:
                    _hex = '0{}'.format(_hex.upper())
                else:
                    _hex = '{}'.format(_hex.upper())
                hexcode += _hex
            colors.append(hexcode)
        return colors

    def write_legend(self, filename = 'legend.txt'):
        '''
        Creates a legend for the community node map as a TXT file.
        Herein, the object composition of each cluster of the node map as well as its frequencies are recorded.
        Since this function is internally called by :py:func:`pyGCluster.Cluster.write_dot`, it is typically not necessary to call this function.

        :param filename: name of the legend TXT file, best given with extension ".txt".
        :type filename: string

        :rtype: none
        '''
        with open(  os.path.join( self[ 'Working directory' ] , filename ), 'w') as legend:
            print( "Frequency order:\n{0}\n".format( ', '.join( sorted( self[ 'Distance-linkage combinations' ] ) ) ), file = legend )
            for name in self[ 'Communities' ]:
                cluster, level = name
                if len( cluster ) > self[ 'for IO skip clusters bigger than' ]:
                    continue
                if cluster in self[ 'Cluster 2 clusterID' ]:
                    cFreq, cFreqDict = self.frequencies( cluster = cluster )
                else:
                    cFreqDict = { None : -99 }
                nodeID = '{0}, {1}'.format( self[ 'Communities' ][ name ][ 'cluster ID' ], level )
                print( 'label = "{nodeID:0>3}", size = {size:0>3}, frequencies = {frequencies}'.format(
                    nodeID = nodeID,
                    size = len( cluster ),
                    frequencies = ', '.join( [ '{0:5.4f}'.format( f ) for method, f in sorted( cFreqDict.items() ) ] )
                    ), file = legend
                )
                for index in cluster:
                    addOn = ''
                    try:
                        addOn = self[ 'Additional Labels' ][ self[ 'Identifiers' ][ index ] ]
                        if type(addOn) == type(list()) or type(addOn) == type(set()):
                            addOn = ".oOo.".join(list(set(addOn)))
                    except:
                        pass
                    print( '{0}\t{1}'.format( self[ 'Identifiers' ][ index ], addOn ), file = legend )
                print( '+' * 50 , file = legend )
        self._print( '... nodemap saved in "{0}"'.format( self[ 'Working directory' ] ), verbosity_level = 2 )
        return

    def write_dot(self, filename , scaleByFreq = True, min_obcofreq_2_plot = None, n_legend_nodes = 5, min_value_4_expression_map = None, max_value_4_expression_map = None, color_gradient = '1337', box_style = 'classic'):
        '''
        Writes a Graphviz DOT file representing the cluster composition of communities.
        Herein, each node represents a cluster. Its name is a combination of the cluster's ID, followed by the level / iteration it was inserted into the community:

            - The node's size reflects the cluster's cFreq.
            - The node's shape illustrates by which distance metric the cluster was found (if the shape is a point, this illustrates that this cluster was not among the most_frequent_clusters, but only formed during AHC of clusters).
            - The node's color shows the community membership; except for clusters which are larger than self[ 'for IO skip clusters bigger than' ], those are highlighted in grey.
            - The node connecting all clusters is the root (the cluster holding all objects), which is highlighted in white.

        The DOT file may be rendered with "Graphviz" or further processed with other appropriate programs such as e.g. "Gephi".
        If "Graphviz" is available, the DOT file is eventually rendered with "Graphviz"'s dot-algorithm.

        In addition, a expression map for each cluster of the node map is created (via :py:func:`pyGCluster.Cluster.draw_expression_map_for_community_cluster`).

        Those are saved in the sub-folder "communityClusters".

        This function also calls :py:func:`pyGCluster.Cluster.write_legend`,
        which creates a TXT file containing the object composition of all clusters, as well as their frequencies.

        :param filename: file name of the Graphviz DOT file representing the node map, best given with extension ".dot".
        :type filename: string
        :param scaleByFreq: switch to either scale nodes (= clusters) by cFreq or apply a constant size to each node (the latter may be useful to put emphasis on the nodes' shapes).
        :type scaleByFreq: boolean
        :param min_obcofreq_2_plot: if defined, clusters with lower cFreq than this value are skipped, i.e. not plotted.
        :type min_obcofreq_2_plot: float
        :param n_legend_nodes: number of nodes representing the legend for the node sizes. The node sizes themselves encode for the cFreq. "Legend nodes" are drawn as grey boxes.
        :type n_legend_nodes: int

        :param min_value_4_expression_map: lower bound for color coding of values in the expression map. Remember that log2-values are expected, i.e. this value should be < 0.
        :type min_value_4_expression_map: float
        :param max_value_4_expression_map: upper bound for color coding of values in the expression map.
        :type max_value_4_expression_map: float
        :param color_gradient: name of the color gradient used for plotting the expression map.
        :type color_gradient: string
        :param box_style: the way the relative standard deviation is visualized in the expression map. Currently supported are 'modern', 'fusion' or 'classic'.
        :type box_style: string

        :rtype: none
        '''
        self[ 'Function parameters' ][ self.write_dot.__name__ ] = { k : v for k, v in locals().items() if k != 'self' }
        import numpy
        node_templateString = '"{nodeID}" [label="{label}", color="{color}", shape="{shape}", width="{freq}", height="{freq}", fixedsize=true, community={community}, c_members={c_members}, metrix="{metrix}", normalized_max_obCoFreq="{normalized_max_obCoFreq}"];'
        node_templateString_dict = {}
        edge_templateString = '"{parent}" -> "{child}" [color="{color}", arrowsize=2.0];'
        edge_templateString_dict = {}

        if 'Communities' not in self.keys():
            self._print( 'function "build_nodemap()" was not called prior. Building node map with default settings ...', verbosity_level = 0 )
            self.build_nodemap()

        most_frequent_clusters_used_4_nodemap = set( self._get_levelX_clusters( level = 0 ) )

        # assign each distance metric combo a specific shape (if possible):
        metrix2shape = {}
        n_metrices = len( self[ 'Distances' ] )
        if len( self[ 'Distances' ] ) > 3:
            self._print( '[ INFO ] more distance metrics than shapes! All shapes equal "ellipse".', verbosity_level = 1 )
            shapes = [ 'ellipse' for i in range( n_metrices ** n_metrices ) ]
        else:
            shapes = [ 'box', 'ellipse', 'triangle', 'diamond', 'octagon', 'invtriangle', 'invtrapezium' ]
        for i in range( 1, n_metrices + 1 ):
            for metric_combo in itertools.combinations( self[ 'Distances' ] , i ):
                metrix2shape[ ' + '.join( sorted( metric_combo ) ) ] = shapes.pop( 0 )
        self._print( 'metric 2 shape:', metrix2shape , verbosity_level = 1 )
        self[ 'nodemap metric2shape' ] = metrix2shape
        # determine max obCoFreq for proper node scaling:
        sorted_obCoFreqs = sorted( [ self[ 'Communities' ][ name ][ 'highest obCoFreq' ] for name in self[ 'Communities' ] ] )
        max_obCoFreq = float( sorted_obCoFreqs[ -2 ] ) # sorted_obCoFreqs[ -1 ] == root, hence max_obCoFreq would always be cFreq(root) == 2.0!
        # get top, i.e. largest cluster of each community:
        max_level = max( [ name[1] for name in self[ 'Communities' ] ] )
        communities_top_cluster = self._get_levelX_clusters( level = max_level )
        # set colors:
        communities_minus_close2root = [ c for c in  communities_top_cluster if len( c ) < self[ 'for IO skip clusters bigger than' ] ]
        community_colors = self.create_rainbow_colors( n_colors = len( communities_minus_close2root ) )
        name2community_and_color = {}
        for communityID, cluster in enumerate( communities_top_cluster ):
            if cluster in communities_minus_close2root:
                color = community_colors.pop( 0 )
            else:
                color = '#BEBEBE'
            name = ( cluster, max_level )
            communityID_color = ( communityID, color )
            name2community_and_color[ name ] = communityID_color
            for child_name in self._get_descendants_in_community_tree( parent_name = name ):
                name2community_and_color[ child_name ] = communityID_color

        # filter nodes by min_obcofreq_2_plot, and build 'name2nodeID'-dict:
        name2nodeID = {}
        skipped_nodes = set()
        skipped_nodes.add( ( self[ 'Root' ], 0 ) )
        for name in self[ 'Communities' ]:
            name2nodeID[ name ] = len( name2nodeID )
            if min_obcofreq_2_plot > self[ 'Communities' ][ name ][ 'highest obCoFreq' ]:
                community, level = name
                if max_level > level: # prevent that communities are lost if community freq < min_obcofreq_2_plot
                    skipped_nodes.add( name )

        ### write dot file:
        dot_filename = os.path.join( self[ 'Working directory' ], filename )
        with open( dot_filename, 'w' )  as dot:
            ## initialize DOT file:
            print( 'digraph "pyGCluster nodemap" {', file = dot )
            print( 'graph [overlap=Prism, splines=true, ranksep=5.0, nodesep=0.75];', file = dot )
            print( 'node [style=filled]', file = dot )
            scale_factor = 5.
            ## draw nodes:
            for level in range( max_level + 1 ):
                for cluster in self._get_levelX_clusters( level ):
                    name = ( cluster, level )
                    if name in skipped_nodes:
                        continue
                    # draw expression map:
                    if len( cluster ) <= self[ 'for IO skip clusters bigger than' ]:
                        self.draw_expression_map_for_community_cluster( name, min_value_4_expression_map = min_value_4_expression_map, max_value_4_expression_map = max_value_4_expression_map, color_gradient = color_gradient , sub_folder = 'communityClusters', box_style = box_style )

                    node_templateString_dict[ 'nodeID' ] = name2nodeID[ name ]
                    # scale node size:
                    if scaleByFreq:
                        normalized_obCoFreq = self[ 'Communities' ][ name ][ 'highest obCoFreq' ]
                        width = normalized_obCoFreq / max_obCoFreq * scale_factor
                    else:
                        width = 2.5
                    node_templateString_dict[ 'freq' ] = width
                    node_templateString_dict[ 'label' ] = '{0}-{1}'.format( self[ 'Communities' ][ name ][ 'cluster ID' ], level )
                    # determine shape:
                    if cluster in most_frequent_clusters_used_4_nodemap:
                        clusterID = self[ 'Cluster 2 clusterID' ][ cluster ]
                        distances = set()
                        for i in numpy.nonzero( self[ 'Cluster counts' ][ clusterID ] > 0 )[ 0 ]:
                            distance, linkage = self[ 'Distance-linkage combinations' ][ i ].split( '-' )
                            distances.add( distance )
                        distances = ' + '.join( sorted( distances ) )
                        node_templateString_dict[ 'metrix' ] = distances
                        node_templateString_dict[ 'shape' ] = metrix2shape[ distances ]
                    else:
                        node_templateString_dict[ 'metrix' ] = 'None'
                        node_templateString_dict[ 'shape' ] = 'point'
                    # store the cluster's size (in terms of objects describing it), set color and community ID:
                    node_templateString_dict[ 'c_members' ] = len( cluster )
                    communityID, community_color = name2community_and_color[ name ]
                    node_templateString_dict[ 'color' ] = community_color
                    node_templateString_dict[ 'community' ] = communityID
                    node_templateString_dict[ 'normalized_max_obCoFreq' ] = self[ 'Communities' ][ name ][ 'highest obCoFreq' ]
                    # finally insert node into dot-file:
                    print( node_templateString.format( **node_templateString_dict ), file = dot )
            ## insert edges:
            for level in range( 1, max_level + 1 ):
                for parent in self._get_levelX_clusters( level ):
                    parent_name  = ( parent, level )
                    edge_templateString_dict[ 'parent' ] = name2nodeID[ parent_name ]
                    edge_templateString_dict[ 'color' ] = name2community_and_color[ parent_name ][ 1 ]
                    for child in self[ 'Communities' ][ parent_name ][ 'children' ]:
                        child_name  = ( child, level - 1 )
                        if child_name in skipped_nodes:
                            continue
                        edge_templateString_dict[ 'child' ] = name2nodeID[ child_name ]
                        # nut to break: child without direct parent ...
                        if parent_name in skipped_nodes:
                            # find largest parent:
                            communityID, _ = name2community_and_color[ child_name ]
                            # get all community clusters which are attributed the current communityID:
                            community_names = set()
                            for name in name2community_and_color:
                                ID, _ = name2community_and_color[ name ]
                                if ID == communityID and name != child_name and name[ 1 ] > child_name[ 1 ]:
                                    community_names.add( name )
                            # Of those, extract clusters which are NOT to be skipped:
                            potential_parents = community_names - skipped_nodes
                            # get parent with lowest level:
                            min_level = max_level
                            for potential_parent_name in potential_parents:
                                parent, _level = potential_parent_name
                                if min_level > _level:
                                    min_level = _level
                            for potential_parent_name in potential_parents:
                                parent, _level = potential_parent_name
                                if _level == min_level:
                                    edge_templateString_dict[ 'parent' ] = name2nodeID[ potential_parent_name ]
                                    break
                        print( edge_templateString.format( **edge_templateString_dict ), file = dot )

            ## connect largest cluster of each community with root:
            root_name = ( self[ 'Root' ], -1 )
            name2nodeID[ root_name ] = len( name2nodeID )
            node_templateString_dict = {
                'nodeID' : name2nodeID[ root_name ],
                'freq' : scale_factor,
                'label' : 'ROOT',
                'c_members' : self[ 'Root size' ],
                'community' : -1,
                'color' : '#FFFFFF',
                'metrix' : 'ALL',
                'shape' : 'ellipse',
                'normalized_max_obCoFreq' : '-99'
            }
            print( node_templateString.format( **node_templateString_dict ), file = dot )
            edge_templateString_dict = { 'parent' : name2nodeID[ root_name ], 'color' : '#000000' }
            for cluster in communities_top_cluster:
                cluster_name = ( cluster, max_level )
                edge_templateString_dict[ 'child' ] = name2nodeID[ cluster_name ]
                print( edge_templateString.format( **edge_templateString_dict ), file = dot )

            ## add legend for the node size as additional, grey, boxed-sized nodes:
            for i in range( 1, n_legend_nodes + 1 ):
                f =  max_obCoFreq * ( i / float( n_legend_nodes ) ) / max_obCoFreq
                node_templateString_dict = {
                    'nodeID' : 'legend_node_{0}'.format( i ),
                    'freq' : f * scale_factor,
                    'label' : round( f, 4 ),
                    'c_members' : -1,
                    'community' : -1,
                    'color' : '#BEBEBE',
                    'metrix' : 'None',
                    'shape' : 'box',
                    'normalized_max_obCoFreq' : '-99'
                }
                print( node_templateString.format( **node_templateString_dict ), file = dot)
            for i in range( 1, n_legend_nodes ):
                edge_templateString_dict = { 'parent' : 'legend_node_{0}'.format( i ), 'child' : 'legend_node_{0}'.format( i + 1 ), 'color' : '#BEBEBE' }
                print( edge_templateString.format( **edge_templateString_dict ), file = dot )
            ## finalize DOT file:
            print( '}', file = dot )

        self.write_legend(filename = '{0}__legend.txt'.format(filename[:-4]))

        try:
            rendered_filename = os.path.join( self[ 'Working directory' ], '{0}.pdf'.format( filename[ : -4 ] ) )
            out, err = subprocess.Popen( [ 'dot', '-Tpdf', dot_filename, '-o', rendered_filename ], stdout = subprocess.PIPE, stderr = subprocess.PIPE ).communicate()
        except:
            self._print( '[ INFO ] only DOT file created, renderering with Graphviz failed.', verbosity_level = 1 )
        return

    def frequencies(self, identifier = None, clusterID = None, cluster = None):
        '''
        Returns a tuple with (i) the cFreq and (ii) a Collections.DefaultDict containing the DLC:frequency pairs for either
        an identifier, e.g. "JGI4|Chlre4|123456"
        or clusterID
        or cluster.
        Returns 'None' if the identifier is not part of the data set, or clusterID or cluster was not found during iterations.

        Example:

            >>> cFreq, dlc_freq_dict = cluster.frequencies( identifier = 'JGI4|Chlre4|123456' )
            >>> dlc_freq_dict
            ... defaultdict(<type 'float'>,
            ... {'average-correlation': 0.0, 'complete-correlation': 0.0,
            ... 'centroid-euclidean': 0.0015, 'median-euclidean': 0.0064666666666666666,
            ... 'ward-euclidean': 0.0041333333333333335, 'weighted-correlation': 0.0,
            ... 'complete-euclidean': 0.0014, 'weighted-euclidean': 0.0066333333333333331,
            ... 'average-euclidean': 0.0020333333333333332})

        :param identifier: search frequencies by identifier input
        :type identifier: string
        :param clusterID: search frequencies by cluster ID input
        :type clusterID: int
        :param cluster: search frequencies by cluster (tuple of ints) input
        :type cluster: tuple

        :rtype: tuple
        '''
        if identifier == None and clusterID == None and cluster == None:
            self._print( 'invalid call of function "frequencies": neither "identifier", "clusterID" nor "cluster" were given.\n\treturning None ...',
                file = sys.stderr,
                verbosity_level = 0
                )
            return None
        cFreqDict = ddict(float)
        if identifier != None:
            # search by identifier
            ident_index = self[ 'Identifiers' ].index( identifier )
            for cluster, clusterID in self[ 'Cluster 2 clusterID' ].iteritems():
                if ident_index in cluster:
                    for i, method in enumerate(self[ 'Distance-linkage combinations' ]):
                        freq = self[ 'Cluster counts' ][ clusterID ][ i ] / float( self[ 'Iterations' ] )
                        cFreqDict[ method ] += freq
        elif cluster != None:
            clusterID = self[ 'Cluster 2 clusterID' ][ cluster ]
        if clusterID != None:
            for i, dlc in enumerate( self[ 'Distance-linkage combinations' ] ):
                freq = self[ 'Cluster counts' ][ clusterID ][ i ] / float( self[ 'Iterations' ] )
                cFreqDict[ dlc ] = freq
        distance_freqs = { distance : [] for distance in self[ 'Distances' ] }
        for dlc, f in cFreqDict.items():
            distance, linkage = dlc.split( '-' )
            distance_freqs[ distance ].append( f )
        cFreq = sum( [ self.median( f ) for dist, f in distance_freqs.items() ] )
        return cFreq, cFreqDict

    def plot_mean_distributions(self):
        '''
        Creates a density plot of mean values for each condition via rpy2.

        :rtype: none
        '''
        try:
            import rpy2.robjects as robjects
            from rpy2.robjects import r
            from rpy2.robjects.packages import importr
            graphics = importr('graphics')
            grdevices = importr('grDevices')
        except ImportError:
            self._print( '[ WARNING ] since "rpy2" is not available (ImportError), the plot of the distribution of mean values could not be created.', verbosity_level = 0 )
            return

        grdevices.pdf( os.path.join( self[ 'Working directory' ] , 'distribution_of_means.pdf'.format(condition) ) )
        for condition in self[ 'Conditions' ]:
            means = []
            for identifier in self[ 'Data' ]:
                mean, sd = self[ 'Data' ][ identifier ][ condition ]
                means.append( mean )
            graphics.plot(
                                r.density( robjects.FloatVector( means ) ),
                                main = condition,
                                col  = 'blue',
                                xlab = 'Mean values',
                                ylab = 'Density',

            )
        grdevices.dev_off()
        return

    def draw_expression_profiles(self, min_value_4_expression_map = None, max_value_4_expression_map = None):
        '''
        Draws an expression profile plot (SVG) for each community, illustrating the main "expression pattern" of a community.
        Each line in this plot represents an object. The "grey cloud" illustrates the range of the standard deviation of the mean values.
        The plots are named prefixed by "exProf", followed by the community name as it is shown in the node map.

        :param min_value_4_expression_map: minimum of the y-axis (since data should be log2 values, this value should typically be < 0).
        :type min_value_4_expression_map: int
        :param max_value_4_expression_map: maximum for the y-axis.
        :type max_value_4_expression_map: int

        :rtype: none
        '''
        self[ 'Function parameters' ][ self.draw_expression_profiles.__name__ ] = { k : v for k, v in locals().items() if k != 'self' }
        import numpy
        FONT_SIZE = 10
        y_offset = 20
        MIN_V, MAX_V = min_value_4_expression_map, max_value_4_expression_map
        if min_value_4_expression_map == None or max_value_4_expression_map == None:
            # determine min and max for y-axis:
            _yAxisMinMax = set()
            for identifier in self[ 'Data' ]:
                for condition in self[ 'Data' ][ identifier ]:
                    mean, sd = self[ 'Data' ][ identifier ][ condition ]
                    _yAxisMinMax.add( round( mean + sd, 2 ) )
                    _yAxisMinMax.add( round( mean - sd, 2 ) )
            if min_value_4_expression_map == None:
                MIN_V = int( math.ceil( min( _yAxisMinMax ) ) ) - 1
            if max_value_4_expression_map == None:
                MAX_V = int( math.ceil( max( _yAxisMinMax ) ) )
            # give y-axis the same amount in positive and negative direction (e.g. from - 10 to 10):
            if min_value_4_expression_map == None and max_value_4_expression_map == None: # but only if no value is given, otherwise it's probably user-chosen!
                if MAX_V > abs( MIN_V ):
                    MIN_V = MAX_V * -1
                else:
                    MAX_V = MIN_V * -1
        startingX = 100
        startingY = 300 + y_offset # determine lenth of y-axis and y-range, represents zero point
        maxY = ( startingY - y_offset ) * 2
        scalingX = max( [ len( con ) * FONT_SIZE for con in self[ 'Conditions' ] ] ) + 20 # distance between each condition
        scalingY = ( maxY - ( startingY - y_offset ) ) / float( MAX_V ) * -1 # has to be negative!

        def svg_text(x, y, text):
            return '<text x="{0}" y="{1}"> {2} </text>'.format( x, y, text )

        def svg_line(x1, y1, x2, y2):
            return '<line x1="{0}" y1="{1}" x2="{2}" y2="{3}" style="stroke:#000000"/>'.format( x1, y1, x2, y2 )

        def svg_comment(text):
            return '<!-- {0} -->'.format( text )

        def min_max_ratioWithSD(ratios, SDs):
            ratios_plus_SD = [ ratio + SDs[ i ] for i, ratio in enumerate( ratios ) ]
            ratios_minus_SD = [ ratio - SDs[ i ] for i, ratio in enumerate( ratios ) ]
            return min( ratios_minus_SD ), max( ratios_plus_SD )

        n_conditions = len( self[ 'Conditions' ] )
        max_level = max( [ name[1] for name in self[ 'Communities' ] ] )
        for cluster in self._get_levelX_clusters( max_level ):
            if len( cluster ) > self[ 'for IO skip clusters bigger than' ]:
                continue
            shape = ( len( cluster ), len( self[ 'Conditions' ] ) )
            ratios = numpy.zeros( shape )
            SDs = numpy.zeros( shape )
            identifiers = []
            for row_index, identifier_index in enumerate( cluster ):
                identifier = self[ 'Identifiers' ][ identifier_index ]
                for col_index, condition in enumerate( self[ 'Data' ][ identifier ] ):
                    mean, sd = self[ 'Data' ][ identifier ][ condition ]
                    ratios[ row_index ][ col_index ] = mean
                    SDs[ row_index ][ col_index ] = sd
                addOn = ''
                try:
                    addOn = self[ 'Additional Labels' ][ self['Identifiers'][ index ] ]
                    if type( addOn ) == type( list() ) or type( addOn ) == type( set() ):
                        addOn = ".oOo.".join( list( set( addOn ) ) )
                except:
                    pass
                if addOn:
                    identifiers.append( '{0}___{1}'.format( identifier, addOn ) )
                else:
                    identifiers.append( identifier )

            ### draw expression profile:
            communityID = self[ 'Communities' ][ ( cluster, max_level ) ][ 'cluster ID' ]
            n_values = len( ratios )
            with open( os.path.join( self[ 'Working directory' ] , 'exProf_{0}-{1}.svg'.format( communityID, max_level ) ), 'w') as fout:
                width = startingX + scalingX * ( n_conditions -1 ) + len( self[ 'Conditions' ][ -1 ] ) * FONT_SIZE + max( [ len( i ) * FONT_SIZE for i in identifiers ] ) + 10
                s = '<svg xmlns="http://www.w3.org/2000/svg" version="1.1" font-size="{2}px" font-family="Verdana" width="{0}" height="{1}">'
                print( s.format( width, maxY + y_offset + FONT_SIZE, FONT_SIZE ), file = fout )
                ## draw SD-cloud:
                # determine min and max ratio + SD:
                print( svg_comment( 'SD CLOUD:' ), file = fout )
                for i in range( n_conditions - 1 ):
                    y1_min, y1_max = min_max_ratioWithSD( [ ratios[ j ][ i ] for j in range( n_values ) ], [ SDs[ j ][ i ] for j in range( n_values ) ] )
                    y2_min, y2_max = min_max_ratioWithSD( [ ratios[ j ][ i + 1 ] for j in range( n_values ) ], [ SDs[ j ][ i + 1 ] for j in range( n_values ) ] )
                    s = '<path d="M{x1} {y1_min} L{x2} {y2_min} L{x2} {y2_max} L{x1} {y1_max} Z" fill="{fill}"/>'
                    d = { 'fill' : '#D3D3D3'}
                    d[ 'x1' ] = startingX + i*scalingX
                    d[ 'x2' ] = startingX+(i+1)*scalingX
                    d[ 'y1_min' ] = startingY + y1_min*scalingY
                    d[ 'y1_max' ] = startingY + y1_max*scalingY
                    d[ 'y2_min' ] = startingY + y2_min*scalingY
                    d[ 'y2_max' ] = startingY + y2_max*scalingY
                    print( s.format( **d ), file = fout )

                ## draw expression profile lines:
                print( svg_comment( 'EXPRESSION PROFILE LINES:' ), file = fout )
                for i in range( n_conditions - 1 ):
                    for j in range( n_values ):
                        d = {}
                        d[ 'x1' ] = startingX + i * scalingX
                        d[ 'x2' ] = startingX + ( i + 1 ) * scalingX
                        d[ 'y1' ] = startingY + ratios[ j ][ i ] * scalingY
                        d[ 'y2' ] = startingY + ratios[ j ][ i + 1 ] * scalingY
                        print( svg_line( x1 = d[ 'x1' ], y1 = d[ 'y1' ], x2 = d[ 'x2' ], y2 = d[ 'y2' ] ), file = fout )

                ## add legend:
                print( svg_comment( 'LEGEND:' ), file = fout )
                # first, collect all values to plot -> to allow removing overlapping identifiers:
                legend = []
                for i, identifier in enumerate( identifiers ):
                    _last_ratio = ratios[ i ][ -1 ]
                    _x = startingX + scalingX * ( n_conditions - 1 ) + 2
                    _y = startingY + _last_ratio * scalingY
                    legend.append( ( _y, _x, identifier ) )
                legend.sort()
                # get all y-differences:
                y_differences = []
                for i, ( y, x, identifier ) in enumerate( legend[ : -1 ] ):
                    y_differences.append( legend[ i + 1 ][ 0 ] - y )
                # max font size for legend is the minimum y distance -> no overlap!
                legend_maxFontSize = int( round( min( y_differences ) ) )
                if legend_maxFontSize == 0:
                    legend_maxFontSize = 1
                # plot legend
                for y, x, identifier in legend:
                    print( '<text x="{0}" y="{1}" font-size="{3}px">{2}</text>'.format( x, y, identifier, legend_maxFontSize ), file = fout )

                ## plot axis:
                print( svg_comment( 'AXES:' ), file = fout )
                # y-axis:
                print(svg_line( x1 = 50, y1 = startingY + MAX_V * scalingY, x2 = 50, y2 = maxY + y_offset), file = fout )
                # y-axis - ticks:
                y_ticks_per_unit = 2
                for i in range( 1, MAX_V * y_ticks_per_unit + 1 ):
                    _ratio = float( i ) / y_ticks_per_unit
                    _y =  startingY + _ratio * scalingY
                    print( svg_text( x = 0, y = _y + FONT_SIZE // 2, text = '+{0}'.format( _ratio ) ), file = fout )
                    print( svg_line( x1 = 40, y1 = _y, x2 = 60, y2 = _y ), file = fout )
                for i in range( 1, abs( MIN_V ) * y_ticks_per_unit + 1 ):
                    _ratio = float( i ) / y_ticks_per_unit * -1
                    _y =  startingY + _ratio * scalingY
                    print( svg_text( x = 0, y = _y + FONT_SIZE // 2, text = _ratio), file = fout )
                    print( svg_line( x1 = 40, y1 = _y, x2 = 60, y2 = _y), file = fout )
                print( svg_text( x = 0, y = startingY + FONT_SIZE // 2, text = 0.0 ), file = fout )
                print( svg_line( x1 = 40, y1 = startingY, x2 = 60, y2 = startingY ), file = fout )
                # zero-line:
                print( svg_line( x1 = 50, y1 = startingY, x2 = startingX + scalingX * ( n_conditions - 1 ), y2 = startingY ), file = fout )
                # x-axis = conditions:
                for i, condition in enumerate( self[ 'Conditions' ] ):
                    _x = startingX + scalingX * i
                    print( svg_text( x= _x + 2, y = maxY + y_offset + FONT_SIZE, text = condition), file = fout )
                    s = '<line x1="{0}" y1="{1}" x2="{2}" y2="{3}" style="stroke-dasharray: 5, 5; stroke:#000000"/>'
                    print( s.format( _x, startingY + MAX_V * scalingY, _x, maxY + y_offset), file = fout )

                print( '</svg>', file = fout )
        self._print( '... community expression profile plots saved in "{0}"'.format( self[ 'Working directory' ] ), verbosity_level = 1 )
        return

    def do_it_all(self, working_directory = None,
                    distances = None, linkages = None, function_2_generate_noise_injected_datasets = None,
                    min_cluster_size = 4, alphabet = None, force_plotting = False, min_cluster_freq_2_retain = 0.001,
                    pickle_filename = 'pyGCluster_resampled.pkl', cpus_2_use = None, iter_max = 250000,
                    iter_tol = 0.01 / 100000, iter_step = 5000, iter_top_P = 0.001, iter_window = 50000, iter_till_the_end = False,
                    top_X_clusters = 0, threshold_4_the_lowest_max_freq = 0.01,
                    starting_min_overlap = 0.1, increasing_min_overlap = 0.05,
                    color_gradient = '1337', box_style = 'classic',
                    min_value_4_expression_map = None, max_value_4_expression_map = None, additional_labels = None
        ):
        '''
        Evokes all necessary functions which constitute the main functionality of pyGCluster.
        This is AHC clustering with noise injection and a variety of DLCs,
        in order to identify highly reproducible clusters,
        followed by a meta-clustering of highly reproducible clusters into so-called 'communities'.

        The functions that are called are:

           - :py:func:`pyGCluster.Cluster.resample`
           - :py:func:`pyGCluster.Cluster.build_nodemap`
           - :py:func:`pyGCluster.Cluster.write_dot`
           - :py:func:`pyGCluster.Cluster.draw_community_expression_maps`
           - :py:func:`pyGCluster.Cluster.draw_expression_profiles`

        For a complete list of possible
        Distance matrix calculations
        see: http://docs.scipy.org/doc/scipy/reference/spatial.distance.html
        or Linkage methods
        see: http://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html

        .. note ::
            If memory is of concern (e.g. for a large dataset, > 5000 objects), cpus_2_use should be kept low.

        :param distances: list of distance metrices, given as strings, e.g. [ 'correlation', 'euclidean' ]
        :type distances: list
        :param linkages: list of distance metrices, given as strings, e.g. [ 'average', 'complete', 'ward' ]
        :type linkages: list
        :param function_2_generate_noise_injected_datasets: function to generate noise-injected datasets. If None (default), Gaussian distributions are used.
        :type function_2_generate_noise_injected_datasets: function
        :param min_cluster_size: minimum size of a cluster, so that it is included in the assessment of cluster reproducibilities.
        :type min_cluster_size: int
        :param alphabet: alphabet used to convert decimal indices to characters to save memory. Defaults to string.printable, without ','.
        :type alphabet: string

        .. note ::
            If alphabet contains ',', this character is removed from alphabet, because the indices comprising a cluster are saved comma-seperated.

        :param force_plotting: the convergence plot is created after each iter_step iteration (otherwise only when convergence is detected).
        :type force_plotting: boolean
        :param min_cluster_freq_2_retain: ]0, 1[ minimum frequency of a cluster (only the maximum of the dlc-frequencies matters here) it has to exhibit to be stored in pyGCluster once all iterations are finished.
        :type min_cluster_freq_2_retain: float
        :param cpus_2_use: number of threads that are evoked in the re-sampling routine.
        :type cpus_2_use: int
        :param iter_max: maximum number of re-sampling iterations.
        :type iter_max: int

        Convergence determination:

        :param iter_tol: ]0, 1e-3[ value for the threshold of the median of normalized slopes, in order to declare convergence.
        :type iter_tol: float
        :param iter_step: number of iterations each multiprocess performs and simultaneously the interval in which to check for convergence.
        :type iter_step: int
        :param iter_top_P: ]0, 1[ for the convergence estmation, the amount of most frequent clusters is examined. This is the threshold for the minimum frequency of a cluster to be included.
        :type iter_top_P: float
        :param iter_window: size of the sliding window in iterations. The median is obtained from normalized slopes inside this window - *should be a multiple of iter_step*
        :type iter_window: int
        :param iter_till_the_end: if set to True, the convergence determination is switched off; hence, re-sampling is performed until iter_max is reached.
        :type iter_till_the_end: boolean

        Output/Plotting:

        :param pickle_filename: Filename of the output pickle object
        :type pickle_filename: string
        :param top_X_clusters: Plot of the top X clusters in the sorted list (by freq) of clusters having a maximum cluster frequency of at least threshold_4_the_lowest_max_freq (clusterfreq-plot is still sorted by size).
        :type top_X_clusters: int
        :param threshold_4_the_lowest_max_freq: ]0, 1[ Clusters must have a maximum frequency of at least threshold_4_the_lowest_max_freq to appear in the plot.
        :type threshold_4_the_lowest_max_freq: float
        :param min_value_4_expression_map: lower bound for color coding of values in the expression map. Remember that log2-values are expected, i.e. this value should be < 0!
        :type min_value_4_expression_map: float
        :param max_value_4_expression_map: upper bound for color coding of values in the expression map.
        :type max_value_4_expression_map: float
        :param color_gradient: name of the color gradient used for plotting the expression map. Currently supported are default, Daniel, barplot, 1337, BrBG, PiYG, PRGn, PuOr, RdBu, RdGy, RdYlBu, RdYlGn and Spectral
        :type color_gradient: string
        :param expression_map_filename: file name for expression map. .svg will be added if required.
        :type expression_map_filename: string
        :param legend_filename: file name for legend .svg will be added if required.
        :type legend_filename: string
        :param box_style: the way the relative standard deviation is visualized in the expression map. Currently supported are 'modern', 'fusion' or 'classic'.
        :type box_style: string
        :param starting_min_overlap: ]0, 1[ minimum required relative overlap between clusters so that they are assigned the same community. The relative overlap is defined as the size of the overlap between two clusters, divided by the size of the larger cluster.
        :type starting_min_overlap: float
        :param increasing_min_overlap: defines the increase of the required overlap between communities
        :type increasing_min_overlap: float
        :param additional_labels: dictionary, where additional labels can be defined which will be added in the expression map plots to the gene/protein names
        :type additional_labels: dict

        :rtype: None

        For more information to each parameter, please refer to :py:func:`pyGCluster.Cluster.resample`,
        and the subsequent functions:
        :py:func:`pyGCluster.Cluster.build_nodemap`,
        :py:func:`pyGCluster.Cluster.write_dot`,
        :py:func:`pyGCluster.Cluster.draw_community_expression_maps`,
        :py:func:`pyGCluster.Cluster.draw_expression_profiles`.

        '''
        if working_directory != None:
            self[ 'Working directory' ] = working_directory
        if distances == None:
            distances = [ 'euclidean', 'correlation' ]
        if linkages == None:
            linkages = [ 'complete', 'average', 'weighted', 'centroid', 'median', 'ward' ]
        if additional_labels != None:
            self[ 'Additional Labels' ] = additional_labels
        self._print( 'RESAMPLING ...', verbosity_level = 2 )
        self.resample(
            distances                                   = distances,
            linkages                                    = linkages,
            function_2_generate_noise_injected_datasets = function_2_generate_noise_injected_datasets,
            alphabet                                    = alphabet,
            iter_max                                    = iter_max,
            iter_top_P                                  = iter_top_P,
            iter_step                                   = iter_step,
            iter_tol                                    = iter_tol,
            iter_window                                 = iter_window,
            min_cluster_size                            = min_cluster_size,
            min_cluster_freq_2_retain                   = min_cluster_freq_2_retain,
            pickle_filename                             = pickle_filename,
            cpus_2_use                                  = cpus_2_use,
            iter_till_the_end                           = iter_till_the_end
        )
        self._print( 'Resampling done.', verbosity_level = 2 )
        self._print( '\nplotting cluster frequencies, building node map, drawing expression maps ...', verbosity_level = 2 )
        self.plot_clusterfreqs(
            min_cluster_size                    = min_cluster_size,
            top_X_clusters                      = top_X_clusters,
            threshold_4_the_lowest_max_freq     = threshold_4_the_lowest_max_freq,
        )
        self.build_nodemap(
            min_cluster_size = min_cluster_size,
            top_X_clusters                      = top_X_clusters,
            threshold_4_the_lowest_max_freq     = threshold_4_the_lowest_max_freq,
            starting_min_overlap                = starting_min_overlap,
            increasing_min_overlap              = increasing_min_overlap
        )
        dot_filename = 'nodemap_minSize{0}_top{1}_top{2:.0f}promille.dot'.format( min_cluster_size, top_X_clusters, threshold_4_the_lowest_max_freq * 1000 )
        self.write_dot(
            filename                    = dot_filename,
            min_value_4_expression_map  = min_value_4_expression_map,
            max_value_4_expression_map  = max_value_4_expression_map,
            color_gradient              = color_gradient,
            box_style                   = box_style
        )
        self.draw_community_expression_maps(
            min_value_4_expression_map  = min_value_4_expression_map,
            max_value_4_expression_map  = max_value_4_expression_map,
            color_gradient              = color_gradient,
            box_style                   = box_style
        )
        self.draw_expression_profiles(
            min_value_4_expression_map = min_value_4_expression_map,
            max_value_4_expression_map = max_value_4_expression_map
        )
        return

    def info(self):
        '''
        Prints some information about the clustering via pyGCluster:

            - number of genes/proteins clustered
            - number of conditions defined
            - number of distance-linkage combinations
            - number of iterations performed

        as well as some information about the communities, the legend for the shapes of nodes in the node map and the way the functions were called.

        :rtype: none
        '''
        self._print( '[ INFO ] {0:*^100}'.format( ' info function START ' ), verbosity_level = 0 )
        self._print('''
            {0:>9} identifiers were used to cluster
            {1:>9} conditions were defined
            {2:>9} linkage - distance def combos were used
            {3:>9} iterations were peformed

            '''.format(
                len( self[ 'Identifiers' ] ),
                len( self[ 'Conditions' ] ),
                len( self[ 'Distance-linkage combinations' ] ),
                self[ 'Iterations' ]
            ), verbosity_level = 0
        )
        self._print( 'Results are saved in the folder: "{0}"'.format( self[ 'Working directory' ] ), verbosity_level = 0 )
        if 'Communities' in self.keys() and self[ 'Communities' ] != {}:
            max_level = max( [ name[1] for name in self[ 'Communities' ] ] )
            communities_top_cluster = self._get_levelX_clusters( level = max_level )
            communities_minus_close2root = [ c for c in  communities_top_cluster if len( c ) < self[ 'for IO skip clusters bigger than' ] ]
            s = '{3} most_frequent_clusters were combined into {0} communities. {1} of those communities contain more than {2} objects (i.e. are "close to root" communities).'
            n_communities = len( communities_top_cluster)
            self._print( s.format( n_communities, n_communities - len( communities_minus_close2root ), self[ 'for IO skip clusters bigger than' ], len( self._get_levelX_clusters( level = 0 ) ) ), verbosity_level = 0 )
            self._print( 'See below for the parameters that were used to form communities (function "build_nodemap").', verbosity_level = 0 )
        else:
            self._print( 'Communities were not yet formed.', verbosity_level = 0 )
        if 'nodemap metric2shape' in self.keys():
            self._print( 'The legend for the node shapes in the DOT file is:', verbosity_level = 0 )
            for metric, shape in self[ 'nodemap metric2shape' ]:
                self._print( ' - clusters that are found by distance metric(s): "{0}" are visualized as "{1}"'.format( metric, shape ), verbosity_level = 0 )
        self._print( 'Values of the parameters of the functions that were already called:', verbosity_level = 0 )
        for function_name in self[ 'Function parameters' ]:
            self._print( '\t- function {0} was called with ...'.format( function_name ), verbosity_level = 0 )
            for kw, value in sorted( self[ 'Function parameters' ][ function_name ].items() ):
                self._print( '\t\t- - keyword: "{0}", value: "{1}".'.format( kw, value ), verbosity_level = 0 )
        self._print( '[ INFO ] {0:*^100}'.format( ' info function END ' ), verbosity_level = 0 )
        return

    def save(self, filename = 'pyGCluster.pkl'):
        '''
        Saves the current pyGCluster.Cluster object in a Pickle object.

        :param filename: may be either a simple file name ("example.pkl") or a complete path (e.g. "/home/user/Desktop/example.pkl"). In the former case, the pickle is stored in pyGCluster's working directory.
        :type filename: string

        :rtype: none
        '''
        tmp = {}
        for key in self.keys():
            tmp[ key ] = self[ key ]
        if not os.path.split( filename )[ 0 ]:
            with open( os.path.join( self[ 'Working directory' ], filename ), 'wb' ) as fout:
                pickle.dump( tmp, fout )
            self._print( 'pyGCluster pickled in: "{0}"'.format( os.path.join( self[ 'Working directory' ], filename ) ), verbosity_level = 1 )
        else:
            with open( filename, 'wb' ) as fout:
                pickle.dump( tmp, fout )
            self._print( 'pyGCluster pickled in: "{0}"'.format( filename ), verbosity_level = 1 )
        return

    def load(self, filename):
        '''
        Fills a pyGCluster.Cluster object with the session saved as "filename".
        If "filename" is not a complete path, e.g. "example.pkl" (instead of "/home/user/Desktop/example.pkl"), the directory given by self[ 'Working directory' ] is used.

        .. note ::
            Loading of pyGCluster has to be performed as a 2-step-procedure:
                >>> LoadedClustering = pyGCluster.Cluster()
                >>> LoadedClustering.load( "/home/user/Desktop/example.pkl" )

        :param filename: may be either a simple file name ("example.pkl") or a complete path (e.g. "/home/user/Desktop/example.pkl").
        :type filename: string

        :rtype: none
        '''
        _dir, _file = os.path.split( filename )
        if _dir:
            with open( filename, 'rb' ) as fin:
                tmp = pickle.load( fin )
        else:
            with open( os.path.join( self[ 'Working directory' ], filename ), 'rb' ) as fin:
                tmp = pickle.load( fin )
        for key in tmp.keys():
            self[ key ] = tmp[ key ]
        self._print( 'pyGCluster loaded.', verbosity_level = 1 )
        return

    def median(self, _list):
        '''
        Returns the median from a list of numeric values.

        :param _list:
        :type _list: list

        :rtype: int / float
        '''
        _list = sorted( _list )
        length = len( _list )
        if not length % 2:
            return ( _list[ length // 2 ] + _list[ length // 2 - 1 ] ) / 2.0
        return _list[ length / 2 ]

    def _print(self, *args, **kwargs):
        '''
        Internal print function which implements the "verbosity_level" parameter.
        :rtype: none
        '''
        if kwargs[ 'verbosity_level' ] <= self[ 'Verbosity level' ]:
            del kwargs[ 'verbosity_level' ]
            print( *args, **kwargs )
        return

if __name__ == '__main__':
    #invoke the freeze_support funtion for windows based systems
    try:
        sys.getwindowsversion()
        multiprocessing.freeze_support()
    except:
        pass

    x = Cluster()
    exit()

