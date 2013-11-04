#!/usr/bin/env python2.7
'''
Testscript to plot a simple heatmap.

This can be used to visualize the different box styles for the expression maps
and to test the plotting function.

This can be also used as a basis to visualize own datasets by simply defining 
the 'data' dictionary in this script.

Usage::

    ./simpleExpressionMap.py

'''
from __future__ import print_function
import pyGCluster
import os

data = {
            'fastaID1' : {'1':(3.0,0.1),'2':(-3.0,0.4),'3':(-4.0,0.4),'4':(1.0,0.1),'5':(-1.0,0.2)},
            'fastaID2' : {'1':(3.0,0.1),'2':(-3.0,0.4),'3':(-3.0,0.4),'4':(2.0,0.2),'5':(-1.5,0.2)},
            'fastaID3' : {'1':(3.0,0.2),'2':(-3.0,0.4),'3':(-2.0,0.4),'4':(3.0,0.3),'5':(-2.0,0.3)},
            'fastaID4' : {'1':(3.0,0.4),'2':(-3.0,0.4),'3':(-1.0,0.4),'4':(4.0,0.4),'5':(-2.5,0.4)},
            'fastaID5' : {'1':(3.0,1.0),'2':(-3.0,0.4),'3':( 0.0,0.4),'4':(0.0,0.1),'5':(-3.5,0.4)},
            'fastaID6' : {'1':(3.0,2.0),'2':(-3.0,0.4),'3':( 1.0,0.4),'4':(1.5,0.2),'5':(-4.0,0.4)},
            'fastaID7' : {'1':(3.0,1.3),'2':(-3.0,0.4),'3':( 2.0,0.4),'4':(2.5,0.3),'5':(-4.0,0.5)},
        }

if __name__ == '__main__':
    # print( __doc__ )

    working_dir = './simpleExpressionMaps/'
    if not os.path.exists( working_dir ):
        os.mkdir( working_dir )
    print( '[ INFO ] ... the results of the example script are saved in "{0}"'.format( working_dir ) )

    cluster = pyGCluster.Cluster()
    for hm in cluster['Heat map']['Color Gradients'].keys():
        cluster.draw_expression_map(
        data = data,
        # additional_labels = None,
        min_value_4_expression_map                    = -4,
        max_value_4_expression_map                    = +4,
        expression_map_filename = os.path.join( working_dir , 'simpleExpressionMap_{0}.svg'.format( hm )),
        legend_filename         = os.path.join( working_dir , 'legend_hm_{0}.svg'.format( hm )),
        color_gradient          = hm,
        box_style               = 'classic'
        )
    for bs in cluster['Heat map']['SVG box styles'].keys():
        cluster.draw_expression_map(
            data = data,
            # additional_labels = None,
            min_value_4_expression_map                    = -4,
            max_value_4_expression_map                    = +4,
            expression_map_filename = os.path.join( working_dir, 'simpleExpressionMap_{0}.svg'.format(bs)),
            # legend_filename         = 'legend_hm_{0}.svg'.format( hm ),
            color_gradient          = 'Spectral',
            box_style               = bs
            )
