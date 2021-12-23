# -*- coding: utf-8 -*-
"""
/***************************************************************************
        copyright            : (C) 2021 by Pietro Belotti
        email                : pietro.belotti@polimi.it
 ***************************************************************************/

This source file contains functions that create the optimization model
to be solved by Python-MIP.
"""

import mip
import numpy as np
import math

from scipy.spatial import KDTree
from scipy.spatial import distance

import functools

def add_point_layer(group, table, attributes=[], groupname=""):
    """table[:,:2] are the x-y columns, the rest is the attributes to be added
    """

    from qgis.core import QgsLayerTreeGroup, QgsProject, QgsVectorLayer, \
        QgsFields, QgsField, QgsFeature, QgsGeometry, QgsLayerTreeLayer, \
        QgsPointXY

    from PyQt5.QtCore import QVariant

    if group is None:
        group = QgsLayerTreeGroup(groupname)
        QgsProject.instance().layerTreeRoot().insertChildNode(1, group)   # 1 is position in layer directory

    layer = QgsVectorLayer("point?crs=epsg:25832", f"{groupname} points", "memory")

    fields = QgsFields()
    for name in attributes:
        fields.append(QgsField(name, QVariant.Double))
    layer.dataProvider().addAttributes(fields)
    layer.updateFields()

    featlist = []

    for p in table:

        feat = QgsFeature()
        feat.setFields(layer.fields())
        for i,a in enumerate(attributes):
            feat.setAttribute(i, p[2 + i])
        feat.setGeometry(QgsGeometry.fromPointXY(QgsPointXY(p[0], p[1])))
        featlist.append(feat)

    layer.dataProvider().addFeatures(featlist)

    QgsProject.instance().addMapLayer(layer, False)

    treelayer = QgsLayerTreeLayer(layer)
    group.insertChildNode(0, treelayer)


def add_segment_layer(group, table, attributes=[], groupname='', flow=None):
    """table[:,:4] are the x-y columns of the two endpoints, the rest is
    the attributes to be added
    """

    from qgis.core import QgsLayerTreeGroup, QgsProject, \
        QgsVectorLayer, QgsFields, QgsField, QgsFeature, \
        QgsGeometry, QgsLayerTreeLayer, QgsPointXY, QgsLineSymbol

    from PyQt5.QtCore import QVariant
    from qgis.PyQt.QtGui import QColor

    if group is None:
        group = QgsLayerTreeGroup(groupname)
        QgsProject.instance().layerTreeRoot().insertChildNode(1, group)   # 1 is position in layer directory

    layer = QgsVectorLayer("linestring?crs=epsg:25832", f"{groupname} segments", "memory")
    pr = layer.dataProvider()

    fields = QgsFields()
    for name in attributes:
        fields.append(QgsField(name, QVariant.Double))
    pr.addAttributes(fields)
    layer.updateFields()

    symbol = QgsLineSymbol()

    if 'canal' in groupname.lower():
        symbol.setWidth(1.12)
        symbol.setColor(QColor.fromRgb(20, 20, 120))
    elif 'railw' in groupname.lower():
        symbol.setWidth(0.9 )
        symbol.setColor(QColor.fromRgb(90, 90, 90))
    elif 'trail' in groupname.lower():
        symbol.setWidth(0.64)
        symbol.setColor(QColor.fromRgb(200, 200, 100))
    else:
        symbol.setWidth(0.44)
        symbol.setColor(QColor.fromRgb(170, 0, 0))

    if flow:
        symbol.setWidth(.99)

    layer.renderer().setSymbol(symbol)

    featlist = []

    for i,p in enumerate(table):

        if flow is not None and flow[i] <= 1e-7:
            continue

        feat = QgsFeature()
        feat.setFields(layer.fields())
        for j,a in enumerate(attributes):
            feat.setAttribute(j, p[4 + j])
        feat.setGeometry(QgsGeometry.fromPolylineXY([QgsPointXY(p[0], p[1]), QgsPointXY(p[2], p[3])]))
        featlist.append(feat)

    pr.addFeatures(featlist)

    QgsProject.instance().addMapLayer(layer, False)

    treelayer = QgsLayerTreeLayer(layer)
    group.insertChildNode(0, treelayer)


def add_cumul_flow_layer(group, sol, layer_name='', flowtype=''):
    """Visualize flow from a solution
    """

    from qgis.core import QgsProject, \
        QgsVectorLayer, QgsFields, QgsField, QgsFeature, \
        QgsLayerTreeLayer, QgsLineSymbol

    from qgis.PyQt.QtGui import QColor
    from PyQt5.QtCore import QVariant

    layer_orig = QgsProject.instance().mapLayersByName(layer_name)[0]

    layer = QgsVectorLayer("linestring?crs=" + layer_orig.dataProvider().crs().authid(), f"Flusso stimato su {flowtype}", "memory")
    pr = layer.dataProvider()

    fields = QgsFields()
    fields.append(QgsField('flusso', QVariant.Double))
    pr.addAttributes(fields)
    layer.updateFields()

    layer.updateFields()

    symbol = QgsLineSymbol()

    if flowtype == 'canali':
        symbol.setWidth(1.12)
        symbol.setColor(QColor.fromRgb(0,0,100))
    elif flowtype == 'ciclabili':
        symbol.setWidth(0.64)
        symbol.setColor(QColor.fromRgb(150,150,50))
    else:
        symbol.setWidth(0.9)
        symbol.setColor(QColor.fromRgb(50,50,50))

    layer.renderer().setSymbol(symbol)

    featlist = []

    for tup in sol:

        if tup is None:
            continue

        feat = QgsFeature()
        feat.setFields(layer.fields())
        feat.setAttribute(0, tup[1])
        feat.setGeometry(tup[0][5].geometry())
        featlist.append(feat)

    pr.addFeatures(featlist)

    QgsProject.instance().addMapLayer(layer, False)

    treelayer = QgsLayerTreeLayer(layer)
    group.insertChildNode(0, treelayer)


def connect_nodes_edges (V, E, nodetype, edgetype):
    """
    For all canal stations, identify the canal that is closest,
    and then add two edges to the canal's endnodes. Do so by
    defining all canals as segments between their endpoints,
    then finding the canal such that the distance from the canal
    point to the canal segment is minimum. @todo this is a
    heuristic and a comparison should be done on all polyline
    points of the canal instead.

    Given a segment (x1,y1)---(x2,y2) and a point xp,yp, the
    closest point is found by drawing the line through the
    segment

    (x2-x1)*(y-y1) = (y2-y1)*(x-x1)

    The perpendicular line to the above through xp,yp is

    (y2-y1)*(y-yp) = (x1-x2)*(x-xp)

    The intersection of the two is the (unique) solution to the
    system (note that the determinant of A in Ax=b is

    delta = (x2-x1)**2 + (y2-y1)**2

    and is nonzero as it's the sum of two squares):

    (x2-x1) y + (y1-y2) x = (x2-x1) y1 - (y2-y1) x1
    (y2-y1) y + (x2-x1) x = (y2-y1) yp - (x1-x2) xp

    Solve it with Cramer's rule:

    yi = (((x2-x1) * y1 - (y2-y1) * x1) * (x2-x1) - ((y2-y1) * yp - (x1-x2) * xp) * (y1-y2)) / delta
    xi = (((x2-x1) * y1 - (y2-y1) * x1) * (y1-y2) - ((y2-y1) * yp - (x1-x2) * xp) * (x1-x2)) / delta

    Simplifying for numpy's benefit,

    yi = ((x2*y1 - x1*y1 - y2*x1 + y1*x1) * (x2-x1) - ((y2-y1) * yp - (x1-x2) * xp) * (y1-y2)) / delta
    xi = ((x2*y1 - x1*y1 - y2*x1 + y1*x1) * (y1-y2) - ((y2-y1) * yp - (x1-x2) * xp) * (x1-x2)) / delta

    yi = ((x2*y1 - y2*x1) * (x2-x1) - ((y2-y1) * yp - (x1-x2) * xp) * (y1-y2)) / delta
    xi = ((x2*y1 - y2*x1) * (y1-y2) - ((y2-y1) * yp - (x1-x2) * xp) * (x1-x2)) / delta

    prepare common quantities (to be vectorized):

    crossp = x2*y1 - y2*x1
    pprod  = (y2-y1) * yp - (x1-x2) * xp

    yi = (crossp * (x2-x1) - pprod * (y1-y2)) / delta
    xi = (crossp * (y1-y2) - pprod * (x1-x2)) / delta
    """

    statn_id =          [i     for i,v in enumerate(V) if (v[3] & nodetype) != 0]
    statn_x  = np.array([v[:2] for i,v in enumerate(V) if (v[3] & nodetype) != 0])

    segments_id = [[i, j, id]                                    for id, (i, j, _1, eclass, _f) in enumerate(E) if (eclass & edgetype) != 0]
    segments_x  = np.array([[V[i][0], V[i][1], V[j][0], V[j][1]] for     (i, j, _1, eclass, _f) in           E  if (eclass & edgetype) != 0])

    x1 = segments_x[:,0]
    y1 = segments_x[:,1]
    x2 = segments_x[:,2]
    y2 = segments_x[:,3]

    xp = statn_x[:,0]
    yp = statn_x[:,1]

    delta = (x1 - x2)**2 + (y1 - y2)**2

    crossp = x2 * y1 - y2 * x1

    pprod = np.outer (yp, y2 - y1) - np.outer (xp, x1 - x2)

    yi = (crossp * (x2-x1) - pprod * (y1-y2)) / delta
    xi = (crossp * (y1-y2) - pprod * (x1-x2)) / delta

    # If the intersection of the two lines is outside of the segment,
    # i.e. it is not a convex combination of the two, i.e. if
    #
    # xi = x1 + alpha (x2 - x1)
    # yi = y1 + alpha (y2 - y1)
    #
    # with alpha := (xi-x1)/(x2-x1) = (yi-y1)/(y2-y1) < 0 or alpha >
    # 1, the closest point is (x1,y1) if alpha < 0, (x2,y2) if alpha >
    # 1.

    alphas = (yi - y1) / (y2 - y1)
    mask = (alphas == np.nan) | (alphas == np.inf) | (alphas == -np.inf)
    np.putmask(alphas, mask, (xi - x1) / (x2 - x1))

    mask_alphaneg = alphas < 0
    mask_alphabig = alphas > 1
    np.putmask(xi, mask_alphaneg, x1)
    np.putmask(xi, mask_alphabig, x2)
    np.putmask(yi, mask_alphaneg, y1)
    np.putmask(yi, mask_alphabig, y2)

    pier_dist = np.sqrt((xi.T - xp)**2 + (yi.T - yp)**2).T

    closest_edge = np.argmin(pier_dist, axis=1)

    E.extend([[min(statn_id[i], segments_id[s][0]), max(statn_id[i], segments_id[s][0]), 0, nodetype, None] for i,s in enumerate(closest_edge) if statn_id[i] != segments_id[s][0]])
    E.extend([[min(statn_id[i], segments_id[s][1]), max(statn_id[i], segments_id[s][1]), 0, nodetype, None] for i,s in enumerate(closest_edge) if statn_id[i] != segments_id[s][1]])

    # Map from canal station node index in V into canal edge index in E
    return {statn_id[i]: segments_id[s][2] for i,s in enumerate(closest_edge)}


class OptModel():

    def __init__(self, netdata):

        # Nodes:

        self.nodes_canal  = netdata['nodes']['canal']  # list of tuples (x, y, cost, feature)
        self.nodes_infra  = netdata['nodes']['trail']  # list of tuples (x, y, cost, feature)
        self.nodes_railst = netdata['nodes']['railway']  # list of tuples (x, y, cost, feature)

        self.edges_canal = netdata['edges']['canal']    # list of tuples (x1,y1,x2,y2,cost,feature)
        self.edges_trail = netdata['edges']['trail'] # list of tuples (x1,y1,x2,y2,cost,feature)
        self.edges_railway = netdata['edges']['railway'] # list of tuples (x1,y1,x2,y2,cost,feature)

        self.demand = netdata['demand']                       # list of tuples (x,y,aggregate)

        # node capacities: fixed and proportional

        self.cap_base_infra_nodes = 4000

        self.cap_mult_infra_nodes = 1500000
        self.cap_mult_csttn_nodes = 9000000
        self.cap_mult_rsttn_nodes = 9000000

        # edge capacities: fixed and proportional

        self.cap_base_trail_edges = 0
        self.cap_base_canal_edges = 10000000
        self.cap_base_railw_edges = 10000000

        self.cap_mult_trail_edges = 15000000
        self.cap_mult_canal_edges = 0
        self.cap_mult_railw_edges = 0

        print("Urban optimizer solver module",
              f" {len(self.nodes_canal):5d} canal stations (or candidates),",
              f" {len(self.nodes_infra):5d} biking infrastructure (or candidates),",
              f" {len(self.nodes_railst):5d} train stations (or candidates),",
              f" {len(self.edges_canal):5d} canals,",
              f" {len(self.edges_trail):5d} bicycle trail stretches,",
              f" {len(self.edges_railway):5d} railway stretches,",
                   f" {len(self.demand):5d} o/d pairs", sep='\n')


    def create_topology(self, **options):

        from qgis.core import QgsLayerTreeGroup, QgsProject

        # Given the instance information collected, create an
        # undirected graph where nodes and edges are associated a
        # cost, a capacity, and each can have different meaning. Each
        # node can be one of
        #
        # * demand nodes from the aggregate set---these nodes are
        #   sources/destination of traffic;
        # * (candidate) canal stations;
        # * (candidate) cycling trail infrastructure points;
        # * (candidate) railway stations;
        # * areas containing one or more end points of the below edges.
        #
        # Edges can be one of:
        #
        # * (candidate) cycling trails;
        # * railways;
        # * canals;
        # * artificial edges connecting o/d points with some end point
        #   of a canal/cycling trail.
        #
        # All input edges are associated with two end nodes. This
        # function does as follows:
        #
        # 1) Define an overall set V0 of nodes as the set of all
        #    native nodes plus any end-node of all edges above;
        #
        # 2) Collapse all nodes within an Euclidean distance of L into
        #    single nodes; associate each node of V0 to one such node,
        #    forming a new set (with lower cardinality) called V;
        #
        # 3) For all input edge ij, add an edge to E with (fixed,
        #    setup) cost dependent on the input parameter; the cost is
        #    nonzero only for candidate cycling paths;
        #
        # 4) For all aggregate demand node, connect it with the two
        #    closest non-demand nodes with zero-cost edges;
        #
        # 5) For each (candidate) canal station, find the closest
        #    canal and join the station to both of its end nodes with
        #    an edge with cost given by the canal station cost.

        # self.nodes_canal:  list of tuples (x, y, cost, feature)
        # self.nodes_infra:  list of tuples (x, y, cost, feature)

        # self.edges_canal:  list of tuples (x1,y1,x2,y2,cost,feature)
        # self.edges_trail:  list of tuples (x1,y1,x2,y2,cost,feature)

        # self.demand: list of tuples (x,y,#passengers,feature)

        # Bit strings for multi-function nodes/edges
        self.TYPE_CSTTN = 1 << 0
        self.TYPE_INFRA = 1 << 1
        self.TYPE_RSTTN = 1 << 2
        self.TYPE_CANAL = 1 << 3
        self.TYPE_TRAIL = 1 << 4
        self.TYPE_RAILW = 1 << 5
        self.TYPE_ODTRF = 1 << 6

        V0 = [[x,  y,  cost, self.TYPE_CSTTN, i, feature] for i, (x,  y,          cost, feature) in enumerate(self.nodes_canal)] + \
                                                                                                                                   \
             [[x,  y,  cost, self.TYPE_INFRA, i, feature] for i, (x,  y,          cost, feature) in enumerate(self.nodes_infra)] + \
                                                                                                                                   \
             [[x,  y,  cost, self.TYPE_RSTTN, i, feature] for i, (x,  y,          cost, feature) in enumerate(self.nodes_railst)] + \
                                                                                                                                   \
             [[x1, y1, cost, self.TYPE_CANAL, i, feature] for i, (x1, y1, _1, _2, cost, feature) in enumerate(self.edges_canal)] + \
             [[x2, y2, cost, self.TYPE_CANAL, i, feature] for i, (_1, _2, x2, y2, cost, feature) in enumerate(self.edges_canal)] + \
                                                                                                                                   \
             [[x1, y1, cost, self.TYPE_TRAIL, i, feature] for i, (x1, y1, _1, _2, cost, feature) in enumerate(self.edges_trail)] + \
             [[x2, y2, cost, self.TYPE_TRAIL, i, feature] for i, (_1, _2, x2, y2, cost, feature) in enumerate(self.edges_trail)] + \
                                                                                                                                   \
             [[x1, y1, cost, self.TYPE_RAILW, i, feature] for i, (x1, y1, _1, _2, cost, feature) in enumerate(self.edges_railway)] + \
             [[x2, y2, cost, self.TYPE_RAILW, i, feature] for i, (_1, _2, x2, y2, cost, feature) in enumerate(self.edges_railway)] + \
                                                                                                                                   \
             [[x1, y1, dem, self.TYPE_ODTRF, i, feature] for i, (x1, y1, _1, _2, dem,  feature) in enumerate(self.demand)]       + \
             [[x2, y2, dem, self.TYPE_ODTRF, i, feature] for i, (_1, _2, x2, y2, dem,  feature) in enumerate(self.demand)]

        from qgis.core import QgsLayerTreeGroup, QgsProject

        group = QgsLayerTreeGroup("Abstraction level 1")
        QgsProject.instance().layerTreeRoot().insertChildNode(1, group)   # 1 is position in layer directory

        add_point_layer(group, [t[:3] for t in V0 if t[3]==self.TYPE_CSTTN], attributes=['cost'], groupname='canal_station')
        add_point_layer(group, [t[:3] for t in V0 if t[3]==self.TYPE_INFRA], attributes=['cost'], groupname='infrastructure')
        add_point_layer(group, [t[:3] for t in V0 if t[3]==self.TYPE_RSTTN], attributes=['cost'], groupname='railway_station')

        add_segment_layer(group, [[x1,y1,x2,y2,cost] for (x1,y1,x2,y2,cost,_) in self.edges_canal],   attributes=['cost'], groupname='canals')
        add_segment_layer(group, [[x1,y1,x2,y2,cost] for (x1,y1,x2,y2,cost,_) in self.edges_trail],   attributes=['cost'], groupname='trails')
        add_segment_layer(group, [[x1,y1,x2,y2,cost] for (x1,y1,x2,y2,cost,_) in self.edges_railway], attributes=['cost'], groupname='railway')

        add_segment_layer(group, [[x1,y1,x2,y2,dem]  for (x1,y1,x2,y2,dem,_)  in self.demand],        attributes=['demand'], groupname='demand')

        gfeaturesV = {v[3]: v for v in self.nodes_canal + self.nodes_infra + self.nodes_railst}
        gfeaturesE = {e[5]: e for e in self.edges_canal + self.edges_trail + self.edges_railway}

        self.gfeatures = {**gfeaturesV, **gfeaturesE}

        ncanal = len(self.edges_canal)
        ntrail = len(self.edges_trail)
        nrailw = len(self.edges_railway)

        nodtrf = len(self.demand)

        start_edges_canal   = min([i for i,v in enumerate(V0) if v[3] == self.TYPE_CANAL])
        start_edges_trail   = min([i for i,v in enumerate(V0) if v[3] == self.TYPE_TRAIL])
        start_edges_railway = min([i for i,v in enumerate(V0) if v[3] == self.TYPE_RAILW])

        start_edges_odtrf   = min([i for i,v in enumerate(V0) if v[3] == self.TYPE_ODTRF])

        # Generate proximity data structure for all nodes

        tree = KDTree (np.array([v[:2] for v in V0]))

        proximity = 250
        pairs = tree.query_pairs(r=proximity)

        pairs = np.array([[p[0], p[1]] for p in pairs])

        # pairs now contains tuples (i,j) such that the distance
        # between V0[i] and V0[j] is not more than proximity. Use this
        # information to collapse them but don't form large clusters
        # arising from pairs (i1,i2), (i2,i3), ... (i10,i11) where i1
        # and i11 might be far apart. Find the indices with most
        # occurrences in the tuples

        occurrences = np.bincount(pairs.flatten())

        collapse = {}

        for _ in range(pairs.shape[0]):

            most_occurring = np.argmax(occurrences)
            if occurrences[most_occurring] == 0:
                break;

            collapse[most_occurring] = most_occurring
            occurrences[most_occurring] = -1

            for a in pairs[(pairs[:,0] == most_occurring) |
                           (pairs[:,1] == most_occurring), :]:
                if a[0] == most_occurring and occurrences[a[1]] != 0:
                    collapse[a[1]] = most_occurring
                    occurrences[a[1]] = -1
                if a[1] == most_occurring and occurrences[a[0]] != 0:
                    collapse[a[0]] = most_occurring
                    occurrences[a[0]] = -1

        # Fill up nodes that don't appear in any of the KDtree's pairs
        for i,v in enumerate(V0):
            if i not in collapse:
                collapse [i] = i

        print(f"Aggregator found reduced vertex set from {len(collapse.keys())} to {len(set(collapse.values()))}")

        # Collapse[] now has a surjective map of the initial set of
        # nodes into a set of collapsed ones, whose coordinates we
        # retain as those of the map's image.
        #
        # nmap is a bijective map from indices in V0 (of nodes that
        # function as aggregators or of nodes that were not collapsed
        # onto other aggregators) into indices from 0 to n-1, where n
        # is the new set of nodes.

        nmap = {}
        nind = 0
        V = []

        for i, (x, y, cost, attributes, n_index, feature) in enumerate(V0):

            c = collapse[i]

            if c not in nmap:

                assert nind == len(V)

                V.append([x, y, 0, 0, feature])
                nmap[c] = nind
                nind += 1

            # Add entry nodes: landings, trail points, railway
            # stations, o/d, and any point on a bike trail.

            if attributes in [self.TYPE_CSTTN, self.TYPE_INFRA, self.TYPE_RSTTN,
                              self.TYPE_ODTRF,
                              self.TYPE_TRAIL]:

                V[nmap[c]][3] = V[nmap[c]][3] | attributes

                if attributes != self.TYPE_TRAIL:  # trail EDGES have a cost, trail NODES don't
                    V[nmap[c]][2] += cost

        # Create initial demand data structure with coordinates and V[] indices

        Q0 = [(x1, y1, x2, y2, dem, self.TYPE_ODTRF, start_edges_odtrf + i, start_edges_odtrf + nodtrf + i, feature)
              for i, (x1, y1, x2, y2, dem, feature) in enumerate(self.demand)]

        # Create V-based list of O/D pairs

        Q = [(nmap[collapse[i_ind]], nmap[collapse[j_ind]], dem, eclass, feature)
             for (x1, y1, x2, y2, dem, eclass, i_ind, j_ind, feature) in Q0
             if collapse[i_ind] != collapse[j_ind]]

        # Create initial set of edges with indices to be used in the
        # maps below. This will be useful when collapsing edges.

        E0 = [(x1, y1, x2, y2, cost, self.TYPE_CANAL, start_edges_canal + i, start_edges_canal + ncanal + i, feature)
              for i, (x1, y1, x2, y2, cost, feature) in enumerate(self.edges_canal)] + \
             [(x1, y1, x2, y2, cost, self.TYPE_TRAIL, start_edges_trail + i, start_edges_trail + ntrail + i, feature)
              for i, (x1, y1, x2, y2, cost, feature) in enumerate(self.edges_trail)] + \
             [(x1, y1, x2, y2, cost, self.TYPE_RAILW, start_edges_railway + i, start_edges_railway + nrailw + i, feature)
              for i, (x1, y1, x2, y2, cost, feature) in enumerate(self.edges_railway)]

        # Create a set of edges (we assume the graph to be undirected)
        # based on the collapsed nodes: for each initial edge, if its
        # end points p[i], p[j] collapse to the same node we simply
        # eliminate the edge: it is a canal/trail that is too short
        # (its direct distance is below the proximity parameter above)
        # to be considered.

        E1 = [(nmap[collapse[i_ind]], nmap[collapse[j_ind]], cost, eclass, feature)
             for (x1, y1, x2, y2, cost, eclass, i_ind, j_ind, feature) in E0
             if collapse[i_ind] != collapse[j_ind]]

        endnodes = set([(min(i,j), max(i,j)) for (i, j, _1, _2, _3) in E1])

        E = []

        for (ii,jj) in endnodes:
            E1l = [(i,j,c,e,f) for (i,j,c,e,f) in E1 if min(i,j)==ii and max(i,j)==jj]
            E.append((ii, jj,
                      functools.reduce(lambda a, b: a+b, [c for (i,j,c,e,f) in E1l]),
                      functools.reduce(lambda a, b: a|b, [e for (i,j,c,e,f) in E1l]),
                      [f for (i,j,c,e,f) in E1l]))

        self.landing2canal = connect_nodes_edges (V, E, self.TYPE_CSTTN, self.TYPE_CANAL)
        self.station2railw = connect_nodes_edges (V, E, self.TYPE_RSTTN, self.TYPE_RAILW)

        for e in E:
            assert e[0] < len(V)
            assert e[1] < len(V)

        #
        # Add edges between each demand point and its n_neighbors closest canal station or bike trail
        #

        access_type = (self.TYPE_CSTTN | self.TYPE_TRAIL | self.TYPE_RSTTN)

        demand_x  = np.array([v[:2] for   v in           V  if (v[3] & self.TYPE_ODTRF) != 0 and (v[3] & access_type) == 0])
        demand_id =          [i     for i,v in enumerate(V) if (v[3] & self.TYPE_ODTRF) != 0 and (v[3] & access_type) == 0]

        access_x  = np.array([v[:2] for   v in           V  if (v[3] & access_type) != 0])
        access_id =          [i     for i,v in enumerate(V) if (v[3] & access_type) != 0]

        ad_distance = distance.cdist(demand_x, access_x)

        n_neighbors = 3

        for i in range(n_neighbors):
            closest_access = np.argmin(ad_distance, axis=1)
            E.extend([[min(demand_id[i], access_id[s]),
                       max(demand_id[i], access_id[s]),
                       0, self.TYPE_ODTRF, None] for i,s in enumerate(closest_access)
                      if demand_id[i] != access_id[s]])
            for i,c in enumerate(closest_access):
                ad_distance[i,c] = 1e100

        group = QgsLayerTreeGroup("Equivalent network")
        QgsProject.instance().layerTreeRoot().insertChildNode(1, group)   # 1 is position in layer directory

        add_point_layer(group, [t[:4] for t in V if (t[3] & self.TYPE_CSTTN) != 0], attributes=['cost', 'type'], groupname='CANAL_STATN')
        add_point_layer(group, [t[:4] for t in V if (t[3] & self.TYPE_INFRA) != 0], attributes=['cost', 'type'], groupname='INFRASTRUCT')
        add_point_layer(group, [t[:4] for t in V if (t[3] & self.TYPE_RSTTN) != 0], attributes=['cost', 'type'], groupname='TRAIN_STATIONS')
        add_point_layer(group, [t[:4] for t in V if (t[3] & self.TYPE_TRAIL) != 0], attributes=['cost', 'type'], groupname='TRAIL_ENDS')
        add_point_layer(group, [t[:4] for t in V if (t[3] & self.TYPE_ODTRF) != 0], attributes=['cost', 'type'], groupname='DEMAND')

        add_segment_layer(group, [[V[i][0],V[i][1],V[j][0],V[j][1],c,a] for (i,j,c,a,_) in E if (a & self.TYPE_CSTTN) != 0], attributes=['cost', 'type'], groupname='JOIN_CANAL')
        add_segment_layer(group, [[V[i][0],V[i][1],V[j][0],V[j][1],c,a] for (i,j,c,a,_) in E if (a & self.TYPE_RSTTN) != 0], attributes=['cost', 'type'], groupname='JOIN_RAILWAY')
        add_segment_layer(group, [[V[i][0],V[i][1],V[j][0],V[j][1],c,a] for (i,j,c,a,_) in E if (a & self.TYPE_ODTRF) != 0], attributes=['cost', 'type'], groupname='DEMAND_LINK')
        add_segment_layer(group, [[V[i][0],V[i][1],V[j][0],V[j][1],c,a] for (i,j,c,a,_) in E if (a & self.TYPE_CANAL) != 0], attributes=['cost', 'type'], groupname='CANAL')
        add_segment_layer(group, [[V[i][0],V[i][1],V[j][0],V[j][1],c,a] for (i,j,c,a,_) in E if (a & self.TYPE_TRAIL) != 0], attributes=['cost', 'type'], groupname='TRAIL')
        add_segment_layer(group, [[V[i][0],V[i][1],V[j][0],V[j][1],c,a] for (i,j,c,a,_) in E if (a & self.TYPE_RAILW) != 0], attributes=['cost', 'type'], groupname='RAILWAY')

        add_segment_layer(group, [[V[i][0],V[i][1],V[j][0],V[j][1],c,a] for (i,j,c,a,_) in Q if (a & self.TYPE_ODTRF) != 0], attributes=['cost', 'type'], groupname='OD_PAIRS')

        #
        # Create traffic demand
        #

        print(f"Routing network: {len(V)} nodes, {len(E)} edges, {len(Q)} o/d pairs")

        self.V = V
        self.E = E
        self.Q = Q


    def create_model(self, **options):

        # Model for routing demands

        V = self.V
        E = self.E
        Q = self.Q

        ODpairs = {(i1, i2): dem for (i1, i2, dem, _e, _f) in Q}

        neighbors = {i: [e[0] + e[1] - i for e in E if e[0]==i or e[1]==i] for i,v in enumerate(V)}

        p = mip.Model()

        flowF = {(e[0],e[1],orig,dest): p.add_var(lb=0, ub=1, name=f'flow_{e[0]}_{e[1]}_{orig}_{dest}') for e in E for (orig,dest) in ODpairs}
        flowB = {(e[1],e[0],orig,dest): p.add_var(lb=0, ub=1, name=f'flow_{e[1]}_{e[0]}_{orig}_{dest}') for e in E for (orig,dest) in ODpairs}
        flow = {**flowF, **flowB}

        edge = {(e[0],e[1]): p.add_var(var_type=mip.BINARY, name=f'edge_{e[0]}_{e[1]}') for e in E}
        node = {i:           p.add_var(var_type=mip.BINARY, name=f'node_{i}') for i,v in enumerate(V)}

        satisf = {(orig, dest): p.add_var(var_type=mip.BINARY, name=f'satisf_{orig}_{dest}') for (orig,dest) in ODpairs}

        cap_null_nodes = {i:           -1 for i,v in enumerate(V)}
        cap_null_edges = {(e[0],e[1]): -1 for e in E}

        cap_base_csttn_nodes = {i: 0                         for i,v in enumerate(V) if (v[3] & self.TYPE_CSTTN) != 0}
        cap_base_infra_nodes = {i: self.cap_base_infra_nodes for i,v in enumerate(V) if (v[3] & self.TYPE_INFRA) != 0}
        cap_base_rsttn_nodes = {i: 0                         for i,v in enumerate(V) if (v[3] & self.TYPE_RSTTN) != 0}

        cap_mult_csttn_nodes = {i: self.cap_mult_csttn_nodes for i,v in enumerate(V) if (v[3] & self.TYPE_CSTTN) != 0}
        cap_mult_infra_nodes = {i: self.cap_mult_infra_nodes for i,v in enumerate(V) if (v[3] & self.TYPE_INFRA) != 0}
        cap_mult_rsttn_nodes = {i: self.cap_mult_rsttn_nodes for i,v in enumerate(V) if (v[3] & self.TYPE_RSTTN) != 0}

        cap_base_canal_edges = {(e[0], e[1]): self.cap_base_canal_edges for e in E if (e[3] & self.TYPE_CANAL) != 0}
        cap_base_trail_edges = {(e[0], e[1]): self.cap_base_trail_edges for e in E if (e[3] & self.TYPE_TRAIL) != 0}
        cap_base_railw_edges = {(e[0], e[1]): self.cap_base_railw_edges for e in E if (e[3] & self.TYPE_RAILW) != 0}

        cap_mult_canal_edges = {(e[0], e[1]): self.cap_mult_canal_edges for e in E if (e[3] & self.TYPE_CANAL) != 0}
        cap_mult_trail_edges = {(e[0], e[1]): self.cap_mult_trail_edges for e in E if (e[3] & self.TYPE_TRAIL) != 0}
        cap_mult_railw_edges = {(e[0], e[1]): self.cap_mult_railw_edges for e in E if (e[3] & self.TYPE_RAILW) != 0}

        cap_base = {**cap_null_nodes,
                    **cap_null_edges,
                    **cap_base_csttn_nodes,
                    **cap_base_infra_nodes,
                    **cap_base_rsttn_nodes,
                    **cap_base_canal_edges,
                    **cap_base_trail_edges,
                    **cap_base_railw_edges}

        cap_mult = {**cap_null_nodes,
                    **cap_null_edges,
                    **cap_mult_csttn_nodes,
                    **cap_mult_infra_nodes,
                    **cap_mult_rsttn_nodes,
                    **cap_mult_canal_edges,
                    **cap_mult_trail_edges,
                    **cap_mult_railw_edges}

        # Flow conservation constraints: for each o/d pair and for
        # each node.

        for i,v in enumerate(V):
            if len(neighbors[i]) > 0:
                for (orig,dest) in ODpairs:
                    p.add_constr(mip.xsum(flow[i,j,orig,dest] - flow[j,i,orig,dest] for j in neighbors[i]) ==
                                 (satisf[orig,dest] if i == orig else
                                 -satisf[orig,dest] if i == dest else 0), name='flowcons_{0}_{1}_{2}'.format(i, orig, dest))

        # Flow limiting: at most one unit of flow entering/leaving every node

        for i,v in enumerate(V):
            if len(neighbors[i]) > 0:
                for (orig,dest) in ODpairs:
                    if i in [orig,dest]:
                        rhs = 1
                    elif len(neighbors[i]) > 1:
                        rhs = 2
                    else:
                        continue
                    p.add_constr(mip.xsum(flow[i,j,orig,dest] + flow[j,i,orig,dest] for j in neighbors[i]) <= rhs,
                                 name='limitflow_{0}_{1}_{2}'.format(i, orig, dest))

        # Edge capacity constraints

        for e in E:
            if cap_base[e[0],e[1]] >= 0 or cap_mult[e[0],e[1]] >= 0:
                p.add_constr(mip.xsum(ODpairs[orig,dest] * flow[e[0],e[1],orig,dest] for (orig,dest) in ODpairs) <=
                             max(0, cap_base[e[0],e[1]]) + max(0, cap_mult[e[0],e[1]]) * edge[e[0],e[1]], name='edge_capacityF_{0}_{1}'.format(e[0], e[1]))
                p.add_constr(mip.xsum(ODpairs[orig,dest] * flow[e[1],e[0],orig,dest] for (orig,dest) in ODpairs) <=
                             max(0, cap_base[e[0],e[1]]) + max(0, cap_mult[e[0],e[1]]) * edge[e[0],e[1]], name='edge_capacityB_{0}_{1}'.format(e[0], e[1]))

        # Node capacity constraints

        for i,v in enumerate(V):
            if cap_base[i] >= 0 or cap_mult[i] >= 0:
                p.add_constr(mip.xsum(ODpairs[orig,dest] * (flow[i,j,orig,dest] + flow[j,i,orig,dest])
                                      for (orig,dest) in ODpairs
                                      for j in neighbors[i]) <= max(0, cap_base[i]) + max(0, cap_mult[i]) * node[i],
                             name='node_capacity_{0}'.format(i))

        # # A subtle constraint: for each canal station node, a transit
        # # flow (i.e. a flow that does not originate or end at that
        # # node) can only enter or exit, but not conserve flow, on the
        # # canal link edges, i.e., on the fictitious links from the
        # # canal station node to the endpoints of the canal.

        # for i,v in enumerate(V):
        #     if (v[3] & self.type_CSTTN):
        #         for (orig,dest) in ODpairs:
        #             if i not in [orig,dest]:
        #                 p.add_constr(mip.xsum(flow[h,k,orig,dest] + flow[k,h,orig,dest]
        #                                       for (h,k,c,e,f) in E
        #                                       if i==k and (e & self.TYPE_CSTTN) != 0) <= 1,
        #                              name='use_canal_{0}_{1}_{2}'.format(i, orig, dest))

        # Another constraint that should fix the mess made by virtual
        # links: demand links, i.e. access of each source to a point
        # of the actual network, should only be used on exit at the
        # source by the flow that originates there, and similarly for
        # the destination.

        for (i,j,c,e,f) in E:
            if e == self.TYPE_ODTRF:
                for (orig,dest) in ODpairs:
                    if i != orig and j != dest:
                        p.add_constr(flow[i,j,orig,dest] == 0, name='avoid_vlinksF_{0}_{1}_{2}_{3}'.format(i, j, orig, dest))
                    if j != orig and i != dest:
                        p.add_constr(flow[j,i,orig,dest] == 0, name='avoid_vlinksB_{0}_{1}_{2}_{3}'.format(i, j, orig, dest))

        if options['objective'] == 'min_cost':

            p.objective = mip.minimize(mip.xsum(e[2] * edge[e[0],e[1]] for e in E) +
                                       mip.xsum(v[2] * node[i]         for i,v in enumerate(V) if v[3] != self.TYPE_ODTRF))

            p.add_constr(mip.xsum(                     satisf[orig,dest] for (orig, dest) in ODpairs) >= options['min_demands'] * len(ODpairs), name='min_demands')

            p.add_constr(mip.xsum(ODpairs[orig,dest] * satisf[orig,dest] for (orig, dest) in ODpairs) >= options['min_traffic'] * sum(ODpairs.values()), name='min_traffic')

        elif options['objective'] == 'max_pairs':

            p.add_constr(mip.xsum(e[2] * edge[e[0],e[1]] for e in E) +
                         mip.xsum(v[2] * node[i]         for i,v in enumerate(V)) <= options['budget'], name='budget_constr')

            p.objective = mip.maximize(mip.xsum(       satisf[orig,dest] for (orig, dest) in ODpairs))

            p.add_constr(mip.xsum(ODpairs[orig,dest] * satisf[orig,dest] for (orig, dest) in ODpairs) >= options['min_traffic'] * sum(ODpairs.values()), name='min_traffic')

        elif options['objective'] == 'max_traffic':

            p.add_constr(mip.xsum(e[2] * edge[e[0],e[1]] for e in E) +
                         mip.xsum(v[2] * node[i]         for i,v in enumerate(V)) <= options['budget'], name='budget_constr')

            p.add_constr(mip.xsum(satisf[orig,dest] for (orig, dest) in ODpairs) >= options['min_demands'] * len(ODpairs), name='min_demands')

            p.objective = mip.maximize(mip.xsum(ODpairs[orig,dest] * satisf[orig,dest] for (orig, dest) in ODpairs))

        self.model = p
        self.variables = {'flow': flow, 'edge': edge, 'node': node, 'satisf': satisf}
        self.ODpairs = ODpairs


    def formulate(self, **options):
        """Given the data from the layers after import, populate a
        mixed-integer linear optimization model to design the urban
        infrastructure.
        """

        print("Creating topology")
        self.create_topology(**options)
        print("Creating model")
        self.create_model(**options)
        print("Optimization model ready")

        self.options = options


    def solve(self):

        # self.model.write('urban.mps')

        status = self.model.optimize(max_seconds=10)

        V = self.V
        E = self.E

        flow = self.variables['flow']
        edge = self.variables['edge']
        node = self.variables['node']
        satisf = self.variables['satisf']

        ODpairs = self.ODpairs

        if status == mip.OptimizationStatus.OPTIMAL:
            print('Optimal solution of value {} found'.format(self.model.objective_value))
        elif status == mip.OptimizationStatus.FEASIBLE:
            print('Solution of value {} found, best possible: {}'.format(self.model.objective_value, self.model.objective_bound))
        elif status == mip.OptimizationStatus.NO_SOLUTION_FOUND:
            print('No feasible solution found, lower bound is: {}'.format(self.model.objective_bound))

        if status not in [mip.OptimizationStatus.OPTIMAL, mip.OptimizationStatus.FEASIBLE]:
            retval = None
        else:

            from qgis.core import QgsLayerTreeGroup, QgsProject

            group = QgsLayerTreeGroup("Solution flow")
            QgsProject.instance().layerTreeRoot().insertChildNode(1, group)   # 1 is position in layer directory

            # add_point_layer(group, [t[:4] for t in V if (t[3] & self.TYPE_CSTTN) != 0], attributes=['cost', 'type'], groupname='CANAL_STATN')
            # add_point_layer(group, [t[:4] for t in V if (t[3] & self.TYPE_INFRA) != 0], attributes=['cost', 'type'], groupname='INFRASTRUCT')
            # add_point_layer(group, [t[:4] for t in V if (t[3] & self.TYPE_TRAIL) != 0], attributes=['cost', 'type'], groupname='TRAIL_ENDS')
            # add_point_layer(group, [t[:4] for t in V if (t[3] & self.TYPE_ODTRF) != 0], attributes=['cost', 'type'], groupname='DEMAND')

            add_segment_layer(group, [[V[i][0],V[i][1],V[j][0],V[j][1],c,a] for (i,j,c,a,_) in E if (a & self.TYPE_CSTTN) != 0], attributes=['cost', 'type'], groupname='sol_join_canal',  flow=[sum(flow[i,j,o,d].x + flow[j,i,o,d].x for o,d in ODpairs) for (i,j,c,a,_) in E if (a & self.TYPE_CSTTN) != 0])
            add_segment_layer(group, [[V[i][0],V[i][1],V[j][0],V[j][1],c,a] for (i,j,c,a,_) in E if (a & self.TYPE_RSTTN) != 0], attributes=['cost', 'type'], groupname='sol_join_statn',  flow=[sum(flow[i,j,o,d].x + flow[j,i,o,d].x for o,d in ODpairs) for (i,j,c,a,_) in E if (a & self.TYPE_RSTTN) != 0])
            add_segment_layer(group, [[V[i][0],V[i][1],V[j][0],V[j][1],c,a] for (i,j,c,a,_) in E if (a & self.TYPE_ODTRF) != 0], attributes=['cost', 'type'], groupname='sol_demand_link', flow=[sum(flow[i,j,o,d].x + flow[j,i,o,d].x for o,d in ODpairs) for (i,j,c,a,_) in E if (a & self.TYPE_ODTRF) != 0])
            add_segment_layer(group, [[V[i][0],V[i][1],V[j][0],V[j][1],c,a] for (i,j,c,a,_) in E if (a & self.TYPE_CANAL) != 0], attributes=['cost', 'type'], groupname='sol_canal',       flow=[sum(flow[i,j,o,d].x + flow[j,i,o,d].x for o,d in ODpairs) for (i,j,c,a,_) in E if (a & self.TYPE_CANAL) != 0])
            add_segment_layer(group, [[V[i][0],V[i][1],V[j][0],V[j][1],c,a] for (i,j,c,a,_) in E if (a & self.TYPE_TRAIL) != 0], attributes=['cost', 'type'], groupname='sol_trail',       flow=[sum(flow[i,j,o,d].x + flow[j,i,o,d].x for o,d in ODpairs) for (i,j,c,a,_) in E if (a & self.TYPE_TRAIL) != 0])
            add_segment_layer(group, [[V[i][0],V[i][1],V[j][0],V[j][1],c,a] for (i,j,c,a,_) in E if (a & self.TYPE_RAILW) != 0], attributes=['cost', 'type'], groupname='sol_railway',     flow=[sum(flow[i,j,o,d].x + flow[j,i,o,d].x for o,d in ODpairs) for (i,j,c,a,_) in E if (a & self.TYPE_RAILW) != 0])

            # add_segment_layer(group, [[V[i][0],V[i][1],V[j][0],V[j][1],c,a] for (i,j,c,a,_) in Q if (a & self.TYPE_ODTRF) != 0], attributes=['cost', 'type'], groupname='OD_PAIRS')

            # for e in E:
            #     if edge[e[0],e[1]].x > 1e-6:
            #         print(f"{e[0]}-{e[1]}: (cost {e[2]}) {edge[e[0],e[1]].x} type {e[3]}")

            # for i,v in enumerate(V):
            #     if node[i].x > 1e-6:
            #         print(f"{i}: (cost {v[2]}) {node[i].x} type: {v[3]}")

            total_cost = sum(e[2] * edge[e[0],e[1]].x for e in E) + \
                         sum(v[2] * node[i].x         for i,v in enumerate(V) if v[3] != self.TYPE_ODTRF)

            ratio_demands = sum(satisf[orig,dest].x for (orig, dest) in ODpairs if satisf[orig,dest].x > 1e-6)

            ratio_traffic = sum(ODpairs[orig,dest] * satisf[orig,dest].x for (orig, dest) in ODpairs if satisf[orig,dest].x > 1e-6)

            V = self.V
            E = self.E

            featmap = self.gfeatures
            featmap[None] = None

            retval = {"cost": total_cost,
                      "#paths": ratio_demands,
                      "traffic": ratio_traffic,

                      "nodes":
                      {
                          "landing":
                          [featmap[f] for i, (_1,_2,_3,attributes,f) in enumerate(V) if node[i].x > 1e-6 and (attributes & self.TYPE_CSTTN) != 0],
                          "trail_end":
                          [featmap[f] for i, (_1,_2,_3,attributes,f) in enumerate(V) if node[i].x > 1e-6 and (attributes & self.TYPE_INFRA) != 0],
                          "railway_st":
                          [featmap[f] for i, (_1,_2,_3,attributes,f) in enumerate(V) if node[i].x > 1e-6 and (attributes & self.TYPE_RSTTN) != 0]
                      },

                      "edges":
                      {
                          "canal":
                          [featmap[l] for (i,j,_3,attributes,f) in E if (attributes & self.TYPE_CANAL) != 0 and edge[i,j].x > 1e-6 for l in f],
                          "trail":
                          [featmap[l] for (i,j,_3,attributes,f) in E if (attributes & self.TYPE_TRAIL) != 0 and edge[i,j].x > 1e-6 for l in f],
                          "railway":
                          [featmap[l] for (i,j,_3,attributes,f) in E if (attributes & self.TYPE_RAILW) != 0 and edge[i,j].x > 1e-6 for l in f]
                      },

                      "flow":
                      {
                          "canal":
                               [[featmap[l], sum(ODpairs[o,d] * (flow[i,j,o,d].x + flow[j,i,o,d].x) for (o,d) in ODpairs)]
                                for (i,j,_1,attributes,f) in E
                                if (attributes & self.TYPE_CANAL) != 0 and
                                   sum(flow[i,j,o,d].x + flow[j,i,o,d].x for (o,d) in ODpairs) > 1e-6 and
                                   f is not None
                                for l in f] + \
                               [[featmap[l], sum(ODpairs[o,d] * (flow[i,j,o,d].x + flow[j,i,o,d].x) for (o,d) in ODpairs)]
                                for id,(i,j,_1,attributes,f) in enumerate(E)
                                if (attributes & self.TYPE_CSTTN) != 0 and
                                   (V[i][3]    & self.TYPE_CSTTN) != 0 and
                                   sum(flow[i,j,o,d].x + flow[j,i,o,d].x for (o,d) in ODpairs) > 1e-6
                                for l in E[self.landing2canal[i]][4]] + \
                                [[featmap[l], sum(ODpairs[o,d] * (flow[i,j,o,d].x + flow[j,i,o,d].x) for (o,d) in ODpairs)]
                                for id,(i,j,_1,attributes,f) in enumerate(E)
                                if (attributes & self.TYPE_CSTTN) != 0 and
                                   (V[j][3]    & self.TYPE_CSTTN) != 0 and
                                   sum(flow[i,j,o,d].x + flow[j,i,o,d].x for (o,d) in ODpairs) > 1e-6
                                for l in E[self.landing2canal[j]][4]],

                          "trail":
                               [[featmap[l], sum(ODpairs[o,d] * (flow[i,j,o,d].x + flow[j,i,o,d].x) for (o,d) in ODpairs)]
                                 for (i,j,_1,attributes,f) in E
                                if (attributes & self.TYPE_TRAIL) != 0 and
                                   sum(flow[i,j,o,d].x + flow[j,i,o,d].x for (o,d) in ODpairs) > 1e-6 and
                                   f is not None
                                for l in f],

                          "railway":
                               [[featmap[l], sum(ODpairs[o,d] * (flow[i,j,o,d].x + flow[j,i,o,d].x) for (o,d) in ODpairs)]
                                for (i,j,_1,attributes,f) in E
                                if (attributes & self.TYPE_RAILW) != 0 and
                                   sum(flow[i,j,o,d].x + flow[j,i,o,d].x for (o,d) in ODpairs) > 1e-6 and
                                   f is not None
                                for l in f] + \
                               [[featmap[l], sum(ODpairs[o,d] * (flow[i,j,o,d].x + flow[j,i,o,d].x) for (o,d) in ODpairs)]
                                for id,(i,j,_1,attributes,f) in enumerate(E)
                                if (attributes & self.TYPE_RSTTN) != 0 and
                                   (V[i][3]    & self.TYPE_RSTTN) != 0 and
                                   sum(flow[i,j,o,d].x + flow[j,i,o,d].x for (o,d) in ODpairs) > 1e-6
                                for l in E[self.station2railw[i]][4]] + \
                               [[featmap[l], sum(ODpairs[o,d] * (flow[i,j,o,d].x + flow[j,i,o,d].x) for (o,d) in ODpairs)]
                                for id,(i,j,_1,attributes,f) in enumerate(E)
                                if (attributes & self.TYPE_RSTTN) != 0 and
                                   (V[j][3]    & self.TYPE_RSTTN) != 0 and
                                   sum(flow[i,j,o,d].x + flow[j,i,o,d].x for (o,d) in ODpairs) > 1e-6
                                for l in E[self.station2railw[j]][4]]
                      }
            }

            # print("flusso su canali:", retval['flow_canal'])

        return retval
