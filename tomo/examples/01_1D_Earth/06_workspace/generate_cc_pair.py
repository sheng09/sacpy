#!/usr/bin/env python3
import numpy as np
import random
import sacpy.geomath as GM

npair = 3000

vol = np.loadtxt( "../03_workspace/ev-sta.txt", dtype= {'names':   ('id',    'evdp',    'evla',    'evlo',    'stla',    'stlo',   'phase', 'tag' ),
                                                  'formats': ('int32',  'float64', 'float64', 'float64', 'float64', 'float64', object, object) } )
map_vol = dict()
for line in vol:
    map_vol[line['id']] = line

set_pair = set()
for id1, ph1 in zip(vol['id'], vol['phase']):
    for id2, ph2 in zip(vol['id'], vol['phase'] ):
        if ( ph1[-1] == 'S' ) and ( (ph1[:-1] + 'P') == ph2 ) : # xS - xP
            set_pair.add( (id1, id2) )

# random select pair
pairs = sorted( random.sample(set_pair, npair ) )

interdis =lambda id1, id2 : GM.haversine( map_vol[id1]['stlo'], map_vol[id1]['stla'], map_vol[id2]['stlo'], map_vol[id2]['stla']  )
epidis =  lambda id : GM.haversine(map_vol[id]['evlo'], map_vol[id]['evla'], map_vol[id]['stlo'], map_vol[id]['stla'] )
azimuth = lambda id : GM.azimuth(  map_vol[id]['evlo'], map_vol[id]['evla'], map_vol[id]['stlo'], map_vol[id]['stla'] )

fid = open('cc.txt', 'w')
fid.write("#ccid  id1  id2  tag(string_without_whitespaces)\n")
for idx, (id1, id2) in enumerate(pairs):
    d1 = epidis(id1)
    d2 = epidis(id2)
    inter_d = interdis(id1, id2)
    az1 =azimuth(id1)
    az2 =azimuth(id2)
    tag = 'd1(%.2f)_d2(%.2f)_interd(%.2f)_az1(%.2f)_az2(%.2f)_daz(%.2f)_cc(%s-%s)' % (d1, d2, inter_d, az1, az2, az1-az2, map_vol[id1]['tag'], map_vol[id2]['tag'] )

    line = "%d %d %d  %s\n" % (idx+1, id1, id2, tag)
    fid.write(line)
