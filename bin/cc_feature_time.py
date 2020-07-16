#!/usr/bin/env python3

import obspy.taup as taup
import sys
import getopt
import numpy as np
import copy
class cc_feature_time:
    local_mod_ak135 = taup.TauPyModel('ak135')
    def __init__(self):
        pass
    def get_feature_time(self, phase1, phase2, inter_rcv_distance_deg, evlo_min=0.0, evlo_max= 180.0, max_interation= 100):
        """
        Given two phases observed at two receivers, the distance between
        those two receivers, return the theoretical feature time for the
        cross-term of those two phases under the same slowness condition.

        evlo_min, evlo_max: event longitude range for search.

        """
        stlo1, stlo2 =  0.0, inter_rcv_distance_deg
        ev1, ev2, ev3 = evlo_min, 0.5*(evlo_min+evlo_max), evlo_max
        #
        a1, a3 = [cc_feature_time.local_mod_ak135.get_travel_times(0.0, stlo1-it, [phase1])[-1] for it in [ev1, ev3] ]
        b1, b3 = [cc_feature_time.local_mod_ak135.get_travel_times(0.0, stlo2-it, [phase2])[-1] for it in [ev1, ev3] ]
        for interation in range(max_interation):
            ev2 = 0.5*(ev1+ev3)
            ###
            a2 = cc_feature_time.local_mod_ak135.get_travel_times(0.0, stlo1-ev2, [phase1])[-1]
            b2 = cc_feature_time.local_mod_ak135.get_travel_times(0.0, stlo2-ev2, [phase2])[-1]
            ###
            rpa = np.array([a1.ray_param, a2.ray_param, a3.ray_param])
            rpb = np.array([b1.ray_param, b2.ray_param, b3.ray_param])
            dista  = np.array([a1.distance, a2.distance, a3.distance])
            distb  = np.array([b1.distance, b2.distance, b3.distance])
            ###
            drp = rpb-rpa
            #print(dista, distb, a1.name, b1.name, rpa, rpb)
            #print(drp, ev2)
            #print( '%6.2f %5s %5s %6.2f %6.2f %6.2f %6.2f' % (ev2, a2.name, b2.name, a2.distance, b2.distance,  a2.ray_param, b2.ray_param ) )
            ###
            if abs(drp[1]) < 1.0e-18:
                break
            elif drp[0]*drp[1]< 0.0:
                ev3 = ev2
                a3 = copy.deepcopy(a2)
                b3 = copy.deepcopy(b2)
            else:
                ev1 = ev2
                a1 = copy.deepcopy(a2)
                b1 = copy.deepcopy(b2)
        ##
        arr1 = cc_feature_time.local_mod_ak135.get_travel_times(0.0, stlo1-ev2, [phase1])[-1]
        arr2 = cc_feature_time.local_mod_ak135.get_travel_times(0.0, stlo2-ev2, [phase2])[-1]
        return ev2, arr1.time-arr2.time, arr1.ray_param_sec_degree

if __name__ == "__main__":
    # -F PcS-PcP -D 10 -E -30/0
    ph1, ph2 = None, None
    dist = None
    ev1, ev2 = 0.0, 180.0
    options, remainder = getopt.getopt(sys.argv[1:], 'F:D:E:' )
    for opt, arg in options:
        if opt in ('-F'):
            ph1, ph2 = arg.split('-') 
        elif opt in ('-D'):
            dist = float(arg)
        elif opt in ('-E'):
            ev1, ev2 = [float(it) for it in arg.split('/')]
        else:
            print('invalid options: %s' % (opt) )
    if ph1 == None or ph2 == None or dist == None:
        print('e.g.: %s -F PcS-PcP -D 10 -E -30/0' % (sys.argv[0]) )
        sys.exit(-1)
    app = cc_feature_time()
    v1, v2, v3 = app.get_feature_time(ph1, ph2, dist, ev1, ev2)
    print('%f %f %f %f'% (dist, v1, v2, v3) )