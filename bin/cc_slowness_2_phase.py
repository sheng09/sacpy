#!/usr/bin/env python3

import obspy.taup as taup
import sys
import getopt
import numpy as np
import copy
class cc_slowness_2_phase:
    local_mod_ak135 = taup.TauPyModel('ak135')
    def __init__(self):
        pass
    def get_phase_from_slowness(self, phase, slowness_sec_degree, dist_min=0.0, dist_max= 180.0, max_interation= 100):
        """
        Return phase information given a slowness (sec per degree)
        """
        d1, d2, d3 = dist_min, 0.5*(dist_min+dist_max), dist_max
        a1, a3 = [cc_slowness_2_phase.local_mod_ak135.get_travel_times(0.0, it, [phase])[0] for it in [d1, d3] ]
        for iteration in range(max_interation):
            d2 = 0.5*(d1+d3)
            a2 = cc_slowness_2_phase.local_mod_ak135.get_travel_times(0.0, d2, [phase])[0]
            ###
            rp = np.array( [a1.ray_param_sec_degree, a2.ray_param_sec_degree, a3.ray_param_sec_degree] )
            drp = rp - slowness_sec_degree
            print((d1, d2, d3), drp)
            if abs(drp[1]) < 1.0e-18:
                break
            elif drp[0]*drp[1]< 0.0:
                d3 = d2
                a3 = copy.deepcopy(a2)
            else:
                d1 = d2
                a1 = copy.deepcopy(a2)
        return d2, a2.time, a2.ray_param_sec_degree

if __name__ == "__main__":
    # -P PKIKS -D 170/179 -S 3.1
    ph = None
    d1, d2 = None, None
    rp = None
    options, remainder = getopt.getopt(sys.argv[1:], 'P:D:S:' )
    for opt, arg in options:
        if opt in ('-P'):
            ph = arg
        elif opt in ('-D'):
            d1, d2 = [float(it) for it in arg.split('/')]
        elif opt in ('-S'):
            rp = float(arg)
        else:
            print('invalid options: %s' % (opt) )
    if ph == None or d1 == None or d2 == None or rp == None:
        print('e.g.: %s -P PKIKS -D 170/179 -S 3.1' % (sys.argv[0]) )
        sys.exit(-1)
    app = cc_slowness_2_phase()
    v1, v2, v3 = app.get_phase_from_slowness(ph, rp, d1, d2)
    print('%f %f %f'% (v1, v2, v3) )