#!/usr/bin/env python3

import sys
import getopt
import matplotlib.pyplot as plt
import sacpy.correlation as correlation
import numpy as np
from obspy import taup
from sacpy.utils import Timer

class cc_feature_time:
    def __init__(self, model_name, evdp_km, rcv1dp_km, rcv2dp_km, debug=False):
        self.tau_mod = taup.TauPyModel(model_name).model
        self.tau_modc1 = correlation.get_corrected_model(self.tau_mod, evdp_km, rcv1dp_km)
        self.tau_modc2 = self.tau_modc1
        if rcv1dp_km != rcv2dp_km:
            self.tau_modc2 = correlation.get_corrected_model(self.tau_mod, evdp_km, rcv2dp_km)
        self.rcv1dp, self.rcv2dp = rcv1dp_km, rcv2dp_km
        self.debug = debug
        pass
    def run(self, distances, phase1, phase2, show=False, figname=None, print_result=True, dist_accuracy_deg=0.5, max_interation=10):
        if distances is None:
            rps, cc_time, trvt1, trvt2, cc_pd, cc_d, pd1, pd2 = correlation.get_ccs(self.tau_modc1, self.tau_modc2, phase1, phase2, self.rcv1dp, self.rcv2dp, dist_accuracy_deg, max_interation)
        else:
            rps, cc_time, trvt1, trvt2, cc_pd, cc_d, pd1, pd2 = correlation.get_ccs_from_cc_dist(distances, self.tau_modc1, self.tau_modc2, phase1, phase2, self.rcv1dp, self.rcv2dp, dist_accuracy_deg, max_interation)
        if self.verbose:
            self.verbose(phase1, phase2, rps, cc_time, trvt1, trvt2, cc_pd, cc_d, pd1, pd2)
    def verbose(self, phase1, phase2, ray_params, cc_time, trvt1, trvt2, cc_purist_distance, cc_distance, purist_distance1, purist_distance2):
        print('#############################################################################')
        print('################################# Result ####################################')
        print('#############################################################################')
        print('cross-term inter-dist(deg)  dist1(deg)  dist2(deg)  ray_param(s/deg)  t1(s)  t2(s)  cc_time(s)')
        #PcS-PcP 10.0 28.244999 38.244999 3.100130 770.354077 576.068671 194.285406
        feature_name = '%s-%s' % (phase1, phase2)
        distance1= correlation.round_degree_180(purist_distance1)
        distance2= correlation.round_degree_180(purist_distance2)
        for rp, ct, it1, it2, cpd, cd, d1, d2 in zip(ray_params, cc_time, trvt1, trvt2, cc_purist_distance, cc_distance, distance1, distance2):
            line = '%s %.3f %.3f %.3f %.6f %.3f %.3f %.3f' % (feature_name, cd, d1, d2, rp, it1, it2, ct)
            print(line)
    def __plot(self, found, rp1, rp2, pd1, pd2, phase1, phase2, show=False, figname=None):
        """
        """
        with Timer(message='plot', color='C5', verbose=self.debug, summary=self.time_summary):
            ##############################################################################################################
            # Init axes
            ##############################################################################################################
            nsol = len(found)
            fig = plt.figure(figsize=(11, 4*nsol) )
            axmat = [[None, None] for junk in range(nsol)]
            for isol in range(nsol):
                axmat[isol][0] = plt.subplot(nsol, 2, isol*2+1, projection='polar')
                axmat[isol][1] = plt.subplot(nsol, 2, isol*2+2)
            #fig, axmat = plt.subplots(nsol, 2, figsize=(nsol*5.5, 8) )
            #radius = self.mod.model.radius_of_planet
            #for (ax1, ax2), it_found in zip(axmat, found):
            #    r1, r2, s_found, rp_found, n_overlap, t1_found, t1_found, ct_found = it_found
            #    clrs, lss = ('C0', 'k'), ('-', '--')
            #    ##############################################################################################################
            #    # Plot ray paths
            #    ##############################################################################################################
            #    for rcv, clr, ls, phase in zip((r1, r2), clrs, lss, (phase1, phase2)):
            #        plotStation(ax1, self.mod, (0.0,), (rcv,), clr)
            #        arrs = self.mod.get_ray_paths(0.0, s_found-rcv, [phase[::-1]]) # the phase name are inverted as we emit the waves from the source to the receiver
            #        idx = argmin( [abs(it.ray_param_sec_degree-rp_found) for it in arrs] )
            #        arr = arrs[idx]
            #        lons = arr.path['dist'] + deg2rad(rcv)
            #        rs   = radius-arr.path['depth']
            #        ax1.plot(lons, rs, color=clr, linestyle=ls)
            #    #plotEq(ax1, self.mod, (0.0,), (s_found,), markersize=16)
            #    plotPrettyEarth(ax1, self.mod, True, 'core')
            #    ##############################################################################################################
            #    # Plot ray_param versus src_loc curves for the two seismic waves to show the intersection, the found source
            #    ##############################################################################################################
            #    s1 = r1+pd1
            #    s2 = r2+pd2+n_overlap*360
            #    for _rp, _s, clr, ls, phase in zip( (rp1, rp2), (s1, s2), clrs, lss, (phase1, phase2) ):
            #        ax2.plot(_rp, _s, label=phase, color=clr, linestyle=ls)
            #    ax2.plot(rp_found, s_found, 'o', color='k')
            #    #
            #    rp_min = min(rp1.min(), rp2.min() )
            #    rp_max = max(rp1.max(), rp2.max() )
            #    rp_min = rp_min-0.09*(rp_max-rp_min)
            #    rp_max = rp_max+0.09*(rp_max-rp_min)
            #    s_min = min(s1.min(), s2.min() )
            #    s_max = max(s1.max(), s2.max() )
            #    s_min = s_min - 0.09*(s_max-s_min)
            #    s_max = s_max + 0.09*(s_max-s_min)
            #    ax2.plot((rp_min, rp_max), (s_found, s_found), 'k:', linewidth=0.6)
            #    s_found_valid = s_found%360
            #    if s_found == s_found_valid:
            #        ax2.text(rp_min, s_found, '%.2f$\degree$' % s_found  )
            #    else:
            #        ax2.text(rp_min, s_found, '%.2f$\degree$(%.2f$\degree$)' % (s_found_valid, s_found) )
            #    #
            #    ax2.plot((rp_found, rp_found), (s_min, s_max), 'k:', linewidth=0.6)
            #    ax2.text(rp_found, s_min, '%.2f$\degree$/s' % rp_found  )
            #    #
            #    ax2.text(rp_min, s_min, '$T_{cc}=$%.2fsec' % ct_found )
            #    #
            #    ax2.set_xlim((rp_min, rp_max) )
            #    ax2.set_xlabel('Ray parameter (s/$\degree$)')
            #    yticks = ax2.get_yticks()
            #    ax2.set_yticks(yticks)
            #    ax2.set_yticklabels( [ '%d(%03d)' % (it%360, it) for it in yticks] )
            #    ax2.set_ylabel('Source location ($\degree$)')
            #    ax2.set_ylim((s_min, s_max) )
            #axmat[0][1].legend(loc=(0.0, 1.01) )
        #    ##############################################################################################################
        #    # Finished
        #    ##############################################################################################################
        #    if figname!=None:
        #        plt.savefig(figname, bbox_inches = 'tight', pad_inches = 0.2)
        #if show:
        #    plt.show()
        plt.close()

if __name__ == "__main__":
    # -F PcS-PcP -D 10 -E -30/0
    evdp_km = 0.0
    rcv1dp_km, rcv2dp_km = 0.0, 0.0
    ph1, ph2 = None, None
    distances = None
    show = False
    figname = None
    max_interation = 10
    dist_accuracy = 0.5
    verbose = False
    debug = False
    options, remainder = getopt.getopt(sys.argv[1:], 'F:D:L:SV', ['max_interation=', 'accuracy=', 'debug',] )
    for opt, arg in options:
        if opt in ('-F', ):
            ph1, ph2 = arg.split('-')
        elif opt in ('-D',):
            distances = np.array( [float(it) for it in arg.split(',')] )
        elif opt in ('-S', ):
            show = True
        elif opt in ('-L', ):
            figname = arg
        elif opt in ('--accuracy', ):
            dist_accuracy = float(arg)
        elif opt in ('--max_interation', ):
            max_interation = int(arg)
        elif opt in ('-V', ):
            verbose = True
        elif opt in ('--debug', ):
            debug = True
        else:
            print('invalid options: %s' % (opt) )
    if ph1 == None or ph2 == None:
        print('e.g.: %s -F PcS-PcP [-D 7.5,8.5,10,11,12,13] [-L out.png] [--accuracy=0.5] [--max_interation=5] [-S] [-V]' % (sys.argv[0]) )
        sys.exit(-1)
    with Timer(message='#run main(...)', verbose=True):
        app = cc_feature_time('ak135', evdp_km, rcv1dp_km, rcv2dp_km, debug=debug)
        app.run(distances, ph1, ph2, show=show, figname=figname, print_result=True, dist_accuracy_deg=dist_accuracy, max_interation=max_interation)
      #self, distances, phase1, phase2, show=False, figname=None, print_result=True, dist_accuracy_deg=0.5, max_interation=10
    if debug:
        pass
        #print(app.time_summary )
        #app.time_summary.plot_rough()
        #app.time_summary.plot(show=True)
    if verbose:
        app.verbose()
    