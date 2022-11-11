#!/usr/bin/env python3

import sys
import getopt
import matplotlib.pyplot as plt
from time import time as compute_time
from obspy import taup
from numpy import arange, array, linspace, interp, argsort, argmin, deg2rad, column_stack, where
from numpy.random import randint
from time import time as compute_time
import sacpy
import os.path
from sacpy.taupplotlib import plotPrettyEarth, plotStation, plotEq
import pickle

print(taup.__path__)

class cc_feature_time:
    def __init__(self, model_name='ak135'):
        app_id = randint(0, 999999)
        self.mod = taup.TauPyModel(model_name)
        self.summary = list()
        pass
    def __get_rp_dist_time(self, phase, dist_start=0.0, dist_end=180.0, dist_step=0.5, evdp_km= 0.0):
        """
        """
        ttmp = compute_time()
        ###
        filename = '%s/bin/dataset/cc_feature_time.pkl' % sacpy.__path__[0]
        evdp_meter = int(evdp_km*1000)
        search_key = (phase, evdp_meter)
        vol = dict()
        if os.path.exists(filename):
            with open(filename, 'rb') as fid:
                vol = pickle.load(fid)
        if search_key not in vol:
            tmp  = [self.mod.get_travel_times(evdp_km, it, [phase]) for it in arange(dist_start, dist_end+dist_step, dist_step)]
            arrs = [it for sub in tmp for it in sub]
            rp          = array( [it.ray_param_sec_degree for it in arrs] )
            purist_dist = array( [it.purist_distance for it in arrs] ) # purist_distance other than actual distance
            time        = array( [it.time for it in arrs] )
            # sort
            idx = argsort(rp)
            rp          = rp[idx]
            purist_dist = purist_dist[idx]
            time        = time[idx]
            vol[search_key] = { 'purist_dist': purist_dist, 'rp': rp, 'time': time }
            with open(filename, 'wb') as fid:
                pickle.dump(vol, fid)###
        ###
        return vol[search_key], compute_time()-ttmp
    def __plot(self, found, rp1, rp2, pd1, pd2, phase1, phase2, show=False, figname=None):
        """
        """
        ttmp = compute_time()
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
        radius = self.mod.model.radius_of_planet
        for (ax1, ax2), it_found in zip(axmat, found):
            r1, r2, s_found, rp_found, n_overlap, t1_found, t1_found, ct_found = it_found
            clrs, lss = ('C0', 'k'), ('-', '--')
            ##############################################################################################################
            # Plot ray paths
            ##############################################################################################################
            for rcv, clr, ls, phase in zip((r1, r2), clrs, lss, (phase1, phase2)):
                plotStation(ax1, self.mod, (0.0,), (rcv,), clr)
                arrs = self.mod.get_ray_paths(0.0, s_found-rcv, [phase[::-1]]) # the phase name are inverted as we emit the waves from the source to the receiver
                idx = argmin( [abs(it.ray_param_sec_degree-rp_found) for it in arrs] )
                arr = arrs[idx]
                lons = arr.path['dist'] + deg2rad(rcv)
                rs   = radius-arr.path['depth']
                ax1.plot(lons, rs, color=clr, linestyle=ls)
            plotEq(ax1, self.mod, (0.0,), (s_found,), markersize=16)
            plotPrettyEarth(ax1, self.mod, True, 'core')
            ##############################################################################################################
            # Plot ray_param versus src_loc curves for the two seismic waves to show the intersection, the found source
            ##############################################################################################################
            s1 = r1+pd1
            s2 = r2+pd2+n_overlap*360
            for _rp, _s, clr, ls, phase in zip( (rp1, rp2), (s1, s2), clrs, lss, (phase1, phase2) ):
                ax2.plot(_rp, _s, label=phase, color=clr, linestyle=ls)
            ax2.plot(rp_found, s_found, 'o', color='k')
            #
            rp_min = min(rp1.min(), rp2.min() )
            rp_max = max(rp1.max(), rp2.max() )
            s_min = min(s1.min(), s2.min() )
            s_max = max(s1.max(), s2.max() )
            s_min = s_min - 0.03*(s_max-s_min)
            s_max = s_max + 0.03*(s_max-s_min)
            ax2.plot((rp_min, rp_max), (s_found, s_found), 'k:', linewidth=0.6)
            s_found_valid = s_found%360
            if s_found == s_found_valid:
                ax2.text(rp_min, s_found, '%.2f$\degree$' % s_found  )
            else:
                ax2.text(rp_min, s_found, '%.2f$\degree$(%.2f$\degree$)' % (s_found_valid, s_found) )
            #
            ax2.plot((rp_found, rp_found), (s_min, s_max), 'k:', linewidth=0.6)
            ax2.text(rp_found, s_min, '%.2f$\degree$/s' % rp_found  )
            #
            ax2.text(rp_min, s_min, '$T_{cc}=$%.2fsec' % ct_found )
            #
            ax2.set_ylim((s_min, s_max) )
            ax2.set_xlabel('Ray parameter (s/$\degree$)')
            yticks = ax2.get_yticks()
            ax2.set_yticks(yticks)
            ax2.set_yticklabels( [ '%03d %03d' % (it, it%360) for it in yticks] )
            ax2.set_ylabel('Source location ($\degree$)')
        axmat[0][1].legend(loc=(0.0, 1.01) )
        ##############################################################################################################
        # Finished
        ##############################################################################################################
        if figname!=None:
            plt.savefig(figname, bbox_inches = 'tight', pad_inches = 0.2)
        time_consumption = compute_time() - ttmp
        if show:
            plt.show()
        plt.close()
        return time_consumption
    def search_inter_rcv(self, inter_rcv_dist, phase1, phase2, evdp_km, show=False, figname= None, print_result=False, accuracy_degree=0.1):
        """
        """
        search_id = randint(0, 999999)
        info = '%s-%s %6.2f(dist-$\degree$) %7.2f(evdp-km)' % (phase1, phase2, inter_rcv_dist, evdp_km)
        single_search_summary = {'0-search_id': search_id, '1-info': info}
        self.summary.append( single_search_summary )
        ##############################################################################################################
        #  Step1:
        #  We set two receivers r1 and r2, and we emit two seismic waves of the same slowness rp from the receivers.
        #  The waves will travel purist distance pd1 and pd2, and the waves will arrive at the r1+pd1 and r2+pd2
        #  that are the locations of the two sources. s1=r1+pd1, s2=r2+pd2
        #
        #  Step2:
        #  If s1 equals to s2 (s1-s2=0), or (s1-s2)%360=0, then the sources are at the same location. In that way,
        #  we find the slowness rp, the purist distances, and the location of the same source.
        ##############################################################################################################
        # step 1:  Get rp-dist-time data for the two seismic phases
        vol1, com_t1 = self.__get_rp_dist_time(phase1[::-1], evdp_km=evdp_km) # the phase name are inverted as we emit the waves
        vol2, com_t2 = self.__get_rp_dist_time(phase2[::-1], evdp_km=evdp_km) # from the source to the receiver
        rp1, pd1, t1 = vol1['rp'], vol1['purist_dist'], vol1['time']
        rp2, pd2, t2 = vol2['rp'], vol2['purist_dist'], vol2['time']
        single_search_summary['2-search for phase #1 (ms)'] = com_t1*1000
        single_search_summary['3-search for phase #2 (ms)'] = com_t1*1000
        ##############################################################################################################
        ttmp = compute_time()
        # step 2:
        rp_min = max(rp1.min(), rp2.min() )
        rp_max = min(rp1.max(), rp2.max() )
        denser_rp  = linspace(rp_min, rp_max, int(1440/accuracy_degree)+1)
        denser_pd1 = interp(denser_rp, rp1, pd1)
        denser_pd2 = interp(denser_rp, rp2, pd2)
        sta_loc_pair= ( (0.0, inter_rcv_dist), (inter_rcv_dist, 0.0) ) # rotate between the two receivers
        found = list()
        for r1, r2 in sta_loc_pair:
            s1 = r1+denser_pd1
            s2 = r2+denser_pd2
            diff = s1-s2
            for n_overlap in arange(diff.min()//360, diff.max()//360 + 1 ):
                relative_diff = diff - n_overlap*360 # look for diff across 0 degree or multiple of 360 degree
                product = relative_diff[:-1]*relative_diff[1:]
                idx = where(product<=0.0)[0]
                if idx.size > 0:
                    found.extend( [(r1, r2, s_found, rp_found, n_overlap) for (s_found, rp_found) in zip(s1[idx], denser_rp[idx]) ] )
        found = array(found)
        rps = found[:,3]
        t1_found = interp(rps, rp1, t1)
        t2_found = interp(rps, rp2, t2)
        ct_found = t1_found-t2_found
        found = column_stack( [ found, t1_found, t2_found, ct_found ] )
        # sort with respect to rp
        rp_found = found[:,3]
        idx = argsort(rp_found)
        found = found[idx]
        #
        single_search_summary['4-search for correlation feature (ms)'] = (compute_time()-ttmp)*1000
        ##############################################################################################################
        # Print the result
        ##############################################################################################################
        if print_result:
            print('#cross-term inter-dist(deg)  dist1(deg)  dist2(deg)  ray_param(s/deg)  t1(s)  t2(s)  cc_time(s)')
            for row in found:
                r1, r2, s, rp, n_overlap, t1, t2, ct = row
                print('%s-%s %6.2f %7.2f %7.2f %7.2f %9.2f %9.2f %9.2f' % (phase1, phase2, inter_rcv_dist, s-r1, s-r2, rp, t1, t2, ct) )
        ##############################################################################################################
        # Optional plot
        ##############################################################################################################
        if show or figname:
            p_time = self.__plot(found, rp1, rp2, pd1, pd2, phase1, phase2, show=show, figname=figname)
            single_search_summary['5-plot (ms)'] = p_time*1000
    def verbose(self):
        print('#############################################################################')
        print('################################# Summary ###################################')
        print('#############################################################################')
        for vol in self.summary:
            for key in sorted( vol.keys() ):
                print('#%-38s:' % key, vol[key])


if __name__ == "__main__":
    # -F PcS-PcP -D 10 -E -30/0
    ph1, ph2 = None, None
    dist = None
    show = False
    figname = None
    verbose = False
    options, remainder = getopt.getopt(sys.argv[1:], 'F:D:L:SV' )
    for opt, arg in options:
        if opt in ('-F'):
            ph1, ph2 = arg.split('-')
        elif opt in ('-D'):
            dist = float(arg)
        elif opt in ('-S'):
            show = True
        elif opt in ('-L'):
            figname = arg
        elif opt in ('-V'):
            verbose = True
        else:
            print('invalid options: %s' % (opt) )
    if ph1 == None or ph2 == None or dist == None:
        print('e.g.: %s -F PcS-PcP -D 10 [-L out.png] [-S] [-V]' % (sys.argv[0]) )
        sys.exit(-1)
    app = cc_feature_time()
    app.search_inter_rcv(dist, ph1, ph2, evdp_km=0.0, show=show, figname=figname, print_result=True)
    if verbose:
        app.verbose()