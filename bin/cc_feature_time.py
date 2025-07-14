#!/usr/bin/env python3

import sys
import getopt
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import sacpy.correlation as correlation
from sacpy.correlation import get_all_interrcv_ccs
from taupplotlib import plotStation, plotPrettyEarth, plot_raypath_geo_arrivals, colored_line, geo_arrival
import numpy as np
from obspy import taup
from sacpy.utils import Timer

class cc_feature_time:
    def __init__(self, model_name, evdp_km, rcv1dp_km, rcv2dp_km, show, figname, debug=False):
        self.model_name= model_name
        self.evdp_km   = evdp_km
        self.rcv1dp_km = rcv1dp_km
        self.rcv2dp_km = rcv2dp_km
        self.show      = show
        self.figname   = figname
        self.debug = debug
    def run(self, distances, feature_name, show=False, figname=None, print_result=True, dist_accuracy_deg=0.5, max_interation=10):
        ####
        if distances is not None:
            taupy_mod = taup.TauPyModel(self.model_name)
            tau_mod   = taupy_mod.model
            if '-' in feature_name:
                tau_modc1 = correlation.get_corrected_model(tau_mod, self.evdp_km, self.rcv1dp_km)
                tau_modc2 = correlation.get_corrected_model(tau_mod, self.evdp_km, self.rcv2dp_km) if rcv1dp_km != rcv2dp_km else tau_modc1
                phase1, phase2 = feature_name.split('-')
                rps, cc_time, trvt1, trvt2, cc_pd, cc_d, pd1, pd2 = correlation.get_ccs_from_cc_dist(distances, tau_modc1, tau_modc2, phase1, phase2, self.rcv1dp_km, self.rcv2dp_km, dist_accuracy_deg, max_interation)
                #####
                self.print_result(feature_name, rps*(np.pi/180), cc_time, trvt1, trvt2, cc_pd, cc_d, pd1, pd2)
                self.plot(feature_name, pxt_data=(rps, cc_d, cc_time), vol_lst=None)
            elif feature_name[-1] == '*':
                arrs = list()
                for d in distances:
                    arrs.extend( taupy_mod.get_travel_times(self.rcv1dp_km, d, [ feature_name[:-1] ], receiver_depth_in_km=self.rcv2dp_km) )
                rps     = np.array([it.ray_param for it in arrs], dtype=np.float64 )
                cc_time = np.array([it.time for it in arrs], dtype=np.float64 )
                cc_d    = np.array([it.distance for it in arrs], dtype=np.float64 )
                pd      = np.array([it.purist_distance for it in arrs], dtype=np.float64 )
                #
                tmp     = np.argsort(rps)
                rps, cc_time, cc_d, pd = rps[tmp], cc_time[tmp], cc_d[tmp], pd[tmp]
                #
                trvt1, cc_pd, pd1   = cc_time, pd, pd
                trvt2   = np.zeros(rps.size, dtype=np.float64)
                pd2     = trvt2 # all zeros
                ####
                self.print_result(feature_name, rps*(np.pi/180), cc_time, trvt1, trvt2, cc_pd, cc_d, pd1, pd2)
                self.plot(feature_name, pxt_data=(rps, cc_d, cc_time), vol_lst=None)
        else:
            vol_lst  = get_all_interrcv_ccs(   cc_feature_names=[feature_name], evdp_km=self.evdp_km, model_name=self.model_name,
                                               rcvdp1_km=self.rcv1dp_km, rcvdp2_km=self.rcv2dp_km, keep_feature_names=True, compute_geo_arrivals=True, log=None )
            if len(vol_lst) == 1:
                rps     = vol_lst[0]['ray_param']
                cc_time = vol_lst[0]['time']
                cc_d    = vol_lst[0]['distance']
                trvt1   = np.array([it.time for it in vol_lst[0]['geo_arr']], dtype=np.float64)
                trvt2   = np.array([it.time for it in vol_lst[0]['geo_arr2']], dtype=np.float64) if '-' in feature_name else np.zeros(rps.size, dtype=np.float64)
                pd1     = np.array([it.purist_distance for it in vol_lst[0]['geo_arr']], dtype=np.float64)
                pd2     = np.array([it.purist_distance for it in vol_lst[0]['geo_arr2']], dtype=np.float64) if '-' in feature_name else np.zeros(rps.size, dtype=np.float64)
                cc_pd   = pd1-pd2
                ####
                self.print_result(feature_name, rps*(np.pi/180), cc_time, trvt1, trvt2, cc_pd, cc_d, pd1, pd2)
                self.plot(feature_name, pxt_data=None, vol_lst=vol_lst)
    def plot(self, feature_name, pxt_data=None, vol_lst=None):
        if (not self.show) and (not self.figname):
            return
        if vol_lst is None:
            vol_lst  = get_all_interrcv_ccs(   cc_feature_names=[feature_name], evdp_km=self.evdp_km, model_name=self.model_name,
                                               rcvdp1_km=self.rcv1dp_km, rcvdp2_km=self.rcv2dp_km, keep_feature_names=True, compute_geo_arrivals=True, log=None )
        if len(vol_lst) > 0:
            if True: # basemap
                fig = plt.figure(figsize=(12, 7) )
                grd = fig.add_gridspec(2, 4, wspace=0.35, width_ratios = (1, 1, 1, 0.03) )
                ax1 = fig.add_subplot(grd[0, 0], projection='polar')
                ax2 = fig.add_subplot(grd[0, 1], projection='polar')
                ax3 = fig.add_subplot(grd[0, 2], projection='polar')
                ax4 = fig.add_subplot(grd[1, 0])
                ax5 = fig.add_subplot(grd[1, 1])
                ax6 = fig.add_subplot(grd[1, 2])
                cax = fig.add_subplot(grd[1, 3])
            ############################
            if pxt_data is not None: # plot for the rps obtained above
                rps, cc_d, cc_time = [np.array(lst) for lst in pxt_data]
                ax4.scatter(cc_d,    cc_time,         c='r', zorder=100)
                ax5.scatter(cc_d,    rps*(np.pi/180), c='r', zorder=100)
                ax6.scatter(cc_time, rps*(np.pi/180), c='r', zorder=100)
                phase_names = [feature_name[:-1]] if feature_name[-1] == '*' else feature_name.split('-')
                for ax, phase_name in zip( (ax1, ax2),  phase_names):
                    for it_p in rps:
                        geo_ar = geo_arrival(self.rcv1dp_km, 0.0, self.rcv2dp_km, phase_name=phase_name, ray_param=it_p, model=taup.TauPyModel(self.model_name) )
                        geo_ar.plot_raypath(ax=ax, color='r', zorder=100)
                        ax4.scatter(geo_ar.distance, geo_ar.time,                 c='r', zorder=100)
                        ax5.scatter(geo_ar.distance, geo_ar.ray_param_sec_degree, c='r', zorder=100)
                        ax6.scatter(geo_ar.time, geo_ar.ray_param_sec_degree,     c='r', zorder=100)
            ############################
            if True: # plot for all possible obtained from get_all_interrcv_ccs(...)
                element = vol_lst[0]
                ps = element['ray_param'] * (np.pi/180) # from s/radian to s/deg
                xs = element['distance']
                ts = element['time']
                pmin, pmax = ps.min(), ps.max()
                ############################
                colored_line(xs, ts, c=ps, ax=ax4, lw=4, vmin=pmin, vmax=pmax, cmap='plasma')
                colored_line(xs, ps, c=ps, ax=ax5, lw=4, vmin=pmin, vmax=pmax, cmap='plasma')
                colored_line(ts, ps, c=ps, ax=ax6, lw=4, vmin=pmin, vmax=pmax, cmap='plasma')
                # plot the colorbar with vmin and vmax in cax
                norm = mcolors.Normalize(vmin=pmin, vmax=pmax)
                cmap = plt.get_cmap('plasma')
                sm = cm.ScalarMappable(norm=norm, cmap=cmap)
                sm.set_array([])  # Dummy array for the ScalarMappable
                plt.colorbar(sm, label=r'Slowness (s/$\degree$)', cax=cax)
                if True: # adjust
                    for ax in (ax4, ax5, ax6):
                        ax.grid(True, ls='--', color='gray', lw=0.6)
                        ax.set_xlabel(r'Distance ($\degree$)')
                    ax4.set_ylabel('Time (s)')
                    ax5.set_ylabel(r'Slowness (s/$\degree$)')
                    ax6.set_ylabel(r'Slowness (s/$\degree$)')
                    ax4.set_title('X-T')
                    ax5.set_title('X-P')
                    ax6.set_title('T-P')
                    ax4.text(xs[0], ts[0], feature_name, clip_on=False)
                    ax5.text(xs[0], ps[0], feature_name, clip_on=False)
                    ax6.text(ts[0], ps[0], feature_name, clip_on=False)
            ############################
            if True: # plot body waves for the cross-term
                if feature_name[-1] == '*':
                    ax2.axis('off')
                    ax3.axis('off')
                    phase_name = feature_name[:-1]
                    ax1.set_title(feature_name)
                    plot_raypath_geo_arrivals(element['geo_arr'][::5], c=ps[::5], ax=ax1, plot_pretty_earth=True, vmin=pmin, vmax=pmax, cmap='plasma')
                elif '-' in feature_name:
                    ax3.axis('off')
                    plot_raypath_geo_arrivals(element['geo_arr'][::5], c=ps[::5], ax=ax1, plot_pretty_earth=True, vmin=pmin, vmax=pmax, cmap='plasma')
                    plot_raypath_geo_arrivals(element['geo_arr2'][::5], c=ps[::5], ax=ax2, plot_pretty_earth=True, vmin=pmin, vmax=pmax, cmap='plasma')
                    phase1_name, phase2_name = feature_name.split('-')
                    ax1.set_title(phase1_name)
                    ax2.set_title(phase2_name)
                    #####
                    for lst in (element['geo_arr'], element['geo_arr2']): # plot ray path for the cross term
                        if len(lst) <=0:
                            continue
                        xs = [it.distance for it in lst]
                        ts = [it.time for it in lst]
                        ps = np.abs( [it.ray_param_sec_degree for it in lst] )
                        colored_line(xs, ts, c=ps, ax=ax4, lw=2, vmin=pmin, vmax=pmax, cmap='plasma')
                        colored_line(xs, ps, c=ps, ax=ax5, lw=2, vmin=pmin, vmax=pmax, cmap='plasma')
                        colored_line(ts, ps, c=ps, ax=ax6, lw=2, vmin=pmin, vmax=pmax, cmap='plasma')
                        ax4.text(xs[0], ts[0], lst[0].name, clip_on=False)
                        ax5.text(xs[0], ps[0], lst[0].name, clip_on=False)
                        ax6.text(ts[0], ps[0], lst[0].name, clip_on=False)
            if self.figname:
                plt.savefig(self.figname, transparent=True, )
            if self.show:
                plt.show()
    def print_result(self, feature_name, ray_params, cc_time, trvt1, trvt2, cc_purist_distance, cc_distance, purist_distance1, purist_distance2):
        if not self.print_result:
            return
        print('#############################################################################')
        print('################################# Result ####################################')
        print('#############################################################################')
        print('#Name inter-dist(deg)  dist1(deg)  dist2(deg)  ray_param(s/deg)  t1(s)  t2(s)  cc_time(s)')
        #PcS-PcP 10.0 28.244999 38.244999 3.100130 770.354077 576.068671 194.285406
        distance1= correlation.round_degree_180(purist_distance1)
        distance2= correlation.round_degree_180(purist_distance2)
        for rp, ct, it1, it2, cpd, cd, d1, d2 in zip(ray_params, cc_time, trvt1, trvt2, cc_purist_distance, cc_distance, distance1, distance2):
            line = '%s %.3f %.3f %.3f %.6f %.3f %.3f %.3f' % (feature_name, cd, d1, d2, rp, it1, it2, ct)
            print(line)
if __name__ == "__main__":
    # -F PcS-PcP -D 10 -E -30/0
    evdp_km = 0.0
    rcv1dp_km, rcv2dp_km = (0.0, 0.0)
    cc_feature_name = None
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
            cc_feature_name = arg
            #ph1, ph2 = arg.split('-')
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
    if cc_feature_name == None:
        print('e.g.: %s -F PcS-PcP [-D 7.5,8.5,10,11,12,13] [-L out.png] [--accuracy=0.5] [--max_interation=5] [-S] [-V]' % (sys.argv[0]) )
        sys.exit(-1)
    with Timer(tag='#run main(...)', verbose=True):
        app = cc_feature_time('ak135', evdp_km, rcv1dp_km, rcv2dp_km, show=show, figname=figname, debug=debug)
        app.run(distances, cc_feature_name, show=show, figname=figname, print_result=True, dist_accuracy_deg=dist_accuracy, max_interation=max_interation)
      #self, distances, phase1, phase2, show=False, figname=None, print_result=True, dist_accuracy_deg=0.5, max_interation=10
    if debug:
        pass
        #print(app.time_summary )
        #app.time_summary.plot_rough()
        #app.time_summary.plot(show=True)
    #if verbose:
    #    app.verbose()
    