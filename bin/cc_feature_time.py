#!/usr/bin/env python3

import obspy.taup as taup
import sys
import getopt
import numpy as np
import copy
import matplotlib.pyplot as plt
import matplotlib.transforms
import pickle
import os.path
import sacpy
import sys

mod_ak135 = taup.TauPyModel('ak135')

def get_dist_rayparam_time(phase, dist_start=0.0, dist_end=180.0, dist_step=0.5, evdp_km= 0.0):
    filename = '%s/bin/dataset/cc_feature_time.pkl' % sacpy.__path__[0]
    vol = dict()
    if os.path.exists(filename):
        with open(filename, 'rb') as fid:
            vol = pickle.load(fid)
    if phase not in vol:
        ###
        arrs = []
        for dist in np.arange(dist_start, dist_end+dist_step, dist_step):
            arrs.extend( mod_ak135.get_travel_times(evdp_km, dist, [phase] )  )
        tmp = np.argsort( [it.ray_param_sec_degree for it in arrs] )
        distance = np.array( [arrs[idx].distance for idx in tmp] )
        rp       = np.array( [arrs[idx].ray_param_sec_degree for idx in tmp] )
        time     = np.array( [arrs[idx].time for idx in tmp] )
        vol[phase] = { 'distance': distance, 'rp': rp, 'time': time }
        ###
        with open(filename, 'wb') as fid:
            pickle.dump(vol, fid)
    #distance, rp, time = vol[phase]['distance'], vol[phase]['rp'], vol[phase]['time']
    return vol[phase]
def get_intersection(pt1, pt2, qt1, qt2):
    """
    Check if two line segments intersect and return the intersection point.
    return (Ture/False, (x, y) )
    """
    x1, y1 = pt1
    x2, y2 = pt2
    x3, y3 = qt1
    x4, y4 = qt2
    if x1 == x2 or x3 == x4:
        return False, (None, None)
    ##
    nx =   ((x1*y2 - x2*y1)*(x3 - x4)) - ((x3*y4 - x4*y3)*(x1 - x2) ) # x = nx/d
    ny =   ((x1*y2 - x2*y1)*(y3 - y4)) - ((x3*y4 - x4*y3)*(y1 - y2) ) # y = ny/d
    
    d  = (x1*y3 - x3*y1 - x1*y4 - x2*y3 + x3*y2 + x4*y1 + x2*y4 - x4*y2)
    
    #if d == 0.0:
    #    print(nx, ny, d, pt1, pt2, qt1, qt2)
    #    return False, (None, None)
    x = nx/d # if d != 0.0 else 0.0
    y = ny/d # if d != 0.0 else 0.0
    ## x1 <= x    <=  x2
    #  x1 <= nx/d <=  x2
    # dx1 <= nx   M= dx2
    flag1 = (x1-x)*(x2-x)
    flag2 = (x3-x)*(x4-x)
    if flag1 <=0 and flag2 <=0:
        #print(flag1, flag2)
        return True, (x, y)
    else:
        return False, (x, y)
def get_intersection_between_list_points(pts, qts):
    ips, iqs, xs, ys = list(), list(), list(), list()
    flag = False
    for ip, (pt1, pt2) in enumerate( zip( pts[:-1], pts[1:] ) ):
        for iq, (qt1, qt2) in enumerate(zip(qts[:-1], qts[1:] ) ):
            flag, (x, y) = get_intersection(pt1, pt2, qt1, qt2)
            #print(flag, x, y)
            #sys.exit()
            if flag:
                ips.append(ip)
                iqs.append(iq)
                xs.append(x)
                ys.append(y)
                flag=False
    if len(xs)>0:
        return True, ip, iq, xs, ys
    else:
        return False, ips, iqs, xs, ys
def plotPrettyEarth(ax, model, distlabel=False):
    """
    Plot Earth with discontinuities.
    ax: axes to plot;
    model: taup.model object;
    distlabel: False/True for degree label;
    """
    ax.set_theta_zero_location('N')
    #ax.set_theta_direction(-1)
    ax.set_xticks([])
    ax.set_yticks([])
    
    if distlabel:
        ticks = np.arange(0, 360, 45)
        ticklabels = [it if it <=180.0 else it-360.0 for it in ticks ]
        ax.set_xticks( np.deg2rad(ticks) )
        ax.set_xticklabels( ticklabels )
    ax.xaxis.grid(False)
    #ax.grid(False)
    radius = model.model.radius_of_planet
    discons = model.model.s_mod.v_mod.get_discontinuity_depths()
    ax.set_yticks(radius - discons)
    ax.yaxis.grid(True)
    #ax.xaxis.set_major_formatter(plt.NullFormatter())
    ax.yaxis.set_major_formatter(plt.NullFormatter())
    ax.set_ylim([0.0, model.model.radius_of_planet]) 
def plotStation(ax, model, stdp_list, stlo_list):
    """
    Plot stations.
    ax: axes to plot;
    model: taup.model object;
    stlo_list, stdp_list: list of longitude (degree) and depth for several stations;
    """
    stlo_list = np.deg2rad(stlo_list)
    radius = model.model.radius_of_planet
    station_radius_list = [ radius-stdp for stdp in stdp_list]
    #arrowprops = dict(arrowstyle='-|>,head_length=0.8,' 'head_width=0.5', color='#C95241', lw=1.5)
    arrowprops = dict(arrowstyle='-|>,head_length=0.8,' 'head_width=0.5', color='#E5E5E5', lw=1, alpha= 0.8)
    for stlo, station_radius in zip(stlo_list, station_radius_list):
        #arrowprops = dict(arrowstyle='-|>,head_length=0.8,' 'head_width=0.5', color='#C95241', lw=1.5)
        arrowprops = dict(arrowstyle='-|>,head_length=0.8,' 'head_width=0.5', color='#E5E5E5', lw=1, alpha= 0.8)
        ax.annotate('',
                    xy=(stlo, station_radius),
                    xycoords='data',
                    xytext=(stlo, station_radius + radius * 0.02),
                    textcoords='data',
                    arrowprops=arrowprops,
                    clip_on=False)
        #arrowprops = dict(arrowstyle='-|>,head_length=1.0,' 'head_width=0.6', color='0.3', lw=1.5, fill=False)
        arrowprops = dict(arrowstyle='-|>,head_length=0.8,' 'head_width=0.5', color='k', lw=0.8, fill=False)
        ax.annotate('',
                    xy=(stlo, station_radius),
                    xycoords='data',
                    xytext=(stlo, station_radius + radius * 0.01),
                    textcoords='data',
                    arrowprops=arrowprops,
                    clip_on=False)
def plotEq(ax, model, eqdp_list, eqlo_list):
    """
    Plot Earthquake.
    ax: axes to plot;
    model: taup.model object;
    eqdp_list, eqlo_list: list of longitude (degree) and depth for several earthquakes.
    """
    radius = model.model.radius_of_planet
    eq_radius = [radius-eqdp for eqdp in eqdp_list]
    eqlo_list = np.deg2rad(eqlo_list)
    #ax.plot(eqlo_list, eq_radius,
    #                marker='*', color='#FEF200', markersize=14, zorder=10, linestyle='None',
    #                markeredgewidth=0.5, markeredgecolor='0.3',
    #                clip_on=False)
    ax.plot(eqlo_list, eq_radius,
                marker='*', color='0.9', markersize=16, zorder=10, linestyle='None',
                markeredgewidth=1, markeredgecolor='k', alpha= 0.9,
                clip_on=False)
    #if self.d_eqcor:
    #    for dep, lon in zip(source_depth, source_lon):
    #        ax.annotate('%3.0f' % np.rad2deg(lon),
    #                xy=(lon, dep),
    #                xycoords='data',
    #                xytext=(lon, dep+1000),
    #                textcoords='data',
    #                clip_on=False,
    #                horizontalalignment = 'center',
    #                verticalalignment = 'center',
    #                multialignment = 'center')
    #    pass

class cc_feature_time:
    local_mod_ak135 = taup.TauPyModel('ak135')
    def __init__(self):
        pass
    def internel_get_feature(self, phase1, phase2, stlo1, stlo2, fig=None, ax1=None, ax2=None, evdp_km=0.0, show=True):
        """
        Given fixed two receivers, so that phase1 must arrive at stlo1, and phase2 at stlo2
        """
        vol1 = get_dist_rayparam_time(phase1, evdp_km=evdp_km)
        vol2 = get_dist_rayparam_time(phase2, evdp_km=evdp_km)
        d1, rp1, t1 = vol1['distance'], vol1['rp'], vol1['time'] # 0<=d1<=180
        d2, rp2, t2 = vol2['distance'], vol2['rp'], vol2['time'] # 0<=d2<=180
        ####
        evlo1 = stlo1-d1 # dist = stlo-evlo, and hence evlo=stlo-dist
        evlo2 = stlo2-d2 
        pts = [(x,y) for x, y in zip(evlo1, rp1) ] 
        qts = [(x,y) for x, y in zip(evlo2, rp2) ]
        ####
        ak_ev = list()
        ak_rp = list()
        flag, ip, iq, ak_ev, ak_rp= get_intersection_between_list_points(pts, qts)
        #print(ak_ev, ak_rp)
        time1 = list()
        time2 = list()
        cc_time = list()
        if flag:
            #print('flag:', flag, ak_ev, ak_rp)
            #ak_ev.append( single_evlo%360.0 )
            #ak_rp.append( single_rc )
            for (e, rp) in zip(ak_ev, ak_rp ):
                arrs = mod_ak135.get_travel_times(evdp_km, stlo1-e, [phase1] )
                itmp = np.argmin( [abs(it.ray_param_sec_degree-rp) for it in arrs] )
                arr1 = arrs[itmp]

                arrs = mod_ak135.get_travel_times(evdp_km, stlo2-e, [phase2] )
                itmp = np.argmin( [abs(it.ray_param_sec_degree-rp) for it in arrs] )
                arr2 = arrs[itmp]
                
                dt = arr1.time - arr2.time
                time1.append(arr1.time)
                time2.append(arr2.time)
                cc_time.append(dt)
        if show:
            if ax1 == None or ax2 == None:
                pass
            else:
                ########
                radius = mod_ak135.model.radius_of_planet
                for e, rp in zip(ak_ev, ak_rp):
                    #
                    for idx, (phase, stlo, clr) in enumerate(zip([phase1, phase2], [stlo1, stlo2], ['C0', 'C1'])):
                        arrs = mod_ak135.get_ray_paths(evdp_km, stlo-e, [phase] )
                        itmp = np.argmin( [abs(it.ray_param_sec_degree-rp) for it in arrs] )
                        arr = arrs[itmp]
                        evlo = stlo-arr.distance
                        #print(phase, stlo, arr.distance, e, evlo)
                        xs= arr.path['dist'] #+ np.deg2rad(evlo)
                        ys = radius-arr.path['depth']
                        ax1.plot(xs, ys, color=clr, label=phase)
                        plotEq(ax1, mod_ak135, [evdp_km],  [np.rad2deg(xs[0] )] )
                        plotStation(ax1, mod_ak135, [0.0], [np.rad2deg(xs[-1])] )
                if ak_ev:
                    ax1.axis('on')
                    ax1.legend(loc='lower left')
                    plotPrettyEarth(ax1, mod_ak135, True)
                else:
                    ax1.axis('off')
                    #fig.delaxes(ax1)
                ########   
                for e, rp in zip(ak_ev, ak_rp):
                    ax2.plot([0.0, 180.0], [rp, rp], 'k--', alpha= 0.6)
                    ax2.text(5.0, rp, '%.2f s/$\degree$' % rp, bbox=dict(facecolor='white', alpha=0.95, edgecolor='white'), verticalalignment='center' )
                    for stlo, clr in zip([stlo1, stlo2], ['C0', 'C1'] ):
                        dist = (stlo-e)%360
                        dist = dist if dist<=180 else 360-dist
                        ax2.plot([dist, dist], [0, rp], 'k--', alpha= 0.6)
                        ax2.plot([dist], [rp], 'o', zorder = 12, color='k')
                        ax2.text(dist, 0.0, '%.1f$\degree$' % dist)
                if ak_ev: 
                    #ax2.set_title('dist1: %.2f $\degree$ dist2: %.2f $\degree$' % () )
                    ax2.axis('on')
                    ax2.plot(d1, rp1, 'C0', label=phase1)
                    ax2.plot(d2, rp2, 'C1', label=phase2)
                    ax2.set_xlabel('Distance ($\degree$)')
                    ax2.set_ylabel('Ray paramter (s/$\degree$)')
                    ax2.legend(loc='upper right' )
                    ax2.set_xlim([0, 180])
                    ax2.set_ylim(bottom=0)
                else:
                    ax2.axis('off')
                    #fig.delaxes(ax2)
        #
        return ak_ev, ak_rp, time1, time2, cc_time
    def get_feature_time(self, phase1, phase2, inter_rcv_distance_deg, evdp_km=0.0, max_interation= 100, show=True):
        """
        Given two phases observed at two receivers, the distance between
        those two receivers, return the theoretical feature time for the
        cross-term of those two phases under the same slowness condition.

        evlo_min, evlo_max: event longitude range for search.

        """
        fig = plt.figure(figsize=(11, 8) )
        ax1 = plt.subplot(2, 2, 1, projection='polar')
        ax2 = plt.subplot(2, 2, 2)
        ax3 = plt.subplot(2, 2, 3, projection='polar')
        ax4 = plt.subplot(2, 2, 4)
        ###############################################################################################
        ak_ev, ak_rp, cc_time = list(), list(), list()
        time1, time2 = list(), list()
        dist1, dist2 = list(), list()
        # search for evlo that present the same slowness for two body waves
        #1. d2-d1 = inter_dist
        stlo1, stlo2 = 0.0, inter_rcv_distance_deg
        tmp1, tmp2, tmp4, tmp5, tmp3 = self.internel_get_feature(phase1, phase2, stlo1, stlo2, fig, ax1, ax2, evdp_km, show)
        ak_ev.extend(tmp1)
        ak_rp.extend(tmp2)
        cc_time.extend(tmp3)
        time1.extend(tmp4)
        time2.extend(tmp5)
        dist1.extend( [(stlo1-it)%360 for it in tmp1] )
        dist2.extend( [(stlo2-it)%360 for it in tmp1] )
        #print('tmp', tmp1, tmp2, tmp4, tmp5, tmp3)
        #2. d1-d2 = inter_dist
        stlo1, stlo2 = inter_rcv_distance_deg, 0.0
        if len(ak_ev) <= 0:
            ax3.axis('off')
            ax4.axis('off')
            ax3, ax4 = ax1, ax2
        tmp1, tmp2, tmp4, tmp5, tmp3 = self.internel_get_feature(phase1, phase2, stlo1, stlo2, fig, ax3, ax4, evdp_km, show)
        ak_ev.extend(tmp1)
        ak_rp.extend(tmp2)
        cc_time.extend(tmp3)
        time1.extend(tmp4)
        time2.extend(tmp5)
        dist1.extend( [stlo1-it for it in tmp1] )
        dist2.extend( [stlo2-it for it in tmp1] )
        #print('tmp', tmp1, tmp2, tmp4, tmp5, tmp3)
        ###############################################################################################
        if show:
            plt.tight_layout()
            plt.show()
        #print(dist1, dist2, ak_rp, time1, time2, cc_time)
        return dist1, dist2, ak_rp, time1, time2, cc_time

if __name__ == "__main__":
    # -F PcS-PcP -D 10 -E -30/0
    ph1, ph2 = None, None
    dist = None
    #ev1, ev2 = 0.0, 180.0
    show=False
    options, remainder = getopt.getopt(sys.argv[1:], 'F:D:E:S' )
    for opt, arg in options:
        if opt in ('-F'):
            ph1, ph2 = arg.split('-') 
        elif opt in ('-D'):
            dist = float(arg)
        elif opt in ('-S'):
            show=True
        else:
            print('invalid options: %s' % (opt) )
    if ph1 == None or ph2 == None or dist == None:
        print('e.g.: %s -F PcS-PcP -D 10 -E -30/0' % (sys.argv[0]) )
        sys.exit(-1)
    get_dist_rayparam_time('PcP')
    app = cc_feature_time()
    dist1, dist2, rp, time1, time2, cc_time = app.get_feature_time(ph1, ph2, dist, show=show)
    #
    print('#cross-term inter-dist(deg)  dist1(deg)  dist2(deg)  ray_param(s/deg)  t1(s)  t2(s)  cc_time(s)')
    for x in zip(dist1, dist2, rp, time1, time2, cc_time):
        print(ph1+'-'+ph2, dist, ' '.join([ '%f' % it for it in x]))
    #print('%f %f %f %f'% (dist, v1, v2, v3) )